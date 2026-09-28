from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Generic, TypeVar

from app.schemas.company import (
    BusinessItem,
    CompanyCapital,
    CompanyData,
    CompanySearchItem,
    CompanyStatus,
)
from app.services.bizscore import (
    STATUS_CONFLICT_DESCRIPTION,
    normalized_status_description,
    registration_status_conflicts,
)


JSONRecord = dict[str, Any]
ModelT = TypeVar("ModelT")

ROC_DATE_PATTERN = re.compile(r"^(?P<year>\d{3})(?P<month>\d{2})(?P<day>\d{2})$")
TAX_ID_PATTERN = re.compile(r"^\d{8}$")


class CompanyNormalizationError(ValueError):
    """A required GCIS field cannot be converted to the public API model."""


@dataclass(frozen=True, slots=True)
class NormalizationResult(Generic[ModelT]):
    data: ModelT
    warnings: tuple[str, ...] = ()
    partial: bool = False


def normalize_roc_date(value: object) -> date | None:
    """Convert a seven-digit Minguo date (YYYMMDD) to a Gregorian date."""

    if value is None:
        return None

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value < 0 or value > 9_999_999:
            return None
        normalized = f"{value:07d}"
    else:
        normalized = str(value).strip()
    if not normalized:
        return None

    match = ROC_DATE_PATTERN.fullmatch(normalized)
    if match is None:
        return None

    year = int(match.group("year")) + 1911
    month = int(match.group("month"))
    day = int(match.group("day"))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def calculate_company_age_years(
    established_at: date | None,
    *,
    as_of: date | None = None,
) -> float | None:
    """Return age in years rounded to one decimal, or None for bad dates."""

    if established_at is None:
        return None

    reference_date = as_of or date.today()
    if established_at > reference_date:
        return None

    return round((reference_date - established_at).days / 365.2425, 1)


def normalize_company_data(
    basic_record: JSONRecord,
    business_record: JSONRecord | None,
    *,
    as_of: date | None = None,
) -> NormalizationResult[CompanyData]:
    """Merge GCIS A1 and A3 records into the canonical company model.

    A1 is authoritative for identity, capital, address and dates. A3 is
    authoritative for the status code and business items. When A3 is not
    available, A1 is still returned and the result is marked as partial.
    """

    warnings: list[str] = []
    tax_id = _required_tax_id(basic_record.get("Business_Accounting_NO"))
    name = _required_string(basic_record.get("Company_Name"), "Company_Name")

    if business_record is not None:
        _warn_if_mismatch(
            warnings,
            field="Business_Accounting_NO",
            primary=tax_id,
            secondary=_clean_string(business_record.get("Business_Accounting_NO")),
        )
        _warn_if_mismatch(
            warnings,
            field="Company_Name",
            primary=name,
            secondary=_clean_string(business_record.get("Company_Name")),
        )
        business_tax_id = _clean_string(business_record.get("Business_Accounting_NO"))
        if business_tax_id is not None and business_tax_id != tax_id:
            warnings.append("GCIS_A3_IDENTITY_MISMATCH: unrelated A3 record was ignored.")
            business_record = None

    status_code = _optional_string(
        business_record,
        "Company_Status",
        warnings,
        warn_if_missing=business_record is not None,
    )
    if status_code is None:
        status_code = _optional_string(
            basic_record,
            "Company_Status",
            warnings,
            warn_if_missing=False,
        )

    status_description = _optional_string(
        basic_record,
        "Company_Status_Desc",
        warnings,
    )
    if status_description is None:
        status_description = _optional_string(
            business_record,
            "Company_Status_Desc",
            warnings,
            warn_if_missing=business_record is not None,
        )

    # Endpoints may update at different times. Do not combine a current code
    # with a contradictory description and then award a score.
    sources = [basic_record] + ([business_record] if business_record is not None else [])
    codes = [_clean_string(item.get("Company_Status")) for item in sources]
    descriptions = [_clean_string(item.get("Company_Status_Desc")) for item in sources]
    status_conflict = (
        len({value for value in codes if value is not None}) > 1
        or len({normalized_status_description(value) for value in descriptions if value is not None}) > 1
        or registration_status_conflicts(status_code, status_description)
    )
    if status_conflict:
        warnings.append(
            "GCIS_STATUS_CONFLICT: A1/A3 status values disagree; "
            f"A1.code={codes[0]!r}, A1.description={descriptions[0]!r}; "
            f"A3.code={(codes[1] if len(codes) > 1 else None)!r}, "
            f"A3.description={(descriptions[1] if len(descriptions) > 1 else None)!r}. "
            "status.code was withheld pending source verification."
        )
        status_code = None
        status_description = STATUS_CONFLICT_DESCRIPTION

    established_at = _optional_roc_date(
        basic_record,
        "Company_Setup_Date",
        warnings,
    )
    last_changed_at = _optional_roc_date(
        basic_record,
        "Change_Of_Approval_Data",
        warnings,
    )
    company_age_years = calculate_company_age_years(
        established_at,
        as_of=as_of,
    )
    if established_at is not None and company_age_years is None:
        warnings.append("Company_Setup_Date is later than the reference date.")

    business_items = _normalize_business_items(business_record, warnings)
    partial = business_record is None or status_conflict
    if business_record is None:
        warnings.append("GCIS A3 business record is unavailable; response is partial.")

    data = CompanyData(
        tax_id=tax_id,
        name=name,
        status=CompanyStatus(
            code=status_code,
            description=status_description,
        ),
        capital=CompanyCapital(
            registered=_optional_non_negative_integer(
                basic_record,
                "Capital_Stock_Amount",
                warnings,
            ),
            paid_in=_optional_non_negative_integer(
                basic_record,
                "Paid_In_Capital_Amount",
                warnings,
            ),
        ),
        established_at=established_at,
        company_age_years=company_age_years,
        last_changed_at=last_changed_at,
        responsible_name=_optional_string(
            basic_record,
            "Responsible_Name",
            warnings,
        ),
        address=_optional_string(
            basic_record,
            "Company_Location",
            warnings,
        ),
        registration_authority=_optional_string(
            basic_record,
            "Register_Organization_Desc",
            warnings,
        ),
        business_items=business_items,
    )
    return NormalizationResult(
        data=data,
        warnings=tuple(warnings),
        partial=partial,
    )


def normalize_company_search_item(
    record: JSONRecord,
) -> NormalizationResult[CompanySearchItem]:
    """Normalize one GCIS keyword-search record into its compact API model."""

    warnings: list[str] = []
    data = CompanySearchItem(
        tax_id=_required_tax_id(record.get("Business_Accounting_NO")),
        name=_required_string(record.get("Company_Name"), "Company_Name"),
        status=CompanyStatus(
            code=_optional_string(record, "Company_Status", warnings),
            description=_optional_string(
                record,
                "Company_Status_Desc",
                warnings,
            ),
        ),
        registered_capital=_optional_non_negative_integer(
            record,
            "Capital_Stock_Amount",
            warnings,
        ),
        established_at=_optional_roc_date(
            record,
            "Company_Setup_Date",
            warnings,
        ),
        last_changed_at=_optional_roc_date(
            record,
            "Change_Of_Approval_Data",
            warnings,
        ),
    )
    return NormalizationResult(data=data, warnings=tuple(warnings))


def _required_tax_id(value: object) -> str:
    normalized = _clean_string(value)
    if normalized is None or TAX_ID_PATTERN.fullmatch(normalized) is None:
        raise CompanyNormalizationError(
            "Business_Accounting_NO must contain exactly 8 digits."
        )
    return normalized


def _required_string(value: object, field: str) -> str:
    normalized = _clean_string(value)
    if normalized is None:
        raise CompanyNormalizationError(f"{field} is required and cannot be blank.")
    return normalized


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _optional_string(
    record: JSONRecord | None,
    field: str,
    warnings: list[str],
    *,
    warn_if_missing: bool = True,
) -> str | None:
    if record is None:
        return None
    if field not in record:
        if warn_if_missing:
            warnings.append(f"{field} is missing.")
        return None

    value = record[field]
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        warnings.append(f"{field} must be a string; value was ignored.")
        return None
    return value.strip() or None


def _optional_roc_date(
    record: JSONRecord,
    field: str,
    warnings: list[str],
) -> date | None:
    if field not in record:
        warnings.append(f"{field} is missing.")
        return None

    value = record[field]
    normalized = normalize_roc_date(value)
    if normalized is None and value is not None and str(value).strip():
        warnings.append(f"{field} is not a valid seven-digit Minguo date.")
    return normalized


def _optional_non_negative_integer(
    record: JSONRecord,
    field: str,
    warnings: list[str],
) -> int | None:
    if field not in record:
        warnings.append(f"{field} is missing.")
        return None

    value = record[field]
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        warnings.append(f"{field} must be a non-negative integer; value was ignored.")
        return None

    try:
        number = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        warnings.append(f"{field} must be a non-negative integer; value was ignored.")
        return None

    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        warnings.append(f"{field} must be a non-negative integer; value was ignored.")
        return None
    return int(number)


def _normalize_business_items(
    business_record: JSONRecord | None,
    warnings: list[str],
) -> list[BusinessItem]:
    if business_record is None:
        return []

    if "Cmp_Business" not in business_record:
        warnings.append("Cmp_Business is missing; business_items is empty.")
        return []

    raw_items = business_record["Cmp_Business"]
    if not isinstance(raw_items, list):
        warnings.append("Cmp_Business is not an array; business_items is empty.")
        return []

    normalized_items: list[BusinessItem] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            warnings.append(f"Cmp_Business[{index}] is not an object; item was ignored.")
            continue

        sequence = _clean_string(raw_item.get("Business_Seq_NO"))
        code = _clean_string(raw_item.get("Business_Item"))
        if sequence is None or code is None:
            warnings.append(
                f"Cmp_Business[{index}] is missing sequence or code; item was ignored."
            )
            continue

        name = _clean_string(raw_item.get("Business_Item_Desc"))
        normalized_items.append(
            BusinessItem(sequence=sequence, code=code, name=name)
        )

    normalized_items.sort(key=_business_item_sort_key)
    return normalized_items


def _business_item_sort_key(item: BusinessItem) -> tuple[int, int | str]:
    if item.sequence.isdigit():
        return (0, int(item.sequence))
    return (1, item.sequence)


def _warn_if_mismatch(
    warnings: list[str],
    *,
    field: str,
    primary: str,
    secondary: str | None,
) -> None:
    if secondary is not None and primary != secondary:
        warnings.append(f"A1 and A3 {field} values do not match; A1 was used.")
