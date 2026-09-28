import json
from datetime import date
from pathlib import Path

import pytest

from app.services.company_normalizer import (
    CompanyNormalizationError,
    calculate_company_age_years,
    normalize_company_data,
    normalize_company_search_item,
    normalize_roc_date,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = WORKSPACE_ROOT / "samples" / "gcis"


def load_first_sample(filename: str) -> dict[str, object]:
    with (SAMPLE_DIR / filename).open(encoding="utf-8") as sample_file:
        payload = json.load(sample_file)
    assert isinstance(payload, list)
    assert isinstance(payload[0], dict)
    return payload[0]


def test_recorded_samples_are_merged_into_canonical_company_data() -> None:
    basic = load_first_sample("company-basic-20828393.json")
    business = load_first_sample("company-business-20828393.json")

    result = normalize_company_data(
        basic,
        business,
        as_of=date(2026, 8, 20),
    )

    assert result.partial is False
    assert result.warnings == ()
    assert result.data.tax_id == "20828393"
    assert result.data.name == "宏碁股份有限公司"
    assert result.data.status.code == "01"
    assert result.data.status.description == "核准設立"
    assert result.data.capital.registered == 40_000_000_000
    assert result.data.capital.paid_in == 30_478_538_280
    assert result.data.capital.currency == "TWD"
    assert result.data.established_at == date(1979, 7, 18)
    assert result.data.company_age_years == 47.1
    assert result.data.last_changed_at == date(2026, 6, 22)
    assert result.data.responsible_name == "陳俊聖"
    assert result.data.address == "臺北市松山區民福里復興北路369號7樓之5"
    assert result.data.registration_authority == "商業發展署"
    assert len(result.data.business_items) == 22
    assert result.data.business_items[0].sequence == "0001"
    assert result.data.business_items[-1].code == "ZZ99999"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("0680718", date(1979, 7, 18)),
        (" 1150622 ", date(2026, 6, 22)),
        (680718, date(1979, 7, 18)),
        ("", None),
        (None, None),
        ("1150230", None),
        ("20260622", None),
    ],
)
def test_normalize_roc_date(raw_value: object, expected: date | None) -> None:
    assert normalize_roc_date(raw_value) == expected


def test_company_age_handles_missing_and_future_dates() -> None:
    assert calculate_company_age_years(None, as_of=date(2026, 8, 20)) is None
    assert (
        calculate_company_age_years(
            date(2026, 8, 21),
            as_of=date(2026, 8, 20),
        )
        is None
    )


def test_missing_business_record_returns_partial_basic_data() -> None:
    basic = load_first_sample("company-basic-20828393.json")

    result = normalize_company_data(
        basic,
        None,
        as_of=date(2026, 8, 20),
    )

    assert result.partial is True
    assert result.data.status.code is None
    assert result.data.business_items == []
    assert result.warnings == (
        "GCIS A3 business record is unavailable; response is partial.",
    )


def test_a1_identity_wins_and_mismatches_are_reported() -> None:
    basic = load_first_sample("company-basic-20828393.json")
    business = load_first_sample("company-business-20828393.json")
    business["Business_Accounting_NO"] = "12345678"
    business["Company_Name"] = "不同公司"

    result = normalize_company_data(basic, business)

    assert result.data.tax_id == "20828393"
    assert result.data.name == "宏碁股份有限公司"
    assert result.warnings[:2] == (
        "A1 and A3 Business_Accounting_NO values do not match; A1 was used.",
        "A1 and A3 Company_Name values do not match; A1 was used.",
    )
    assert result.partial is True
    assert result.data.status.code is None
    assert result.data.business_items == []


@pytest.mark.parametrize("raw_items", [None, {}, "not-an-array"])
def test_invalid_business_item_collection_becomes_empty_with_warning(
    raw_items: object,
) -> None:
    basic = load_first_sample("company-basic-20828393.json")
    business = load_first_sample("company-business-20828393.json")
    business["Cmp_Business"] = raw_items

    result = normalize_company_data(basic, business)

    assert result.data.business_items == []
    assert "Cmp_Business is not an array" in result.warnings[-1]


def test_bad_optional_values_become_none_and_generate_warnings() -> None:
    basic = load_first_sample("company-basic-20828393.json")
    business = load_first_sample("company-business-20828393.json")
    basic["Capital_Stock_Amount"] = "not-a-number"
    basic["Paid_In_Capital_Amount"] = -1
    basic["Company_Setup_Date"] = "1150230"
    basic["Responsible_Name"] = 123
    basic["Company_Location"] = "  "

    result = normalize_company_data(basic, business)

    assert result.data.capital.registered is None
    assert result.data.capital.paid_in is None
    assert result.data.established_at is None
    assert result.data.company_age_years is None
    assert result.data.responsible_name is None
    assert result.data.address is None
    assert len(result.warnings) == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Business_Accounting_NO", "123"),
        ("Company_Name", "  "),
    ],
)
def test_invalid_required_fields_raise_normalization_error(
    field: str,
    value: object,
) -> None:
    basic = load_first_sample("company-basic-20828393.json")
    business = load_first_sample("company-business-20828393.json")
    basic[field] = value

    with pytest.raises(CompanyNormalizationError):
        normalize_company_data(basic, business)


def test_search_item_is_trimmed_and_normalized() -> None:
    record = {
        "Business_Accounting_NO": " 20828393 ",
        "Company_Name": " 宏碁股份有限公司 ",
        "Company_Status": " 01 ",
        "Company_Status_Desc": " 核准設立 ",
        "Capital_Stock_Amount": "40,000,000,000",
        "Company_Setup_Date": "0680718",
        "Change_Of_Approval_Data": "1150622",
    }

    result = normalize_company_search_item(record)

    assert result.warnings == ()
    assert result.data.tax_id == "20828393"
    assert result.data.name == "宏碁股份有限公司"
    assert result.data.status.code == "01"
    assert result.data.registered_capital == 40_000_000_000
    assert result.data.established_at == date(1979, 7, 18)
    assert result.data.last_changed_at == date(2026, 6, 22)


def test_business_items_are_sorted_by_sequence_without_losing_zeroes() -> None:
    basic = load_first_sample("company-basic-20828393.json")
    business = load_first_sample("company-business-20828393.json")
    business["Cmp_Business"] = [
        {
            "Business_Seq_NO": "0010",
            "Business_Item": "B",
            "Business_Item_Desc": "第二項",
        },
        {
            "Business_Seq_NO": "0002",
            "Business_Item": "A",
            "Business_Item_Desc": "第一項",
        },
    ]

    result = normalize_company_data(basic, business)

    assert [item.sequence for item in result.data.business_items] == [
        "0002",
        "0010",
    ]
