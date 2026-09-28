from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
import unicodedata
from typing import Literal

from app.schemas.bizscore import (
    BenchmarkReference,
    BizScoreDimension,
    BizScoreResult,
    IndustryClassification,
    IndustryGroup,
    PeerBenchmarkRankCounts,
    PeerBenchmarkSample,
    PeerRelativePositionScore,
    RegistrationStatusScore,
)
from app.schemas.company import BusinessItem


COMPANY_AGE_KEY = "company_age"
COMPANY_AGE_LABEL = "成立年資"
COMPANY_AGE_MAX_SCORE = 20

REGISTRATION_STATUS_KEY = "registration_status"
REGISTRATION_STATUS_LABEL = "登記狀態"
REGISTRATION_STATUS_MAX_SCORE = 25

REGISTERED_CAPITAL_KEY = "registered_capital_scale"
REGISTERED_CAPITAL_LABEL = "登記資本規模"
REGISTERED_CAPITAL_MAX_SCORE = 20

REGISTRATION_CHANGE_RECENCY_KEY = "registration_change_recency"
REGISTRATION_CHANGE_RECENCY_LABEL = "登記異動距今"
REGISTRATION_CHANGE_RECENCY_MAX_SCORE = 15

PEER_RELATIVE_POSITION_KEY = "peer_relative_position"
PEER_RELATIVE_POSITION_LABEL = "同業相對位置"
PEER_RELATIVE_POSITION_MAX_SCORE = 20
MINIMUM_PEER_SAMPLE_SIZE = 30

INDUSTRY_MAPPING_VERSION = "gcis-business-category-v1"
BUSINESS_ITEM_CODE_PATTERN = re.compile(r"^(?:[A-Z]\d{6}|[A-Z]{2}\d{5})$")

INDUSTRY_CATEGORY_NAMES: dict[str, str] = {
    "A": "農、林、漁、牧業",
    "B": "礦業及土石採取業",
    "C": "製造業",
    "D": "水電燃氣業",
    "E": "營造及工程業",
    "F": "零售、批發及餐飲業",
    "G": "運輸、倉儲及通信業",
    "H": "金融、保險及不動產業",
    "I": "專業、科學及技術服務業",
    "J": "文化、運動、休閒及其他服務業",
    "Z": "其他未分類業",
}


StatusOutcome = Literal[
    "scored",
    "review_required",
    "non_current",
    "insufficient_data",
]


@dataclass(frozen=True, slots=True)
class _RegistrationStatusRule:
    description: str
    score: int | None
    total_score_eligible: bool
    status_cap: int | None
    outcome: StatusOutcome


_REGISTRATION_STATUS_RULES: dict[str, _RegistrationStatusRule] = {
    "01": _RegistrationStatusRule("核准設立", 25, True, 100, "scored"),
    "02": _RegistrationStatusRule(
        "核准設立，但已命令解散", 0, False, None, "review_required"
    ),
    "03": _RegistrationStatusRule("重整", 0, False, None, "review_required"),
    "04": _RegistrationStatusRule("解散", 0, False, None, "non_current"),
    "05": _RegistrationStatusRule("撤銷", 0, False, None, "non_current"),
    "06": _RegistrationStatusRule("破產", 0, False, None, "non_current"),
    "07": _RegistrationStatusRule("合併解散", 0, False, None, "non_current"),
    "08": _RegistrationStatusRule("撤回登記", 0, False, None, "non_current"),
    "09": _RegistrationStatusRule("廢止", 0, False, None, "non_current"),
    "10": _RegistrationStatusRule("廢止認許", 0, False, None, "non_current"),
    "11": _RegistrationStatusRule("解散已清算完結", 0, False, None, "non_current"),
    "12": _RegistrationStatusRule("撤銷已清算完結", 0, False, None, "non_current"),
    "13": _RegistrationStatusRule("廢止已清算完結", 0, False, None, "non_current"),
    "14": _RegistrationStatusRule(
        "撤回登記已清算完結", 0, False, None, "non_current"
    ),
    "15": _RegistrationStatusRule(
        "撤銷登記已清算完結", 0, False, None, "non_current"
    ),
    "16": _RegistrationStatusRule(
        "廢止登記已清算完結", 0, False, None, "non_current"
    ),
    "17": _RegistrationStatusRule("撤銷登記", 0, False, None, "non_current"),
    "18": _RegistrationStatusRule("分割解散", 0, False, None, "non_current"),
    "19": _RegistrationStatusRule("終止破產", 0, False, None, "non_current"),
    "20": _RegistrationStatusRule("中止破產", 0, False, None, "non_current"),
    "21": _RegistrationStatusRule("塗銷破產", 0, False, None, "non_current"),
    "22": _RegistrationStatusRule(
        "破產程序終結(終止)", 0, False, None, "non_current"
    ),
    "23": _RegistrationStatusRule(
        "破產程序終結(終止)清算中", 0, False, None, "review_required"
    ),
    "24": _RegistrationStatusRule("破產已清算完結", 0, False, None, "non_current"),
    "25": _RegistrationStatusRule("接管", 0, False, None, "review_required"),
    "26": _RegistrationStatusRule("撤銷無需清算", 0, False, None, "non_current"),
    "27": _RegistrationStatusRule("撤銷許可", 0, False, None, "non_current"),
    "28": _RegistrationStatusRule("廢止許可", 0, False, None, "non_current"),
    "29": _RegistrationStatusRule(
        "撤銷許可已清算完結", 0, False, None, "non_current"
    ),
    "30": _RegistrationStatusRule(
        "廢止許可已清算完結", 0, False, None, "non_current"
    ),
    "31": _RegistrationStatusRule("清理", 0, False, None, "review_required"),
    "32": _RegistrationStatusRule("撤銷公司設立", 0, False, None, "non_current"),
    "33": _RegistrationStatusRule("清理完結", 0, False, None, "non_current"),
}


STATUS_CONFLICT_DESCRIPTION = "來源登記狀態不一致，待確認"


def normalized_status_description(value: str) -> str:
    """Ignore display punctuation/spacing, never meaningful qualifiers."""
    return "".join(char for char in unicodedata.normalize("NFKC", value) if char.isalnum())


def registration_status_conflicts(code: str | None, description: str | None) -> bool:
    rule = _REGISTRATION_STATUS_RULES.get(code or "")
    return bool(
        rule and description and description.strip()
        and normalized_status_description(description) != normalized_status_description(rule.description)
    )


def score_company_age(company_age_years: object) -> BizScoreDimension:
    """Score the observable registration history on the BizScore v1 scale."""

    age, invalid_reason = _normalize_non_negative_number(company_age_years)
    if age is None:
        warning = (
            "company_age_years is unavailable; company_age was not scored."
            if invalid_reason == "missing"
            else "company_age_years must be a finite non-negative number; "
            "company_age was not scored."
        )
        return BizScoreDimension(
            key=COMPANY_AGE_KEY,
            label=COMPANY_AGE_LABEL,
            max_score=COMPANY_AGE_MAX_SCORE,
            available=False,
            warnings=[warning],
        )

    if age >= Decimal("10"):
        score, rule = 20, "age>=10"
    elif age >= Decimal("5"):
        score, rule = 16, "5<=age<10"
    elif age >= Decimal("3"):
        score, rule = 12, "3<=age<5"
    elif age >= Decimal("1"):
        score, rule = 8, "1<=age<3"
    else:
        score, rule = 4, "0<=age<1"

    return BizScoreDimension(
        key=COMPANY_AGE_KEY,
        label=COMPANY_AGE_LABEL,
        score=score,
        max_score=COMPANY_AGE_MAX_SCORE,
        available=True,
        evidence=[
            f"company_age_years={_format_decimal(age)}",
            f"rule={rule}",
        ],
    )


def score_registration_status(
    status_code: object,
    status_description: object = None,
) -> RegistrationStatusScore:
    """Score a GCIS status code and return its total-score eligibility rules."""

    code = _normalize_status_text(status_code)
    description = _normalize_status_text(status_description)
    evidence = [f"status.code={code}"] if code is not None else []
    if description is not None:
        evidence.append(f"status.description={description}")

    rule = _REGISTRATION_STATUS_RULES.get(code or "")
    conflict = registration_status_conflicts(code, description)
    if rule is None or conflict:
        warning = (
            "GCIS_STATUS_CONFLICT: status code and description disagree; "
            "registration_status and total BizScore were not scored."
            if conflict else
            "status.code is unavailable; registration_status was not scored."
            if code is None
            else f"status.code={code} is not recognized by BizScore v1; "
            "registration_status was not scored."
        )
        return RegistrationStatusScore(
            dimension=BizScoreDimension(
                key=REGISTRATION_STATUS_KEY,
                label=REGISTRATION_STATUS_LABEL,
                max_score=REGISTRATION_STATUS_MAX_SCORE,
                available=False,
                evidence=evidence,
                warnings=[warning],
            ),
            total_score_eligible=False,
            outcome="insufficient_data",
        )

    warnings: list[str] = []
    if rule.outcome == "review_required":
        warnings.append(
            "Registration status requires enhanced review; total BizScore must not be "
            "produced."
        )
    elif rule.outcome == "non_current":
        warnings.append(
            "Registration status is non-current; total BizScore must not be produced."
        )
    evidence.extend(
        [
            f"official_status={rule.description}",
            f"rule=status_{code}",
        ]
    )
    return RegistrationStatusScore(
        dimension=BizScoreDimension(
            key=REGISTRATION_STATUS_KEY,
            label=REGISTRATION_STATUS_LABEL,
            score=rule.score,
            max_score=REGISTRATION_STATUS_MAX_SCORE,
            available=rule.score is not None,
            evidence=evidence,
            warnings=warnings,
        ),
        total_score_eligible=rule.total_score_eligible,
        status_cap=rule.status_cap,
        outcome=rule.outcome,
    )


def score_registered_capital(registered_capital: object) -> BizScoreDimension:
    """Score GCIS registered capital on the BizScore v1 TWD bands."""

    capital, invalid_reason = _normalize_non_negative_integer(registered_capital)
    if capital is None:
        warning = (
            "capital.registered is unavailable; registered_capital_scale was not scored."
            if invalid_reason == "missing"
            else "capital.registered must be a finite non-negative integer; "
            "registered_capital_scale was not scored."
        )
        return BizScoreDimension(
            key=REGISTERED_CAPITAL_KEY,
            label=REGISTERED_CAPITAL_LABEL,
            max_score=REGISTERED_CAPITAL_MAX_SCORE,
            available=False,
            warnings=[warning],
        )

    if capital >= 20_000_000:
        score, rule = 20, "capital>=20000000"
    elif capital >= 5_000_000:
        score, rule = 16, "5000000<=capital<20000000"
    elif capital >= 1_000_000:
        score, rule = 12, "1000000<=capital<5000000"
    elif capital >= 500_000:
        score, rule = 8, "500000<=capital<1000000"
    elif capital > 0:
        score, rule = 4, "0<capital<500000"
    else:
        score, rule = 0, "capital=0"

    return BizScoreDimension(
        key=REGISTERED_CAPITAL_KEY,
        label=REGISTERED_CAPITAL_LABEL,
        score=score,
        max_score=REGISTERED_CAPITAL_MAX_SCORE,
        available=True,
        evidence=[
            f"capital.registered={capital}",
            "capital.currency=TWD",
            f"rule={rule}",
        ],
    )


def score_registration_change_recency(
    last_changed_at: object,
    *,
    as_of: object,
    established_at: object = None,
) -> BizScoreDimension:
    """Score elapsed days since the latest approved registration change."""

    reference_date = _normalize_date(as_of)
    if reference_date is None:
        return _unavailable_change_recency(
            "as_of must be a date or datetime; registration_change_recency "
            "was not scored."
        )

    change_date = _normalize_date(last_changed_at)
    if change_date is None:
        warning = (
            "last_changed_at is unavailable; registration_change_recency "
            "was not scored."
            if last_changed_at is None
            else "last_changed_at must be a date or datetime; "
            "registration_change_recency was not scored."
        )
        return _unavailable_change_recency(warning)

    setup_date: date | None = None
    if established_at is not None:
        setup_date = _normalize_date(established_at)
        if setup_date is None:
            return _unavailable_change_recency(
                "established_at must be a date or datetime when provided; "
                "registration_change_recency was not scored."
            )
        if setup_date > reference_date:
            return _unavailable_change_recency(
                "established_at is later than as_of; registration_change_recency "
                "was not scored."
            )
        if change_date < setup_date:
            return _unavailable_change_recency(
                "last_changed_at is earlier than established_at; "
                "registration_change_recency was not scored."
            )

    if change_date > reference_date:
        return _unavailable_change_recency(
            "last_changed_at is later than as_of; registration_change_recency "
            "was not scored."
        )

    days_since_last_change = (reference_date - change_date).days
    if days_since_last_change >= 1_825:
        score, rule = 15, "days>=1825"
    elif days_since_last_change >= 1_095:
        score, rule = 12, "1095<=days<1825"
    elif days_since_last_change >= 365:
        score, rule = 9, "365<=days<1095"
    elif days_since_last_change >= 90:
        score, rule = 6, "90<=days<365"
    else:
        score, rule = 3, "0<=days<90"

    evidence = [
        f"last_changed_at={change_date.isoformat()}",
        f"as_of={reference_date.isoformat()}",
        f"days_since_last_change={days_since_last_change}",
        f"rule={rule}",
    ]
    if setup_date is not None:
        evidence.insert(1, f"established_at={setup_date.isoformat()}")

    return BizScoreDimension(
        key=REGISTRATION_CHANGE_RECENCY_KEY,
        label=REGISTRATION_CHANGE_RECENCY_LABEL,
        score=score,
        max_score=REGISTRATION_CHANGE_RECENCY_MAX_SCORE,
        available=True,
        evidence=evidence,
    )


def classify_industry(
    business_items: list[BusinessItem] | None,
) -> IndustryClassification:
    """Map canonical business items to versioned GCIS industry categories."""

    if not business_items:
        return IndustryClassification(
            version=INDUSTRY_MAPPING_VERSION,
            available=False,
            warnings=[
                "business_items is empty; an industry comparison group could not be assigned."
            ],
        )

    groups: list[IndustryGroup] = []
    seen_categories: set[str] = set()
    excluded_codes: list[str] = []
    unmapped_codes: list[str] = []

    for item in sorted(business_items, key=_business_item_sort_key):
        code = item.code.strip().upper()
        if code == "ZZ99999":
            excluded_codes.append(code)
            continue
        if BUSINESS_ITEM_CODE_PATTERN.fullmatch(code) is None:
            unmapped_codes.append(code or "<blank>")
            continue

        category_code = code[0]
        category_name = INDUSTRY_CATEGORY_NAMES.get(category_code)
        if category_name is None:
            unmapped_codes.append(code)
            continue
        if category_code in seen_categories:
            continue

        seen_categories.add(category_code)
        groups.append(
            IndustryGroup(
                category_code=category_code,
                category_name=category_name,
                source_sequence=item.sequence,
                source_business_item_code=code,
                source_business_item_name=item.name,
                benchmark_eligible=category_code != "Z",
            )
        )

    primary_group = next(
        (group for group in groups if group.benchmark_eligible),
        None,
    )
    warnings: list[str] = []
    if excluded_codes:
        warnings.append(
            "Generic business item ZZ99999 was excluded from industry classification."
        )
    if unmapped_codes:
        warnings.append(
            "Some business item codes could not be mapped by "
            f"{INDUSTRY_MAPPING_VERSION}."
        )
    if primary_group is None:
        warnings.append(
            "No benchmark-eligible industry comparison group could be assigned."
        )

    return IndustryClassification(
        version=INDUSTRY_MAPPING_VERSION,
        available=primary_group is not None,
        primary_group=primary_group,
        groups=groups,
        excluded_business_item_codes=excluded_codes,
        unmapped_business_item_codes=unmapped_codes,
        warnings=warnings,
    )


def score_peer_relative_position(
    company_age_years: object,
    registered_capital: object,
    peer_samples: list[PeerBenchmarkSample] | None,
    *,
    industry_code: object,
    benchmark_version: object,
) -> PeerRelativePositionScore:
    """Score age and capital midrank percentiles within one benchmark group."""

    normalized_industry_code = _normalize_industry_code(industry_code)
    normalized_benchmark_version = _normalize_status_text(benchmark_version)
    valid_samples, excluded_sample_count = _normalize_peer_samples(peer_samples)
    sample_size = len(valid_samples)

    if normalized_industry_code is None:
        return _unavailable_peer_relative_position(
            "industry_code must be a benchmark-eligible GCIS category A-J; "
            "peer_relative_position was not scored.",
            industry_code=None,
            benchmark_version=normalized_benchmark_version,
            sample_size=sample_size,
            excluded_sample_count=excluded_sample_count,
        )
    if normalized_benchmark_version is None:
        return _unavailable_peer_relative_position(
            "benchmark_version is unavailable; peer_relative_position was not scored.",
            industry_code=normalized_industry_code,
            benchmark_version=None,
            sample_size=sample_size,
            excluded_sample_count=excluded_sample_count,
        )

    company_age_median = _median_decimal([age for age, _ in valid_samples])
    registered_capital_median = _median_decimal(
        [Decimal(capital) for _, capital in valid_samples]
    )
    target_age, age_invalid_reason = _normalize_non_negative_number(
        company_age_years
    )
    target_capital, capital_invalid_reason = _normalize_non_negative_integer(
        registered_capital
    )
    if target_age is None or target_capital is None:
        missing_fields: list[str] = []
        invalid_fields: list[str] = []
        if target_age is None:
            (missing_fields if age_invalid_reason == "missing" else invalid_fields).append(
                "company_age_years"
            )
        if target_capital is None:
            (
                missing_fields
                if capital_invalid_reason == "missing"
                else invalid_fields
            ).append("capital.registered")

        reasons: list[str] = []
        if missing_fields:
            reasons.append(f"missing target fields: {', '.join(missing_fields)}")
        if invalid_fields:
            reasons.append(f"invalid target fields: {', '.join(invalid_fields)}")
        return _unavailable_peer_relative_position(
            "; ".join(reasons) + "; peer_relative_position was not scored.",
            industry_code=normalized_industry_code,
            benchmark_version=normalized_benchmark_version,
            sample_size=sample_size,
            excluded_sample_count=excluded_sample_count,
            company_age_median=company_age_median,
            registered_capital_median=registered_capital_median,
        )

    rank_counts = PeerBenchmarkRankCounts(
        sample_size=sample_size,
        age_lower_count=sum(age < target_age for age, _ in valid_samples),
        age_equal_count=sum(age == target_age for age, _ in valid_samples),
        capital_lower_count=sum(capital < target_capital for _, capital in valid_samples),
        capital_equal_count=sum(capital == target_capital for _, capital in valid_samples),
    )
    return score_peer_relative_position_from_rank_counts(
        target_age,
        target_capital,
        rank_counts,
        industry_code=normalized_industry_code,
        benchmark_version=normalized_benchmark_version,
        excluded_sample_count=excluded_sample_count,
        company_age_median=company_age_median,
        registered_capital_median=registered_capital_median,
    )


def score_peer_relative_position_from_rank_counts(
    company_age_years: object,
    registered_capital: object,
    rank_counts: PeerBenchmarkRankCounts,
    *,
    industry_code: object,
    benchmark_version: object,
    excluded_sample_count: int = 0,
    company_age_median: Decimal | None = None,
    registered_capital_median: Decimal | None = None,
) -> PeerRelativePositionScore:
    """Score peer position from SQLite rank counts for large formal snapshots."""

    normalized_industry_code = _normalize_industry_code(industry_code)
    normalized_benchmark_version = _normalize_status_text(benchmark_version)
    sample_size = rank_counts.sample_size
    if normalized_industry_code is None:
        return _unavailable_peer_relative_position(
            "industry_code must be a benchmark-eligible GCIS category A-J; "
            "peer_relative_position was not scored.",
            industry_code=None,
            benchmark_version=normalized_benchmark_version,
            sample_size=sample_size,
            excluded_sample_count=excluded_sample_count,
            company_age_median=company_age_median,
            registered_capital_median=registered_capital_median,
        )
    if normalized_benchmark_version is None:
        return _unavailable_peer_relative_position(
            "benchmark_version is unavailable; peer_relative_position was not scored.",
            industry_code=normalized_industry_code,
            benchmark_version=None,
            sample_size=sample_size,
            excluded_sample_count=excluded_sample_count,
            company_age_median=company_age_median,
            registered_capital_median=registered_capital_median,
        )

    target_age, age_invalid_reason = _normalize_non_negative_number(company_age_years)
    target_capital, capital_invalid_reason = _normalize_non_negative_integer(
        registered_capital
    )
    if target_age is None or target_capital is None:
        missing_fields: list[str] = []
        invalid_fields: list[str] = []
        if target_age is None:
            (missing_fields if age_invalid_reason == "missing" else invalid_fields).append(
                "company_age_years"
            )
        if target_capital is None:
            (
                missing_fields if capital_invalid_reason == "missing" else invalid_fields
            ).append("capital.registered")
        reasons: list[str] = []
        if missing_fields:
            reasons.append(f"missing target fields: {', '.join(missing_fields)}")
        if invalid_fields:
            reasons.append(f"invalid target fields: {', '.join(invalid_fields)}")
        return _unavailable_peer_relative_position(
            "; ".join(reasons) + "; peer_relative_position was not scored.",
            industry_code=normalized_industry_code,
            benchmark_version=normalized_benchmark_version,
            sample_size=sample_size,
            excluded_sample_count=excluded_sample_count,
            company_age_median=company_age_median,
            registered_capital_median=registered_capital_median,
        )

    excluded_warning = _excluded_peer_sample_warning(excluded_sample_count)
    if sample_size < MINIMUM_PEER_SAMPLE_SIZE:
        warnings = [
            f"peer sample size {sample_size} is below the required minimum "
            f"of {MINIMUM_PEER_SAMPLE_SIZE}; peer_relative_position was not scored."
        ]
        if excluded_warning is not None:
            warnings.append(excluded_warning)
        return _unavailable_peer_relative_position(
            *warnings,
            industry_code=normalized_industry_code,
            benchmark_version=normalized_benchmark_version,
            sample_size=sample_size,
            excluded_sample_count=excluded_sample_count,
            company_age_median=company_age_median,
            registered_capital_median=registered_capital_median,
        )

    age_percentile = _midrank_percentile_from_counts(
        rank_counts.age_lower_count,
        rank_counts.age_equal_count,
        sample_size,
    )
    capital_percentile = _midrank_percentile_from_counts(
        rank_counts.capital_lower_count,
        rank_counts.capital_equal_count,
        sample_size,
    )
    peer_index = (age_percentile + capital_percentile) / Decimal(2)

    if peer_index >= Decimal(80):
        score, rule = 20, "peer_index>=80"
    elif peer_index >= Decimal(60):
        score, rule = 16, "60<=peer_index<80"
    elif peer_index >= Decimal(40):
        score, rule = 12, "40<=peer_index<60"
    elif peer_index >= Decimal(20):
        score, rule = 8, "20<=peer_index<40"
    else:
        score, rule = 4, "0<=peer_index<20"

    warnings = []
    if excluded_warning is not None:
        warnings.append(excluded_warning)
    dimension = BizScoreDimension(
        key=PEER_RELATIVE_POSITION_KEY,
        label=PEER_RELATIVE_POSITION_LABEL,
        score=score,
        max_score=PEER_RELATIVE_POSITION_MAX_SCORE,
        available=True,
        evidence=[
            f"industry.code={normalized_industry_code}",
            f"benchmark.version={normalized_benchmark_version}",
            f"peer.sample_size={sample_size}",
            f"peer.company_age_median={_format_optional_decimal(company_age_median)}",
            "peer.registered_capital_median="
            f"{_format_optional_decimal(registered_capital_median)}",
            f"age_pr={_format_decimal(age_percentile)}",
            f"capital_pr={_format_decimal(capital_percentile)}",
            f"peer_index={_format_decimal(peer_index)}",
            f"rule={rule}",
        ],
        warnings=warnings,
    )
    return PeerRelativePositionScore(
        dimension=dimension,
        industry_code=normalized_industry_code,
        benchmark_version=normalized_benchmark_version,
        sample_size=sample_size,
        excluded_sample_count=excluded_sample_count,
        minimum_sample_size=MINIMUM_PEER_SAMPLE_SIZE,
        age_percentile=float(age_percentile),
        capital_percentile=float(capital_percentile),
        peer_index=float(peer_index),
        company_age_median=_optional_decimal_float(company_age_median),
        registered_capital_median=_optional_decimal_float(
            registered_capital_median
        ),
    )


def calculate_bizscore_total(
    registration_status: RegistrationStatusScore,
    company_age: BizScoreDimension,
    registered_capital: BizScoreDimension,
    registration_change_recency: BizScoreDimension,
    peer_relative_position: PeerRelativePositionScore,
    *,
    as_of: date,
    industry: IndustryClassification,
    benchmark: BenchmarkReference,
    input_partial: bool = False,
    input_warnings: list[str] | None = None,
) -> BizScoreResult:
    """Combine all five v1 dimensions using coverage and status guardrails."""

    dimensions = [
        registration_status.dimension,
        company_age,
        registered_capital,
        registration_change_recency,
        peer_relative_position.dimension,
    ]
    available_dimensions = [
        dimension
        for dimension in dimensions
        if dimension.available and dimension.score is not None
    ]
    available_weight = sum(dimension.max_score for dimension in available_dimensions)
    earned_points = sum(dimension.score or 0 for dimension in available_dimensions)
    coverage = available_weight / 100
    missing_dimensions = [
        dimension.key
        for dimension in dimensions
        if not dimension.available or dimension.score is None
    ]

    warnings = _deduplicate_strings(
        [
            *(input_warnings or []),
            *industry.warnings,
            *(
                warning
                for dimension in dimensions
                for warning in dimension.warnings
            ),
        ]
    )

    score: int | None = None
    if registration_status.total_score_eligible and coverage >= 0.8:
        normalized_score = int(
            (
                Decimal(earned_points)
                / Decimal(available_weight)
                * Decimal(100)
            ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
        status_cap = registration_status.status_cap or 100
        score = min(normalized_score, status_cap)

    provisional = score is not None and (
        coverage < 1 or input_partial or bool(input_warnings)
    )
    return BizScoreResult(
        score=score,
        band=_bizscore_band(score),
        coverage=coverage,
        provisional=provisional,
        as_of=as_of,
        status_cap=registration_status.status_cap,
        dimensions=dimensions,
        missing_dimensions=missing_dimensions,
        industry=industry,
        benchmark=benchmark,
        peer_benchmark=peer_relative_position,
        warnings=warnings,
    )


def _normalize_non_negative_number(
    value: object,
) -> tuple[Decimal | None, Literal["missing", "invalid"] | None]:
    if value is None:
        return None, "missing"
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None, "invalid"

    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None, "invalid"
    if not number.is_finite() or number < 0:
        return None, "invalid"
    return number, None


def _normalize_non_negative_integer(
    value: object,
) -> tuple[int | None, Literal["missing", "invalid"] | None]:
    number, invalid_reason = _normalize_non_negative_number(value)
    if number is None:
        return None, invalid_reason
    if number != number.to_integral_value():
        return None, "invalid"
    return int(number), None


def _normalize_status_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_industry_code(value: object) -> str | None:
    code = _normalize_status_text(value)
    if code is None:
        return None
    normalized = code.upper()
    if normalized == "Z" or normalized not in INDUSTRY_CATEGORY_NAMES:
        return None
    return normalized


def _normalize_peer_samples(
    peer_samples: list[PeerBenchmarkSample] | None,
) -> tuple[list[tuple[Decimal, int]], int]:
    valid_samples: list[tuple[Decimal, int]] = []
    excluded_sample_count = 0
    for sample in peer_samples or []:
        if not isinstance(sample, PeerBenchmarkSample):
            excluded_sample_count += 1
            continue
        age, _ = _normalize_non_negative_number(sample.company_age_years)
        capital, _ = _normalize_non_negative_integer(sample.registered_capital)
        if age is None or capital is None:
            excluded_sample_count += 1
            continue
        valid_samples.append((age, capital))
    return valid_samples, excluded_sample_count


def _midrank_percentile(target: Decimal, samples: list[Decimal]) -> Decimal:
    lower_count = sum(sample < target for sample in samples)
    equal_count = sum(sample == target for sample in samples)
    return _midrank_percentile_from_counts(
        lower_count,
        equal_count,
        len(samples),
    )


def _midrank_percentile_from_counts(
    lower_count: int,
    equal_count: int,
    sample_size: int,
) -> Decimal:
    return (
        Decimal(100)
        * (Decimal(lower_count) + Decimal("0.5") * Decimal(equal_count))
        / Decimal(sample_size)
    )


def _median_decimal(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _bizscore_band(score: int | None) -> str | None:
    if score is None:
        return None
    if score >= 80:
        return "公開資料呈現較穩健"
    if score >= 60:
        return "公開資料呈現一般"
    if score >= 40:
        return "建議進一步查核"
    return "需優先查核"


def _deduplicate_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _excluded_peer_sample_warning(excluded_sample_count: int) -> str | None:
    if excluded_sample_count == 0:
        return None
    return (
        f"{excluded_sample_count} incomplete benchmark sample(s) were excluded "
        "before percentile calculation."
    )


def _normalize_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _unavailable_change_recency(warning: str) -> BizScoreDimension:
    return BizScoreDimension(
        key=REGISTRATION_CHANGE_RECENCY_KEY,
        label=REGISTRATION_CHANGE_RECENCY_LABEL,
        max_score=REGISTRATION_CHANGE_RECENCY_MAX_SCORE,
        available=False,
        warnings=[warning],
    )


def _unavailable_peer_relative_position(
    *warnings: str,
    industry_code: str | None,
    benchmark_version: str | None,
    sample_size: int,
    excluded_sample_count: int,
    company_age_median: Decimal | None = None,
    registered_capital_median: Decimal | None = None,
) -> PeerRelativePositionScore:
    return PeerRelativePositionScore(
        dimension=BizScoreDimension(
            key=PEER_RELATIVE_POSITION_KEY,
            label=PEER_RELATIVE_POSITION_LABEL,
            max_score=PEER_RELATIVE_POSITION_MAX_SCORE,
            available=False,
            warnings=list(warnings),
        ),
        industry_code=industry_code,
        benchmark_version=benchmark_version,
        sample_size=sample_size,
        excluded_sample_count=excluded_sample_count,
        minimum_sample_size=MINIMUM_PEER_SAMPLE_SIZE,
        company_age_median=_optional_decimal_float(company_age_median),
        registered_capital_median=_optional_decimal_float(
            registered_capital_median
        ),
    )


def _format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized, "f")


def _format_optional_decimal(value: Decimal | None) -> str:
    return "unavailable" if value is None else _format_decimal(value)


def _optional_decimal_float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _business_item_sort_key(item: BusinessItem) -> tuple[int, int | str]:
    sequence = item.sequence.strip()
    if sequence.isdigit():
        return (0, int(sequence))
    return (1, sequence)
