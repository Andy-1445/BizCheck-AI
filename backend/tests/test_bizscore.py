import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from app.services.bizscore import (
    INDUSTRY_CATEGORY_NAMES,
    classify_industry,
    score_company_age,
    score_registered_capital,
    score_registration_change_recency,
    score_registration_status,
)
from app.schemas.company import BusinessItem


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = WORKSPACE_ROOT / "samples" / "gcis"


@pytest.mark.parametrize(
    ("age", "expected_score"),
    [
        (0, 4),
        (0.999, 4),
        (1, 8),
        (2.999, 8),
        (3, 12),
        (4.999, 12),
        (5, 16),
        (9.999, 16),
        (10, 20),
        (47.1, 20),
        (Decimal("10.0"), 20),
    ],
)
def test_company_age_score_boundaries(age: object, expected_score: int) -> None:
    result = score_company_age(age)

    assert result.key == "company_age"
    assert result.label == "成立年資"
    assert result.score == expected_score
    assert result.max_score == 20
    assert result.available is True
    assert result.evidence[0].startswith("company_age_years=")
    assert result.warnings == []


@pytest.mark.parametrize(
    "age",
    [None, -0.1, float("nan"), float("inf"), True, "5", object()],
)
def test_company_age_invalid_values_are_not_scored(age: object) -> None:
    result = score_company_age(age)

    assert result.score is None
    assert result.max_score == 20
    assert result.available is False
    assert result.evidence == []
    assert len(result.warnings) == 1


@pytest.mark.parametrize(
    (
        "status_code",
        "expected_score",
        "expected_eligible",
        "expected_cap",
        "expected_outcome",
    ),
    [
        ("01", 25, True, 100, "scored"),
        ("02", 0, False, None, "review_required"),
        ("03", 0, False, None, "review_required"),
        ("04", 0, False, None, "non_current"),
        ("05", 0, False, None, "non_current"),
        ("06", 0, False, None, "non_current"),
        ("07", 0, False, None, "non_current"),
        ("08", 0, False, None, "non_current"),
        ("09", 0, False, None, "non_current"),
        ("10", 0, False, None, "non_current"),
        ("11", 0, False, None, "non_current"),
        ("12", 0, False, None, "non_current"),
        ("13", 0, False, None, "non_current"),
        ("14", 0, False, None, "non_current"),
        ("15", 0, False, None, "non_current"),
        ("16", 0, False, None, "non_current"),
        ("17", 0, False, None, "non_current"),
        ("18", 0, False, None, "non_current"),
        ("19", 0, False, None, "non_current"),
        ("20", 0, False, None, "non_current"),
        ("21", 0, False, None, "non_current"),
        ("22", 0, False, None, "non_current"),
        ("23", 0, False, None, "review_required"),
        ("24", 0, False, None, "non_current"),
        ("25", 0, False, None, "review_required"),
        ("26", 0, False, None, "non_current"),
        ("27", 0, False, None, "non_current"),
        ("28", 0, False, None, "non_current"),
        ("29", 0, False, None, "non_current"),
        ("30", 0, False, None, "non_current"),
        ("31", 0, False, None, "review_required"),
        ("32", 0, False, None, "non_current"),
        ("33", 0, False, None, "non_current"),
    ],
)
def test_registration_status_rules(
    status_code: str,
    expected_score: int | None,
    expected_eligible: bool,
    expected_cap: int | None,
    expected_outcome: str,
) -> None:
    result = score_registration_status(status_code)

    assert result.dimension.key == "registration_status"
    assert result.dimension.label == "登記狀態"
    assert result.dimension.score == expected_score
    assert result.dimension.max_score == 25
    assert result.dimension.available is (expected_score is not None)
    assert result.total_score_eligible is expected_eligible
    assert result.status_cap == expected_cap
    assert result.outcome == expected_outcome
    assert f"status.code={status_code}" in result.dimension.evidence


@pytest.mark.parametrize("status_code", [None, "", "  ", "00", "34", "1", 1, True])
def test_unknown_or_missing_status_is_not_scored(status_code: object) -> None:
    result = score_registration_status(status_code, "核准設立")

    assert result.dimension.score is None
    assert result.dimension.available is False
    assert result.total_score_eligible is False
    assert result.status_cap is None
    assert result.outcome == "insufficient_data"
    assert len(result.dimension.warnings) == 1


def test_status_description_never_replaces_a_missing_code() -> None:
    result = score_registration_status(None, "核准設立")

    assert result.dimension.score is None
    assert "status.description=核准設立" in result.dimension.evidence
    assert result.outcome == "insufficient_data"


def test_status_text_is_trimmed_and_evidence_is_serializable() -> None:
    result = score_registration_status(" 01 ", " 核准設立 ")

    assert result.dimension.score == 25
    assert result.dimension.evidence[:2] == [
        "status.code=01",
        "status.description=核准設立",
    ]
    assert result.model_dump(mode="json")["status_cap"] == 100


@pytest.mark.parametrize("status_code", ["02", "03", "23", "25", "31"])
def test_review_required_statuses_block_total_score(status_code: str) -> None:
    result = score_registration_status(status_code)

    assert result.total_score_eligible is False
    assert result.dimension.score == 0
    assert result.outcome == "review_required"
    assert "enhanced review" in result.dimension.warnings[0]


@pytest.mark.parametrize("status_code", ["04", "09", "24", "33"])
def test_non_current_statuses_block_total_score(status_code: str) -> None:
    result = score_registration_status(status_code)

    assert result.total_score_eligible is False
    assert result.dimension.score == 0
    assert result.outcome == "non_current"
    assert "must not be produced" in result.dimension.warnings[0]


def test_company_status_02_uses_official_company_meaning_not_business_meaning() -> None:
    result = score_registration_status("02", "核准設立，但已命令解散")

    assert "official_status=核准設立，但已命令解散" in result.dimension.evidence
    assert "official_status=停業" not in result.dimension.evidence
    assert result.total_score_eligible is False
    assert result.status_cap is None


@pytest.mark.parametrize(
    ("status_code", "official_description"),
    [
        ("01", "核准設立"),
        ("02", "核准設立，但已命令解散"),
        ("03", "重整"),
        ("04", "解散"),
        ("05", "撤銷"),
        ("06", "破產"),
        ("07", "合併解散"),
        ("08", "撤回登記"),
        ("09", "廢止"),
        ("10", "廢止認許"),
        ("11", "解散已清算完結"),
        ("12", "撤銷已清算完結"),
        ("13", "廢止已清算完結"),
        ("14", "撤回登記已清算完結"),
        ("15", "撤銷登記已清算完結"),
        ("16", "廢止登記已清算完結"),
        ("17", "撤銷登記"),
        ("18", "分割解散"),
        ("19", "終止破產"),
        ("20", "中止破產"),
        ("21", "塗銷破產"),
        ("22", "破產程序終結(終止)"),
        ("23", "破產程序終結(終止)清算中"),
        ("24", "破產已清算完結"),
        ("25", "接管"),
        ("26", "撤銷無需清算"),
        ("27", "撤銷許可"),
        ("28", "廢止許可"),
        ("29", "撤銷許可已清算完結"),
        ("30", "廢止許可已清算完結"),
        ("31", "清理"),
        ("32", "撤銷公司設立"),
        ("33", "清理完結"),
    ],
)
def test_all_company_status_codes_match_gcis_table_7(
    status_code: str,
    official_description: str,
) -> None:
    result = score_registration_status(status_code)

    assert f"official_status={official_description}" in result.dimension.evidence


@pytest.mark.parametrize(
    ("capital", "expected_score"),
    [
        (0, 0),
        (1, 4),
        (499_999, 4),
        (500_000, 8),
        (999_999, 8),
        (1_000_000, 12),
        (4_999_999, 12),
        (5_000_000, 16),
        (19_999_999, 16),
        (20_000_000, 20),
        (40_000_000_000, 20),
        (Decimal("5000000"), 16),
    ],
)
def test_registered_capital_score_boundaries(
    capital: object,
    expected_score: int,
) -> None:
    result = score_registered_capital(capital)

    assert result.key == "registered_capital_scale"
    assert result.label == "登記資本規模"
    assert result.score == expected_score
    assert result.max_score == 20
    assert result.available is True
    assert result.evidence[0].startswith("capital.registered=")
    assert result.evidence[1] == "capital.currency=TWD"
    assert result.warnings == []


@pytest.mark.parametrize(
    "capital",
    [None, -1, 1.5, float("nan"), float("inf"), True, "1000000", object()],
)
def test_invalid_registered_capital_is_not_scored(capital: object) -> None:
    result = score_registered_capital(capital)

    assert result.score is None
    assert result.available is False
    assert result.evidence == []
    assert len(result.warnings) == 1


@pytest.mark.parametrize(
    ("category_code", "category_name", "business_item_code"),
    [
        ("A", "農、林、漁、牧業", "A101011"),
        ("B", "礦業及土石採取業", "B102010"),
        ("C", "製造業", "C114010"),
        ("D", "水電燃氣業", "D101060"),
        ("E", "營造及工程業", "E101011"),
        ("F", "零售、批發及餐飲業", "F113050"),
        ("G", "運輸、倉儲及通信業", "G101021"),
        ("H", "金融、保險及不動產業", "H201010"),
        ("I", "專業、科學及技術服務業", "I301010"),
        ("J", "文化、運動、休閒及其他服務業", "J101010"),
    ],
)
def test_official_gcis_categories_map_to_benchmark_groups(
    category_code: str,
    category_name: str,
    business_item_code: str,
) -> None:
    result = classify_industry(
        [
            BusinessItem(
                sequence="0001",
                code=business_item_code,
                name="測試營業項目",
            )
        ]
    )

    assert INDUSTRY_CATEGORY_NAMES[category_code] == category_name
    assert result.version == "gcis-business-category-v1"
    assert result.available is True
    assert result.primary_group is not None
    assert result.primary_group.category_code == category_code
    assert result.primary_group.category_name == category_name
    assert result.primary_group.source_business_item_code == business_item_code
    assert result.primary_group.benchmark_eligible is True


def test_industry_classification_sorts_items_and_keeps_unique_groups() -> None:
    result = classify_industry(
        [
            BusinessItem(sequence="0010", code="I301010", name="資訊軟體服務業"),
            BusinessItem(sequence="0002", code="F213030", name="電腦零售業"),
            BusinessItem(sequence="0001", code="F113050", name="電腦批發業"),
            BusinessItem(sequence="0003", code="C114010", name="食品製造業"),
        ]
    )

    assert result.available is True
    assert result.primary_group is not None
    assert result.primary_group.category_code == "F"
    assert result.primary_group.source_business_item_code == "F113050"
    assert [group.category_code for group in result.groups] == ["F", "C", "I"]


def test_generic_and_unknown_items_are_not_used_as_primary_group() -> None:
    result = classify_industry(
        [
            BusinessItem(sequence="0001", code="ZZ99999", name="概括項目"),
            BusinessItem(sequence="0002", code="K123456", name="未知大類"),
            BusinessItem(sequence="0003", code="bad-code", name="格式錯誤"),
        ]
    )

    assert result.available is False
    assert result.primary_group is None
    assert result.groups == []
    assert result.excluded_business_item_codes == ["ZZ99999"]
    assert result.unmapped_business_item_codes == ["K123456", "BAD-CODE"]
    assert len(result.warnings) == 3


def test_z_category_is_visible_but_not_benchmark_eligible() -> None:
    result = classify_industry(
        [BusinessItem(sequence="0001", code="Z123456", name="其他未分類業")]
    )

    assert result.available is False
    assert result.primary_group is None
    assert len(result.groups) == 1
    assert result.groups[0].category_code == "Z"
    assert result.groups[0].benchmark_eligible is False


@pytest.mark.parametrize("business_items", [None, []])
def test_empty_business_items_cannot_assign_industry(
    business_items: list[BusinessItem] | None,
) -> None:
    result = classify_industry(business_items)

    assert result.available is False
    assert result.primary_group is None
    assert result.groups == []
    assert len(result.warnings) == 1


def test_recorded_company_sample_maps_to_traceable_industry_groups() -> None:
    with (SAMPLE_DIR / "company-normalized-20828393.json").open(
        encoding="utf-8"
    ) as sample_file:
        payload = json.load(sample_file)
    items = [
        BusinessItem.model_validate(item)
        for item in payload["data"]["business_items"]
    ]

    result = classify_industry(items)

    assert result.available is True
    assert result.primary_group is not None
    assert result.primary_group.category_code == "F"
    assert result.primary_group.source_business_item_code == "F113050"
    assert [group.category_code for group in result.groups] == [
        "F",
        "I",
        "J",
        "C",
        "E",
    ]
    assert result.excluded_business_item_codes == ["ZZ99999"]
    assert result.unmapped_business_item_codes == []


@pytest.mark.parametrize(
    ("days_since_last_change", "expected_score"),
    [
        (0, 3),
        (89, 3),
        (90, 6),
        (364, 6),
        (365, 9),
        (1_094, 9),
        (1_095, 12),
        (1_824, 12),
        (1_825, 15),
        (5_000, 15),
    ],
)
def test_registration_change_recency_boundaries(
    days_since_last_change: int,
    expected_score: int,
) -> None:
    as_of = date(2026, 8, 22)
    result = score_registration_change_recency(
        as_of - timedelta(days=days_since_last_change),
        as_of=as_of,
    )

    assert result.key == "registration_change_recency"
    assert result.label == "登記異動距今"
    assert result.score == expected_score
    assert result.max_score == 15
    assert result.available is True
    assert f"days_since_last_change={days_since_last_change}" in result.evidence
    assert result.warnings == []


def test_change_recency_accepts_datetime_and_keeps_cross_field_evidence() -> None:
    result = score_registration_change_recency(
        datetime(2026, 6, 22, 15, 30, tzinfo=timezone(timedelta(hours=8))),
        as_of=datetime(2026, 8, 20, 9, 0, tzinfo=timezone(timedelta(hours=8))),
        established_at=date(1979, 7, 18),
    )

    assert result.score == 3
    assert result.evidence == [
        "last_changed_at=2026-06-22",
        "established_at=1979-07-18",
        "as_of=2026-08-20",
        "days_since_last_change=59",
        "rule=0<=days<90",
    ]


@pytest.mark.parametrize("last_changed_at", [None, "2026-06-22", 20260622, object()])
def test_missing_or_invalid_change_date_is_not_scored(
    last_changed_at: object,
) -> None:
    result = score_registration_change_recency(
        last_changed_at,
        as_of=date(2026, 8, 22),
    )

    assert result.score is None
    assert result.available is False
    assert result.evidence == []
    assert len(result.warnings) == 1


@pytest.mark.parametrize("as_of", [None, "2026-08-22", 20260822, object()])
def test_missing_or_invalid_reference_date_is_not_scored(as_of: object) -> None:
    result = score_registration_change_recency(
        date(2026, 6, 22),
        as_of=as_of,
    )

    assert result.score is None
    assert result.available is False
    assert "as_of must be" in result.warnings[0]


def test_future_change_date_is_not_scored() -> None:
    result = score_registration_change_recency(
        date(2026, 8, 23),
        as_of=date(2026, 8, 22),
    )

    assert result.score is None
    assert result.available is False
    assert "later than as_of" in result.warnings[0]


def test_change_date_earlier_than_establishment_is_not_scored() -> None:
    result = score_registration_change_recency(
        date(2019, 12, 31),
        as_of=date(2026, 8, 22),
        established_at=date(2020, 1, 1),
    )

    assert result.score is None
    assert result.available is False
    assert "earlier than established_at" in result.warnings[0]


def test_future_establishment_date_is_not_scored() -> None:
    result = score_registration_change_recency(
        date(2026, 8, 22),
        as_of=date(2026, 8, 22),
        established_at=date(2026, 8, 23),
    )

    assert result.score is None
    assert result.available is False
    assert "established_at is later than as_of" in result.warnings[0]


def test_change_recency_is_reproducible_for_the_same_reference_date() -> None:
    first = score_registration_change_recency(
        date(2025, 8, 22),
        as_of=date(2026, 8, 22),
    )
    second = score_registration_change_recency(
        date(2025, 8, 22),
        as_of=date(2026, 8, 22),
    )

    assert first == second
    assert first.score == 9
