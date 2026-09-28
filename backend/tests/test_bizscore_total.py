from datetime import date

import pytest

from app.schemas.bizscore import (
    BenchmarkReference,
    BizScoreDimension,
    IndustryClassification,
    PeerRelativePositionScore,
    RegistrationStatusScore,
)
from app.services.bizscore import calculate_bizscore_total


AS_OF = date(2026, 8, 1)


def _dimension(key: str, score: int | None, max_score: int) -> BizScoreDimension:
    return BizScoreDimension(
        key=key,
        label=key,
        score=score,
        max_score=max_score,
        available=score is not None,
    )


def _result_for_total(
    total: int,
    *,
    status_cap: int = 100,
    status_eligible: bool = True,
    missing: set[str] | None = None,
    input_partial: bool = False,
    input_warnings: list[str] | None = None,
):
    missing = missing or set()
    weights = [
        ("registration_status", 25),
        ("company_age", 20),
        ("registered_capital_scale", 20),
        ("registration_change_recency", 15),
        ("peer_relative_position", 20),
    ]
    remaining = total
    dimensions: dict[str, BizScoreDimension] = {}
    for key, weight in weights:
        if key in missing:
            dimensions[key] = _dimension(key, None, weight)
            continue
        score = min(remaining, weight)
        remaining -= score
        dimensions[key] = _dimension(key, score, weight)
    assert remaining == 0

    registration = RegistrationStatusScore(
        dimension=dimensions["registration_status"],
        total_score_eligible=status_eligible,
        status_cap=status_cap if status_eligible else None,
        outcome="scored" if status_eligible else "non_current",
    )
    peer = PeerRelativePositionScore(
        dimension=dimensions["peer_relative_position"],
        industry_code="F",
        benchmark_version="benchmark-F",
        sample_size=30,
    )
    return calculate_bizscore_total(
        registration,
        dimensions["company_age"],
        dimensions["registered_capital_scale"],
        dimensions["registration_change_recency"],
        peer,
        as_of=AS_OF,
        industry=IndustryClassification(version="map-v1", available=False),
        benchmark=BenchmarkReference(
            catalog_version="catalog-v1",
            catalog_as_of=AS_OF,
            catalog_industry_mapping_version="membership-v1",
            classification_version="map-v1",
        ),
        input_partial=input_partial,
        input_warnings=input_warnings,
    )


@pytest.mark.parametrize(
    ("score", "expected_band"),
    [
        (0, "需優先查核"),
        (39, "需優先查核"),
        (40, "建議進一步查核"),
        (59, "建議進一步查核"),
        (60, "公開資料呈現一般"),
        (79, "公開資料呈現一般"),
        (80, "公開資料呈現較穩健"),
        (100, "公開資料呈現較穩健"),
    ],
)
def test_total_score_band_boundaries(score: int, expected_band: str) -> None:
    result = _result_for_total(score)

    assert result.score == score
    assert result.band == expected_band
    assert result.coverage == 1
    assert result.provisional is False


def test_exactly_eighty_percent_coverage_produces_a_provisional_score() -> None:
    result = _result_for_total(50, missing={"peer_relative_position"})

    assert result.coverage == 0.8
    assert result.score == 63
    assert result.provisional is True
    assert result.missing_dimensions == ["peer_relative_position"]


def test_less_than_eighty_percent_coverage_does_not_produce_a_score() -> None:
    result = _result_for_total(
        50,
        missing={"registration_change_recency", "peer_relative_position"},
    )

    assert result.coverage == 0.65
    assert result.score is None
    assert result.band is None
    assert result.provisional is False


def test_ineligible_registration_status_blocks_an_otherwise_complete_score() -> None:
    result = _result_for_total(75, status_eligible=False)

    assert result.coverage == 1
    assert result.score is None
    assert result.band is None


def test_status_cap_is_applied_after_normalization() -> None:
    result = _result_for_total(90, status_cap=49)

    assert result.score == 49
    assert result.band == "建議進一步查核"


def test_partial_input_marks_a_complete_score_as_provisional() -> None:
    result = _result_for_total(
        80,
        input_partial=True,
        input_warnings=["GCIS A3 was partial."],
    )

    assert result.score == 80
    assert result.provisional is True
    assert result.warnings == ["GCIS A3 was partial."]
