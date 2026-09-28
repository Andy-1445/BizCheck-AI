from decimal import Decimal

import pytest

from app.schemas.bizscore import PeerBenchmarkRankCounts, PeerBenchmarkSample
from app.services.bizscore import (
    score_peer_relative_position,
    score_peer_relative_position_from_rank_counts,
)


BENCHMARK_VERSION = "benchmark-2026-08-22-v1"


def _ranked_samples(count: int = 100) -> list[PeerBenchmarkSample]:
    return [
        PeerBenchmarkSample(
            company_age_years=rank,
            registered_capital=rank * 10,
        )
        for rank in range(1, count + 1)
    ]


@pytest.mark.parametrize(
    ("lower_count", "expected_score"),
    [
        (0, 4),
        (19, 4),
        (20, 8),
        (39, 8),
        (40, 12),
        (59, 12),
        (60, 16),
        (79, 16),
        (80, 20),
        (100, 20),
    ],
)
def test_peer_index_score_boundaries(
    lower_count: int,
    expected_score: int,
) -> None:
    result = score_peer_relative_position(
        Decimal(lower_count) + Decimal("0.5"),
        lower_count * 10 + 5,
        _ranked_samples(),
        industry_code="I",
        benchmark_version=BENCHMARK_VERSION,
    )

    assert result.dimension.key == "peer_relative_position"
    assert result.dimension.label == "同業相對位置"
    assert result.dimension.score == expected_score
    assert result.dimension.max_score == 20
    assert result.dimension.available is True
    assert result.age_percentile == lower_count
    assert result.capital_percentile == lower_count
    assert result.peer_index == lower_count
    assert result.sample_size == 100
    assert result.minimum_sample_size == 30


def test_midrank_percentile_gives_tied_values_half_of_equal_count() -> None:
    samples = [
        *[
            PeerBenchmarkSample(company_age_years=1, registered_capital=100)
            for _ in range(10)
        ],
        *[
            PeerBenchmarkSample(company_age_years=5, registered_capital=500)
            for _ in range(10)
        ],
        *[
            PeerBenchmarkSample(company_age_years=10, registered_capital=1_000)
            for _ in range(10)
        ],
    ]

    result = score_peer_relative_position(
        5,
        500,
        samples,
        industry_code="I",
        benchmark_version=BENCHMARK_VERSION,
    )

    assert result.age_percentile == 50
    assert result.capital_percentile == 50
    assert result.peer_index == 50
    assert result.company_age_median == 5
    assert result.registered_capital_median == 500
    assert result.dimension.score == 12
    assert result.dimension.evidence == [
        "industry.code=I",
        f"benchmark.version={BENCHMARK_VERSION}",
        "peer.sample_size=30",
        "peer.company_age_median=5",
        "peer.registered_capital_median=500",
        "age_pr=50",
        "capital_pr=50",
        "peer_index=50",
        "rule=40<=peer_index<60",
    ]


def test_rank_count_scorer_matches_in_memory_midrank_result() -> None:
    samples = _ranked_samples()
    in_memory = score_peer_relative_position(
        Decimal("20.5"),
        805,
        samples,
        industry_code="I",
        benchmark_version=BENCHMARK_VERSION,
    )
    from_counts = score_peer_relative_position_from_rank_counts(
        Decimal("20.5"),
        805,
        PeerBenchmarkRankCounts(
            sample_size=100,
            age_lower_count=20,
            age_equal_count=0,
            capital_lower_count=80,
            capital_equal_count=0,
        ),
        industry_code="I",
        benchmark_version=BENCHMARK_VERSION,
        company_age_median=Decimal("50.5"),
        registered_capital_median=Decimal("505"),
    )

    assert from_counts == in_memory


def test_even_peer_group_uses_average_of_two_middle_values() -> None:
    result = score_peer_relative_position(
        2,
        250,
        [
            PeerBenchmarkSample(company_age_years=1, registered_capital=100),
            PeerBenchmarkSample(company_age_years=2, registered_capital=200),
            PeerBenchmarkSample(company_age_years=3, registered_capital=300),
            PeerBenchmarkSample(company_age_years=4, registered_capital=400),
        ],
        industry_code="I",
        benchmark_version=BENCHMARK_VERSION,
    )

    assert result.company_age_median == 2.5
    assert result.registered_capital_median == 250


def test_peer_index_averages_age_and_capital_percentiles() -> None:
    result = score_peer_relative_position(
        Decimal("20.5"),
        805,
        _ranked_samples(),
        industry_code="I",
        benchmark_version=BENCHMARK_VERSION,
    )

    assert result.age_percentile == 20
    assert result.capital_percentile == 80
    assert result.peer_index == 50
    assert result.dimension.score == 12


def test_peer_sample_minimum_uses_valid_sample_count() -> None:
    result = score_peer_relative_position(
        10,
        100,
        _ranked_samples(29),
        industry_code="I",
        benchmark_version=BENCHMARK_VERSION,
    )

    assert result.dimension.score is None
    assert result.dimension.available is False
    assert result.sample_size == 29
    assert result.age_percentile is None
    assert "below the required minimum of 30" in result.dimension.warnings[0]


@pytest.mark.parametrize(
    ("company_age_years", "registered_capital", "expected_warning"),
    [
        (None, 100, "missing target fields: company_age_years"),
        (5, None, "missing target fields: capital.registered"),
        (None, None, "company_age_years, capital.registered"),
        (-1, 100, "invalid target fields: company_age_years"),
        (5, -1, "invalid target fields: capital.registered"),
        ("5", 100, "invalid target fields: company_age_years"),
        (5, 100.5, "invalid target fields: capital.registered"),
    ],
)
def test_missing_or_invalid_target_fields_are_not_scored(
    company_age_years: object,
    registered_capital: object,
    expected_warning: str,
) -> None:
    result = score_peer_relative_position(
        company_age_years,
        registered_capital,
        _ranked_samples(30),
        industry_code="I",
        benchmark_version=BENCHMARK_VERSION,
    )

    assert result.dimension.score is None
    assert result.dimension.available is False
    assert expected_warning in result.dimension.warnings[0]
    assert result.company_age_median == 15.5
    assert result.registered_capital_median == 155


def test_incomplete_benchmark_rows_are_excluded_and_reported() -> None:
    samples = [
        *_ranked_samples(30),
        PeerBenchmarkSample(company_age_years=None, registered_capital=100),
        PeerBenchmarkSample(company_age_years=5, registered_capital=None),
    ]

    result = score_peer_relative_position(
        15,
        155,
        samples,
        industry_code=" i ",
        benchmark_version=f" {BENCHMARK_VERSION} ",
    )

    assert result.dimension.available is True
    assert result.industry_code == "I"
    assert result.benchmark_version == BENCHMARK_VERSION
    assert result.sample_size == 30
    assert result.excluded_sample_count == 2
    assert "2 incomplete benchmark sample(s)" in result.dimension.warnings[0]


@pytest.mark.parametrize("industry_code", [None, "", "Z", "K", 1])
def test_invalid_or_ineligible_industry_group_is_not_scored(
    industry_code: object,
) -> None:
    result = score_peer_relative_position(
        5,
        100,
        _ranked_samples(30),
        industry_code=industry_code,
        benchmark_version=BENCHMARK_VERSION,
    )

    assert result.dimension.score is None
    assert result.dimension.available is False
    assert result.industry_code is None
    assert "benchmark-eligible GCIS category A-J" in result.dimension.warnings[0]


@pytest.mark.parametrize("benchmark_version", [None, "", "   ", 1])
def test_missing_benchmark_version_is_not_scored(benchmark_version: object) -> None:
    result = score_peer_relative_position(
        5,
        100,
        _ranked_samples(30),
        industry_code="I",
        benchmark_version=benchmark_version,
    )

    assert result.dimension.score is None
    assert result.dimension.available is False
    assert result.benchmark_version is None
    assert "benchmark_version is unavailable" in result.dimension.warnings[0]


def test_peer_score_is_reproducible_and_json_serializable() -> None:
    samples = _ranked_samples(30)
    first = score_peer_relative_position(
        Decimal("12.5"),
        125,
        samples,
        industry_code="I",
        benchmark_version=BENCHMARK_VERSION,
    )
    second = score_peer_relative_position(
        Decimal("12.5"),
        125,
        samples,
        industry_code="I",
        benchmark_version=BENCHMARK_VERSION,
    )

    assert first == second
    payload = first.model_dump(mode="json")
    assert isinstance(payload["age_percentile"], float)
    assert isinstance(payload["capital_percentile"], float)
    assert isinstance(payload["peer_index"], float)
    assert isinstance(payload["company_age_median"], float)
    assert isinstance(payload["registered_capital_median"], float)
    assert payload["benchmark_version"] == BENCHMARK_VERSION
