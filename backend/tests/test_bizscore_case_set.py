import json
from pathlib import Path

import pytest

from scripts.build_bizscore_case_set import verify_case_set


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = WORKSPACE_ROOT / "backend"
CASE_SET_PATH = (
    WORKSPACE_ROOT
    / "samples"
    / "bizscore"
    / "bizscore-v1-10-company-test-set-2026-08-23.json"
)
CATALOG_PATH = (
    BACKEND_ROOT
    / "data"
    / "benchmarks"
    / "benchmark-catalog-2026-08-01-v1.json"
)
DATABASE_PATH = (
    BACKEND_ROOT / "data" / "benchmarks" / "bizcheck-benchmark.sqlite3"
)


def test_recorded_case_set_has_ten_unique_traceable_companies() -> None:
    payload = json.loads(CASE_SET_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]

    assert payload["case_count"] == 10
    assert len(cases) == 10
    assert len({case["tax_id"] for case in cases}) == 10
    assert len(payload["summary"]["band_counts"]) == 3
    assert len(payload["summary"]["industry_counts"]) == 7
    assert sum(payload["summary"]["band_counts"].values()) == 10

    for case in cases:
        result = case["expected_bizscore"]
        peer = result["peer_benchmark"]
        assert case["input_company"]["tax_id"] == case["tax_id"]
        assert result["benchmark"]["catalog_version"] == (
            "benchmark-catalog-2026-08-01-v1"
        )
        assert result["as_of"] == "2026-08-01"
        assert 0 <= result["score"] <= 100
        assert peer["sample_size"] >= 30
        assert peer["company_age_median"] is not None
        assert peer["registered_capital_median"] is not None
        assert 0 <= peer["age_percentile"] <= 100
        assert 0 <= peer["capital_percentile"] <= 100
        assert 0 <= peer["peer_index"] <= 100


@pytest.mark.skipif(
    not DATABASE_PATH.exists(),
    reason="Formal Benchmark SQLite artifact is not available.",
)
def test_recorded_case_set_replays_against_formal_catalog() -> None:
    result = verify_case_set(
        CASE_SET_PATH,
        catalog_path=CATALOG_PATH,
        database_path=DATABASE_PATH,
        catalog_version="benchmark-catalog-2026-08-01-v1",
    )

    assert result["verified_case_count"] == 10
    assert result["reproducible"] is True
