from datetime import date, datetime, timedelta, timezone
import json

import pytest

from app.schemas.company import (
    BusinessItem,
    CompanyCapital,
    CompanyData,
    CompanyStatus,
)
from app.services.benchmark_catalog import (
    BenchmarkCatalogInvalidError,
    BenchmarkCatalogService,
    BenchmarkCatalogUnavailableError,
)
from app.services.benchmark_snapshot import BenchmarkSnapshotStore


CATALOG_VERSION = "benchmark-catalog-2026-08-01-v1"
SNAPSHOT_VERSION = "benchmark-2026-08-01-F-v1"
SOURCE_VERSION = "gcis-benchmark-F-2026-08-v1"
AS_OF = date(2026, 8, 1)


def _company(
    number: int,
    *,
    business_items: list[BusinessItem] | None = None,
) -> CompanyData:
    return CompanyData(
        tax_id=f"{number:08d}",
        name=f"測試公司 {number}",
        status=CompanyStatus(code="01", description="核准設立"),
        capital=CompanyCapital(registered=number * 100_000),
        established_at=AS_OF - timedelta(days=365 * number),
        company_age_years=999,
        last_changed_at=date(2020, 1, 1),
        business_items=business_items or [],
    )


def _catalog_payload(metadata) -> dict[str, object]:
    categories = {
        code: {
            "snapshot_version": (
                metadata.version if code == "F" else f"unused-{code}-snapshot"
            ),
            "source_version": (
                metadata.source_version if code == "F" else f"unused-{code}-source"
            ),
            "sample_count": metadata.sample_count,
            "checksum_sha256": metadata.checksum_sha256,
        }
        for code in "ABCDEFGHIJ"
    }
    return {
        "catalog_version": CATALOG_VERSION,
        "as_of": AS_OF.isoformat(),
        "industry_mapping_version": "gcis-regional-category-membership-v1",
        "membership_sample_count": metadata.sample_count * 10,
        "membership_count_note": "Test membership rows are not unique companies.",
        "sqlite_integrity_check": "ok",
        "categories": categories,
    }


def _service(tmp_path) -> tuple[BenchmarkCatalogService, dict[str, object]]:
    database_path = tmp_path / "benchmark.sqlite3"
    metadata = BenchmarkSnapshotStore(database_path).create_snapshot(
        [_company(number) for number in range(1, 31)],
        version=SNAPSHOT_VERSION,
        as_of=AS_OF,
        source="GCIS nationwide F-category batch manifest",
        source_version=SOURCE_VERSION,
        created_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
        industry_code_override="F",
        industry_mapping_version="gcis-regional-category-membership-v1",
    )
    payload = _catalog_payload(metadata)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return (
        BenchmarkCatalogService(
            catalog_path,
            database_path,
            expected_catalog_version=CATALOG_VERSION,
        ),
        payload,
    )


def test_catalog_selects_primary_industry_snapshot_and_scores_company(tmp_path) -> None:
    service, _ = _service(tmp_path)
    company = _company(
        15,
        business_items=[
            BusinessItem(
                sequence="0001",
                code="F113050",
                name="電腦及事務性機器設備批發業",
            )
        ],
    )

    result = service.calculate_company_bizscore(company)

    assert result.version == "1.0"
    assert result.as_of == AS_OF
    assert result.score is not None
    assert result.coverage == 1
    assert len(result.dimensions) == 5
    assert result.dimensions[1].evidence[0] == "company_age_years=15"
    assert result.benchmark.catalog_version == CATALOG_VERSION
    assert result.benchmark.industry_code == "F"
    assert result.benchmark.snapshot_version == SNAPSHOT_VERSION
    assert result.benchmark.snapshot_source_version == SOURCE_VERSION
    assert result.benchmark.sample_count == 30
    assert result.dimensions[-1].available is True
    assert result.peer_benchmark.industry_code == "F"
    assert result.peer_benchmark.benchmark_version == SNAPSHOT_VERSION
    assert result.peer_benchmark.sample_size == 30
    assert result.peer_benchmark.company_age_median == 15.5
    assert result.peer_benchmark.registered_capital_median == 1_550_000
    assert result.peer_benchmark.age_percentile is not None
    assert result.peer_benchmark.capital_percentile is not None
    assert result.peer_benchmark.peer_index is not None

    repeated = service.calculate_company_bizscore(company)
    assert repeated == result


def test_missing_industry_keeps_peer_dimension_missing_at_eighty_percent(
    tmp_path,
) -> None:
    service, _ = _service(tmp_path)

    result = service.calculate_company_bizscore(
        _company(15),
        input_partial=True,
        input_warnings=["GCIS A3 business request timed out."],
    )

    assert result.score is not None
    assert result.coverage == 0.8
    assert result.provisional is True
    assert result.missing_dimensions == ["peer_relative_position"]
    assert result.benchmark.snapshot_version is None
    assert result.benchmark.sample_count == 0
    assert result.peer_benchmark.age_percentile is None
    assert result.peer_benchmark.company_age_median is None


def test_catalog_checksum_mismatch_is_rejected_before_scoring(tmp_path) -> None:
    service, payload = _service(tmp_path)
    payload["categories"]["F"]["checksum_sha256"] = "0" * 64
    service.catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    service._catalog = None

    with pytest.raises(BenchmarkCatalogInvalidError, match="checksum_sha256"):
        service.calculate_company_bizscore(
            _company(
                15,
                business_items=[BusinessItem(sequence="1", code="F113050")],
            )
        )


def test_catalog_must_contain_exactly_categories_a_through_j(tmp_path) -> None:
    service, payload = _service(tmp_path)
    del payload["categories"]["J"]
    payload["membership_sample_count"] -= 30
    service.catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    service._catalog = None

    with pytest.raises(BenchmarkCatalogInvalidError, match="exactly A-J"):
        _ = service.catalog


def test_missing_catalog_is_reported_as_unavailable(tmp_path) -> None:
    service = BenchmarkCatalogService(
        tmp_path / "missing.json",
        tmp_path / "missing.sqlite3",
        expected_catalog_version=CATALOG_VERSION,
    )

    with pytest.raises(BenchmarkCatalogUnavailableError, match="was not found"):
        _ = service.catalog


def test_configured_catalog_version_must_match_file(tmp_path) -> None:
    service, _ = _service(tmp_path)
    service.expected_catalog_version = "another-catalog"
    service._catalog = None

    with pytest.raises(BenchmarkCatalogInvalidError, match="does not match"):
        _ = service.catalog
