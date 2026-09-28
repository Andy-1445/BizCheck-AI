from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import sqlite3

import pytest

from app.schemas.company import (
    BusinessItem,
    CompanyCapital,
    CompanyData,
    CompanyResponse,
    CompanyStatus,
    ResponseMeta,
)
from app.services.benchmark_snapshot import (
    BenchmarkSnapshotExistsError,
    BenchmarkSnapshotInputError,
    BenchmarkSnapshotNotFoundError,
    BenchmarkSnapshotStore,
    load_companies_from_jsonl,
)
from app.services.bizscore import score_peer_relative_position
from app.services.company_normalizer import calculate_company_age_years


SNAPSHOT_VERSION = "benchmark-2026-08-22-v1"
CREATED_AT = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)


def _company(
    number: int,
    *,
    industry_code: str = "I",
    status_code: str | None = "01",
    established_at: date | None = date(2020, 1, 1),
    registered_capital: int | None = 1_000_000,
    include_industry: bool = True,
) -> CompanyData:
    business_items = []
    if include_industry:
        business_items.append(
            BusinessItem(
                sequence="0001",
                code=f"{industry_code}301010",
                name="測試營業項目",
            )
        )
    return CompanyData(
        tax_id=f"{number:08d}",
        name=f"測試公司 {number}",
        status=CompanyStatus(code=status_code, description="核准設立"),
        capital=CompanyCapital(registered=registered_capital),
        established_at=established_at,
        company_age_years=999,
        business_items=business_items,
    )


def _create_snapshot(
    store: BenchmarkSnapshotStore,
    companies: list[CompanyData],
    *,
    version: str = SNAPSHOT_VERSION,
    as_of: date = date(2026, 8, 22),
):
    return store.create_snapshot(
        companies,
        version=version,
        as_of=as_of,
        source="GCIS canonical export",
        source_version="2026-08-22",
        created_at=CREATED_AT,
    )


def test_snapshot_filters_records_and_persists_audit_metadata(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    companies = [
        _company(1, industry_code="I"),
        _company(2, industry_code="F", registered_capital=2_000_000),
        _company(3, status_code="02"),
        _company(4, established_at=None),
        _company(5, registered_capital=None),
        _company(6, include_industry=False),
    ]

    metadata = _create_snapshot(store, companies)

    assert metadata.input_record_count == 6
    assert metadata.sample_count == 2
    assert metadata.excluded_record_count == 4
    assert metadata.exclusion_counts == {
        "industry_unavailable": 1,
        "invalid_established_at": 1,
        "invalid_registered_capital": 1,
        "status_not_active": 1,
    }
    assert metadata.group_sample_sizes == {"F": 1, "I": 1}
    assert metadata.industry_mapping_version == "gcis-business-category-v1"
    assert len(metadata.checksum_sha256) == 64
    assert store.get_snapshot(SNAPSHOT_VERSION) == metadata

    expected_age = calculate_company_age_years(
        date(2020, 1, 1),
        as_of=date(2026, 8, 22),
    )
    samples = store.get_peer_samples(SNAPSHOT_VERSION, "i")
    assert len(samples) == 1
    assert samples[0].company_age_years == Decimal(str(expected_age))
    assert samples[0].registered_capital == 1_000_000


def test_snapshot_uses_reference_date_instead_of_input_age(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    company = _company(1)
    company.company_age_years = 999

    _create_snapshot(store, [company])

    sample = store.get_peer_samples(SNAPSHOT_VERSION, "I")[0]
    assert sample.company_age_years != 999
    assert sample.company_age_years == Decimal("6.6")


def test_snapshot_version_is_immutable(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    original = _create_snapshot(store, [_company(1)])

    with pytest.raises(BenchmarkSnapshotExistsError):
        _create_snapshot(store, [_company(2)])

    assert store.get_snapshot(SNAPSHOT_VERSION) == original
    assert store.get_peer_samples(SNAPSHOT_VERSION, "I")[0].registered_capital == 1_000_000


def test_checksum_is_independent_of_input_order_and_snapshot_metadata(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    companies = [_company(1), _company(2, industry_code="F")]

    first = _create_snapshot(store, companies, version="snapshot-a")
    second = _create_snapshot(store, list(reversed(companies)), version="snapshot-b")

    assert first.checksum_sha256 == second.checksum_sha256


def test_duplicate_tax_id_aborts_snapshot_without_partial_write(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")

    with pytest.raises(BenchmarkSnapshotInputError, match="Duplicate tax_id"):
        _create_snapshot(store, [_company(1), _company(1, industry_code="F")])

    with pytest.raises(BenchmarkSnapshotNotFoundError):
        store.get_snapshot(SNAPSHOT_VERSION)


def test_snapshot_with_no_eligible_records_is_rejected(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")

    with pytest.raises(BenchmarkSnapshotInputError, match="no eligible"):
        _create_snapshot(store, [_company(1, status_code="02")])

    assert not store.database_path.exists()


def test_latest_snapshot_uses_as_of_date_before_creation_time(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    _create_snapshot(
        store,
        [_company(1)],
        version="older",
        as_of=date(2026, 8, 21),
    )
    _create_snapshot(
        store,
        [_company(2)],
        version="newer",
        as_of=date(2026, 8, 22),
    )

    assert store.get_latest_snapshot().version == "newer"


def test_snapshot_schema_and_foreign_keys_are_initialized(tmp_path) -> None:
    database_path = tmp_path / "nested" / "benchmark.sqlite3"
    store = BenchmarkSnapshotStore(database_path)

    store.initialize()

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"benchmark_snapshots", "benchmark_samples"} <= table_names


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("version", " ", "version must be"),
        ("source", "", "source must be"),
        ("source_version", "", "source_version must be"),
        ("as_of", datetime(2026, 8, 22), "as_of must be a date"),
        ("created_at", datetime(2026, 8, 22), "timezone-aware"),
    ],
)
def test_invalid_snapshot_metadata_is_rejected(
    tmp_path,
    field: str,
    value: object,
    expected_message: str,
) -> None:
    arguments = {
        "version": SNAPSHOT_VERSION,
        "as_of": date(2026, 8, 22),
        "source": "GCIS",
        "source_version": "2026-08-22",
        "created_at": CREATED_AT,
    }
    arguments[field] = value

    with pytest.raises(BenchmarkSnapshotInputError, match=expected_message):
        BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3").create_snapshot(
            [_company(1)],
            **arguments,
        )


@pytest.mark.parametrize("industry_code", ["", "Z", "K", "AA", 1])
def test_peer_sample_reader_only_accepts_benchmark_industries(
    tmp_path,
    industry_code: object,
) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    _create_snapshot(store, [_company(1)])

    with pytest.raises(BenchmarkSnapshotInputError, match="industry_code must be A-J"):
        store.get_peer_samples(SNAPSHOT_VERSION, industry_code)


def test_peer_sample_count_matches_stored_group_without_loading_rows(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    _create_snapshot(store, [_company(1), _company(2)])

    assert store.count_peer_samples(SNAPSHOT_VERSION, "I") == 2
    assert store.count_peer_samples(SNAPSHOT_VERSION, "A") == 0


def test_peer_rank_counts_match_midrank_inputs_without_loading_rows(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    companies = [
        _company(1, registered_capital=100_000),
        _company(2, registered_capital=500_000),
        _company(3, registered_capital=500_000),
    ]
    _create_snapshot(store, companies)
    target_age = Decimal(
        str(
            calculate_company_age_years(
                date(2020, 1, 1),
                as_of=date(2026, 8, 22),
            )
        )
    )

    counts = store.get_peer_rank_counts(
        SNAPSHOT_VERSION,
        "I",
        company_age_years=target_age,
        registered_capital=500_000,
    )

    assert counts.sample_size == 3
    assert counts.age_lower_count == 0
    assert counts.age_equal_count == 3
    assert counts.capital_lower_count == 1
    assert counts.capital_equal_count == 2


def test_peer_medians_are_calculated_for_even_and_empty_groups(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    _create_snapshot(
        store,
        [
            _company(1, registered_capital=100_000),
            _company(2, registered_capital=200_000),
            _company(3, registered_capital=300_000),
            _company(4, registered_capital=400_000),
        ],
    )

    medians = store.get_peer_medians(SNAPSHOT_VERSION, "I")
    empty = store.get_peer_medians(SNAPSHOT_VERSION, "A")

    assert medians.sample_size == 4
    assert medians.company_age_median == Decimal("6.6")
    assert medians.registered_capital_median == Decimal("250000")
    assert empty.sample_size == 0
    assert empty.company_age_median is None
    assert empty.registered_capital_median is None


def test_snapshot_samples_feed_peer_pr_calculation(tmp_path) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    as_of = date(2026, 8, 22)
    companies = [
        _company(
            number,
            established_at=as_of - timedelta(days=365 * number),
            registered_capital=number * 100_000,
        )
        for number in range(1, 31)
    ]
    metadata = _create_snapshot(store, companies, as_of=as_of)

    result = score_peer_relative_position(
        15,
        1_500_000,
        store.get_peer_samples(metadata.version, "I"),
        industry_code="I",
        benchmark_version=metadata.version,
    )

    assert metadata.group_sample_sizes == {"I": 30}
    assert result.dimension.available is True
    assert result.sample_size == 30
    assert result.benchmark_version == metadata.version


def test_official_batch_industry_override_does_not_fabricate_business_items(
    tmp_path,
) -> None:
    store = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3")
    company = _company(1, include_industry=False)

    metadata = store.create_snapshot(
        [company],
        version=SNAPSHOT_VERSION,
        as_of=date(2026, 8, 1),
        source="GCIS nationwide I-category batch manifest",
        source_version="gcis-benchmark-I-2026-08-v1",
        created_at=CREATED_AT,
        industry_code_override="I",
        industry_mapping_version="gcis-regional-category-membership-v1",
    )

    assert company.business_items == []
    assert metadata.group_sample_sizes == {"I": 1}
    assert (
        metadata.industry_mapping_version
        == "gcis-regional-category-membership-v1"
    )
    assert len(store.get_peer_samples(SNAPSHOT_VERSION, "I")) == 1


def test_jsonl_loader_accepts_company_data_and_company_response(tmp_path) -> None:
    company = _company(1)
    response = CompanyResponse(
        data=_company(2),
        meta=ResponseMeta(
            fetched_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        ),
    )
    input_path = tmp_path / "companies.jsonl"
    input_path.write_text(
        "\n".join(
            [
                company.model_dump_json(),
                response.model_dump_json(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    companies = load_companies_from_jsonl(input_path)

    assert [item.tax_id for item in companies] == ["00000001", "00000002"]


def test_jsonl_loader_reports_the_invalid_line(tmp_path) -> None:
    input_path = tmp_path / "companies.jsonl"
    input_path.write_text(
        _company(1).model_dump_json() + "\nnot-json\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkSnapshotInputError, match="line 2"):
        load_companies_from_jsonl(input_path)
