import csv
from datetime import date
import hashlib
import json

import pytest

from app.schemas.company import CompanyData
from app.services.benchmark_snapshot import (
    BenchmarkSnapshotStore,
    load_companies_from_jsonl,
)
from app.services.gcis_batch_importer import (
    GCISBatchImportError,
    download_gcis_batch_sources,
    import_gcis_batch_to_canonical_jsonl,
    load_gcis_batch_manifest,
)


CSV_HEADERS = [
    "統一編號",
    "公司名稱",
    "公司地址",
    "資本總額",
    "實收資本額",
    "在境內營運資金",
    "核准設立日期",
    "登記狀態",
    "營業地址（財政資訊中心匯入）",
]


def _manifest_payload(local_filename: str = "source.csv") -> dict:
    return {
        "version": "gcis-benchmark-I-2026-08-v1",
        "as_of": "2026-08-01",
        "industry_code": "I",
        "industry_mapping_version": "gcis-regional-category-membership-v1",
        "provider": "經濟部商業發展署",
        "coverage": "測試涵蓋範圍",
        "catalog_downloaded_at": "2026-08-22",
        "sources": [
            {
                "dataset_id": "54285",
                "title": "新北市公司登記資料-I專業、科學及技術服務業",
                "url": "https://data.gcis.nat.gov.tw/od/file?oid=test",
                "metadata_modified_at": "2026-08-03 04:00:06",
                "local_filename": local_filename,
            }
        ],
    }


def _write_manifest(tmp_path, payload: dict | None = None):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(payload or _manifest_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _write_source_csv(tmp_path, rows: list[list[str]], headers=CSV_HEADERS):
    path = tmp_path / "source.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def _valid_row(
    tax_id: str,
    *,
    status: str = "核准設立",
    capital: str = "500000",
    setup_date: str = "1140203",
) -> list[str]:
    return [
        tax_id,
        f"測試公司 {tax_id}",
        "新北市測試路 1 號",
        capital,
        " ",
        " ",
        setup_date,
        status,
        "",
    ]


def test_official_csv_is_normalized_to_canonical_company_data_jsonl(
    tmp_path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    active_row = _valid_row("00000001")
    _write_source_csv(
        tmp_path,
        [
            active_row,
            _valid_row("00000002", status="解散", capital="0", setup_date="1000101"),
            _valid_row("bad-tax"),
            active_row,
        ],
    )
    output_path = tmp_path / "canonical.jsonl"
    report_path = tmp_path / "import-report.json"

    report = import_gcis_batch_to_canonical_jsonl(
        manifest_path,
        source_directory=tmp_path,
        output_path=output_path,
        report_path=report_path,
    )

    assert report.input_row_count == 4
    assert report.canonical_company_count == 2
    assert report.duplicate_row_count == 1
    assert report.rejected_row_count == 1
    assert report.rejection_counts == {"invalid_required_identity": 1}
    assert report.status_description_counts == {"核准設立": 3, "解散": 1}
    assert report.source_file_count == 1
    assert len(report.canonical_jsonl_sha256) == 64

    payloads = [
        CompanyData.model_validate_json(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [company.tax_id for company in payloads] == ["00000001", "00000002"]
    assert payloads[0].status.code == "01"
    assert payloads[0].status.description == "核准設立"
    assert payloads[0].capital.registered == 500_000
    assert payloads[0].established_at == date(2025, 2, 3)
    assert payloads[0].business_items == []
    assert payloads[1].status.code is None
    assert payloads[1].status.description == "解散"
    assert payloads[1].capital.registered == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))[
        "canonical_jsonl_sha256"
    ] == report.canonical_jsonl_sha256


def test_imported_jsonl_builds_snapshot_with_manifest_membership(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    _write_source_csv(
        tmp_path,
        [
            _valid_row("00000001"),
            _valid_row("00000002", status="解散"),
        ],
    )
    output_path = tmp_path / "canonical.jsonl"
    report_path = tmp_path / "import-report.json"
    report = import_gcis_batch_to_canonical_jsonl(
        manifest_path,
        source_directory=tmp_path,
        output_path=output_path,
        report_path=report_path,
    )

    metadata = BenchmarkSnapshotStore(tmp_path / "benchmark.sqlite3").create_snapshot(
        load_companies_from_jsonl(output_path),
        version="benchmark-2026-08-01-v1",
        as_of=report.as_of,
        source="GCIS official batch manifest",
        source_version=report.manifest_version,
        industry_code_override=report.industry_code,
        industry_mapping_version=report.industry_mapping_version,
    )

    assert metadata.input_record_count == 2
    assert metadata.sample_count == 1
    assert metadata.exclusion_counts == {"status_not_active": 1}
    assert metadata.group_sample_sizes == {"I": 1}


def test_existing_source_file_can_be_reused_by_download_workflow(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    source_path = _write_source_csv(tmp_path, [_valid_row("00000001")])
    report_path = tmp_path / "download-report.json"

    report = download_gcis_batch_sources(
        manifest_path,
        destination_directory=tmp_path,
        report_path=report_path,
    )

    source_bytes = source_path.read_bytes()
    assert report.files[0].reused_existing_file is True
    assert report.files[0].byte_size == len(source_bytes)
    assert report.files[0].sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert report_path.exists()


def test_conflicting_duplicate_company_aborts_final_output(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    _write_source_csv(
        tmp_path,
        [
            _valid_row("00000001", capital="500000"),
            _valid_row("00000001", capital="600000"),
        ],
    )
    output_path = tmp_path / "canonical.jsonl"

    with pytest.raises(GCISBatchImportError, match="Conflicting duplicate"):
        import_gcis_batch_to_canonical_jsonl(
            manifest_path,
            source_directory=tmp_path,
            output_path=output_path,
            report_path=tmp_path / "import-report.json",
        )

    assert not output_path.exists()


def test_missing_required_csv_header_is_rejected(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    headers = [header for header in CSV_HEADERS if header != "核准設立日期"]
    _write_source_csv(tmp_path, [], headers=headers)

    with pytest.raises(GCISBatchImportError, match="核准設立日期"):
        import_gcis_batch_to_canonical_jsonl(
            manifest_path,
            source_directory=tmp_path,
            output_path=tmp_path / "canonical.jsonl",
            report_path=tmp_path / "import-report.json",
        )


def test_manifest_rejects_duplicate_dataset_ids(tmp_path) -> None:
    payload = _manifest_payload()
    payload["sources"].append(dict(payload["sources"][0], local_filename="two.csv"))
    manifest_path = _write_manifest(tmp_path, payload)

    with pytest.raises(GCISBatchImportError, match="duplicate dataset_id"):
        load_gcis_batch_manifest(manifest_path)


def test_manifest_rejects_unsafe_local_filename(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path, _manifest_payload("../source.csv"))

    with pytest.raises(GCISBatchImportError, match="Unsafe local filename"):
        download_gcis_batch_sources(
            manifest_path,
            destination_directory=tmp_path,
            report_path=tmp_path / "download-report.json",
        )


def test_import_never_overwrites_existing_output(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path)
    _write_source_csv(tmp_path, [_valid_row("00000001")])
    output_path = tmp_path / "canonical.jsonl"
    output_path.write_text("owned by user\n", encoding="utf-8")

    with pytest.raises(GCISBatchImportError, match="Output already exists"):
        import_gcis_batch_to_canonical_jsonl(
            manifest_path,
            source_directory=tmp_path,
            output_path=output_path,
            report_path=tmp_path / "import-report.json",
        )

    assert output_path.read_text(encoding="utf-8") == "owned by user\n"
