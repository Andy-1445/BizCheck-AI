from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import BinaryIO
from urllib.request import Request, urlopen

from pydantic import ValidationError

from app.schemas.benchmark_import import (
    GCISBatchDownloadReport,
    GCISBatchImportReport,
    GCISBatchManifest,
    GCISBatchSource,
    GCISBatchSourceFileReport,
)
from app.schemas.company import CompanyData
from app.services.company_normalizer import (
    CompanyNormalizationError,
    normalize_company_data,
)


REQUIRED_CSV_COLUMNS = frozenset(
    {
        "統一編號",
        "公司名稱",
        "公司地址",
        "資本總額",
        "實收資本額",
        "核准設立日期",
        "登記狀態",
    }
)
ACTIVE_STATUS_DESCRIPTION = "核准設立"


class GCISBatchImportError(RuntimeError):
    """The official GCIS batch source cannot be downloaded or normalized safely."""


def load_gcis_batch_manifest(path: str | Path) -> GCISBatchManifest:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = GCISBatchManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise GCISBatchImportError(f"Invalid GCIS batch manifest: {exc}") from exc

    industry_code = manifest.industry_code.strip().upper()
    if industry_code not in set("ABCDEFGHIJ"):
        raise GCISBatchImportError("Manifest industry_code must be A-J.")
    if industry_code != manifest.industry_code:
        manifest = manifest.model_copy(update={"industry_code": industry_code})

    dataset_ids = [source.dataset_id for source in manifest.sources]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise GCISBatchImportError("Manifest contains duplicate dataset_id values.")
    local_filenames = [source.local_filename for source in manifest.sources]
    if len(local_filenames) != len(set(local_filenames)):
        raise GCISBatchImportError("Manifest contains duplicate local filenames.")
    return manifest


def download_gcis_batch_sources(
    manifest_path: str | Path,
    *,
    destination_directory: str | Path,
    report_path: str | Path,
    timeout_seconds: float = 120,
) -> GCISBatchDownloadReport:
    manifest = load_gcis_batch_manifest(manifest_path)
    destination = Path(destination_directory)
    destination.mkdir(parents=True, exist_ok=True)
    output_report_path = Path(report_path)
    if output_report_path.exists():
        raise GCISBatchImportError(
            f"Download report already exists: {output_report_path}"
        )

    file_reports: list[GCISBatchSourceFileReport] = []
    for source in manifest.sources:
        target = _safe_source_path(destination, source)
        reused_existing_file = target.exists()
        if not reused_existing_file:
            partial_path = target.with_suffix(target.suffix + ".part")
            if partial_path.exists():
                raise GCISBatchImportError(
                    f"Partial download already exists and was not deleted: {partial_path}"
                )
            request = Request(
                source.url,
                headers={"User-Agent": "BizCheck-AI-Benchmark/1.0"},
            )
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    with partial_path.open("xb") as output_file:
                        _copy_stream(response, output_file)
                os.replace(partial_path, target)
            except (OSError, TimeoutError) as exc:
                raise GCISBatchImportError(
                    f"Could not download dataset {source.dataset_id}: {exc}"
                ) from exc

        byte_size, sha256 = _file_fingerprint(target)
        file_reports.append(
            GCISBatchSourceFileReport(
                dataset_id=source.dataset_id,
                title=source.title,
                local_filename=source.local_filename,
                byte_size=byte_size,
                sha256=sha256,
                reused_existing_file=reused_existing_file,
            )
        )
        print(
            f"dataset={source.dataset_id} bytes={byte_size} "
            f"reused={str(reused_existing_file).lower()}"
        )

    report = GCISBatchDownloadReport(
        manifest_version=manifest.version,
        downloaded_at=datetime.now(timezone.utc),
        files=file_reports,
    )
    _write_json_once(output_report_path, report.model_dump(mode="json"))
    return report


def import_gcis_batch_to_canonical_jsonl(
    manifest_path: str | Path,
    *,
    source_directory: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> GCISBatchImportReport:
    manifest = load_gcis_batch_manifest(manifest_path)
    source_root = Path(source_directory)
    canonical_output_path = Path(output_path)
    import_report_path = Path(report_path)
    _ensure_output_paths_available(canonical_output_path, import_report_path)

    temporary_output_path = canonical_output_path.with_suffix(
        canonical_output_path.suffix + ".tmp"
    )
    temporary_report_path = import_report_path.with_suffix(
        import_report_path.suffix + ".tmp"
    )
    if temporary_output_path.exists() or temporary_report_path.exists():
        raise GCISBatchImportError(
            "Temporary import output already exists and was not deleted."
        )

    input_row_count = 0
    canonical_company_count = 0
    duplicate_row_count = 0
    rejection_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    seen_companies: dict[str, str] = {}
    source_file_reports: list[GCISBatchSourceFileReport] = []
    canonical_digest = hashlib.sha256()

    canonical_output_path.parent.mkdir(parents=True, exist_ok=True)
    import_report_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary_output_path.open("x", encoding="utf-8", newline="\n") as output:
        for source in manifest.sources:
            source_path = _safe_source_path(source_root, source)
            if not source_path.is_file():
                raise GCISBatchImportError(
                    f"Official source file is missing: {source_path}"
                )
            byte_size, source_sha256 = _file_fingerprint(source_path)
            source_file_reports.append(
                GCISBatchSourceFileReport(
                    dataset_id=source.dataset_id,
                    title=source.title,
                    local_filename=source.local_filename,
                    byte_size=byte_size,
                    sha256=source_sha256,
                    reused_existing_file=True,
                )
            )

            with source_path.open(encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                _validate_csv_headers(reader.fieldnames, source)
                for row_number, row in enumerate(reader, start=2):
                    input_row_count += 1
                    status_description = _clean_text(row.get("登記狀態"))
                    status_counts[status_description or "<missing>"] += 1
                    try:
                        company = _normalize_batch_row(
                            row,
                            as_of=manifest.as_of,
                            status_description=status_description,
                        )
                    except CompanyNormalizationError:
                        rejection_counts["invalid_required_identity"] += 1
                        continue

                    fingerprint = _company_fingerprint(company)
                    previous_fingerprint = seen_companies.get(company.tax_id)
                    if previous_fingerprint is not None:
                        if previous_fingerprint != fingerprint:
                            raise GCISBatchImportError(
                                "Conflicting duplicate company data for tax_id "
                                f"{company.tax_id} in dataset {source.dataset_id}, "
                                f"row {row_number}."
                            )
                        duplicate_row_count += 1
                        continue
                    seen_companies[company.tax_id] = fingerprint

                    line = company.model_dump_json() + "\n"
                    output.write(line)
                    canonical_digest.update(line.encode("utf-8"))
                    canonical_company_count += 1

    report = GCISBatchImportReport(
        manifest_version=manifest.version,
        as_of=manifest.as_of,
        industry_code=manifest.industry_code,
        industry_mapping_version=manifest.industry_mapping_version,
        provider=manifest.provider,
        coverage=manifest.coverage,
        generated_at=datetime.now(timezone.utc),
        source_file_count=len(manifest.sources),
        input_row_count=input_row_count,
        canonical_company_count=canonical_company_count,
        duplicate_row_count=duplicate_row_count,
        rejected_row_count=sum(rejection_counts.values()),
        rejection_counts=dict(sorted(rejection_counts.items())),
        status_description_counts=dict(sorted(status_counts.items())),
        source_files=source_file_reports,
        canonical_jsonl_sha256=canonical_digest.hexdigest(),
    )
    _write_json_once(temporary_report_path, report.model_dump(mode="json"))
    os.replace(temporary_output_path, canonical_output_path)
    os.replace(temporary_report_path, import_report_path)
    return report


def _normalize_batch_row(
    row: dict[str, str],
    *,
    as_of: date,
    status_description: str | None,
) -> CompanyData:
    status_code = "01" if status_description == ACTIVE_STATUS_DESCRIPTION else None
    basic_record = {
        "Business_Accounting_NO": _clean_text(row.get("統一編號")),
        "Company_Name": _clean_text(row.get("公司名稱")),
        "Company_Status": status_code,
        "Company_Status_Desc": status_description,
        "Capital_Stock_Amount": _clean_text(row.get("資本總額")),
        "Paid_In_Capital_Amount": _clean_text(row.get("實收資本額")),
        "Company_Location": _clean_text(row.get("公司地址")),
        "Company_Setup_Date": _clean_text(row.get("核准設立日期")),
        "Change_Of_Approval_Data": None,
    }
    return normalize_company_data(
        basic_record,
        None,
        as_of=as_of,
    ).data


def _validate_csv_headers(
    fieldnames: list[str] | None,
    source: GCISBatchSource,
) -> None:
    available = set(fieldnames or [])
    missing = sorted(REQUIRED_CSV_COLUMNS - available)
    if missing:
        raise GCISBatchImportError(
            f"Dataset {source.dataset_id} is missing required columns: "
            f"{', '.join(missing)}"
        )


def _safe_source_path(root: Path, source: GCISBatchSource) -> Path:
    if Path(source.local_filename).name != source.local_filename:
        raise GCISBatchImportError(
            f"Unsafe local filename in dataset {source.dataset_id}."
        )
    root_resolved = root.resolve()
    candidate = (root / source.local_filename).resolve()
    if candidate.parent != root_resolved:
        raise GCISBatchImportError(
            f"Source path escapes the import directory: {source.local_filename}"
        )
    return candidate


def _company_fingerprint(company: CompanyData) -> str:
    return hashlib.sha256(company.model_dump_json().encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            byte_size += len(chunk)
            digest.update(chunk)
    return byte_size, digest.hexdigest()


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    while chunk := source.read(1024 * 1024):
        destination.write(chunk)


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _ensure_output_paths_available(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise GCISBatchImportError(f"Output already exists: {path}")


def _write_json_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise GCISBatchImportError(f"Output already exists: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
