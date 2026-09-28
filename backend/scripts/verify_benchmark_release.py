from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from app.schemas.benchmark_import import (
    GCISBatchDownloadReport,
    GCISBatchImportReport,
)
from app.services.benchmark_snapshot import BenchmarkSnapshotStore
from app.services.gcis_batch_importer import load_gcis_batch_manifest


class BenchmarkReleaseVerificationError(RuntimeError):
    """A production Benchmark artifact does not match its audit metadata."""


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkReleaseVerificationError(
            f"Could not read JSON artifact {path}: {exc}"
        ) from exc


def _fingerprint(path: Path, *, count_lines: bool = False) -> tuple[int, str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    line_count = 0
    try:
        with path.open("rb") as input_file:
            while chunk := input_file.read(1024 * 1024):
                byte_size += len(chunk)
                digest.update(chunk)
                if count_lines:
                    line_count += chunk.count(b"\n")
    except OSError as exc:
        raise BenchmarkReleaseVerificationError(
            f"Could not fingerprint {path}: {exc}"
        ) from exc
    return byte_size, digest.hexdigest(), line_count


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkReleaseVerificationError(message)


def verify_release(
    *,
    imports_directory: Path,
    database_path: Path,
    catalog_path: Path,
) -> dict[str, object]:
    catalog = _load_json(catalog_path)
    _require(isinstance(catalog, dict), "Benchmark catalog must be a JSON object.")
    categories = catalog.get("categories")
    _require(
        isinstance(categories, dict) and set(categories) == set("ABCDEFGHIJ"),
        "Benchmark catalog must contain categories A-J exactly.",
    )

    with sqlite3.connect(database_path) as connection:
        integrity_check = connection.execute("PRAGMA integrity_check").fetchone()[0]
    _require(integrity_check == "ok", f"SQLite integrity check failed: {integrity_check}")

    store = BenchmarkSnapshotStore(database_path)
    summaries: dict[str, dict[str, object]] = {}
    for category in sorted(categories):
        prefix = f"gcis-benchmark-{category}-2026-08"
        manifest_path = imports_directory / f"{prefix}-manifest.json"
        download_path = imports_directory / f"{prefix}-download-report.json"
        import_path = imports_directory / f"{prefix}-import-report.json"
        canonical_path = imports_directory / f"{prefix}-canonical.jsonl"

        manifest = load_gcis_batch_manifest(manifest_path)
        try:
            download_report = GCISBatchDownloadReport.model_validate(
                _load_json(download_path)
            )
            import_report = GCISBatchImportReport.model_validate(_load_json(import_path))
        except ValidationError as exc:
            raise BenchmarkReleaseVerificationError(
                f"Category {category} has an invalid audit report: {exc}"
            ) from exc

        _require(manifest.industry_code == category, f"Manifest category mismatch: {category}")
        _require(len(manifest.sources) == 10, f"Category {category} must have 10 sources.")
        _require(
            download_report.manifest_version == manifest.version
            and import_report.manifest_version == manifest.version,
            f"Category {category} manifest versions do not match.",
        )
        _require(
            import_report.source_file_count == 10,
            f"Category {category} import report must contain 10 sources.",
        )

        downloads = {item.dataset_id: item for item in download_report.files}
        imports = {item.dataset_id: item for item in import_report.source_files}
        source_bytes = 0
        for source in manifest.sources:
            _require(
                source.dataset_id in downloads and source.dataset_id in imports,
                f"Category {category} is missing source audit metadata for "
                f"{source.dataset_id}.",
            )
            actual_size, actual_sha256, _ = _fingerprint(
                imports_directory / source.local_filename
            )
            source_bytes += actual_size
            download_item = downloads[source.dataset_id]
            import_item = imports[source.dataset_id]
            _require(
                actual_size == download_item.byte_size == import_item.byte_size,
                f"Dataset {source.dataset_id} byte size does not match reports.",
            )
            _require(
                actual_sha256 == download_item.sha256 == import_item.sha256,
                f"Dataset {source.dataset_id} SHA-256 does not match reports.",
            )

        canonical_bytes, canonical_sha256, canonical_line_count = _fingerprint(
            canonical_path,
            count_lines=True,
        )
        _require(
            canonical_sha256 == import_report.canonical_jsonl_sha256,
            f"Category {category} canonical JSONL SHA-256 does not match report.",
        )
        _require(
            canonical_line_count == import_report.canonical_company_count,
            f"Category {category} canonical JSONL line count does not match report.",
        )

        catalog_entry = categories[category]
        version = catalog_entry["snapshot_version"]
        metadata = store.get_snapshot(version)
        stored_count = store.count_peer_samples(version, category)
        _require(
            metadata.sample_count == stored_count == catalog_entry["sample_count"],
            f"Category {category} snapshot sample counts do not match.",
        )
        _require(
            metadata.checksum_sha256 == catalog_entry["checksum_sha256"],
            f"Category {category} snapshot checksum does not match catalog.",
        )
        _require(
            metadata.source_version == manifest.version,
            f"Category {category} snapshot source version does not match manifest.",
        )

        summaries[category] = {
            "source_file_count": len(manifest.sources),
            "source_bytes": source_bytes,
            "canonical_bytes": canonical_bytes,
            "canonical_company_count": canonical_line_count,
            "eligible_sample_count": stored_count,
            "snapshot_version": version,
        }

    return {
        "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "catalog_version": catalog["catalog_version"],
        "as_of": catalog["as_of"],
        "sqlite_integrity_check": integrity_check,
        "categories": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify every source and derived artifact in an A-J release.",
    )
    parser.add_argument("--imports-directory", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        report = verify_release(
            imports_directory=args.imports_directory,
            database_path=args.database,
            catalog_path=args.catalog,
        )
    except BenchmarkReleaseVerificationError as exc:
        parser.error(str(exc))

    output_json = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_json + "\n", encoding="utf-8")
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
