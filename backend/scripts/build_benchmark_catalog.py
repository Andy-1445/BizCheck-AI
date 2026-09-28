from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

from app.services.benchmark_snapshot import (
    BENCHMARK_ELIGIBLE_INDUSTRY_CODES,
    BenchmarkSnapshotStore,
)


class BenchmarkCatalogError(RuntimeError):
    """The selected snapshot versions cannot form one production catalog."""


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _parse_snapshot(value: str) -> tuple[str, str]:
    category, separator, version = value.partition("=")
    normalized_category = category.strip().upper()
    normalized_version = version.strip()
    if (
        separator != "="
        or normalized_category not in BENCHMARK_ELIGIBLE_INDUSTRY_CODES
        or not normalized_version
    ):
        raise argparse.ArgumentTypeError("snapshot must use CATEGORY=VERSION")
    return normalized_category, normalized_version


def build_catalog(
    database_path: Path,
    *,
    catalog_version: str,
    as_of: date,
    snapshots: list[tuple[str, str]],
) -> dict[str, object]:
    selected = dict(snapshots)
    if len(selected) != len(snapshots):
        raise BenchmarkCatalogError("Snapshot categories must not repeat.")

    required_categories = set(BENCHMARK_ELIGIBLE_INDUSTRY_CODES)
    missing = sorted(required_categories - set(selected))
    extra = sorted(set(selected) - required_categories)
    if missing or extra:
        raise BenchmarkCatalogError(
            f"Catalog must contain A-J exactly; missing={missing}, extra={extra}."
        )

    with sqlite3.connect(database_path) as connection:
        integrity_check = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity_check != "ok":
        raise BenchmarkCatalogError(
            f"SQLite integrity check failed: {integrity_check}"
        )

    store = BenchmarkSnapshotStore(database_path)
    categories: dict[str, dict[str, object]] = {}
    mapping_versions: set[str] = set()
    membership_sample_count = 0
    for category in sorted(selected):
        version = selected[category]
        metadata = store.get_snapshot(version)
        if metadata.as_of != as_of:
            raise BenchmarkCatalogError(
                f"Snapshot {version} uses {metadata.as_of}, expected {as_of}."
            )
        group_sample_sizes = metadata.group_sample_sizes
        if set(group_sample_sizes) != {category}:
            raise BenchmarkCatalogError(
                f"Snapshot {version} must contain only category {category}."
            )
        stored_count = store.count_peer_samples(version, category)
        if stored_count != metadata.sample_count:
            raise BenchmarkCatalogError(
                f"Snapshot {version} metadata count {metadata.sample_count} "
                f"does not match stored count {stored_count}."
            )

        mapping_versions.add(metadata.industry_mapping_version)
        membership_sample_count += metadata.sample_count
        categories[category] = {
            "snapshot_version": version,
            "sample_count": metadata.sample_count,
            "checksum_sha256": metadata.checksum_sha256,
            "source_version": metadata.source_version,
        }

    if len(mapping_versions) != 1:
        raise BenchmarkCatalogError(
            f"Snapshots use inconsistent industry mappings: {sorted(mapping_versions)}"
        )

    return {
        "catalog_version": catalog_version,
        "as_of": as_of.isoformat(),
        "industry_mapping_version": mapping_versions.pop(),
        "sqlite_integrity_check": integrity_check,
        "membership_sample_count": membership_sample_count,
        "membership_count_note": (
            "A company may appear in more than one category membership snapshot; "
            "this total is not a unique-company count."
        ),
        "categories": categories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a write-once A-J production Benchmark catalog.",
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--catalog-version", required=True)
    parser.add_argument("--as-of", type=_parse_date, required=True)
    parser.add_argument(
        "--snapshot",
        action="append",
        type=_parse_snapshot,
        required=True,
        help="Repeat for all categories using CATEGORY=VERSION.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        catalog = build_catalog(
            args.database,
            catalog_version=args.catalog_version,
            as_of=args.as_of,
            snapshots=args.snapshot,
        )
    except BenchmarkCatalogError as exc:
        parser.error(str(exc))

    output_json = json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_json + "\n", encoding="utf-8")
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
