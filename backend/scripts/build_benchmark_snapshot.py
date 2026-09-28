from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from app.services.benchmark_snapshot import (
    BenchmarkSnapshotError,
    BenchmarkSnapshotStore,
    iter_companies_from_jsonl,
)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an immutable BizCheck AI peer Benchmark snapshot.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="UTF-8 JSONL containing canonical CompanyData or CompanyResponse records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="SQLite snapshot database path.",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--as-of", type=_parse_date, required=True)
    parser.add_argument("--source", default="GCIS")
    parser.add_argument(
        "--source-version",
        required=True,
        help="Official export date or another immutable source identifier.",
    )
    parser.add_argument(
        "--industry-code",
        help="A-J category supplied by an official category-specific batch source.",
    )
    parser.add_argument(
        "--industry-mapping-version",
        default="gcis-business-category-v1",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional write-once JSON file for the stored snapshot metadata.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.metadata_output is not None and args.metadata_output.exists():
        parser.error(f"Metadata output already exists: {args.metadata_output}")
    try:
        companies = iter_companies_from_jsonl(args.input)
        metadata = BenchmarkSnapshotStore(args.output).create_snapshot(
            companies,
            version=args.version,
            as_of=args.as_of,
            source=args.source,
            source_version=args.source_version,
            industry_code_override=args.industry_code,
            industry_mapping_version=args.industry_mapping_version,
        )
    except (OSError, BenchmarkSnapshotError) as exc:
        parser.error(str(exc))

    metadata_json = json.dumps(
        metadata.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(metadata_json + "\n", encoding="utf-8")
    print(metadata_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
