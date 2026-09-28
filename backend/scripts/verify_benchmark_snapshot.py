from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from app.services.benchmark_snapshot import BenchmarkSnapshotStore
from app.services.bizscore import score_peer_relative_position


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a Benchmark SQLite snapshot and optionally score a target.",
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--industry-code", required=True)
    parser.add_argument("--company-age-years", type=float)
    parser.add_argument("--registered-capital", type=int)
    args = parser.parse_args()

    with sqlite3.connect(args.database) as connection:
        integrity_check = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity_check != "ok":
        parser.error(f"SQLite integrity check failed: {integrity_check}")

    store = BenchmarkSnapshotStore(args.database)
    metadata = store.get_snapshot(args.version)
    samples = store.get_peer_samples(args.version, args.industry_code)
    expected_count = metadata.group_sample_sizes.get(
        args.industry_code.strip().upper(),
        0,
    )
    if len(samples) != expected_count:
        parser.error(
            f"Stored sample count {len(samples)} does not match metadata {expected_count}."
        )

    payload: dict[str, object] = {
        "sqlite_integrity_check": integrity_check,
        "metadata": metadata.model_dump(mode="json"),
        "loaded_peer_sample_count": len(samples),
    }
    if args.company_age_years is not None or args.registered_capital is not None:
        if args.company_age_years is None or args.registered_capital is None:
            parser.error("Both target age and registered capital are required.")
        result = score_peer_relative_position(
            args.company_age_years,
            args.registered_capital,
            samples,
            industry_code=args.industry_code,
            benchmark_version=args.version,
        )
        payload["smoke_score"] = result.model_dump(mode="json")

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
