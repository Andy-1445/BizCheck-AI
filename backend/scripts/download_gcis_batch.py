from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.gcis_batch_importer import (
    GCISBatchImportError,
    download_gcis_batch_sources,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download official GCIS CSV files declared by a manifest.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = download_gcis_batch_sources(
            args.manifest,
            destination_directory=args.destination,
            report_path=args.report,
        )
    except (OSError, GCISBatchImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
