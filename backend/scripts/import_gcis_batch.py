from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.gcis_batch_importer import (
    GCISBatchImportError,
    import_gcis_batch_to_canonical_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize official GCIS batch CSV files to CompanyData JSONL.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = import_gcis_batch_to_canonical_jsonl(
            args.manifest,
            source_directory=args.source_directory,
            output_path=args.output,
            report_path=args.report,
        )
    except (OSError, GCISBatchImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
