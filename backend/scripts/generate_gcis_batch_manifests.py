from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import date, datetime
from pathlib import Path


CATEGORY_PATTERN = re.compile(r"公司登記資料-([A-J])")
REQUIRED_COVERAGE_AREAS = (
    "台北市",
    "新北市",
    "桃園市",
    "台中市",
    "台南市",
    "高雄市",
    "北台灣排除六都",
    "中台灣排除六都",
    "南台灣排除六都",
    "東台灣排除六都",
)
CATALOG_COLUMNS = {
    "資料集識別碼",
    "資料集名稱",
    "資料下載網址",
    "提供機關",
    "詮釋資料更新時間",
}


class ManifestGenerationError(RuntimeError):
    """The official catalog cannot produce an unambiguous regional manifest."""


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _parse_categories(value: str) -> tuple[str, ...]:
    categories = tuple(dict.fromkeys(value.strip().upper()))
    invalid = sorted(set(categories) - set("ABCDEFGHIJ"))
    if invalid or not categories:
        raise argparse.ArgumentTypeError("categories must contain only A-J")
    return categories


def _coverage_area(title: str) -> str | None:
    for area in REQUIRED_COVERAGE_AREAS[:6]:
        if title.startswith(f"{area}公司登記資料-"):
            return area
    for area in REQUIRED_COVERAGE_AREAS[6:]:
        if title.startswith(f"({area})公司登記資料-"):
            return area
    return None


def _catalog_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as catalog_file:
        reader = csv.DictReader(catalog_file)
        missing = sorted(CATALOG_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ManifestGenerationError(
                f"Catalog is missing required columns: {', '.join(missing)}"
            )
        return list(reader)


def build_manifest(
    rows: list[dict[str, str]],
    *,
    category: str,
    as_of: date,
    catalog_downloaded_at: date,
) -> dict[str, object]:
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        title = row["資料集名稱"].strip()
        match = CATEGORY_PATTERN.search(title)
        if match is None or match.group(1) != category:
            continue

        area = _coverage_area(title)
        if area is None:
            # In particular, exclude the overlapping 六都 aggregate dataset.
            continue
        if area in selected:
            raise ManifestGenerationError(
                f"Category {category} has duplicate coverage area {area}: "
                f"{selected[area]['資料集識別碼']} and {row['資料集識別碼']}"
            )
        selected[area] = row

    missing_areas = [area for area in REQUIRED_COVERAGE_AREAS if area not in selected]
    if missing_areas:
        raise ManifestGenerationError(
            f"Category {category} is missing coverage areas: "
            f"{', '.join(missing_areas)}"
        )

    year_month = as_of.strftime("%Y-%m")
    sources: list[dict[str, str]] = []
    providers: set[str] = set()
    for area in REQUIRED_COVERAGE_AREAS:
        row = selected[area]
        dataset_id = row["資料集識別碼"].strip()
        title = row["資料集名稱"].strip()
        url = row["資料下載網址"].strip()
        provider = row["提供機關"].strip()
        modified = row["詮釋資料更新時間"].strip()
        if not dataset_id or not url or not provider or not modified:
            raise ManifestGenerationError(
                f"Category {category}, area {area} has incomplete catalog metadata."
            )
        try:
            datetime.fromisoformat(modified)
        except ValueError as exc:
            raise ManifestGenerationError(
                f"Dataset {dataset_id} has invalid modified timestamp {modified!r}."
            ) from exc

        providers.add(provider)
        sources.append(
            {
                "dataset_id": dataset_id,
                "title": title,
                "url": url,
                "metadata_modified_at": modified,
                "local_filename": (
                    f"gcis-dataset-{dataset_id}-{category}-{year_month}.csv"
                ),
            }
        )

    if len(providers) != 1:
        raise ManifestGenerationError(
            f"Category {category} has inconsistent providers: {sorted(providers)}"
        )

    return {
        "version": f"gcis-benchmark-{category}-{year_month}-v1",
        "as_of": as_of.isoformat(),
        "industry_code": category,
        "industry_mapping_version": "gcis-regional-category-membership-v1",
        "provider": providers.pop(),
        "coverage": (
            "全臺公司登記產業分類批次資料；六都個別資料加上北、中、南、東非六都區域"
        ),
        "catalog_downloaded_at": catalog_downloaded_at.isoformat(),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate non-overlapping A-J GCIS regional batch manifests from the "
            "official data.gov.tw catalog export."
        )
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--as-of", type=_parse_date, required=True)
    parser.add_argument(
        "--catalog-downloaded-at",
        type=_parse_date,
        required=True,
    )
    parser.add_argument(
        "--categories",
        type=_parse_categories,
        default=tuple("ABCDEFGHIJ"),
    )
    args = parser.parse_args()

    rows = _catalog_rows(args.catalog)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, object]] = []
    for category in args.categories:
        manifest = build_manifest(
            rows,
            category=category,
            as_of=args.as_of,
            catalog_downloaded_at=args.catalog_downloaded_at,
        )
        output_path = (
            args.output_directory
            / f"gcis-benchmark-{category}-{args.as_of.strftime('%Y-%m')}-manifest.json"
        )
        if output_path.exists():
            raise ManifestGenerationError(f"Output already exists: {output_path}")
        output_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        generated.append(
            {
                "category": category,
                "path": str(output_path),
                "source_count": len(manifest["sources"]),
            }
        )

    print(json.dumps(generated, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
