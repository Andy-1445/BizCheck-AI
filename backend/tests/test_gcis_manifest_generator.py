from datetime import date

import pytest

from scripts.generate_gcis_batch_manifests import (
    ManifestGenerationError,
    REQUIRED_COVERAGE_AREAS,
    build_manifest,
)


def _title(area: str, category: str = "A") -> str:
    prefix = area if area.endswith("市") else f"({area})"
    return f"{prefix}公司登記資料-{category}測試產業"


def _row(area: str, index: int, category: str = "A") -> dict[str, str]:
    return {
        "資料集識別碼": str(1000 + index),
        "資料集名稱": _title(area, category),
        "資料下載網址": f"https://data.gcis.nat.gov.tw/od/file?oid={index}",
        "提供機關": "經濟部商業發展署",
        "詮釋資料更新時間": "2026-08-03 06:00:00",
    }


def _complete_rows(category: str = "A") -> list[dict[str, str]]:
    return [
        _row(area, index, category)
        for index, area in enumerate(REQUIRED_COVERAGE_AREAS)
    ]


def test_manifest_uses_ten_non_overlapping_areas_and_excludes_aggregate() -> None:
    rows = _complete_rows()
    rows.append(
        {
            **_row("台北市", 99),
            "資料集識別碼": "9999",
            "資料集名稱": "六都公司登記資料-A測試產業",
        }
    )

    manifest = build_manifest(
        rows,
        category="A",
        as_of=date(2026, 8, 1),
        catalog_downloaded_at=date(2026, 8, 22),
    )

    sources = manifest["sources"]
    assert len(sources) == 10
    assert all(source["dataset_id"] != "9999" for source in sources)
    assert manifest["version"] == "gcis-benchmark-A-2026-08-v1"


def test_manifest_rejects_missing_coverage_area() -> None:
    rows = _complete_rows()[:-1]

    with pytest.raises(ManifestGenerationError, match="missing coverage areas"):
        build_manifest(
            rows,
            category="A",
            as_of=date(2026, 8, 1),
            catalog_downloaded_at=date(2026, 8, 22),
        )


def test_manifest_rejects_duplicate_coverage_area() -> None:
    rows = _complete_rows()
    rows.append({**rows[0], "資料集識別碼": "duplicate"})

    with pytest.raises(ManifestGenerationError, match="duplicate coverage area"):
        build_manifest(
            rows,
            category="A",
            as_of=date(2026, 8, 1),
            catalog_downloaded_at=date(2026, 8, 22),
        )
