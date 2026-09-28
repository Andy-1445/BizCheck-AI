from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

import httpx

from app.schemas.company import CompanyData
from app.services.benchmark_catalog import BenchmarkCatalogService


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/api/v1"
DEFAULT_CATALOG_PATH = Path(
    "data/benchmarks/benchmark-catalog-2026-08-01-v1.json"
)
DEFAULT_DATABASE_PATH = Path("data/benchmarks/bizcheck-benchmark.sqlite3")
DEFAULT_CATALOG_VERSION = "benchmark-catalog-2026-08-01-v1"
PROFILE_QUERIES = {
    "low": (
        "CAST(company_age_years AS REAL) <= 1.5 "
        "AND registered_capital <= 500000",
        "registered_capital ASC, CAST(company_age_years AS REAL) ASC, tax_id",
    ),
    "mid": (
        "CAST(company_age_years AS REAL) BETWEEN 3 AND 10 "
        "AND registered_capital BETWEEN 500000 AND 5000000",
        "ABS(CAST(company_age_years AS REAL) - 6.5), "
        "ABS(registered_capital - 2000000), tax_id",
    ),
    "high": (
        "CAST(company_age_years AS REAL) >= 15 "
        "AND registered_capital >= 20000000",
        "registered_capital DESC, CAST(company_age_years AS REAL) DESC, tax_id",
    ),
}


class CaseSetError(RuntimeError):
    """The requested case set could not be built or reproduced."""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe, build, or verify a reproducible BizScore v1 case set."
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--catalog-version", default=DEFAULT_CATALOG_VERSION)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--max-requests", type=int, default=30)
    parser.add_argument("--tax-id", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()

    try:
        if args.verify is not None:
            result = verify_case_set(
                args.verify,
                catalog_path=args.catalog,
                database_path=args.database,
                catalog_version=args.catalog_version,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        if args.probe:
            results = probe_candidates(
                api_base_url=args.api_base_url,
                catalog_path=args.catalog,
                database_path=args.database,
                max_requests=args.max_requests,
            )
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0

        if len(args.tax_id) != 10:
            raise CaseSetError("Build mode requires exactly ten --tax-id values.")
        if args.output is None or args.report_output is None:
            raise CaseSetError(
                "Build mode requires --output and --report-output paths."
            )
        result = build_case_set(
            args.tax_id,
            api_base_url=args.api_base_url,
            catalog_path=args.catalog,
            database_path=args.database,
            catalog_version=args.catalog_version,
            output_path=args.output,
            report_output_path=args.report_output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (CaseSetError, httpx.HTTPError, OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def probe_candidates(
    *,
    api_base_url: str,
    catalog_path: Path,
    database_path: Path,
    max_requests: int,
) -> list[dict[str, Any]]:
    if max_requests < 1:
        raise CaseSetError("max_requests must be positive.")
    candidates = _candidate_pool(catalog_path, database_path)
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=30) as client:
        for candidate in candidates[:max_requests]:
            try:
                payload = _fetch_bizscore(
                    client,
                    api_base_url=api_base_url,
                    tax_id=candidate["tax_id"],
                )
            except CaseSetError as exc:
                results.append({**candidate, "error": str(exc)})
                continue
            company = payload["data"]["company"]
            score = payload["data"]["bizscore"]
            peer = score["peer_benchmark"]
            results.append(
                {
                    **candidate,
                    "company_name": company["name"],
                    "status_code": company["status"]["code"],
                    "score": score["score"],
                    "band": score["band"],
                    "coverage": score["coverage"],
                    "provisional": score["provisional"],
                    "actual_industry_code": peer["industry_code"],
                    "sample_size": peer["sample_size"],
                }
            )
            time.sleep(0.05)
    return results


def build_case_set(
    tax_ids: list[str],
    *,
    api_base_url: str,
    catalog_path: Path,
    database_path: Path,
    catalog_version: str,
    output_path: Path,
    report_output_path: Path,
) -> dict[str, Any]:
    if len(set(tax_ids)) != 10:
        raise CaseSetError("The ten tax IDs must be unique.")
    service = BenchmarkCatalogService(
        catalog_path,
        database_path,
        expected_catalog_version=catalog_version,
    )
    service.validate_all_snapshots()

    cases: list[dict[str, Any]] = []
    with httpx.Client(timeout=30) as client:
        for index, tax_id in enumerate(tax_ids, start=1):
            payload = _fetch_bizscore(
                client,
                api_base_url=api_base_url,
                tax_id=tax_id,
            )
            company = CompanyData.model_validate(payload["data"]["company"])
            first = service.calculate_company_bizscore(
                company,
                input_partial=payload["meta"]["partial"],
                input_warnings=payload["meta"]["warnings"],
            )
            second = service.calculate_company_bizscore(
                company,
                input_partial=payload["meta"]["partial"],
                input_warnings=payload["meta"]["warnings"],
            )
            if first != second:
                raise CaseSetError(f"BizScore was not reproducible for tax ID {tax_id}.")
            api_score = payload["data"]["bizscore"]
            expected_score = first.model_dump(mode="json")
            if api_score != expected_score:
                raise CaseSetError(
                    f"Live API and direct engine output differ for tax ID {tax_id}."
                )
            cases.append(
                {
                    "case_id": f"BZV1-{index:02d}",
                    "tax_id": tax_id,
                    "company_name": company.name,
                    "input_company": company.model_dump(mode="json"),
                    "input_meta": payload["meta"],
                    "expected_bizscore": expected_score,
                }
            )
            time.sleep(0.05)

    summary = _case_summary(cases)
    _validate_case_distribution(summary)
    artifact = {
        "case_set_version": "bizscore-v1-10-company-2026-08-23-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "Live GCIS A1/A3 normalized by BizCheck AI",
        "catalog_version": catalog_version,
        "catalog_as_of": service.catalog.as_of.isoformat(),
        "case_count": len(cases),
        "selection_note": (
            "Ten real companies selected from formal Benchmark candidates to cover "
            "multiple score bands and industries. Inputs are recorded for replay."
        ),
        "summary": summary,
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(_markdown_report(artifact), encoding="utf-8")
    return {
        "case_set_version": artifact["case_set_version"],
        "case_count": len(cases),
        "summary": summary,
        "output": str(output_path),
        "report_output": str(report_output_path),
    }


def verify_case_set(
    path: Path,
    *,
    catalog_path: Path,
    database_path: Path,
    catalog_version: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise CaseSetError("Case set must contain exactly ten cases.")
    service = BenchmarkCatalogService(
        catalog_path,
        database_path,
        expected_catalog_version=catalog_version,
    )
    service.validate_all_snapshots()
    verified: list[str] = []
    for case in cases:
        company = CompanyData.model_validate(case["input_company"])
        meta = case["input_meta"]
        result = service.calculate_company_bizscore(
            company,
            input_partial=meta["partial"],
            input_warnings=meta["warnings"],
        )
        if result.model_dump(mode="json") != case["expected_bizscore"]:
            raise CaseSetError(
                f"Replayed BizScore differs for case {case.get('case_id')}."
            )
        verified.append(case["case_id"])
    summary = _case_summary(cases)
    _validate_case_distribution(summary)
    return {
        "case_set_version": payload.get("case_set_version"),
        "verified_case_count": len(verified),
        "verified_case_ids": verified,
        "summary": summary,
        "reproducible": True,
    }


def _candidate_pool(
    catalog_path: Path,
    database_path: Path,
) -> list[dict[str, Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    categories = catalog["categories"]
    candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        for category_code in sorted(categories):
            version = categories[category_code]["snapshot_version"]
            for profile, (where_clause, order_clause) in PROFILE_QUERIES.items():
                rows = connection.execute(
                    f"""
                    SELECT tax_id, company_age_years, registered_capital
                    FROM benchmark_samples
                    WHERE snapshot_version = ? AND industry_code = ?
                      AND {where_clause}
                    ORDER BY {order_clause}
                    LIMIT 4
                    """,
                    (version, category_code),
                ).fetchall()
                candidates_by_key[(category_code, profile)] = [
                    {
                        "tax_id": row["tax_id"],
                        "source_industry_code": category_code,
                        "source_profile": profile,
                        "snapshot_age_years": float(row["company_age_years"]),
                        "snapshot_registered_capital": row["registered_capital"],
                    }
                    for row in rows
                ]
    finally:
        connection.close()

    pool: list[dict[str, Any]] = []
    for rank in range(4):
        for profile in ("low", "mid", "high"):
            for category_code in sorted(categories):
                rows = candidates_by_key[(category_code, profile)]
                if rank < len(rows):
                    pool.append(rows[rank])
    return pool


def _fetch_bizscore(
    client: httpx.Client,
    *,
    api_base_url: str,
    tax_id: str,
) -> dict[str, Any]:
    response = client.get(
        f"{api_base_url.rstrip('/')}/companies/{tax_id}/bizscore"
    )
    if response.status_code != 200:
        detail = response.text[:300]
        raise CaseSetError(
            f"Tax ID {tax_id} returned HTTP {response.status_code}: {detail}"
        )
    payload = response.json()
    if not isinstance(payload, dict) or "data" not in payload or "meta" not in payload:
        raise CaseSetError(f"Tax ID {tax_id} returned an invalid API payload.")
    return payload


def _case_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    bands: Counter[str] = Counter()
    industries: Counter[str] = Counter()
    missing_dimensions: Counter[str] = Counter()
    scored_count = 0
    provisional_count = 0
    full_coverage_scores: list[int] = []
    for case in cases:
        result = case["expected_bizscore"]
        band = result["band"] or "無數字總分"
        bands[band] += 1
        industry_code = result["peer_benchmark"]["industry_code"] or "未分類"
        industries[industry_code] += 1
        scored_count += result["score"] is not None
        provisional_count += result["provisional"] is True
        missing_dimensions.update(result["missing_dimensions"])
        if result["coverage"] == 1 and result["score"] is not None:
            full_coverage_scores.append(result["score"])
    return {
        "scored_count": scored_count,
        "unscored_count": len(cases) - scored_count,
        "provisional_count": provisional_count,
        "band_counts": dict(sorted(bands.items())),
        "industry_counts": dict(sorted(industries.items())),
        "missing_dimension_counts": dict(sorted(missing_dimensions.items())),
        "full_coverage_score_range": {
            "minimum": min(full_coverage_scores) if full_coverage_scores else None,
            "maximum": max(full_coverage_scores) if full_coverage_scores else None,
        },
    }


def _validate_case_distribution(summary: dict[str, Any]) -> None:
    numeric_band_count = len(
        [band for band in summary["band_counts"] if band != "無數字總分"]
    )
    if numeric_band_count < 3:
        raise CaseSetError(
            "Case set must cover at least three numeric BizScore bands."
        )
    if len(summary["industry_counts"]) < 3:
        raise CaseSetError("Case set must cover at least three primary industries.")


def _markdown_report(artifact: dict[str, Any]) -> str:
    lines = [
        "# BizScore v1 — 10 家正式公司測試集",
        "",
        f"版本：`{artifact['case_set_version']}`  ",
        f"建立時間：`{artifact['created_at']}`  ",
        f"Benchmark：`{artifact['catalog_version']}`  ",
        f"基準日：`{artifact['catalog_as_of']}`",
        "",
        "| 案例 | 公司 | 統編 | 狀態 | 產業 | 分數 | 區間 | 覆蓋率 | 年資中位數 | 資本中位數 | 年資 PR | 資本 PR |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for case in artifact["cases"]:
        company = case["input_company"]
        result = case["expected_bizscore"]
        peer = result["peer_benchmark"]
        lines.append(
            "| {case_id} | {name} | `{tax_id}` | {status} | {industry} | "
            "{score} | {band} | {coverage:.0%} | {age_median} | "
            "{capital_median} | {age_pr} | {capital_pr} |".format(
                case_id=case["case_id"],
                name=case["company_name"].replace("|", "／"),
                tax_id=case["tax_id"],
                status=company["status"]["description"] or company["status"]["code"],
                industry=peer["industry_code"] or "未分類",
                score=result["score"] if result["score"] is not None else "—",
                band=result["band"] or "資料不足／非現行",
                coverage=result["coverage"],
                age_median=_display_number(peer["company_age_median"]),
                capital_median=_display_number(
                    peer["registered_capital_median"], thousands=True
                ),
                age_pr=_display_number(peer["age_percentile"]),
                capital_pr=_display_number(peer["capital_percentile"]),
            )
        )
    summary = artifact["summary"]
    lines.extend(
        [
            "",
            "## 驗收摘要",
            "",
            f"- 有數字總分：{summary['scored_count']} 家。",
            f"- 無數字總分：{summary['unscored_count']} 家。",
            f"- 暫定分數：{summary['provisional_count']} 家。",
            "- 分數區間分布："
            + "、".join(
                f"{band} {count} 家"
                for band, count in summary["band_counts"].items()
            )
            + "。",
            "- 主要產業分布："
            + "、".join(
                f"{code} {count} 家"
                for code, count in summary["industry_counts"].items()
            )
            + "。",
            "- 缺少構面："
            + (
                "、".join(
                    f"{key} {count} 家"
                    for key, count in summary["missing_dimension_counts"].items()
                )
                if summary["missing_dimension_counts"]
                else "無"
            )
            + "。",
            "- 完整覆蓋案例分數範圍："
            f"{summary['full_coverage_score_range']['minimum']}–"
            f"{summary['full_coverage_score_range']['maximum']}。",
            "- BZV1-09 的 100 分為 85% 覆蓋率的暫定分數，缺少登記異動距今；"
            "此案用於驗證介面必須同時顯示部分資料提示，不可當成完整 100 分。",
            "- 每案均以保存的 CompanyData 連續重算兩次，並與 HTTP API 結果完全比對。",
            "",
            "> 本測試集只驗證固定規則、資料覆蓋與可重現性，不代表信用評等或合作安全性。",
            "",
        ]
    )
    return "\n".join(lines)


def _display_number(value: Any, *, thousands: bool = False) -> str:
    if value is None:
        return "—"
    if thousands:
        return f"{value:,.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
