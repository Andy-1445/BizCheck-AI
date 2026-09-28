"""Capture local API demo snapshots or replay them without GCIS/LLM calls.

Run from the repository root: backend/.venv/Scripts/python.exe backend/scripts/verify_demo_cases.py
Capture is explicit, refuses to overwrite existing evidence, and validates all cases first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas.bizscore import CompanyBizScoreResponse
from app.services.benchmark_catalog import BenchmarkCatalogService

MANIFEST = ROOT / "samples/demo/demo-cases-v1.json"
SNAPSHOTS = ROOT / "samples/demo/snapshots-2026-09-24.json"


def verify_snapshot(case, raw, service):
    response = CompanyBizScoreResponse.model_validate(raw)
    company, score = response.data.company, response.data.bizscore
    assert company.tax_id == case["tax_id"], "Company identifier mismatch"
    assert company.name == case["name"], "Company name changed; review demo selection"
    for key, expected in case["expected"].items():
        assert getattr(score, key) == expected, f'{case["id"]}: {key} changed'
    replay = service.calculate_company_bizscore(
        company, input_partial=response.meta.partial,
        input_warnings=response.meta.warnings,
    )
    assert replay.model_dump(mode="json") == score.model_dump(mode="json"), "Replay mismatch"
    return response


def verify_all(bundle):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert bundle["manifest_version"] == manifest["version"]
    service = BenchmarkCatalogService(
        ROOT / "backend/data/benchmarks/benchmark-catalog-2026-08-01-v1.json",
        ROOT / "backend/data/benchmarks/bizcheck-benchmark.sqlite3",
        expected_catalog_version=manifest["catalog_version"],
    )
    assert service.catalog.as_of.isoformat() == manifest["as_of"]
    assert set(bundle["responses"]) == {c["id"] for c in manifest["cases"]}
    results = []
    for case in manifest["cases"]:
        response = verify_snapshot(case, bundle["responses"][case["id"]], service)
        results.append({"case": case["id"], "score": response.data.bizscore.score,
                        "coverage": response.data.bizscore.coverage, "replay": "passed"})

    # Isolate missing data from status gating. This is NOT a real company snapshot.
    company = CompanyBizScoreResponse.model_validate(bundle["responses"]["DEMO-01"]).data.company
    controlled = company.model_copy(update={"last_changed_at": None, "business_items": []})
    score = service.calculate_company_bizscore(controlled, input_partial=True)
    assert controlled.status.code == "01"
    assert score.score is None and score.band is None and score.coverage == 0.65
    assert set(score.missing_dimensions) == {"registration_change_recency", "peer_relative_position"}
    results.append({"case": "CONTROLLED-MISSING-ONLY", "synthetic": True,
                    "score": score.score, "coverage": score.coverage, "replay": "passed"})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if args.capture:
        import httpx
        if SNAPSHOTS.exists():
            raise SystemExit("Snapshot already exists; refusing to overwrite evidence.")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        bundle = {"manifest_version": manifest["version"], "source": "Local BizCheck API / GCIS",
                  "responses": {}}
        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=60) as client:
            for case in manifest["cases"]:
                response = client.get(f'/api/v1/companies/{case["tax_id"]}/bizscore')
                response.raise_for_status()
                bundle["responses"][case["id"]] = response.json()
        results = verify_all(bundle)
        with SNAPSHOTS.open("x", encoding="utf-8") as output:
            json.dump(bundle, output, ensure_ascii=False, indent=2)
    else:
        results = verify_all(json.loads(SNAPSHOTS.read_text(encoding="utf-8")))
    print(json.dumps({"passed": len(results), "results": results}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
