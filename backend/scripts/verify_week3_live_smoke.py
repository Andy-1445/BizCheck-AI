"""Read-only GCIS/API smoke; does not call a paid AI provider.

Checks health, search, scores, comparisons and validation errors.
Does not send analysis requests, regardless of the configured AI provider.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = []
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=45) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200, ready.status_code
        assert all(item["status"] == "ok" for item in ready.json()["checks"].values())
        checks.append({"name": "readiness", "http_status": 200, "passed": True})

        search = client.get("/api/v1/companies/search", params={"q": "宏碁"})
        assert search.status_code == 200, search.status_code
        assert search.json()["data"]
        checks.append({"name": "name_search", "http_status": 200, "passed": True})

        for tax_id in ("20828393", "22099131", "16108153"):
            response = client.get(f"/api/v1/companies/{tax_id}/bizscore")
            assert response.status_code == 200, response.status_code
            data = response.json()["data"]
            assert data["company"]["tax_id"] == tax_id
            score = data["bizscore"]
            if data["company"]["status"]["code"] != "01":
                assert score["score"] is None and score["band"] is None
            checks.append({"name": f"bizscore_{tax_id}", "http_status": 200,
                           "status": data["company"]["status"], "score": score["score"],
                           "coverage": score["coverage"], "provisional": score["provisional"],
                           "passed": True})

        for tax_ids in (["20828393", "22099131"], ["20828393", "22099131", "16108153"]):
            response = client.post("/api/v1/companies/compare", json={"tax_ids": tax_ids})
            assert response.status_code == 200, response.status_code
            items = response.json()["data"]["items"]
            assert [item["company"]["tax_id"] for item in items] == tax_ids
            checks.append({"name": f"comparison_{len(tax_ids)}", "http_status": 200, "passed": True})

        invalid = client.get("/api/v1/companies/123/bizscore")
        assert invalid.status_code == 422
        assert invalid.json()["detail"]["code"] == "INVALID_TAX_ID"
        checks.append({"name": "invalid_tax_id", "http_status": 422, "passed": True})

    # No analysis POSTs: this reusable smoke must not unexpectedly incur model costs.
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "passed": True,
              "live_gcis": True, "ai_provider_calls": 0, "checks": checks,
              "boundary": "GCIS and deterministic APIs only; not live THU acceptance."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
