from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import httpx


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/api/v1"


class AcceptanceError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the Week 2 live BizScore API against the formal case set."
    )
    parser.add_argument("--case-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    args = parser.parse_args()

    try:
        result = verify_live_api(
            case_set_path=args.case_set,
            api_base_url=args.api_base_url,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (AcceptanceError, httpx.HTTPError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def verify_live_api(
    *,
    case_set_path: Path,
    api_base_url: str,
) -> dict[str, Any]:
    case_set = json.loads(case_set_path.read_text(encoding="utf-8"))
    cases = case_set.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise AcceptanceError("The formal case set must contain exactly 10 cases.")

    base_url = api_base_url.rstrip("/")
    service_url = (
        base_url[: -len("/api/v1")]
        if base_url.endswith("/api/v1")
        else base_url
    )
    verified_cases: list[dict[str, Any]] = []
    with httpx.Client(timeout=30) as client:
        health = client.get(f"{service_url}/health/ready")
        _expect_status(health, 200, "health")

        for case in cases:
            tax_id = case["tax_id"]
            response = client.get(f"{base_url}/companies/{tax_id}/bizscore")
            _expect_status(response, 200, tax_id)
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, dict):
                raise AcceptanceError(f"{tax_id}: response.data is missing.")

            company = data.get("company")
            bizscore = data.get("bizscore")
            if company != case["input_company"]:
                raise AcceptanceError(
                    f"{tax_id}: live normalized CompanyData differs from the case set."
                )
            if bizscore != case["expected_bizscore"]:
                raise AcceptanceError(
                    f"{tax_id}: live BizScore differs from the recorded expectation."
                )

            dimensions = bizscore.get("dimensions")
            peer = bizscore.get("peer_benchmark")
            if not isinstance(dimensions, list) or len(dimensions) != 5:
                raise AcceptanceError(f"{tax_id}: expected exactly five dimensions.")
            if not isinstance(peer, dict):
                raise AcceptanceError(f"{tax_id}: peer_benchmark is missing.")

            verified_cases.append(
                {
                    "case_id": case["case_id"],
                    "tax_id": tax_id,
                    "company_name": case["company_name"],
                    "score": bizscore["score"],
                    "band": bizscore["band"],
                    "coverage": bizscore["coverage"],
                    "provisional": bizscore["provisional"],
                    "industry_code": bizscore["benchmark"]["industry_code"],
                    "sample_size": peer["sample_size"],
                    "exact_match": True,
                }
            )

        invalid_tax_id = client.get(f"{base_url}/companies/123/bizscore")
        _expect_error(
            invalid_tax_id,
            expected_status=422,
            expected_code="INVALID_TAX_ID",
            expected_retryable=False,
        )

        not_found = client.get(f"{base_url}/companies/00000000/bizscore")
        _expect_error(
            not_found,
            expected_status=404,
            expected_code="COMPANY_NOT_FOUND",
            expected_retryable=False,
        )

    return {
        "passed": True,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "api_base_url": base_url,
        "case_set_version": case_set["case_set_version"],
        "catalog_version": case_set["catalog_version"],
        "verified_case_count": len(verified_cases),
        "exact_match_count": sum(case["exact_match"] for case in verified_cases),
        "cases": verified_cases,
        "error_contracts": [
            {
                "scenario": "invalid_tax_id",
                "http_status": 422,
                "code": "INVALID_TAX_ID",
                "retryable": False,
                "passed": True,
            },
            {
                "scenario": "company_not_found",
                "http_status": 404,
                "code": "COMPANY_NOT_FOUND",
                "retryable": False,
                "passed": True,
            },
        ],
    }


def _expect_status(response: httpx.Response, expected: int, label: str) -> None:
    if response.status_code != expected:
        raise AcceptanceError(
            f"{label}: expected HTTP {expected}, got {response.status_code}: "
            f"{response.text[:300]}"
        )


def _expect_error(
    response: httpx.Response,
    *,
    expected_status: int,
    expected_code: str,
    expected_retryable: bool,
) -> None:
    _expect_status(response, expected_status, expected_code)
    detail = response.json().get("detail")
    if not isinstance(detail, dict):
        raise AcceptanceError(f"{expected_code}: error detail is missing.")
    actual = {
        "code": detail.get("code"),
        "retryable": detail.get("retryable"),
    }
    expected = {
        "code": expected_code,
        "retryable": expected_retryable,
    }
    if actual != expected:
        raise AcceptanceError(
            f"{expected_code}: expected error contract {expected}, got {actual}."
        )


if __name__ == "__main__":
    raise SystemExit(main())
