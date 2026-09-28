"""Local-only OpenAI Responses stub for browser acceptance.

This script never calls an external model. It validates the minimum outgoing
request shape, waits briefly so the loading state can be observed, then returns
the repository's reviewed Prompt v1 output fixture through an OpenAI-compatible
``POST /v1/responses`` endpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FIXTURE = (
    PROJECT_ROOT / "samples" / "llm" / "company-analysis-output-v1.example.json"
)
DISCLAIMER = (
    "AI 內容僅解釋已提供資料與計算結果，可能有錯誤或遺漏，"
    "不應取代獨立查核與專業意見。"
)


class ResponsesHandler(BaseHTTPRequestHandler):
    server_version = "BizCheckAcceptanceStub/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._send_json(HTTPStatus.OK, {"status": "ready"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/responses":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        request = self._read_json()
        verified_input = self._extract_verified_input(request)
        if (
            request is None
            or not self._valid_request(request)
            or verified_input is None
        ):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": "invalid acceptance request"}},
            )
            return

        time.sleep(self.server.response_delay_seconds)  # type: ignore[attr-defined]
        analysis = self._analysis_for_input(verified_input)
        self._send_json(
            HTTPStatus.OK,
            {
                "id": "resp_bizcheck_browser_acceptance",
                "status": "completed",
                "model": "gpt-5.4-browser-acceptance-stub",
                "created_at": int(time.time()),
                "service_tier": "local-acceptance",
                "output_text": json.dumps(analysis, ensure_ascii=False),
                "usage": {
                    "input_tokens": 1720,
                    "output_tokens": 612,
                    "total_tokens": 2332,
                },
            },
            extra_headers={"x-request-id": "req_bizcheck_browser_acceptance"},
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            value = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _valid_request(request: dict[str, Any]) -> bool:
        metadata = request.get("metadata")
        text = request.get("text")
        return (
            request.get("store") is False
            and isinstance(request.get("instructions"), str)
            and isinstance(request.get("input"), str)
            and isinstance(metadata, dict)
            and metadata.get("task") == "company_analysis"
            and isinstance(text, dict)
            and isinstance(text.get("format"), dict)
            and text["format"].get("strict") is True
        )

    @staticmethod
    def _extract_verified_input(
        request: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if request is None or not isinstance(request.get("input"), str):
            return None
        user_prompt = request["input"]
        start_marker = "<verified_input_json>\n"
        end_marker = "\n</verified_input_json>"
        start = user_prompt.find(start_marker)
        end = user_prompt.find(end_marker)
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(user_prompt[start + len(start_marker) : end])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _analysis_for_input(verified_input: dict[str, Any]) -> dict[str, Any]:
        company = verified_input["company"]
        bizscore = verified_input["bizscore"]
        source_meta = verified_input["source_meta"]
        no_numeric_score = (
            bizscore.get("score") is None or bizscore.get("coverage", 0) < 0.8
        )
        if company.get("tax_id") == "20828393" and not no_numeric_score:
            output = json.loads(OUTPUT_FIXTURE.read_text(encoding="utf-8"))
            age_finding = next(
                item for item in output["findings"] if item["topic"] == "company_age"
            )
            age_paths = [
                "/company/company_age_years",
                "/bizscore/dimensions/1/evidence/1",
            ]
            age_finding["evidence_paths"].extend(age_paths)
            output["provenance"]["evidence_paths_used"].extend(age_paths)
            return output

        evidence_paths: list[str] = ["/company/status/description"]
        limitations: list[dict[str, Any]] = [
            {
                "code": "public_data_only",
                "message": "分析僅使用政府公開登記資料與固定規則計算結果。",
                "related_evidence_paths": ["/source_meta/source"],
            },
            {
                "code": "ai_generated",
                "message": "文字由 AI 產生，可能有錯誤或遺漏。",
                "related_evidence_paths": [],
            },
        ]
        evidence_paths.append("/source_meta/source")

        if source_meta.get("partial") or source_meta.get("warnings"):
            partial_path = (
                "/source_meta/warnings/0"
                if source_meta.get("warnings")
                else "/source_meta/partial"
            )
            limitations.append(
                {
                    "code": "partial_source_data",
                    "message": "部分來源欄位缺漏或附有資料提醒。",
                    "related_evidence_paths": [partial_path],
                }
            )
            evidence_paths.append(partial_path)

        dimensions = bizscore.get("dimensions", [])
        missing_dimensions = bizscore.get("missing_dimensions", [])
        unavailable_index = next(
            (
                index
                for index, dimension in enumerate(dimensions)
                if not dimension.get("available")
            ),
            None,
        )
        if missing_dimensions or unavailable_index is not None:
            missing_path = (
                "/bizscore/missing_dimensions/0"
                if missing_dimensions
                else f"/bizscore/dimensions/{unavailable_index}/available"
            )
            limitations.append(
                {
                    "code": "missing_dimension",
                    "message": "至少一個 BizScore 構面缺少可用資料。",
                    "related_evidence_paths": [missing_path],
                }
            )
            evidence_paths.append(missing_path)

        if bizscore.get("provisional"):
            limitations.append(
                {
                    "code": "provisional_score",
                    "message": "BizScore 因資料覆蓋狀況標記為暫定結果。",
                    "related_evidence_paths": ["/bizscore/provisional"],
                }
            )
            evidence_paths.append("/bizscore/provisional")

        if no_numeric_score:
            limitations.append(
                {
                    "code": "no_numeric_score",
                    "message": "目前資料不足以顯示數字總分。",
                    "related_evidence_paths": ["/bizscore/score"],
                }
            )
            evidence_paths.append("/bizscore/score")

        peer_dimension = bizscore.get("peer_benchmark", {}).get("dimension", {})
        benchmark_unavailable = (
            bizscore.get("benchmark", {}).get("snapshot_version") is None
            or not peer_dimension.get("available")
        )
        if benchmark_unavailable:
            limitations.append(
                {
                    "code": "benchmark_unavailable",
                    "message": "本次沒有可用的正式同業比較結果。",
                    "related_evidence_paths": [
                        "/bizscore/benchmark/snapshot_version"
                    ],
                }
            )
            evidence_paths.append("/bizscore/benchmark/snapshot_version")

        findings = []
        if not no_numeric_score:
            findings.append(
                {
                    "topic": "registration_status",
                    "title": "登記狀態資料",
                    "observation": "本次報告已納入官方登記狀態欄位。",
                    "evidence_paths": ["/company/status/description"],
                    "caveat": "登記狀態不代表實際營運、付款或履約狀況。",
                }
            )

        return {
            "schema_version": "1.0",
            "status": "insufficient_data" if no_numeric_score else "completed",
            "headline": (
                "目前公開資料不足，先依查核清單補充資訊"
                if no_numeric_score
                else "公開登記資料已整理，仍需完成合作條件查核"
            ),
            "overall_observation": (
                "部分 BizScore 構面缺少可用資料，因此本次不顯示數字總分；可先核對簽約主體、代表權限與付款條件。"
                if no_numeric_score
                else "本次只整理輸入中的公開登記資料與固定規則結果，不包含個別合約、付款安排或實際履約紀錄。"
            ),
            "findings": findings,
            "verification_items": [
                {
                    "priority": "優先",
                    "question": "簽約主體、代表權限與付款條件是否已逐項確認？",
                    "reason": "公開登記資料不包含本次交易的合約與付款安排。",
                    "related_evidence_paths": ["/company/status/description"],
                }
            ],
            "limitations": limitations,
            "provenance": {
                "input_schema_version": verified_input["schema_version"],
                "prompt_version": "1.0",
                "company_tax_id": company["tax_id"],
                "bizscore_version": bizscore["version"],
                "benchmark_catalog_version": bizscore["benchmark"][
                    "catalog_version"
                ],
                "data_as_of": bizscore["as_of"],
                "disclaimer_version": verified_input["disclaimer_version"],
                "evidence_paths_used": list(dict.fromkeys(evidence_paths)),
            },
            "disclaimer": DISCLAIMER,
        }

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--delay-seconds", type=float, default=1.5)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ResponsesHandler)
    server.response_delay_seconds = max(0.0, args.delay_seconds)  # type: ignore[attr-defined]
    server.serve_forever()


if __name__ == "__main__":
    main()
