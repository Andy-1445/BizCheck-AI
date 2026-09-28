import json
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.bizscore import CompanyBizScoreResponse
from app.schemas.company_comparison import CompanyComparisonRequest
from app.schemas.company import ResponseMeta
from app.schemas.llm_analysis import (
    AI_ANALYSIS_DISCLAIMER,
    CompanyAnalysisLLMInput,
    CompanyAnalysisLLMOutput,
    build_llm_input_json_schema,
    build_llm_output_json_schema,
)
from app.schemas.llm_comparison_analysis import (
    AI_COMPARISON_DISCLAIMER,
    CompanyComparisonLLMInput,
    CompanyComparisonLLMOutput,
    build_comparison_llm_input_json_schema,
    build_comparison_llm_output_json_schema,
)
from app.services.company_comparison import build_company_comparison
from app.services.llm_comparison_prompt import validate_company_comparison_output
from app.services.llm_prompt import validate_company_analysis_output


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_SET_PATH = (
    PROJECT_ROOT
    / "samples"
    / "bizscore"
    / "bizscore-v1-10-company-test-set-2026-08-23.json"
)
SCHEMA_DIRECTORY = PROJECT_ROOT / "docs" / "schemas"
SAMPLE_DIRECTORY = PROJECT_ROOT / "samples" / "llm"


def main() -> None:
    case_set = _read_json(CASE_SET_PATH)
    case = next(item for item in case_set["cases"] if item["case_id"] == "BZV1-10")

    input_example = CompanyAnalysisLLMInput.model_validate(
        {
            "schema_version": "1.0",
            "task": "company_analysis",
            "language": "zh-TW",
            "disclaimer_version": "1.0",
            "company": case["input_company"],
            "bizscore": case["expected_bizscore"],
            "source_meta": {
                "source": "GCIS",
                "provider": "經濟部商業發展署",
                "fetched_at": "2026-08-23T12:00:00+08:00",
                "partial": False,
                "warnings": [],
            },
        }
    )

    evidence_paths = [
        "/company/status/description",
        "/company/established_at",
        "/company/capital/registered",
        "/bizscore/peer_benchmark/peer_index",
        "/bizscore/peer_benchmark/sample_size",
        "/source_meta/source",
    ]
    output_example = CompanyAnalysisLLMOutput.model_validate(
        {
            "schema_version": "1.0",
            "status": "completed",
            "headline": "公開登記資料多數完整，合作前仍應補充交易條件查核",
            "overall_observation": (
                "公司目前為核准設立，成立時間與登記資本在本次公開資料中均可取得；"
                "同業相對位置較高，但這些資訊不代表付款或履約能力。"
            ),
            "findings": [
                {
                    "topic": "registration_status",
                    "title": "登記狀態",
                    "observation": "目前官方登記狀態為核准設立。",
                    "evidence_paths": ["/company/status/description"],
                    "caveat": "登記狀態不等同實際營運或履約狀況。",
                },
                {
                    "topic": "company_age",
                    "title": "成立歷史",
                    "observation": "公開資料顯示公司設立日期為 1979 年 7 月 18 日。",
                    "evidence_paths": ["/company/established_at"],
                    "caveat": "成立時間本身不能證明目前財務或營運狀況。",
                },
                {
                    "topic": "registered_capital_scale",
                    "title": "登記資本",
                    "observation": "本次資料中的登記資本額為新臺幣 400 億元。",
                    "evidence_paths": ["/company/capital/registered"],
                    "caveat": "登記資本額不等同可動用現金、營收或清償能力。",
                },
                {
                    "topic": "peer_relative_position",
                    "title": "同業相對位置",
                    "observation": "年資與登記資本的同業綜合相對位置為 PR 99.2。",
                    "evidence_paths": [
                        "/bizscore/peer_benchmark/peer_index",
                        "/bizscore/peer_benchmark/sample_size",
                    ],
                    "caveat": "此指標不是信用、市場或投資排名。",
                },
            ],
            "verification_items": [
                {
                    "priority": "優先",
                    "question": "本次合作的簽約主體、代表權限與付款條件是否已確認？",
                    "reason": "公開登記資料不包含個別合約與付款安排。",
                    "related_evidence_paths": [],
                },
                {
                    "priority": "一般",
                    "question": "是否需要向公司索取近期履約案例或財務佐證？",
                    "reason": "登記資本與成立時間不能取代實際履約能力查核。",
                    "related_evidence_paths": [
                        "/company/established_at",
                        "/company/capital/registered",
                    ],
                },
            ],
            "limitations": [
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
            ],
            "provenance": {
                "input_schema_version": "1.0",
                "prompt_version": "1.0",
                "company_tax_id": "20828393",
                "bizscore_version": "1.0",
                "benchmark_catalog_version": (
                    "benchmark-catalog-2026-08-01-v1"
                ),
                "data_as_of": "2026-08-01",
                "disclaimer_version": "1.0",
                "evidence_paths_used": evidence_paths,
            },
            "disclaimer": AI_ANALYSIS_DISCLAIMER,
        }
    )
    output_example = validate_company_analysis_output(
        input_example,
        output_example,
    )
    comparison_input, comparison_output = _build_comparison_examples(case_set)

    _write_json(
        SCHEMA_DIRECTORY / "llm-company-analysis-input-v1.schema.json",
        build_llm_input_json_schema(),
    )
    _write_json(
        SCHEMA_DIRECTORY / "llm-company-analysis-output-v1.schema.json",
        build_llm_output_json_schema(),
    )
    _write_json(
        SAMPLE_DIRECTORY / "company-analysis-input-v1.example.json",
        input_example.model_dump(mode="json"),
    )
    _write_json(
        SAMPLE_DIRECTORY / "company-analysis-output-v1.example.json",
        output_example.model_dump(mode="json"),
    )
    _write_json(
        SCHEMA_DIRECTORY / "llm-company-comparison-input-v1.schema.json",
        build_comparison_llm_input_json_schema(),
    )
    _write_json(
        SCHEMA_DIRECTORY / "llm-company-comparison-output-v1.schema.json",
        build_comparison_llm_output_json_schema(),
    )
    _write_json(
        SAMPLE_DIRECTORY / "company-comparison-input-v1.example.json",
        comparison_input.model_dump(mode="json"),
    )
    _write_json(
        SAMPLE_DIRECTORY / "company-comparison-output-v1.example.json",
        comparison_output.model_dump(mode="json"),
    )


def _build_comparison_examples(
    case_set: dict,
) -> tuple[CompanyComparisonLLMInput, CompanyComparisonLLMOutput]:
    selected = [
        next(item for item in case_set["cases"] if item["case_id"] == case_id)
        for case_id in ("BZV1-10", "BZV1-09")
    ]
    source_meta = ResponseMeta(
        fetched_at=datetime(2026, 8, 24, 17, 30, tzinfo=timezone.utc),
    )
    responses = [
        CompanyBizScoreResponse.model_validate(
            {
                "data": {
                    "company": item["input_company"],
                    "bizscore": item["expected_bizscore"],
                },
                "meta": source_meta.model_dump(mode="json"),
            }
        )
        for item in selected
    ]
    tax_ids = [response.data.company.tax_id for response in responses]
    comparison = build_company_comparison(
        CompanyComparisonRequest(tax_ids=tax_ids),
        responses,
        generated_at=datetime(2026, 8, 24, 9, 30, tzinfo=timezone.utc),
    )
    analysis_input = CompanyComparisonLLMInput(comparison=comparison)
    company_paths = [
        f"/comparison/data/items/{index}/company/status/description"
        for index in range(len(tax_ids))
    ]
    peer_scope_path = "/comparison/data/context/peer_comparison_scope"
    limitations: list[dict[str, object]] = [
        {
            "code": "public_data_only",
            "message": "內容僅依政府公開登記資料整理。",
            "related_evidence_paths": [],
        },
        {
            "code": "ai_generated",
            "message": "文字由人工智慧依已驗證輸入整理。",
            "related_evidence_paths": [],
        },
        {
            "code": "not_ranked",
            "message": "公司順序只沿用使用者選取順序，不代表名次。",
            "related_evidence_paths": [],
        },
    ]
    conditional_limitations = (
        (
            "partial_source_data",
            comparison.meta.has_partial_source_data,
            "部分來源欄位未完整取得。",
        ),
        (
            "provisional_score",
            comparison.meta.has_provisional_scores,
            "部分分數帶有暫定狀態。",
        ),
        (
            "no_numeric_score",
            comparison.meta.has_unscored_companies,
            "至少一家公司沒有可用總分。",
        ),
        (
            "cross_industry_comparison",
            comparison.data.context.peer_comparison_scope
            == "different_industry_snapshots",
            "各公司使用不同產業同業基準。",
        ),
        (
            "benchmark_unavailable",
            comparison.data.context.peer_comparison_scope == "unavailable",
            "同業基準資料無法完整使用。",
        ),
    )
    for code, required, message in conditional_limitations:
        if required:
            limitations.append(
                {
                    "code": code,
                    "message": message,
                    "related_evidence_paths": [],
                }
            )
    analysis_output = CompanyComparisonLLMOutput.model_validate(
        {
            "status": (
                "insufficient_data"
                if comparison.meta.has_unscored_companies
                else "completed"
            ),
            "headline": "公開登記資料並列觀察",
            "overall_observation": (
                "請並列查看各公司公開欄位，並針對實際合作條件另行查核。"
            ),
            "company_observations": [
                {
                    "tax_id": tax_id,
                    "topic": "registration_status",
                    "title": "登記狀態欄位",
                    "observation": "公開登記狀態已有欄位可供查看。",
                    "evidence_paths": [company_paths[index]],
                    "caveat": (
                        "登記狀態不能代表付款、履約或實際營運狀況。"
                    ),
                }
                for index, tax_id in enumerate(tax_ids)
            ],
            "comparison_observations": [
                {
                    "topic": "peer_scope",
                    "title": "同業比較範圍",
                    "observation": (
                        "各公司同業相對位置應依各自產業基準分別閱讀。"
                    ),
                    "evidence_paths": [peer_scope_path],
                    "caveat": "跨產業同業相對位置不能形成共同排名。",
                }
            ],
            "verification_items": [
                {
                    "priority": "優先",
                    "question": "是否已向各公司核對交易條件與履約文件？",
                    "reason": "公開登記資料不包含個別交易安排。",
                    "related_evidence_paths": [],
                }
            ],
            "limitations": limitations,
            "provenance": {
                "requested_tax_ids": tax_ids,
                "benchmark_catalog_version": (
                    comparison.data.context.benchmark_catalog_version
                ),
                "data_as_of": comparison.data.context.benchmark_as_of,
                "evidence_paths_used": [*company_paths, peer_scope_path],
            },
            "disclaimer": AI_COMPARISON_DISCLAIMER,
        }
    )
    return analysis_input, validate_company_comparison_output(
        analysis_input,
        analysis_output,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
