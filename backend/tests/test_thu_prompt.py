from __future__ import annotations

import json
from pathlib import Path

from app.services.llm_comparison_prompt import (
    build_company_comparison_prompt,
    validate_company_comparison_output,
)
from app.services.llm_prompt import (
    build_company_analysis_prompt,
    validate_company_analysis_output,
)
from app.services.llm_output_normalizer import normalize_evidence_manifest
from app.services.thu_prompt import build_thu_system_instruction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = PROJECT_ROOT / "samples" / "llm"


def _read_sample(name: str) -> dict[str, object]:
    payload = json.loads((SAMPLE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _extract_example(instruction: str) -> dict[str, object]:
    opening = "<complete_output_example>"
    closing = "</complete_output_example>"
    start = instruction.index(opening) + len(opening)
    end = instruction.index(closing)
    payload = json.loads(instruction[start:end].strip())
    assert isinstance(payload, dict)
    return payload


def _extract_schema(instruction: str) -> dict[str, object]:
    opening = "<response_json_schema>"
    closing = "</response_json_schema>"
    start = instruction.index(opening) + len(opening)
    end = instruction.index(closing)
    payload = json.loads(instruction[start:end])
    assert isinstance(payload, dict)
    return payload


def test_thu_company_prompt_contains_complete_input_consistent_valid_example() -> None:
    analysis_input = _read_sample("company-analysis-input-v1.example.json")
    prompt = build_company_analysis_prompt(analysis_input)

    instruction = build_thu_system_instruction(prompt)
    example = _extract_example(instruction)
    response_schema = _extract_schema(instruction)

    assert instruction.startswith("你是 BizCheck AI 的企業公開資料解釋層")
    assert "所有 object 都禁止額外欄位" in instruction
    assert "findings 每一項必須且只能包含" in instruction
    assert "caveat 不得省略" in instruction
    assert "provenance.evidence_paths_used" in instruction
    assert "allowed_evidence_paths_json" in instruction
    assert "不得寫阿拉伯數字" in instruction
    assert "中文數字" in instruction
    assert "即使有 evidence" in instruction
    assert "findings 必須逐項沿用" in instruction
    assert "<response_json_schema>" in instruction
    assert example["provenance"]["evidence_paths_used"] == []
    assert [item["topic"] for item in example["findings"]] == [
        "registration_status",
        "company_age",
        "registered_capital_scale",
        "registration_change_recency",
        "peer_relative_position",
    ]
    assert example["findings"][2]["observation"] == (
        "登記資本已有公開欄位可供核對，規模分數由固定規則計算。"
    )
    assert response_schema["$defs"]["LLMAnalysisProvenance"]["properties"][
        "evidence_paths_used"
    ]["maxItems"] == 0
    evidence_schema = response_schema["$defs"]["LLMAnalysisFinding"][
        "properties"
    ]["evidence_paths"]["items"]
    assert "/company/status/description" in evidence_schema["enum"]
    assert "/company/status" not in evidence_schema["enum"]
    limitation_codes = response_schema["$defs"]["LLMAnalysisLimitation"][
        "properties"
    ]["code"]["enum"]
    assert limitation_codes == ["public_data_only", "ai_generated"]
    assert response_schema["properties"]["overall_observation"][
        "pattern"
    ] == (
        r"^[^0-9０-９%％]*$"
    )
    assert response_schema["$defs"]["LLMAnalysisFinding"]["properties"][
        "observation"
    ]["pattern"] == (
        r"^[^0-9０-９%％]*$"
    )
    normalized = normalize_evidence_manifest(
        example,
        task="company_analysis",
    )
    validated = validate_company_analysis_output(analysis_input, normalized)
    assert validated.provenance.company_tax_id == analysis_input["company"]["tax_id"]


def test_thu_comparison_prompt_contains_complete_input_consistent_valid_example() -> None:
    analysis_input = _read_sample("company-comparison-input-v1.example.json")
    prompt = build_company_comparison_prompt(analysis_input)

    instruction = build_thu_system_instruction(prompt)
    example = _extract_example(instruction)
    response_schema = _extract_schema(instruction)

    assert instruction.startswith("你是 BizCheck AI 的企業公開資料比較解釋層")
    assert "company_observations 每一項必須且只能包含" in instruction
    assert "comparison_observations 每一項必須且只能包含" in instruction
    assert "company_observations 與 comparison_observations 必須逐項沿用" in instruction
    assert "findings 必須逐項沿用" not in instruction
    assert example["provenance"]["evidence_paths_used"] == []
    assert response_schema["$defs"]["LLMComparisonProvenance"]["properties"][
        "evidence_paths_used"
    ]["maxItems"] == 0
    normalized = normalize_evidence_manifest(
        example,
        task="company_comparison_analysis",
    )
    validated = validate_company_comparison_output(analysis_input, normalized)
    assert validated.provenance.requested_tax_ids == (
        analysis_input["comparison"]["meta"]["requested_tax_ids"]
    )
    assert len(validated.company_observations) in {2, 3}
