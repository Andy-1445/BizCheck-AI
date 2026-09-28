import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas.llm_analysis import (
    AI_ANALYSIS_DISCLAIMER,
    CompanyAnalysisLLMInput,
    CompanyAnalysisLLMOutput,
    build_llm_input_json_schema,
    build_llm_output_json_schema,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIRECTORY = PROJECT_ROOT / "docs" / "schemas"
SAMPLE_DIRECTORY = PROJECT_ROOT / "samples" / "llm"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _input_example() -> dict:
    return _read_json(SAMPLE_DIRECTORY / "company-analysis-input-v1.example.json")


def _output_example() -> dict:
    return _read_json(SAMPLE_DIRECTORY / "company-analysis-output-v1.example.json")


def test_exported_input_schema_matches_model() -> None:
    assert _read_json(
        SCHEMA_DIRECTORY / "llm-company-analysis-input-v1.schema.json"
    ) == build_llm_input_json_schema()


def test_exported_output_schema_matches_model() -> None:
    assert _read_json(
        SCHEMA_DIRECTORY / "llm-company-analysis-output-v1.schema.json"
    ) == build_llm_output_json_schema()


def test_saved_examples_validate() -> None:
    input_model = CompanyAnalysisLLMInput.model_validate(_input_example())
    output_model = CompanyAnalysisLLMOutput.model_validate(_output_example())

    assert input_model.company.tax_id == "20828393"
    assert output_model.provenance.company_tax_id == "20828393"
    assert output_model.disclaimer == AI_ANALYSIS_DISCLAIMER


def test_input_rejects_unknown_fields() -> None:
    payload = _input_example()
    payload["unverified_news"] = "不得加入模型輸入"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CompanyAnalysisLLMInput.model_validate(payload)


def test_input_requires_exactly_five_bizscore_dimensions() -> None:
    payload = _input_example()
    payload["bizscore"]["dimensions"] = payload["bizscore"]["dimensions"][:-1]

    with pytest.raises(ValidationError, match="exactly the five BizScore v1 keys"):
        CompanyAnalysisLLMInput.model_validate(payload)


def test_output_rejects_llm_generated_score_or_band() -> None:
    payload = _output_example()
    payload["score"] = 100
    payload["band"] = "安全公司"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CompanyAnalysisLLMOutput.model_validate(payload)


def test_output_rejects_invalid_evidence_root() -> None:
    payload = _output_example()
    payload["findings"][0]["evidence_paths"] = ["/web_search/result/0"]

    with pytest.raises(ValidationError, match="String should match pattern"):
        CompanyAnalysisLLMOutput.model_validate(payload)


def test_output_requires_public_data_and_ai_limitations() -> None:
    payload = _output_example()
    payload["limitations"] = [
        {
            "code": "partial_source_data",
            "message": "部分資料缺漏。",
            "related_evidence_paths": [],
        },
        {
            "code": "missing_dimension",
            "message": "部分構面無法計分。",
            "related_evidence_paths": [],
        },
    ]

    with pytest.raises(ValidationError, match="public_data_only and ai_generated"):
        CompanyAnalysisLLMOutput.model_validate(payload)


def test_output_requires_complete_evidence_manifest() -> None:
    payload = copy.deepcopy(_output_example())
    payload["provenance"]["evidence_paths_used"] = [
        path
        for path in payload["provenance"]["evidence_paths_used"]
        if path != "/company/status/description"
    ]

    with pytest.raises(ValidationError, match="must exactly match all referenced paths"):
        CompanyAnalysisLLMOutput.model_validate(payload)
