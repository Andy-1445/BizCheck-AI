import copy

import pytest

from app.services.company_analysis import CompanyAnalysisInvalidOutputError, CompanyAnalysisService
from app.services.llm_comparison_prompt import build_company_comparison_prompt
from app.services.llm_prompt import LLMOutputSafetyError, build_company_analysis_prompt
from app.services.thu_prompt import (
    THU_VERIFICATION_REASONS, build_thu_response_schema,
    build_thu_system_instruction, validate_thu_output_contract,
)
from test_prompt_safety_reacceptance import THUStub, company_input, example, sample


@pytest.mark.parametrize("task", ["single", "comparison"])
def test_outgoing_schema_and_complete_example_teach_neutral_document_checks(task):
    prompt = (
        build_company_analysis_prompt(company_input()) if task == "single"
        else build_company_comparison_prompt(sample("company-comparison-input-v1.example.json"))
    )
    schema = build_thu_response_schema(prompt)
    name = "LLMVerificationItem" if task == "single" else "LLMComparisonVerificationItem"
    assert schema["$defs"][name]["properties"]["reason"]["enum"] == list(THU_VERIFICATION_REASONS)
    assert schema["properties"]["verification_items"]["maxItems"] == 4
    output = example(prompt)
    assert len(output["verification_items"]) == 3
    assert len({item["question"] for item in output["verification_items"]}) == 3
    assert all(item["reason"] in THU_VERIFICATION_REASONS for item in output["verification_items"])
    assert all(item["related_evidence_paths"] == [] for item in output["verification_items"])
    assert "allowed_verification_reasons_json" in build_thu_system_instruction(prompt)
    validate_thu_output_contract(prompt, output)
    output["verification_items"][0]["reason"] = "這是模型自行改寫的理由。"
    with pytest.raises(LLMOutputSafetyError, match="verification contract"):
        validate_thu_output_contract(prompt, output)


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["json_schema", "json_object", "off"])
async def test_neutral_reason_contract_is_enforced_even_if_gateway_ignores_schema(mode, caplog):
    payload = company_input()
    good = example(build_company_analysis_prompt(payload))
    bad = copy.deepcopy(good)
    bad["verification_items"][0]["reason"] = "這是未核准的自由生成理由。"
    provider = THUStub([bad, good], mode)
    result = await CompanyAnalysisService(provider, max_attempts=2).analyze(payload)
    assert result.attempts == 2
    assert "verification_contract_mismatch" in caplog.text
    assert "/verification_items/0/reason" in caplog.text
    assert bad["verification_items"][0]["reason"] not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("state", ["complete", "missing", "status_blocked"])
async def test_new_examples_generate_valid_reports_for_each_data_state(state):
    payload = company_input(state)
    output = example(build_company_analysis_prompt(payload))
    result = await CompanyAnalysisService(THUStub([output])).analyze(payload)
    assert len(result.analysis.verification_items) == 3
    assert result.analysis.status == ("completed" if state == "complete" else "insufficient_data")


def test_gateway_cannot_exceed_verification_question_limit():
    prompt = build_company_analysis_prompt(company_input())
    output = example(prompt)
    output["verification_items"] *= 2
    with pytest.raises(LLMOutputSafetyError, match="verification contract"):
        validate_thu_output_contract(prompt, output)
