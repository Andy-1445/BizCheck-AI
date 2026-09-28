from __future__ import annotations

import copy

import pytest

from app.services.capital_claim_safety import has_unsupported_funding_claim
from app.services.company_analysis import CompanyAnalysisInvalidOutputError, CompanyAnalysisService
from app.services.company_comparison_analysis import CompanyComparisonAnalysisService
from app.services.llm_cache import AsyncLRUTTLCache
from app.services.llm_comparison_prompt import build_company_comparison_prompt, validate_company_comparison_output
from app.services.llm_prompt import LLMOutputSafetyError, build_company_analysis_prompt, validate_company_analysis_output
from test_prompt_safety_reacceptance import THUStub, company_input, example, sample


LIVE_FAILURE = "實收資本額與登記資本額可能反映實際資金動員能力。"


@pytest.mark.parametrize("text", [
    LIVE_FAILURE,
    "較高的登記資本暗示資金調度能力較強。",
    "實收資本可作為籌資能力的參考。",
    "資本規模可能意味著財務實力。",
    "公司的融資能力不足。",
    "登記資本顯示資金充裕。",
    "資金動員能力不弱。",
    "是否因登記資本较高而具有資金動員能力？",
    "是否已確認登記資本反映資金動員能力？",
    "登記資本不能代表資金動員能力，但實收資本反映資金動員能力。",
    "登記資本不能不代表資金動員能力。",
    "登記資本並非不能反映資金動員能力。",
    "不能代表全部資料顯示資金動員能力。",
    "是否已另行查核公司的資金動員能力較強？",
    "實收資本反映資\u200b金動員能力。",
    "實收資本反映資 金 動 員 能 力。",
    "實收資本反映資金·動員能力。",
    "Capital suggests funding capability.",
])
def test_rejects_assertions_even_hedged_quoted_or_disguised_as_questions(text):
    assert has_unsupported_funding_claim(text, is_question=True)


@pytest.mark.parametrize("text,question", [
    ("登記資本額不能代表實際資金動員能力。", False),
    ("實收資本不足以證明公司的融資能力。", False),
    ("資金調度能力仍需另行查核。", False),
    ("公司的資金動員能力尚未確認。", False),
    ("是否已另行查核公司的資金調度能力？", True),
    ("是否已確認實際籌資能力？", True),
    ("是否已核對登記資本與實收資本的差異？", True),
    ("登記資本額不等同可動用現金或償債能力。", False),
])
def test_keeps_limits_and_independent_verification_questions(text, question):
    assert not has_unsupported_funding_claim(text, is_question=question)


@pytest.mark.parametrize("task", ["single", "comparison"])
@pytest.mark.parametrize("field", ["question", "reason", "overall_observation"])
def test_shared_validator_rejects_capability_claim_with_real_capital_evidence(task, field):
    if task == "single":
        payload = company_input()
        prompt = build_company_analysis_prompt(payload)
        validate = validate_company_analysis_output
        evidence = "/company/capital/registered"
    else:
        payload = sample("company-comparison-input-v1.example.json")
        prompt = build_company_comparison_prompt(payload)
        validate = validate_company_comparison_output
        evidence = "/comparison/data/items/0/company/capital/registered"
    output = example(prompt)
    output["verification_items"][0]["related_evidence_paths"] = [evidence]
    # Canonical manifest, so failure is attributable to content, not schema.
    from app.services.llm_output_normalizer import normalize_evidence_manifest
    if field == "overall_observation":
        output[field] = LIVE_FAILURE
    else:
        output["verification_items"][0][field] = LIVE_FAILURE.rstrip("。") + ("？" if field == "question" else "。")
    output = normalize_evidence_manifest(output, task="company_analysis" if task == "single" else "company_comparison_analysis")
    with pytest.raises(LLMOutputSafetyError, match="unsupported funding capability inference"):
        validate(payload, output)


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["json_schema", "json_object", "off"])
async def test_service_retries_safely_without_leaking_original_or_mutating_it(mode, caplog):
    payload = company_input()
    good = example(build_company_analysis_prompt(payload))
    bad = copy.deepcopy(good)
    bad["verification_items"][0]["reason"] = LIVE_FAILURE
    original = copy.deepcopy(bad)
    provider = THUStub([bad, good], mode)
    result = await CompanyAnalysisService(provider, max_attempts=2).analyze(payload)
    assert result.attempts == 2
    assert bad == original
    feedback = provider.prompts[1].messages[1].content.split("<validation_retry_feedback>")[1]
    assert "unsupported_funding_capability_inference" in feedback
    assert "/verification_items/0/reason" in feedback
    assert LIVE_FAILURE not in caplog.text + feedback
    assert payload["company"]["tax_id"] not in caplog.text + feedback


@pytest.mark.anyio
async def test_repeated_failure_is_closed_and_pk_failure_is_not_cached():
    payload = company_input()
    bad = example(build_company_analysis_prompt(payload))
    bad["verification_items"][0]["reason"] = LIVE_FAILURE
    with pytest.raises(CompanyAnalysisInvalidOutputError):
        await CompanyAnalysisService(THUStub([bad, bad]), max_attempts=2).analyze(payload)
    payload = sample("company-comparison-input-v1.example.json")
    good = example(build_company_comparison_prompt(payload))
    bad = copy.deepcopy(good)
    bad["verification_items"][0]["reason"] = LIVE_FAILURE
    provider = THUStub([bad, good])
    service = CompanyComparisonAnalysisService(provider, AsyncLRUTTLCache(max_entries=4, ttl_seconds=60, stale_if_error_seconds=60))
    assert (await service.analyze(payload["comparison"])).data.status == "fallback"
    assert (await service.analyze(payload["comparison"])).data.analysis is not None
    assert len(provider.prompts) == 2
