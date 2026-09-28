import copy

import pytest

from app.schemas.company_comparison import CompanyComparisonRequest
from app.services.company_comparison import build_company_comparison
from app.services.company_comparison_analysis import CompanyComparisonAnalysisService
from app.services.llm_cache import AsyncLRUTTLCache
from app.services.llm_comparison_prompt import build_company_comparison_prompt, validate_company_comparison_output
from app.services.llm_output_normalizer import normalize_evidence_manifest
from app.services.thu_prompt import build_thu_response_schema
from test_company_comparison import _company_bizscore_response, GENERATED_AT, TAX_ID_A, TAX_ID_B, TAX_ID_C
from test_prompt_safety_reacceptance import THUStub, example


PARTIAL = "部分公司缺少可用的同業基準資料；其他公司的同業資料仍可各自查看，無法對全部公司進行同業位置比較。"
ALL = "所有公司都缺少可用的同業基準資料，無法進行同業位置比較。"
OLD = "本次比較沒有可用的同業基準資料。"


def comparison(availability):
    tax_ids = [TAX_ID_A, TAX_ID_B, TAX_ID_C][:len(availability)]
    responses = []
    for tax_id, available in zip(tax_ids, availability):
        response = _company_bizscore_response(tax_id)
        if not available:
            score = response.data.bizscore
            score.dimensions[-1].available = False
            score.dimensions[-1].score = None
            score.peer_benchmark.dimension = score.dimensions[-1].model_copy(deep=True)
            score.peer_benchmark.peer_index = None
            score.benchmark.snapshot_version = None
            score.peer_benchmark.benchmark_version = None
            score.coverage = 0.8
            score.provisional = True
            score.missing_dimensions = ["peer_relative_position"]
        responses.append(response)
    return build_company_comparison(CompanyComparisonRequest(tax_ids=tax_ids), responses, generated_at=GENERATED_AT)


@pytest.mark.parametrize("availability", [
    (True, True), (False, False), (True, False), (False, True),
    (True, True, True), (False, False, False),
    (True, True, False), (True, False, True), (False, True, True),
    (True, False, False), (False, True, False), (False, False, True),
])
def test_limitation_matches_each_company_in_any_selection_order(availability):
    source = comparison(availability)
    payload = {"comparison": source.model_dump(mode="json")}
    prompt = build_company_comparison_prompt(payload)
    output = example(prompt)
    limitations = [x for x in output["limitations"] if x["code"] == "benchmark_unavailable"]
    if all(availability):
        assert not limitations
    else:
        assert len(limitations) == 1
        assert limitations[0]["message"] == (PARTIAL if any(availability) else ALL)
    assert build_thu_response_schema(prompt)["properties"]["limitations"]["enum"] == [output["limitations"]]
    validate_company_comparison_output(payload, normalize_evidence_manifest(output, task="company_comparison_analysis"))


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["json_schema", "json_object", "off"])
async def test_old_overgeneralization_is_rejected_before_caching_and_repaired(mode):
    source = comparison((True, True, False))
    prompt = build_company_comparison_prompt({"comparison": source.model_dump(mode="json")})
    good = example(prompt)
    bad = copy.deepcopy(good)
    next(x for x in bad["limitations"] if x["code"] == "benchmark_unavailable")["message"] = OLD
    provider = THUStub([bad, good], mode)
    service = CompanyComparisonAnalysisService(provider, AsyncLRUTTLCache(max_entries=4, ttl_seconds=60, stale_if_error_seconds=60), max_attempts=2)
    response = await service.analyze(source)
    assert response.meta.attempts == 2
    assert response.meta.fallback.active is False
    assert next(x.message for x in response.data.analysis.limitations if x.code == "benchmark_unavailable") == PARTIAL
    cached = await service.analyze(source)
    assert cached.meta.cache.status == "hit"
    assert cached.data.analysis == response.data.analysis
    assert len(provider.prompts) == 2
