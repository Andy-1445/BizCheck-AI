from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas.llm_analysis import (
    CompanyAnalysisLLMInput,
    CompanyAnalysisLLMOutput,
)
from app.services.company_analysis import (
    CompanyAnalysisInvalidOutputError,
    CompanyAnalysisService,
)
from app.services.llm_prompt import (
    CompanyAnalysisPromptPackage,
    LLMOutputSafetyError,
)
from app.services.llm_provider import (
    LLMInvalidResponseError,
    LLMProviderResult,
    LLMProviderUsage,
    LLMTimeoutError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIRECTORY = PROJECT_ROOT / "samples" / "llm"
FIXED_CREATED_AT = datetime(2026, 8, 23, 4, 30, tzinfo=timezone.utc)


def _read_json(filename: str) -> dict[str, Any]:
    return json.loads(
        (SAMPLE_DIRECTORY / filename).read_text(encoding="utf-8")
    )


def _input_payload() -> dict[str, Any]:
    return _read_json("company-analysis-input-v1.example.json")


def _output_payload() -> dict[str, Any]:
    return _read_json("company-analysis-output-v1.example.json")


def _provider_result(
    output: dict[str, Any],
    *,
    response_id: str = "resp_test_123",
) -> LLMProviderResult:
    return LLMProviderResult(
        output=output,
        provider="fake-openai",
        model="test-structured-model",
        response_id=response_id,
        request_id="req_test_456",
        created_at=FIXED_CREATED_AT,
        service_tier="default",
        usage=LLMProviderUsage(
            input_tokens=701,
            output_tokens=211,
            total_tokens=912,
        ),
    )


def _canonical_model_sha256(
    model: CompanyAnalysisLLMInput | CompanyAnalysisLLMOutput,
) -> str:
    canonical_json = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class FakeCompanyAnalysisProvider:
    provider_name = "fake-openai"
    model_name = "test-structured-model"

    def __init__(
        self,
        outcomes: list[LLMProviderResult | Exception],
    ) -> None:
        self.outcomes = list(outcomes)
        self.prompts: list[CompanyAnalysisPromptPackage] = []
        self.ensure_available_calls = 0

    def ensure_available(self) -> None:
        self.ensure_available_calls += 1

    async def generate(
        self,
        prompt: CompanyAnalysisPromptPackage,
    ) -> LLMProviderResult:
        self.prompts.append(prompt)
        outcome_index = len(self.prompts) - 1
        if outcome_index >= len(self.outcomes):
            raise AssertionError("Fake provider received an unexpected extra call.")
        outcome = self.outcomes[outcome_index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        return None


@pytest.mark.anyio
async def test_success_preserves_provider_metadata_hashes_and_prompt() -> None:
    analysis_input = CompanyAnalysisLLMInput.model_validate(_input_payload())
    expected_output = CompanyAnalysisLLMOutput.model_validate(_output_payload())
    provider_result = _provider_result(_output_payload())
    provider = FakeCompanyAnalysisProvider([provider_result])
    service = CompanyAnalysisService(provider)

    service.ensure_available()
    execution = await service.analyze(analysis_input)

    assert provider.ensure_available_calls == 1
    assert provider.prompts == [execution.prompt]
    assert execution.prompt.prompt_version == "1.0"
    assert execution.prompt.messages[0].role == "system"
    assert execution.prompt.messages[1].role == "user"
    assert '"tax_id": "20828393"' in execution.prompt.messages[1].content
    assert execution.analysis_input == analysis_input
    assert execution.analysis == expected_output
    assert execution.provider_result is provider_result
    assert execution.generated_at == FIXED_CREATED_AT
    assert execution.attempts == 1
    assert execution.duration_ms >= 0
    assert execution.input_sha256 == _canonical_model_sha256(analysis_input)
    assert execution.output_sha256 == _canonical_model_sha256(expected_output)
    assert len(execution.input_sha256) == 64
    assert len(execution.output_sha256) == 64


@pytest.mark.anyio
async def test_service_rebuilds_evidence_manifest_without_mutating_raw_output() -> None:
    raw_output = _output_payload()
    expected_output = CompanyAnalysisLLMOutput.model_validate(
        copy.deepcopy(raw_output)
    )
    raw_output["provenance"]["evidence_paths_used"] = []
    untouched = copy.deepcopy(raw_output)
    provider_result = _provider_result(raw_output)
    provider = FakeCompanyAnalysisProvider([provider_result])

    execution = await CompanyAnalysisService(provider).analyze(_input_payload())

    assert execution.analysis == expected_output
    assert provider_result.output == untouched


def _make_rejected_output(case: str) -> dict[str, Any]:
    output = copy.deepcopy(_output_payload())
    if case == "invalid_schema":
        del output["headline"]
    elif case == "invalid_evidence":
        old_path = output["findings"][0]["evidence_paths"][0]
        new_path = "/company/status"
        output["findings"][0]["evidence_paths"][0] = new_path
        output["provenance"]["evidence_paths_used"] = [
            new_path if path == old_path else path
            for path in output["provenance"]["evidence_paths_used"]
        ]
    elif case == "forbidden_phrase":
        output["overall_observation"] = "模型斷言這家公司值得推薦合作。"
    elif case == "provenance_mismatch":
        output["provenance"]["company_tax_id"] = "12345678"
    else:
        raise AssertionError(f"Unknown rejected-output case: {case}")
    return output


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("case", "expected_cause"),
    [
        pytest.param("invalid_schema", ValidationError, id="schema"),
        pytest.param("invalid_evidence", LLMOutputSafetyError, id="safety"),
        pytest.param("forbidden_phrase", LLMOutputSafetyError, id="forbidden"),
        pytest.param("provenance_mismatch", LLMOutputSafetyError, id="provenance"),
    ],
)
async def test_rejects_schema_and_safety_boundary_violations(
    case: str,
    expected_cause: type[Exception],
) -> None:
    provider = FakeCompanyAnalysisProvider(
        [_provider_result(_make_rejected_output(case))]
    )

    with pytest.raises(CompanyAnalysisInvalidOutputError) as captured:
        await CompanyAnalysisService(provider).analyze(_input_payload())

    assert len(provider.prompts) == 1
    assert isinstance(captured.value.__cause__, expected_cause)
    assert str(captured.value) == (
        "The provider output failed JSON, schema, or safety validation."
    )


@pytest.mark.anyio
async def test_invalid_output_is_retried_then_valid_output_is_accepted() -> None:
    provider = FakeCompanyAnalysisProvider(
        [
            LLMInvalidResponseError("first response was not a JSON object"),
            _provider_result(_output_payload(), response_id="resp_retry_ok"),
        ]
    )

    execution = await CompanyAnalysisService(
        provider,
        max_attempts=3,
    ).analyze(_input_payload())

    assert execution.attempts == 2
    assert execution.provider_result.response_id == "resp_retry_ok"
    assert len(provider.prompts) == 2
    assert provider.prompts[0] != provider.prompts[1]
    assert provider.prompts[1] == execution.prompt
    retry_content = provider.prompts[1].messages[1].content
    assert "<validation_retry_feedback>" in retry_content
    assert '"fields":["$"]' in retry_content
    assert '"reasons":["invalid_json_or_response_shape"]' in retry_content
    assert "first response was not a JSON object" not in retry_content


@pytest.mark.anyio
async def test_retry_count_never_exceeds_configured_max_attempts() -> None:
    provider = FakeCompanyAnalysisProvider(
        [
            LLMInvalidResponseError("invalid response 1"),
            LLMInvalidResponseError("invalid response 2"),
            LLMInvalidResponseError("invalid response 3"),
            _provider_result(_output_payload()),
        ]
    )

    with pytest.raises(CompanyAnalysisInvalidOutputError):
        await CompanyAnalysisService(provider, max_attempts=3).analyze(
            _input_payload()
        )

    assert len(provider.prompts) == 3


@pytest.mark.anyio
async def test_default_is_a_single_attempt_even_if_a_valid_result_is_queued() -> None:
    provider = FakeCompanyAnalysisProvider(
        [
            LLMInvalidResponseError("first response was invalid"),
            _provider_result(_output_payload()),
        ]
    )

    with pytest.raises(CompanyAnalysisInvalidOutputError):
        await CompanyAnalysisService(provider).analyze(_input_payload())

    assert len(provider.prompts) == 1


@pytest.mark.anyio
async def test_provider_timeout_propagates_without_retry() -> None:
    timeout = LLMTimeoutError("provider timed out")
    provider = FakeCompanyAnalysisProvider(
        [timeout, _provider_result(_output_payload())]
    )

    with pytest.raises(LLMTimeoutError) as captured:
        await CompanyAnalysisService(provider, max_attempts=3).analyze(
            _input_payload()
        )

    assert captured.value is timeout
    assert len(provider.prompts) == 1


@pytest.mark.parametrize("max_attempts", [-1, 0, 4, 99])
def test_invalid_max_attempts_is_rejected(max_attempts: int) -> None:
    provider = FakeCompanyAnalysisProvider([_provider_result(_output_payload())])

    with pytest.raises(
        ValueError,
        match="max_attempts must be between 1 and 3",
    ):
        CompanyAnalysisService(provider, max_attempts=max_attempts)


@pytest.mark.anyio
async def test_public_invalid_output_error_does_not_include_raw_output() -> None:
    secret = "RAW_PROVIDER_PAYLOAD_MUST_NOT_ESCAPE_9f31"
    rejected_output = _output_payload()
    rejected_output["provider_debug_payload"] = secret
    provider = FakeCompanyAnalysisProvider([_provider_result(rejected_output)])

    with pytest.raises(CompanyAnalysisInvalidOutputError) as captured:
        await CompanyAnalysisService(provider).analyze(_input_payload())

    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
    assert captured.value.args == (
        "The provider output failed JSON, schema, or safety validation.",
    )
