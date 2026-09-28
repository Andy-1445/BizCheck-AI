from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.services.llm_prompt import CompanyAnalysisPromptPackage, PromptMessage
from app.services.llm_comparison_prompt import CompanyComparisonPromptPackage
from app.services.llm_provider import (
    DisabledCompanyAnalysisProvider,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMIncompleteResponseError,
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMRequestRejectedError,
    LLMTimeoutError,
    LLMUpstreamError,
    MisconfiguredCompanyAnalysisProvider,
    OpenAIResponsesConfig,
    OpenAIResponsesProvider,
    build_openai_strict_schema,
    create_company_analysis_provider,
)
from app.settings import Settings


def _prompt(
    response_schema: dict[str, object] | None = None,
) -> CompanyAnalysisPromptPackage:
    return CompanyAnalysisPromptPackage(
        prompt_version="1.0",
        system_prompt_sha256="a" * 64,
        messages=[
            PromptMessage(role="system", content="system instructions"),
            PromptMessage(role="user", content='{"verified":true}'),
        ],
        response_schema=response_schema
        or {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )


def _config(**overrides: Any) -> OpenAIResponsesConfig:
    values: dict[str, Any] = {
        "api_key": "test-secret",
        "base_url": "https://openai.test/v1/",
        "model": "test-model",
        "timeout_seconds": 3.5,
        "max_output_tokens": 777,
    }
    values.update(overrides)
    return OpenAIResponsesConfig(**values)


async def _generate_with_payload(
    payload: object,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json=payload,
            headers=headers,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(_config(), http_client=client)
        return await provider.generate(_prompt())


def test_build_openai_strict_schema_normalizes_nested_objects_without_mutation() -> None:
    canonical: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://bizcheck.test/schema.json",
        "title": "Canonical contract",
        "type": "object",
        "properties": {
            "fixed": {
                "type": "string",
                "const": "v1",
                "default": "v1",
                "minLength": 1,
                "pattern": "^v1$",
            },
            "optional_date": {
                "anyOf": [
                    {"type": "string", "format": "date"},
                    {"type": "null"},
                ]
            },
            "rows": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        }
                    },
                },
            },
        },
        "required": ["fixed"],
    }
    untouched = copy.deepcopy(canonical)

    strict = build_openai_strict_schema(canonical)

    assert canonical == untouched
    assert strict is not canonical
    assert "$schema" not in strict
    assert "$id" not in strict
    assert "title" not in strict
    assert strict["required"] == ["fixed", "optional_date", "rows"]
    assert strict["additionalProperties"] is False

    properties = strict["properties"]
    assert isinstance(properties, dict)
    fixed = properties["fixed"]
    assert fixed == {"type": "string", "enum": ["v1"]}
    optional_date = properties["optional_date"]
    assert isinstance(optional_date, dict)
    assert optional_date["anyOf"] == [{"type": "string"}, {"type": "null"}]
    rows = properties["rows"]
    assert isinstance(rows, dict)
    assert "minItems" not in rows
    nested = rows["items"]
    assert isinstance(nested, dict)
    assert nested["required"] == ["amount"]
    assert nested["additionalProperties"] is False
    nested_properties = nested["properties"]
    assert isinstance(nested_properties, dict)
    assert nested_properties["amount"] == {"type": "number"}


def test_build_openai_strict_schema_preserves_names_that_match_schema_keywords() -> None:
    canonical: dict[str, object] = {
        "title": "Root schema metadata",
        "type": "object",
        "properties": {
            "title": {
                "title": "Finding title metadata",
                "type": "string",
                "minLength": 1,
            },
            "default": {"type": "string"},
            "format": {"type": "string"},
            "minimum": {"type": "number"},
            "nested": {
                "type": "object",
                "properties": {
                    "examples": {"type": "string"},
                    "title": {"type": "string"},
                },
            },
        },
        "$defs": {
            "title": {
                "type": "object",
                "properties": {"default": {"type": "string"}},
            }
        },
    }

    strict = build_openai_strict_schema(canonical)

    assert "title" not in {key for key in strict if key != "properties"}
    properties = strict["properties"]
    assert isinstance(properties, dict)
    assert list(properties) == ["title", "default", "format", "minimum", "nested"]
    assert properties["title"] == {"type": "string"}
    nested = properties["nested"]
    assert isinstance(nested, dict)
    assert nested["required"] == ["examples", "title"]
    assert nested["additionalProperties"] is False

    definitions = strict["$defs"]
    assert isinstance(definitions, dict)
    assert "title" in definitions
    title_definition = definitions["title"]
    assert isinstance(title_definition, dict)
    assert title_definition["required"] == ["default"]
    assert title_definition["properties"] == {"default": {"type": "string"}}


@pytest.mark.anyio
async def test_generate_sends_responses_api_contract_and_keeps_schema_canonical() -> None:
    canonical: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "answer": {"type": "integer", "minimum": 0},
            "note": {
                "anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}],
                "default": None,
            },
        },
        "required": ["answer"],
    }
    untouched = copy.deepcopy(canonical)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://openai.test/v1/responses")
        assert request.headers["authorization"] == "Bearer test-secret"
        assert request.headers["content-type"].startswith("application/json")

        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["instructions"] == "system instructions"
        assert body["input"] == '{"verified":true}'
        assert body["store"] is False
        assert body["prompt_cache_key"].startswith(
            "bizcheck-company-analysis-1.0-"
        )
        assert body["max_output_tokens"] == 777
        assert body["metadata"] == {
            "app": "bizcheck_ai",
            "task": "company_analysis",
            "input_schema_version": "1.0",
            "prompt_version": "1.0",
        }
        output_format = body["text"]["format"]
        assert output_format["type"] == "json_schema"
        assert output_format["name"] == "bizcheck_company_analysis_v1"
        assert output_format["strict"] is True
        assert output_format["schema"]["required"] == ["answer", "note"]
        assert output_format["schema"]["additionalProperties"] is False
        assert "$schema" not in output_format["schema"]
        assert output_format["schema"]["properties"]["answer"] == {
            "type": "integer"
        }
        return httpx.Response(
            200,
            json={"status": "completed", "output_text": '{"answer":7}'},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(_config(), http_client=client)
        result = await provider.generate(_prompt(canonical))

    assert result.output == {"answer": 7}
    assert canonical == untouched


@pytest.mark.anyio
async def test_comparison_generate_uses_distinct_strict_responses_contract() -> None:
    prompt = CompanyComparisonPromptPackage(
        prompt_version="1.0",
        system_prompt_sha256="b" * 64,
        messages=[
            PromptMessage(role="system", content="comparison system"),
            PromptMessage(role="user", content='{"comparison":true}'),
        ],
        response_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["store"] is False
        assert body["instructions"] == "comparison system"
        assert body["input"] == '{"comparison":true}'
        assert body["prompt_cache_key"].startswith(
            "bizcheck-comparison-analysis-1.0-"
        )
        assert len(body["prompt_cache_key"]) <= 64
        assert body["metadata"] == {
            "app": "bizcheck_ai",
            "task": "company_comparison_analysis",
            "input_schema_version": "1.0",
            "prompt_version": "1.0",
        }
        output_format = body["text"]["format"]
        assert output_format["name"] == (
            "bizcheck_company_comparison_analysis_v1"
        )
        assert output_format["strict"] is True
        assert output_format["schema"]["additionalProperties"] is False
        return httpx.Response(
            200,
            json={"status": "completed", "output_text": '{"answer":"ok"}'},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(_config(), http_client=client)
        result = await provider.generate(prompt)

    assert result.output == {"answer": "ok"}


@pytest.mark.anyio
async def test_generate_parses_top_level_output_text_and_provider_metadata() -> None:
    result = await _generate_with_payload(
        {
            "id": "resp_123",
            "model": "returned-model",
            "status": "completed",
            "created_at": 0,
            "service_tier": "priority",
            "output_text": '{"answer":42}',
            "usage": {
                "input_tokens": 101,
                "output_tokens": 19,
                "total_tokens": 120,
            },
        },
        headers={"x-request-id": "req_456"},
    )

    assert result.output == {"answer": 42}
    assert result.provider == "openai"
    assert result.model == "returned-model"
    assert result.response_id == "resp_123"
    assert result.request_id == "req_456"
    assert result.created_at == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert result.service_tier == "priority"
    assert result.usage is not None
    assert result.usage.input_tokens == 101
    assert result.usage.output_tokens == 19
    assert result.usage.total_tokens == 120


@pytest.mark.anyio
async def test_generate_concatenates_nested_output_text_fragments() -> None:
    result = await _generate_with_payload(
        {
            "status": "completed",
            "model": "",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": '{"answer":'},
                        {"type": "output_text", "text": "9}"},
                    ],
                }
            ],
        }
    )

    assert result.output == {"answer": 9}
    assert result.model == "test-model"


@pytest.mark.anyio
async def test_invalid_usage_or_timestamp_is_ignored() -> None:
    result = await _generate_with_payload(
        {
            "status": "completed",
            "output_text": '{"answer":1}',
            "created_at": True,
            "usage": {
                "input_tokens": 1,
                "output_tokens": -1,
                "total_tokens": 0,
            },
        }
    )

    assert result.created_at is None
    assert result.usage is None


@pytest.mark.anyio
async def test_generate_rejects_non_json_http_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(_config(), http_client=client)
        with pytest.raises(LLMInvalidResponseError, match="body was not valid JSON"):
            await provider.generate(_prompt())


@pytest.mark.anyio
async def test_generate_rejects_non_object_http_json_root() -> None:
    with pytest.raises(LLMInvalidResponseError, match="response root"):
        await _generate_with_payload([{"output_text": '{"answer":1}'}])


@pytest.mark.anyio
async def test_generate_rejects_missing_output_text() -> None:
    with pytest.raises(LLMInvalidResponseError, match="did not contain"):
        await _generate_with_payload({"status": "completed", "output": []})


@pytest.mark.anyio
async def test_generate_rejects_invalid_output_text_json() -> None:
    with pytest.raises(LLMInvalidResponseError, match="output_text was not valid JSON"):
        await _generate_with_payload(
            {"status": "completed", "output_text": "{invalid"}
        )


@pytest.mark.anyio
async def test_generate_rejects_non_object_structured_output_root() -> None:
    with pytest.raises(LLMInvalidResponseError, match="structured output root"):
        await _generate_with_payload(
            {"status": "completed", "output_text": "[1,2,3]"}
        )


@pytest.mark.anyio
async def test_generate_rejects_error_object_in_success_response() -> None:
    with pytest.raises(LLMUpstreamError, match="error response object"):
        await _generate_with_payload(
            {"status": "completed", "error": {"message": "failure"}}
        )


@pytest.mark.anyio
async def test_generate_rejects_nested_refusal() -> None:
    with pytest.raises(LLMIncompleteResponseError, match="refused"):
        await _generate_with_payload(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "refusal", "refusal": "cannot comply"}
                        ],
                    }
                ],
            }
        )


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled", "queued"])
async def test_generate_rejects_non_completed_status(status: str) -> None:
    with pytest.raises(LLMIncompleteResponseError, match=repr(status)):
        await _generate_with_payload(
            {"status": status, "output_text": '{"answer":1}'}
        )


@pytest.mark.anyio
async def test_generate_maps_httpx_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(_config(), http_client=client)
        with pytest.raises(LLMTimeoutError, match="timed out"):
            await provider.generate(_prompt())


@pytest.mark.anyio
async def test_generate_maps_httpx_connection_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIResponsesProvider(_config(), http_client=client)
        with pytest.raises(LLMConnectionError, match="Could not connect"):
            await provider.generate(_prompt())


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, LLMAuthenticationError),
        (403, LLMAuthenticationError),
        (429, LLMRateLimitError),
        (400, LLMRequestRejectedError),
        (404, LLMRequestRejectedError),
        (409, LLMRequestRejectedError),
        (422, LLMRequestRejectedError),
        (500, LLMUpstreamError),
        (503, LLMUpstreamError),
        (418, LLMUpstreamError),
    ],
)
async def test_generate_maps_http_status_errors(
    status_code: int,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        await _generate_with_payload(
            {"error": {"message": "provider failure"}},
            status_code=status_code,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"api_key": "   "}, "api_key"),
        ({"base_url": "ftp://openai.test/v1"}, "base_url"),
        ({"model": "  "}, "model"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
    ],
)
def test_openai_config_rejects_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_openai_config_normalizes_surrounding_whitespace() -> None:
    config = _config(
        api_key="  secret  ",
        base_url="  https://openai.test/v1///  ",
        model="  configured-model  ",
    )

    assert config.api_key == "secret"
    assert config.base_url == "https://openai.test/v1"
    assert config.model == "configured-model"
    assert "secret" not in repr(config)


@pytest.mark.anyio
@pytest.mark.parametrize("provider_name", ["", "disabled", "none", " DISABLED "])
async def test_factory_returns_disabled_provider(provider_name: str) -> None:
    provider = create_company_analysis_provider(
        Settings(llm_provider=provider_name, openai_api_key="unused")
    )

    assert isinstance(provider, DisabledCompanyAnalysisProvider)
    with pytest.raises(LLMNotConfiguredError):
        await provider.generate(_prompt())


@pytest.mark.anyio
async def test_factory_treats_missing_openai_key_as_disabled() -> None:
    provider = create_company_analysis_provider(
        Settings(llm_provider="openai", openai_api_key=None)
    )

    assert isinstance(provider, DisabledCompanyAnalysisProvider)
    with pytest.raises(LLMNotConfiguredError):
        await provider.generate(_prompt())


@pytest.mark.anyio
async def test_factory_returns_misconfigured_provider_for_unknown_name() -> None:
    provider = create_company_analysis_provider(
        Settings(llm_provider="another-provider", openai_api_key="unused")
    )

    assert isinstance(provider, MisconfiguredCompanyAnalysisProvider)
    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
        await provider.generate(_prompt())


@pytest.mark.anyio
async def test_factory_preserves_invalid_openai_config_as_runtime_error() -> None:
    provider = create_company_analysis_provider(
        Settings(
            llm_provider="openai",
            openai_api_key="test-secret",
            openai_base_url="ftp://openai.test",
        )
    )

    assert isinstance(provider, MisconfiguredCompanyAnalysisProvider)
    with pytest.raises(LLMConfigurationError, match="base_url"):
        await provider.generate(_prompt())


@pytest.mark.anyio
async def test_factory_builds_enabled_openai_provider() -> None:
    provider = create_company_analysis_provider(
        Settings(
            llm_provider=" OPENAI ",
            openai_api_key="test-secret",
            openai_base_url="https://openai.test/v1",
            openai_model="configured-model",
        )
    )

    assert isinstance(provider, OpenAIResponsesProvider)
    assert provider.provider_name == "openai"
    assert provider.model_name == "configured-model"
    await provider.aclose()


@pytest.mark.anyio
async def test_provider_closes_only_the_http_client_it_owns() -> None:
    owned_provider = OpenAIResponsesProvider(_config())
    owned_client = owned_provider._http_client
    assert not owned_client.is_closed

    await owned_provider.aclose()
    await owned_provider.aclose()

    assert owned_client.is_closed

    injected_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, request=request)
        )
    )
    injected_provider = OpenAIResponsesProvider(
        _config(),
        http_client=injected_client,
    )
    await injected_provider.aclose()
    assert not injected_client.is_closed
    await injected_client.aclose()
