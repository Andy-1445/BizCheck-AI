from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.services.llm_prompt import (
    CompanyAnalysisPromptPackage,
    build_company_analysis_prompt,
)
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
    THUChatCompletionsConfig,
    THUChatCompletionsProvider,
    create_company_analysis_provider,
)
from app.settings import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prompt() -> CompanyAnalysisPromptPackage:
    payload = json.loads(
        (
            PROJECT_ROOT
            / "samples"
            / "llm"
            / "company-analysis-input-v1.example.json"
        ).read_text(encoding="utf-8")
    )
    return build_company_analysis_prompt(payload)


def _config(**overrides: Any) -> THUChatCompletionsConfig:
    values: dict[str, Any] = {
        "api_key": "thu-test-secret",
        "base_url": "https://thu.test/v1/",
        "model": "gpt-oss-test",
        "timeout_seconds": 3.5,
        "max_output_tokens": 777,
        "json_mode": "off",
        "temperature": 0.1,
    }
    values.update(overrides)
    return THUChatCompletionsConfig(**values)


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
        provider = THUChatCompletionsProvider(_config(), http_client=client)
        return await provider.generate(_prompt())


@pytest.mark.anyio
async def test_thu_generate_sends_documented_chat_completions_contract() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("https://thu.test/v1/chat/completions")
        assert request.headers["authorization"] == "Bearer thu-test-secret"
        assert request.headers["content-type"].startswith("application/json")

        body = json.loads(request.content)
        assert set(body) == {"model", "messages", "max_tokens", "temperature"}
        assert body["model"] == "gpt-oss-test"
        assert body["max_tokens"] == 777
        assert body["temperature"] == 0.1
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"].startswith(
            "你是 BizCheck AI 的企業公開資料解釋層"
        )
        assert "<complete_output_example>" in body["messages"][0]["content"]
        assert "<response_json_schema>" in body["messages"][0]["content"]
        assert '"additionalProperties":false' in body["messages"][0]["content"]
        assert body["messages"][1]["role"] == "user"
        assert "<verified_input_json>" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_thu_123",
                "model": "returned-thu-model",
                "created": 0,
                "service_tier": "default",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"answer":7}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 101,
                    "completion_tokens": 19,
                    "total_tokens": 120,
                },
            },
            headers={"x-request-id": "thu_req_456"},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = THUChatCompletionsProvider(_config(), http_client=client)
        result = await provider.generate(_prompt())

    assert result.output == {"answer": 7}
    assert result.provider == "thu"
    assert result.model == "returned-thu-model"
    assert result.response_id == "chatcmpl_thu_123"
    assert result.request_id == "thu_req_456"
    assert result.created_at == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert result.service_tier == "default"
    assert result.usage is not None
    assert result.usage.input_tokens == 101
    assert result.usage.output_tokens == 19
    assert result.usage.total_tokens == 120


@pytest.mark.anyio
async def test_thu_generate_accepts_missing_optional_completion_metadata() -> None:
    result = await _generate_with_payload(
        {
            "choices": [
                {"message": {"role": "assistant", "content": '{"answer":1}'}}
            ]
        }
    )

    assert result.output == {"answer": 1}
    assert result.model == "gpt-oss-test"
    assert result.created_at is None
    assert result.usage is None


@pytest.mark.anyio
@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
async def test_thu_generate_rejects_incomplete_completion(
    finish_reason: str,
) -> None:
    with pytest.raises(LLMIncompleteResponseError, match="finish_reason"):
        await _generate_with_payload(
            {
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {"content": '{"answer":1}'},
                    }
                ]
            }
        )


@pytest.mark.anyio
async def test_thu_generate_rejects_refusal() -> None:
    with pytest.raises(LLMIncompleteResponseError, match="refused"):
        await _generate_with_payload(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "", "refusal": "cannot comply"},
                    }
                ]
            }
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "not-json"}}]},
        {"choices": [{"message": {"content": "[1,2,3]"}}]},
    ],
)
async def test_thu_generate_rejects_invalid_response_shapes(payload: object) -> None:
    with pytest.raises(LLMInvalidResponseError):
        await _generate_with_payload(payload)


@pytest.mark.anyio
async def test_thu_generate_rejects_non_json_http_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = THUChatCompletionsProvider(_config(), http_client=client)
        with pytest.raises(LLMInvalidResponseError, match="body was not valid JSON"):
            await provider.generate(_prompt())


@pytest.mark.anyio
async def test_thu_generate_rejects_error_object_in_success_response() -> None:
    with pytest.raises(LLMUpstreamError, match="error response object"):
        await _generate_with_payload({"error": {"message": "failure"}})


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, LLMAuthenticationError),
        (403, LLMAuthenticationError),
        (429, LLMRateLimitError),
        (400, LLMRequestRejectedError),
        (404, LLMRequestRejectedError),
        (422, LLMRequestRejectedError),
        (500, LLMUpstreamError),
        (503, LLMUpstreamError),
    ],
)
async def test_thu_generate_maps_http_status_errors(
    status_code: int,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        await _generate_with_payload(
            {"error": {"message": "provider failure"}},
            status_code=status_code,
        )


@pytest.mark.anyio
async def test_thu_generate_maps_httpx_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = THUChatCompletionsProvider(_config(), http_client=client)
        with pytest.raises(LLMTimeoutError, match="timed out"):
            await provider.generate(_prompt())


@pytest.mark.anyio
async def test_thu_generate_maps_httpx_connection_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = THUChatCompletionsProvider(_config(), http_client=client)
        with pytest.raises(LLMConnectionError, match="Could not connect"):
            await provider.generate(_prompt())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"api_key": "   "}, "api_key"),
        ({"base_url": "ftp://thu.test/v1"}, "base_url"),
        ({"model": "  "}, "model"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"json_mode": "unknown"}, "json_mode"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 2.1}, "temperature"),
    ],
)
def test_thu_config_rejects_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_thu_config_normalizes_whitespace_and_hides_key() -> None:
    config = _config(
        api_key="  secret  ",
        base_url="  https://thu.test/v1///  ",
        model="  configured-model  ",
        json_mode="  JSON_OBJECT  ",
    )

    assert config.api_key == "secret"
    assert config.base_url == "https://thu.test/v1"
    assert config.model == "configured-model"
    assert config.json_mode == "json_object"
    assert "secret" not in repr(config)


@pytest.mark.anyio
async def test_factory_treats_missing_thu_key_as_disabled() -> None:
    provider = create_company_analysis_provider(
        Settings(llm_provider="thu", thu_api_key=None)
    )

    assert isinstance(provider, DisabledCompanyAnalysisProvider)
    assert provider.model_name == "gpt-oss-120b"
    with pytest.raises(LLMNotConfiguredError):
        await provider.generate(_prompt())


@pytest.mark.anyio
async def test_factory_preserves_invalid_thu_config_as_runtime_error() -> None:
    provider = create_company_analysis_provider(
        Settings(
            llm_provider="thu",
            thu_api_key="test-secret",
            thu_base_url="ftp://thu.test",
        )
    )

    assert isinstance(provider, MisconfiguredCompanyAnalysisProvider)
    with pytest.raises(LLMConfigurationError, match="base_url"):
        await provider.generate(_prompt())


@pytest.mark.anyio
async def test_factory_builds_enabled_thu_provider() -> None:
    provider = create_company_analysis_provider(
        Settings(
            llm_provider=" THU ",
            thu_api_key="test-secret",
            thu_base_url="https://thu.test/v1",
            thu_model="configured-model",
        )
    )

    assert isinstance(provider, THUChatCompletionsProvider)
    assert provider.provider_name == "thu"
    assert provider.model_name == "configured-model"
    await provider.aclose()


@pytest.mark.anyio
async def test_thu_provider_closes_only_the_http_client_it_owns() -> None:
    owned_provider = THUChatCompletionsProvider(_config())
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
    injected_provider = THUChatCompletionsProvider(
        _config(),
        http_client=injected_client,
    )
    await injected_provider.aclose()
    assert not injected_client.is_closed
    await injected_client.aclose()


@pytest.mark.anyio
async def test_thu_auto_mode_prefers_json_schema_and_reuses_capability() -> None:
    response_formats: list[object] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        response_formats.append(body.get("response_format"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"answer":1}'},
                    }
                ]
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = THUChatCompletionsProvider(
            _config(json_mode="auto"),
            http_client=client,
        )
        await provider.generate(_prompt())
        await provider.generate(_prompt())

    assert len(response_formats) == 2
    for response_format in response_formats:
        assert isinstance(response_format, dict)
        assert response_format["type"] == "json_schema"
        schema_config = response_format["json_schema"]
        assert schema_config["name"] == "bizcheck_company_analysis_v1"
        assert schema_config["strict"] is True
        assert schema_config["schema"]["additionalProperties"] is False
        finding_schema = schema_config["schema"]["$defs"]["LLMAnalysisFinding"]
        assert "title" in finding_schema["properties"]
        assert "title" in finding_schema["required"]
        assert finding_schema["properties"]["observation"]["pattern"] == (
            r"^[^0-9０-９%％]*$"
        )
        assert finding_schema["properties"]["evidence_paths"]["minItems"] == 1
        assert finding_schema["properties"]["evidence_paths"]["maxItems"] == 6
        provenance_schema = schema_config["schema"]["$defs"][
            "LLMAnalysisProvenance"
        ]
        manifest_schema = provenance_schema["properties"]["evidence_paths_used"]
        assert manifest_schema["maxItems"] == 0
    assert provider._resolved_json_mode == "json_schema"
    assert provider.active_json_mode == "json_schema"


@pytest.mark.anyio
async def test_thu_auto_mode_downgrades_only_request_contract_errors() -> None:
    attempted_modes: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        response_format = body.get("response_format")
        mode = (
            response_format["type"]
            if isinstance(response_format, dict)
            else "off"
        )
        attempted_modes.append(mode)
        if mode != "off":
            return httpx.Response(
                400 if mode == "json_schema" else 422,
                json={"error": {"message": "unsupported request field"}},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"answer":1}'},
                    }
                ]
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = THUChatCompletionsProvider(
            _config(json_mode="auto"),
            http_client=client,
        )
        await provider.generate(_prompt())
        await provider.generate(_prompt())

    assert attempted_modes == ["json_schema", "json_object", "off", "off"]
    assert provider._resolved_json_mode == "off"
    assert provider.active_json_mode == "off"


@pytest.mark.anyio
async def test_thu_auto_mode_does_not_downgrade_auth_or_server_errors() -> None:
    for status_code, expected_error in (
        (401, LLMAuthenticationError),
        (500, LLMUpstreamError),
    ):
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                status_code,
                json={"error": {"message": "failure"}},
                request=request,
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = THUChatCompletionsProvider(
                _config(json_mode="auto"),
                http_client=client,
            )
            with pytest.raises(expected_error):
                await provider.generate(_prompt())

        assert calls == 1
        assert provider._resolved_json_mode is None


@pytest.mark.anyio
async def test_thu_explicit_json_object_mode_has_no_implicit_retry() -> None:
    request_bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={"error": {"message": "unsupported"}},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = THUChatCompletionsProvider(
            _config(json_mode="json_object"),
            http_client=client,
        )
        with pytest.raises(LLMRequestRejectedError):
            await provider.generate(_prompt())

    assert len(request_bodies) == 1
    assert request_bodies[0]["response_format"] == {"type": "json_object"}
