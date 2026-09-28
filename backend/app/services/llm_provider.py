from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from app.services.llm_comparison_prompt import CompanyComparisonPromptPackage
from app.services.llm_output_normalizer import build_server_owned_manifest_schema
from app.services.llm_prompt import CompanyAnalysisPromptPackage
from app.services.thu_prompt import (
    apply_thu_narrative_schema_guards,
    build_thu_response_schema,
    build_thu_system_instruction,
    restore_thu_collection_bounds,
)
from app.settings import Settings


JSONValue = dict[str, Any]
AnalysisPromptPackage = (
    CompanyAnalysisPromptPackage | CompanyComparisonPromptPackage
)


class LLMProviderError(RuntimeError):
    """Base class for failures raised by a company-analysis provider."""


class LLMNotConfiguredError(LLMProviderError):
    """Company analysis is intentionally disabled or lacks credentials."""


class LLMConfigurationError(LLMProviderError):
    """The selected provider has an invalid local configuration."""


class LLMTimeoutError(LLMProviderError):
    """The provider did not finish before the configured timeout."""


class LLMConnectionError(LLMProviderError):
    """The provider could not be reached."""


class LLMAuthenticationError(LLMProviderError):
    """The provider rejected the configured credentials."""


class LLMRateLimitError(LLMProviderError):
    """The provider rejected the request because of a rate limit."""


class LLMRequestRejectedError(LLMProviderError):
    """The provider rejected BizCheck's request contract."""


class LLMUpstreamError(LLMProviderError):
    """The provider returned a server-side failure."""


class LLMInvalidResponseError(LLMProviderError):
    """The provider response could not be parsed as the expected JSON object."""


class LLMIncompleteResponseError(LLMProviderError):
    """The provider refused or did not complete the response."""


@dataclass(frozen=True, slots=True)
class LLMProviderUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class LLMProviderResult:
    output: JSONValue
    provider: str
    model: str
    response_id: str | None = None
    request_id: str | None = None
    created_at: datetime | None = None
    service_tier: str | None = None
    usage: LLMProviderUsage | None = None


class CompanyAnalysisProvider(Protocol):
    provider_name: str
    model_name: str

    async def generate(
        self,
        prompt: AnalysisPromptPackage,
    ) -> LLMProviderResult: ...

    def ensure_available(self) -> None: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class OpenAIResponsesConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5.4"
    timeout_seconds: float = 45.0
    max_output_tokens: int = 3000

    def __post_init__(self) -> None:
        api_key = self.api_key.strip()
        base_url = self.base_url.strip().rstrip("/")
        model = self.model.strip()
        if not api_key:
            raise ValueError("OpenAI api_key must not be empty.")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("OpenAI base_url must use http or https.")
        if not model:
            raise ValueError("OpenAI model must not be empty.")
        if self.timeout_seconds <= 0:
            raise ValueError("LLM timeout_seconds must be greater than zero.")
        if self.max_output_tokens <= 0:
            raise ValueError("LLM max_output_tokens must be greater than zero.")

        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)


@dataclass(frozen=True, slots=True)
class THUChatCompletionsConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://api.ithu.tw/v1"
    model: str = "gpt-oss-120b"
    timeout_seconds: float = 45.0
    max_output_tokens: int = 3000
    json_mode: str = "auto"
    temperature: float = 0.1

    def __post_init__(self) -> None:
        api_key = self.api_key.strip()
        base_url = self.base_url.strip().rstrip("/")
        model = self.model.strip()
        json_mode = self.json_mode.strip().lower()
        if not api_key:
            raise ValueError("THU api_key must not be empty.")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("THU base_url must use http or https.")
        if not model:
            raise ValueError("THU model must not be empty.")
        if self.timeout_seconds <= 0:
            raise ValueError("LLM timeout_seconds must be greater than zero.")
        if self.max_output_tokens <= 0:
            raise ValueError("LLM max_output_tokens must be greater than zero.")
        if json_mode not in {"auto", "json_schema", "json_object", "off"}:
            raise ValueError(
                "THU json_mode must be auto, json_schema, json_object, or off."
            )
        if not 0 <= self.temperature <= 2:
            raise ValueError("THU temperature must be between zero and two.")

        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "json_mode", json_mode)


class DisabledCompanyAnalysisProvider:
    provider_name = "disabled"

    def __init__(self, model_name: str = "not-configured") -> None:
        self.model_name = model_name or "not-configured"

    async def __aenter__(self) -> DisabledCompanyAnalysisProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        return None

    def ensure_available(self) -> None:
        raise LLMNotConfiguredError("Company analysis is not configured.")

    async def generate(
        self,
        prompt: AnalysisPromptPackage,
    ) -> LLMProviderResult:
        del prompt
        self.ensure_available()
        raise AssertionError("Disabled provider availability check unexpectedly passed.")


class MisconfiguredCompanyAnalysisProvider(DisabledCompanyAnalysisProvider):
    provider_name = "misconfigured"

    def __init__(self, reason: str, model_name: str = "not-configured") -> None:
        super().__init__(model_name=model_name)
        self._reason = reason

    async def generate(
        self,
        prompt: AnalysisPromptPackage,
    ) -> LLMProviderResult:
        del prompt
        self.ensure_available()
        raise AssertionError(
            "Misconfigured provider availability check unexpectedly passed."
        )

    def ensure_available(self) -> None:
        raise LLMConfigurationError(self._reason)


class OpenAIResponsesProvider:
    """OpenAI Responses API adapter using strict structured output.

    The canonical Draft 2020-12 schema remains provider-neutral. A deep-copied,
    reduced strict-schema variant is generated only for the outgoing request;
    Pydantic and Prompt v1 guardrails still validate the returned object.
    """

    provider_name = "openai"

    def __init__(
        self,
        config: OpenAIResponsesConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.model_name = config.model
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=config.timeout_seconds,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "BizCheckAI/0.1",
            },
        )

    async def __aenter__(self) -> OpenAIResponsesProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def ensure_available(self) -> None:
        return None

    async def generate(
        self,
        prompt: AnalysisPromptPackage,
    ) -> LLMProviderResult:
        system_message, user_message = prompt.messages
        task = getattr(prompt, "task", "company_analysis")
        schema_name = getattr(
            prompt,
            "schema_name",
            "bizcheck_company_analysis_v1",
        )
        description = (
            "BizCheck AI 2–3 company comparison analysis v1"
            if task == "company_comparison_analysis"
            else "BizCheck AI single-company analysis v1"
        )
        request_body = {
            "model": self.config.model,
            "instructions": system_message.content,
            "input": user_message.content,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "description": description,
                    "schema": build_openai_strict_schema(prompt.response_schema),
                    "strict": True,
                }
            },
            "store": False,
            "prompt_cache_key": _provider_prompt_cache_key(prompt),
            "max_output_tokens": self.config.max_output_tokens,
            "metadata": {
                "app": "bizcheck_ai",
                "task": task,
                "input_schema_version": "1.0",
                "prompt_version": prompt.prompt_version,
            },
        }
        try:
            response = await self._http_client.post(
                f"{self.config.base_url}/responses",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json=request_body,
                timeout=self.config.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("OpenAI Responses request timed out.") from exc
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                "Could not connect to the OpenAI Responses API."
            ) from exc

        request_id = response.headers.get("x-request-id")
        _raise_for_provider_status(response.status_code)

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMInvalidResponseError(
                "OpenAI response body was not valid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise LLMInvalidResponseError(
                "OpenAI response root must be a JSON object."
            )
        if payload.get("error") is not None:
            raise LLMUpstreamError("OpenAI returned an error response object.")

        response_status = payload.get("status")
        if response_status not in (None, "completed"):
            raise LLMIncompleteResponseError(
                f"OpenAI response status was {response_status!r}."
            )
        if _contains_refusal(payload):
            raise LLMIncompleteResponseError("OpenAI refused the analysis request.")

        output_text = _extract_output_text(payload)
        try:
            output = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise LLMInvalidResponseError(
                "OpenAI output_text was not valid JSON."
            ) from exc
        if not isinstance(output, dict):
            raise LLMInvalidResponseError(
                "OpenAI structured output root must be a JSON object."
            )

        model = payload.get("model")
        response_id = payload.get("id")
        service_tier = payload.get("service_tier")
        return LLMProviderResult(
            output=output,
            provider=self.provider_name,
            model=model if isinstance(model, str) and model else self.config.model,
            response_id=(
                response_id if isinstance(response_id, str) and response_id else None
            ),
            request_id=request_id,
            created_at=_parse_created_at(payload.get("created_at")),
            service_tier=(
                service_tier
                if isinstance(service_tier, str) and service_tier
                else None
            ),
            usage=_parse_usage(payload.get("usage")),
        )


class THUChatCompletionsProvider:
    """Tunghai University Chat Completions adapter.

    THU's public API contract documents ``/chat/completions`` but does not
    document strict JSON Schema response formatting. The canonical schema is
    therefore supplied as trusted system context, and the existing Pydantic and
    safety validators remain the fail-closed enforcement boundary.
    """

    provider_name = "thu"

    def __init__(
        self,
        config: THUChatCompletionsConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.model_name = config.model
        self._owns_http_client = http_client is None
        self._resolved_json_mode: str | None = (
            None if config.json_mode == "auto" else config.json_mode
        )
        self._json_mode_lock = asyncio.Lock()
        self._http_client = http_client or httpx.AsyncClient(
            timeout=config.timeout_seconds,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "BizCheckAI/0.1",
            },
        )

    async def __aenter__(self) -> THUChatCompletionsProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def ensure_available(self) -> None:
        return None

    @property
    def active_json_mode(self) -> str:
        if self.config.json_mode != "auto":
            return self.config.json_mode
        return self._resolved_json_mode or "auto_unresolved"

    async def generate(
        self,
        prompt: AnalysisPromptPackage,
    ) -> LLMProviderResult:
        _, user_message = prompt.messages
        schema_instruction = build_thu_system_instruction(prompt)
        request_body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": schema_instruction},
                {"role": "user", "content": user_message.content},
            ],
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
        }
        try:
            response = await self._post_with_json_mode(request_body, prompt)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("THU Chat Completions request timed out.") from exc
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                "Could not connect to the THU Chat Completions API."
            ) from exc

        request_id = response.headers.get("x-request-id")
        _raise_for_provider_status(response.status_code, provider_label="THU")

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMInvalidResponseError(
                "THU response body was not valid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise LLMInvalidResponseError(
                "THU response root must be a JSON object."
            )
        if payload.get("error") is not None:
            raise LLMUpstreamError("THU returned an error response object.")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMInvalidResponseError(
                "THU response did not contain a completion choice."
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMInvalidResponseError(
                "THU completion choice must be a JSON object."
            )
        finish_reason = choice.get("finish_reason")
        if finish_reason not in (None, "stop"):
            raise LLMIncompleteResponseError(
                f"THU completion finish_reason was {finish_reason!r}."
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMInvalidResponseError(
                "THU completion did not contain an assistant message."
            )
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise LLMIncompleteResponseError("THU refused the analysis request.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMInvalidResponseError(
                "THU assistant message did not contain output text."
            )
        try:
            output = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMInvalidResponseError(
                "THU assistant message was not valid JSON."
            ) from exc
        if not isinstance(output, dict):
            raise LLMInvalidResponseError(
                "THU structured output root must be a JSON object."
            )

        model = payload.get("model")
        response_id = payload.get("id")
        service_tier = payload.get("service_tier")
        return LLMProviderResult(
            output=output,
            provider=self.provider_name,
            model=model if isinstance(model, str) and model else self.config.model,
            response_id=(
                response_id if isinstance(response_id, str) and response_id else None
            ),
            request_id=request_id,
            created_at=_parse_created_at(payload.get("created")),
            service_tier=(
                service_tier
                if isinstance(service_tier, str) and service_tier
                else None
            ),
            usage=_parse_usage(payload.get("usage")),
        )

    async def _post_with_json_mode(
        self,
        request_body: dict[str, object],
        prompt: AnalysisPromptPackage,
    ) -> httpx.Response:
        if self.config.json_mode != "auto":
            return await self._post_chat_completion(
                request_body,
                prompt,
                json_mode=self.config.json_mode,
            )

        if self._resolved_json_mode is not None:
            return await self._post_chat_completion(
                request_body,
                prompt,
                json_mode=self._resolved_json_mode,
            )

        async with self._json_mode_lock:
            if self._resolved_json_mode is not None:
                return await self._post_chat_completion(
                    request_body,
                    prompt,
                    json_mode=self._resolved_json_mode,
                )

            for json_mode in ("json_schema", "json_object", "off"):
                response = await self._post_chat_completion(
                    request_body,
                    prompt,
                    json_mode=json_mode,
                )
                if response.status_code in {400, 422} and json_mode != "off":
                    continue
                if 200 <= response.status_code < 300:
                    self._resolved_json_mode = json_mode
                return response

        raise AssertionError("THU JSON mode negotiation did not return a response.")

    async def _post_chat_completion(
        self,
        request_body: dict[str, object],
        prompt: AnalysisPromptPackage,
        *,
        json_mode: str,
    ) -> httpx.Response:
        body = dict(request_body)
        response_format = _thu_response_format(prompt, json_mode=json_mode)
        if response_format is not None:
            body["response_format"] = response_format
        return await self._http_client.post(
            f"{self.config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json=body,
            timeout=self.config.timeout_seconds,
        )


def _provider_prompt_cache_key(prompt: AnalysisPromptPackage) -> str:
    """Return a stable, non-identifying key for OpenAI prompt-prefix caching."""

    task = getattr(prompt, "task", "company_analysis")
    task_token = "comparison" if task == "company_comparison_analysis" else "company"
    return (
        f"bizcheck-{task_token}-analysis-{prompt.prompt_version}-"
        f"{prompt.system_prompt_sha256[:16]}"
    )


_PROVIDER_DOCUMENT_KEYS = frozenset(
    {
        "$schema",
        "$id",
        "default",
        "examples",
        "title",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    }
)

_SCHEMA_NAME_MAP_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)


def build_openai_strict_schema(
    canonical_schema: dict[str, object],
) -> dict[str, object]:
    """Return an OpenAI strict-schema copy without mutating the public schema.

    OpenAI strict structured outputs require every object property to be listed
    in ``required``. Optional values therefore remain required-but-nullable in
    this provider-only copy. Constraints removed here are enforced again by the
    canonical Pydantic model after generation.
    """

    schema = copy.deepcopy(canonical_schema)
    normalized = _normalize_strict_schema_node(schema)
    if not isinstance(normalized, dict):
        raise TypeError("The structured-output schema root must be an object.")
    return normalized


def _normalize_strict_schema_node(value: object) -> object:
    if isinstance(value, list):
        return [_normalize_strict_schema_node(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized: dict[str, object] = {}
    for key, item in value.items():
        if key == "enum":
            # Enum members are instance data, never schema nodes. Business
            # fields such as title must survive in fixed-output contracts.
            normalized[key] = copy.deepcopy(item)
            continue
        if key in _PROVIDER_DOCUMENT_KEYS or key == "const":
            continue
        if key in _SCHEMA_NAME_MAP_KEYS and isinstance(item, dict):
            # Keys inside these maps are business property or definition names,
            # not JSON Schema keywords. Preserve names such as ``title`` or
            # ``default`` while still normalizing each child schema.
            normalized[key] = {
                name: _normalize_strict_schema_node(child_schema)
                for name, child_schema in item.items()
            }
            continue
        normalized[key] = _normalize_strict_schema_node(item)
    if "const" in value:
        normalized["enum"] = [copy.deepcopy(value["const"])]

    properties = normalized.get("properties")
    if isinstance(properties, dict):
        normalized["required"] = list(properties)
        normalized["additionalProperties"] = False
    return normalized


def create_company_analysis_provider(
    settings: Settings,
) -> CompanyAnalysisProvider:
    provider_name = settings.llm_provider.strip().lower()
    if provider_name in {"", "disabled", "none"}:
        return DisabledCompanyAnalysisProvider(settings.openai_model)
    if provider_name == "thu":
        if not settings.thu_api_key or not settings.thu_api_key.strip():
            return DisabledCompanyAnalysisProvider(settings.thu_model)
        try:
            config = THUChatCompletionsConfig(
                api_key=settings.thu_api_key,
                base_url=settings.thu_base_url,
                model=settings.thu_model,
                timeout_seconds=settings.llm_timeout_seconds,
                max_output_tokens=settings.llm_max_output_tokens,
                json_mode=settings.thu_json_mode,
                temperature=settings.thu_temperature,
            )
        except ValueError as exc:
            return MisconfiguredCompanyAnalysisProvider(
                str(exc),
                settings.thu_model,
            )
        return THUChatCompletionsProvider(config)
    if provider_name != "openai":
        return MisconfiguredCompanyAnalysisProvider(
            f"Unsupported LLM provider: {provider_name}",
            settings.openai_model,
        )
    if not settings.openai_api_key or not settings.openai_api_key.strip():
        return DisabledCompanyAnalysisProvider(settings.openai_model)

    try:
        config = OpenAIResponsesConfig(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
        )
    except ValueError as exc:
        return MisconfiguredCompanyAnalysisProvider(
            str(exc),
            settings.openai_model,
        )
    return OpenAIResponsesProvider(config)


def _thu_response_format(
    prompt: AnalysisPromptPackage,
    *,
    json_mode: str,
) -> dict[str, object] | None:
    if json_mode == "off":
        return None
    if json_mode == "json_object":
        return {"type": "json_object"}
    if json_mode != "json_schema":
        raise ValueError(f"Unsupported THU JSON mode: {json_mode}")

    task = getattr(prompt, "task", "company_analysis")
    schema_name = getattr(
        prompt,
        "schema_name",
        "bizcheck_company_analysis_v1",
    )
    description = (
        "BizCheck AI 2–3 company comparison analysis v1"
        if task == "company_comparison_analysis"
        else "BizCheck AI single-company analysis v1"
    )
    canonical_thu_schema = build_thu_response_schema(prompt)
    strict_thu_schema = build_openai_strict_schema(canonical_thu_schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "description": description,
            "schema": apply_thu_narrative_schema_guards(
                build_server_owned_manifest_schema(
                    restore_thu_collection_bounds(
                        strict_thu_schema,
                        canonical_thu_schema,
                    )
                )
            ),
            "strict": True,
        },
    }


def _raise_for_provider_status(
    status_code: int,
    *,
    provider_label: str = "OpenAI",
) -> None:
    if 200 <= status_code < 300:
        return
    if status_code in {401, 403}:
        raise LLMAuthenticationError(
            f"{provider_label} rejected credentials with HTTP {status_code}."
        )
    if status_code == 429:
        raise LLMRateLimitError(f"{provider_label} rate limit was reached.")
    if status_code in {400, 404, 409, 422}:
        raise LLMRequestRejectedError(
            f"{provider_label} rejected the request with HTTP {status_code}."
        )
    if status_code >= 500:
        raise LLMUpstreamError(
            f"{provider_label} returned server error HTTP {status_code}."
        )
    raise LLMUpstreamError(
        f"{provider_label} returned unexpected HTTP {status_code}."
    )


def _contains_refusal(payload: dict[str, Any]) -> bool:
    output = payload.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "refusal"
            for part in content
        ):
            return True
    return False


def _extract_output_text(payload: dict[str, Any]) -> str:
    direct_output = payload.get("output_text")
    if isinstance(direct_output, str) and direct_output.strip():
        return direct_output

    fragments: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "output_text":
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    fragments.append(text)

    joined = "".join(fragments)
    if not joined.strip():
        raise LLMInvalidResponseError(
            "OpenAI response did not contain structured output text."
        )
    return joined


def _parse_created_at(value: object) -> datetime | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _parse_usage(value: object) -> LLMProviderUsage | None:
    if not isinstance(value, dict):
        return None
    token_values = (
        value.get("input_tokens", value.get("prompt_tokens")),
        value.get("output_tokens", value.get("completion_tokens")),
        value.get("total_tokens"),
    )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in token_values
    ):
        return None
    return LLMProviderUsage(
        input_tokens=token_values[0],
        output_tokens=token_values[1],
        total_tokens=token_values[2],
    )
