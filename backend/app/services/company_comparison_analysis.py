from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import ValidationError

from app.schemas.company_analysis import LLMTokenUsage
from app.schemas.company_comparison import CompanyComparisonResponse
from app.schemas.company_comparison_analysis import (
    COMPARISON_ANALYSIS_CACHE_KEY_VERSION,
    CompanyComparisonAnalysisCacheMeta,
    CompanyComparisonAnalysisData,
    CompanyComparisonAnalysisFallbackMeta,
    CompanyComparisonAnalysisMeta,
    CompanyComparisonAnalysisResponse,
    CompanyComparisonFallback,
    FallbackCode,
)
from app.schemas.llm_comparison_analysis import (
    CompanyComparisonLLMInput,
    CompanyComparisonLLMOutput,
)
from app.services.llm_cache import AsyncLRUTTLCache, CacheResolution
from app.services.company_analysis import _with_validation_retry_guidance
from app.services.llm_comparison_prompt import (
    CompanyComparisonPromptPackage,
    build_company_comparison_prompt,
    validate_company_comparison_output,
)
from app.services.llm_prompt import LLMOutputSafetyError
from app.services.llm_output_normalizer import normalize_evidence_manifest
from app.services.thu_prompt import THU_SAFETY_REVISION, validate_thu_output_contract
from app.services.llm_provider import (
    CompanyAnalysisProvider,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMIncompleteResponseError,
    LLMInvalidResponseError,
    LLMNotConfiguredError,
    LLMProviderError,
    LLMProviderResult,
    LLMRateLimitError,
    LLMRequestRejectedError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.services.llm_validation_diagnostics import log_llm_validation_failure


class CompanyComparisonAnalysisInvalidOutputError(RuntimeError):
    """All allowed attempts returned malformed, ungrounded, or unsafe output."""

    def __init__(self, attempts: int) -> None:
        super().__init__(
            "The provider output failed JSON, schema, grounding, or safety validation."
        )
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class CompanyComparisonAnalysisExecution:
    analysis_input: CompanyComparisonLLMInput
    analysis: CompanyComparisonLLMOutput
    prompt: CompanyComparisonPromptPackage
    provider_result: LLMProviderResult
    input_sha256: str
    output_sha256: str
    attempts: int
    duration_ms: int
    generated_at: datetime


class CompanyComparisonAnalysisService:
    """Generate safe comparison observations with versioned cache/fallback behavior."""

    def __init__(
        self,
        provider: CompanyAnalysisProvider,
        cache: AsyncLRUTTLCache[CompanyComparisonAnalysisExecution],
        *,
        max_attempts: int = 1,
    ) -> None:
        if max_attempts < 1 or max_attempts > 3:
            raise ValueError("LLM max_attempts must be between 1 and 3.")
        self.provider = provider
        self.cache = cache
        self.max_attempts = max_attempts

    async def analyze(
        self,
        comparison: CompanyComparisonResponse | dict[str, object],
        *,
        force_refresh: bool = False,
    ) -> CompanyComparisonAnalysisResponse:
        verified_comparison = CompanyComparisonResponse.model_validate(comparison)
        analysis_input = CompanyComparisonLLMInput(comparison=verified_comparison)
        prompt = build_company_comparison_prompt(analysis_input)
        input_sha256 = stable_comparison_input_sha256(analysis_input)
        cache_key = build_comparison_analysis_cache_key(
            analysis_input,
            prompt,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
        )
        request_started = time.perf_counter()

        async def generate() -> CompanyComparisonAnalysisExecution:
            return await self._generate(
                analysis_input,
                prompt,
                input_sha256=input_sha256,
            )

        try:
            resolution = await self.cache.resolve(
                cache_key,
                generate,
                force_refresh=force_refresh,
            )
        except (LLMProviderError, CompanyComparisonAnalysisInvalidOutputError) as error:
            duration_ms = max(
                0,
                round((time.perf_counter() - request_started) * 1000),
            )
            return self._fallback_response(
                verified_comparison,
                prompt,
                input_sha256=input_sha256,
                cache_key=cache_key,
                force_refresh=force_refresh,
                error=error,
                duration_ms=duration_ms,
            )

        return self._success_response(
            verified_comparison,
            cache_key=cache_key,
            resolution=resolution,
        )

    async def _generate(
        self,
        analysis_input: CompanyComparisonLLMInput,
        prompt: CompanyComparisonPromptPackage,
        *,
        input_sha256: str,
    ) -> CompanyComparisonAnalysisExecution:
        self.provider.ensure_available()
        started_at = time.perf_counter()
        final_error: Exception | None = None
        attempt_prompt = prompt

        for attempt in range(1, self.max_attempts + 1):
            try:
                provider_result = await self.provider.generate(attempt_prompt)
                normalized_output = normalize_evidence_manifest(
                    provider_result.output,
                    task="company_comparison_analysis",
                )
                analysis = validate_company_comparison_output(
                    analysis_input,
                    normalized_output,
                )
                if self.provider.provider_name == "thu":
                    validate_thu_output_contract(attempt_prompt, normalized_output)
            except (
                LLMInvalidResponseError,
                LLMOutputSafetyError,
                ValidationError,
            ) as error:
                diagnostic = log_llm_validation_failure(
                    task="company_comparison_analysis",
                    provider=self.provider.provider_name,
                    provider_mode=getattr(
                        self.provider,
                        "active_json_mode",
                        "unknown",
                    ),
                    attempt=attempt,
                    error=error,
                    response_schema=prompt.response_schema,
                )
                final_error = error
                if attempt < self.max_attempts:
                    attempt_prompt = _with_validation_retry_guidance(prompt, diagnostic)
                    continue
                raise CompanyComparisonAnalysisInvalidOutputError(attempt) from error

            duration_ms = max(
                0,
                round((time.perf_counter() - started_at) * 1000),
            )
            return CompanyComparisonAnalysisExecution(
                analysis_input=analysis_input,
                analysis=analysis,
                prompt=attempt_prompt,
                provider_result=provider_result,
                input_sha256=input_sha256,
                output_sha256=_model_sha256(analysis),
                attempts=attempt,
                duration_ms=duration_ms,
                generated_at=provider_result.created_at
                or datetime.now(timezone.utc),
            )

        raise CompanyComparisonAnalysisInvalidOutputError(self.max_attempts) from final_error

    def _success_response(
        self,
        comparison: CompanyComparisonResponse,
        *,
        cache_key: str,
        resolution: CacheResolution[CompanyComparisonAnalysisExecution],
    ) -> CompanyComparisonAnalysisResponse:
        execution = resolution.value
        provider_result = execution.provider_result
        usage = _usage(provider_result)
        window = resolution.window
        return CompanyComparisonAnalysisResponse(
            data=CompanyComparisonAnalysisData(
                status=execution.analysis.status,
                comparison=comparison,
                analysis=execution.analysis,
                fallback=None,
            ),
            meta=CompanyComparisonAnalysisMeta(
                provider=provider_result.provider,
                model=provider_result.model,
                provider_response_id=provider_result.response_id,
                provider_request_id=provider_result.request_id,
                service_tier=provider_result.service_tier,
                generated_at=execution.generated_at,
                prompt_version=execution.prompt.prompt_version,
                system_prompt_sha256=execution.prompt.system_prompt_sha256,
                input_sha256=execution.input_sha256,
                output_sha256=execution.output_sha256,
                attempts=execution.attempts,
                duration_ms=execution.duration_ms,
                usage=usage,
                cache=CompanyComparisonAnalysisCacheMeta(
                    status=resolution.status,
                    key_sha256=cache_key,
                    ttl_seconds=self.cache.ttl_seconds,
                    stale_if_error_seconds=self.cache.stale_if_error_seconds,
                    cached_at=window.cached_at,
                    expires_at=window.expires_at,
                    stale_expires_at=window.stale_expires_at,
                ),
                fallback=CompanyComparisonAnalysisFallbackMeta(
                    active=False,
                    code=None,
                    retryable=False,
                ),
            ),
        )

    def _fallback_response(
        self,
        comparison: CompanyComparisonResponse,
        prompt: CompanyComparisonPromptPackage,
        *,
        input_sha256: str,
        cache_key: str,
        force_refresh: bool,
        error: LLMProviderError | CompanyComparisonAnalysisInvalidOutputError,
        duration_ms: int,
    ) -> CompanyComparisonAnalysisResponse:
        fallback = _fallback_for_error(error)
        attempts = (
            error.attempts
            if isinstance(error, CompanyComparisonAnalysisInvalidOutputError)
            else 0
            if isinstance(error, (LLMNotConfiguredError, LLMConfigurationError))
            else 1
        )
        return CompanyComparisonAnalysisResponse(
            data=CompanyComparisonAnalysisData(
                status="fallback",
                comparison=comparison,
                analysis=None,
                fallback=fallback,
            ),
            meta=CompanyComparisonAnalysisMeta(
                provider=self.provider.provider_name,
                model=self.provider.model_name,
                generated_at=datetime.now(timezone.utc),
                prompt_version=prompt.prompt_version,
                system_prompt_sha256=prompt.system_prompt_sha256,
                input_sha256=input_sha256,
                output_sha256=None,
                attempts=attempts,
                duration_ms=duration_ms,
                usage=None,
                cache=CompanyComparisonAnalysisCacheMeta(
                    status="bypass" if force_refresh else "miss",
                    key_sha256=cache_key,
                    ttl_seconds=self.cache.ttl_seconds,
                    stale_if_error_seconds=self.cache.stale_if_error_seconds,
                    cached_at=None,
                    expires_at=None,
                    stale_expires_at=None,
                ),
                fallback=CompanyComparisonAnalysisFallbackMeta(
                    active=True,
                    code=fallback.code,
                    retryable=fallback.retryable,
                ),
            ),
        )


def stable_comparison_input_sha256(
    analysis_input: CompanyComparisonLLMInput | dict[str, object],
) -> str:
    """Hash substantive evidence while excluding fetch/generation timestamps."""

    verified_input = CompanyComparisonLLMInput.model_validate(analysis_input)
    payload = verified_input.model_dump(mode="json")
    comparison = payload["comparison"]
    if isinstance(comparison, dict):
        meta = comparison.get("meta")
        if isinstance(meta, dict):
            meta.pop("generated_at", None)
        data = comparison.get("data")
        if isinstance(data, dict):
            items = data.get("items")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    source_meta = item.get("source_meta")
                    if isinstance(source_meta, dict):
                        source_meta.pop("fetched_at", None)
                        source_meta.pop("data_freshness", None)
                        source_meta.pop("fallback_reason", None)
    return _json_sha256(payload)


def build_comparison_analysis_cache_key(
    analysis_input: CompanyComparisonLLMInput | dict[str, object],
    prompt: CompanyComparisonPromptPackage,
    *,
    provider: str,
    model: str,
) -> str:
    verified_input = CompanyComparisonLLMInput.model_validate(analysis_input)
    material = {
        "key_version": COMPARISON_ANALYSIS_CACHE_KEY_VERSION,
        "task": prompt.task,
        "requested_tax_ids": verified_input.comparison.meta.requested_tax_ids,
        "input_sha256": stable_comparison_input_sha256(verified_input),
        "input_schema_version": verified_input.schema_version,
        "output_schema_version": "1.0",
        "prompt_version": prompt.prompt_version,
        "system_prompt_sha256": prompt.system_prompt_sha256,
        "schema_name": prompt.schema_name,
        "response_schema_sha256": _json_sha256(prompt.response_schema),
        "safety_revision": THU_SAFETY_REVISION,
        "provider": provider,
        "model": model,
    }
    return _json_sha256(material)


def _model_sha256(model: CompanyComparisonLLMOutput) -> str:
    return _json_sha256(model.model_dump(mode="json"))


def _json_sha256(payload: object) -> str:
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _usage(provider_result: LLMProviderResult) -> LLMTokenUsage | None:
    if provider_result.usage is None:
        return None
    return LLMTokenUsage(
        input_tokens=provider_result.usage.input_tokens,
        output_tokens=provider_result.usage.output_tokens,
        total_tokens=provider_result.usage.total_tokens,
    )


def _fallback_for_error(
    error: LLMProviderError | CompanyComparisonAnalysisInvalidOutputError,
) -> CompanyComparisonFallback:
    common_message = (
        "已驗證的公司比較資料仍可使用；請直接查看各欄位，並另行核對交易條件與履約文件。"
    )
    if isinstance(error, LLMNotConfiguredError):
        code: FallbackCode = "not_configured"
        title = "AI 比較重點尚未啟用"
        retryable = False
    elif isinstance(error, LLMTimeoutError):
        code = "timeout"
        title = "AI 比較重點回應逾時"
        retryable = True
    elif isinstance(error, (LLMConnectionError, LLMRateLimitError)):
        code = "provider_unavailable"
        title = "AI 比較服務暫時無法使用"
        retryable = True
    elif isinstance(error, CompanyComparisonAnalysisInvalidOutputError):
        code = "invalid_output"
        title = "AI 比較結果未通過安全驗證"
        retryable = False
    else:
        code = "provider_error"
        title = "AI 比較服務未產生可用結果"
        retryable = isinstance(error, LLMUpstreamError)
        if isinstance(
            error,
            (
                LLMAuthenticationError,
                LLMConfigurationError,
                LLMRequestRejectedError,
                LLMIncompleteResponseError,
            ),
        ):
            retryable = False
    return CompanyComparisonFallback(
        code=code,
        title=title,
        message=common_message,
        retryable=retryable,
    )
