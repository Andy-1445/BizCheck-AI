from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import APIModel
from app.schemas.company_analysis import LLMTokenUsage
from app.schemas.company_comparison import CompanyComparisonResponse, TaxId
from app.schemas.llm_comparison_analysis import CompanyComparisonLLMOutput


COMPARISON_ANALYSIS_VERSION = "1.0"
COMPARISON_ANALYSIS_CACHE_KEY_VERSION = "1.0"

FallbackCode = Literal[
    "not_configured",
    "timeout",
    "provider_unavailable",
    "provider_error",
    "invalid_output",
]
CacheStatus = Literal["hit", "miss", "bypass", "stale"]


class CompanyComparisonAnalysisRequest(APIModel):
    tax_ids: list[TaxId] = Field(
        min_length=2,
        max_length=3,
        description="Two or three unique tax IDs in display order.",
        json_schema_extra={"uniqueItems": True},
    )
    force_refresh: bool = Field(
        default=False,
        strict=True,
        description=(
            "Bypass a fresh application cache entry. A validated stale entry may "
            "still be served if the provider attempt fails."
        ),
    )

    @model_validator(mode="after")
    def validate_unique_tax_ids(self) -> "CompanyComparisonAnalysisRequest":
        if len(self.tax_ids) != len(set(self.tax_ids)):
            raise ValueError("tax_ids must not contain duplicates.")
        return self


class CompanyComparisonFallback(APIModel):
    code: FallbackCode
    title: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=260)
    retryable: bool


class CompanyComparisonAnalysisData(APIModel):
    status: Literal["completed", "insufficient_data", "fallback"]
    comparison: CompanyComparisonResponse
    analysis: CompanyComparisonLLMOutput | None = None
    fallback: CompanyComparisonFallback | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "CompanyComparisonAnalysisData":
        if self.status == "fallback":
            if self.analysis is not None or self.fallback is None:
                raise ValueError("fallback status requires only fallback content.")
        else:
            if (
                self.analysis is None
                or self.fallback is not None
                or self.status != self.analysis.status
            ):
                raise ValueError("AI status must match the validated analysis output.")
        return self


class CompanyComparisonAnalysisCacheMeta(APIModel):
    status: CacheStatus
    key_version: Literal["1.0"] = COMPARISON_ANALYSIS_CACHE_KEY_VERSION
    key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ttl_seconds: int = Field(ge=1)
    stale_if_error_seconds: int = Field(ge=0)
    cached_at: datetime | None = None
    expires_at: datetime | None = None
    stale_expires_at: datetime | None = None

    @model_validator(mode="after")
    def validate_cache_window(self) -> "CompanyComparisonAnalysisCacheMeta":
        timestamps = (self.cached_at, self.expires_at, self.stale_expires_at)
        if any(
            value is not None
            and (value.tzinfo is None or value.utcoffset() is None)
            for value in timestamps
        ):
            raise ValueError("cache timestamps must include a UTC offset.")
        all_empty = all(value is None for value in timestamps)
        all_present = all(value is not None for value in timestamps)
        if not (all_empty or all_present):
            raise ValueError("cache timestamps must be all present or all omitted.")
        if all_present:
            assert self.cached_at is not None
            assert self.expires_at is not None
            assert self.stale_expires_at is not None
            if not self.cached_at <= self.expires_at <= self.stale_expires_at:
                raise ValueError("cache timestamps are out of order.")
        if self.status in {"hit", "stale"} and not all_present:
            raise ValueError("cache hits must include their cache window.")
        return self


class CompanyComparisonAnalysisFallbackMeta(APIModel):
    active: bool
    code: FallbackCode | None = None
    retryable: bool

    @model_validator(mode="after")
    def validate_active_state(self) -> "CompanyComparisonAnalysisFallbackMeta":
        if self.active != (self.code is not None):
            raise ValueError("fallback code must match active state.")
        if not self.active and self.retryable:
            raise ValueError("inactive fallback cannot be retryable.")
        return self


class CompanyComparisonAnalysisMeta(APIModel):
    analysis_version: Literal["1.0"] = COMPARISON_ANALYSIS_VERSION
    provider: str = Field(min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=120)
    provider_response_id: str | None = Field(default=None, max_length=200)
    provider_request_id: str | None = Field(default=None, max_length=200)
    service_tier: str | None = Field(default=None, max_length=80)
    generated_at: datetime
    prompt_version: Literal["1.0"] = "1.0"
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_schema_version: Literal["1.0"] = "1.0"
    output_schema_version: Literal["1.0"] = "1.0"
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    attempts: int = Field(ge=0, le=3)
    duration_ms: int = Field(ge=0)
    usage: LLMTokenUsage | None = None
    cache: CompanyComparisonAnalysisCacheMeta
    fallback: CompanyComparisonAnalysisFallbackMeta

    @model_validator(mode="after")
    def validate_generated_at(self) -> "CompanyComparisonAnalysisMeta":
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must include a UTC offset.")
        return self


class CompanyComparisonAnalysisResponse(APIModel):
    data: CompanyComparisonAnalysisData
    meta: CompanyComparisonAnalysisMeta

    @model_validator(mode="after")
    def validate_fallback_consistency(self) -> "CompanyComparisonAnalysisResponse":
        fallback = self.data.fallback
        if self.meta.fallback.active != (fallback is not None):
            raise ValueError("fallback metadata does not match response data.")
        if fallback is not None:
            if (
                self.meta.fallback.code != fallback.code
                or self.meta.fallback.retryable != fallback.retryable
                or self.meta.output_sha256 is not None
            ):
                raise ValueError("fallback metadata is inconsistent.")
        elif self.meta.output_sha256 is None:
            raise ValueError("AI responses require an output hash.")
        return self
