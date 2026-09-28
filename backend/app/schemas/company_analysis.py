from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel
from app.schemas.llm_analysis import (
    CompanyAnalysisLLMInput,
    CompanyAnalysisLLMOutput,
)


class LLMTokenUsage(APIModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class CompanyAnalysisData(APIModel):
    input: CompanyAnalysisLLMInput
    analysis: CompanyAnalysisLLMOutput


class CompanyAnalysisMeta(APIModel):
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
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempts: int = Field(ge=1)
    duration_ms: int = Field(ge=0)
    usage: LLMTokenUsage | None = None


class CompanyAnalysisResponse(APIModel):
    data: CompanyAnalysisData
    meta: CompanyAnalysisMeta
