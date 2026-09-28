from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypeVar

from pydantic import ValidationError

from app.schemas.llm_analysis import (
    CompanyAnalysisLLMInput,
    CompanyAnalysisLLMOutput,
)
from app.services.llm_prompt import (
    CompanyAnalysisPromptPackage,
    LLMOutputSafetyError,
    build_company_analysis_prompt,
    validate_company_analysis_output,
)
from app.services.llm_output_normalizer import normalize_evidence_manifest
from app.services.thu_prompt import validate_thu_output_contract
from app.services.llm_provider import (
    AnalysisPromptPackage,
    CompanyAnalysisProvider,
    LLMInvalidResponseError,
    LLMProviderResult,
)
from app.services.llm_validation_diagnostics import (
    LLMValidationDiagnostic,
    log_llm_validation_failure,
)


PromptPackage = TypeVar("PromptPackage", bound=AnalysisPromptPackage)


class CompanyAnalysisInvalidOutputError(RuntimeError):
    """All allowed attempts returned malformed or unsafe analysis output."""


@dataclass(frozen=True, slots=True)
class CompanyAnalysisExecution:
    analysis_input: CompanyAnalysisLLMInput
    analysis: CompanyAnalysisLLMOutput
    prompt: CompanyAnalysisPromptPackage
    provider_result: LLMProviderResult
    input_sha256: str
    output_sha256: str
    attempts: int
    duration_ms: int
    generated_at: datetime


class CompanyAnalysisService:
    """Build Prompt v1 and accept only output that passes deterministic guards."""

    def __init__(
        self,
        provider: CompanyAnalysisProvider,
        *,
        max_attempts: int = 1,
    ) -> None:
        if max_attempts < 1 or max_attempts > 3:
            raise ValueError("LLM max_attempts must be between 1 and 3.")
        self.provider = provider
        self.max_attempts = max_attempts

    def ensure_available(self) -> None:
        self.provider.ensure_available()

    async def analyze(
        self,
        analysis_input: CompanyAnalysisLLMInput | dict[str, object],
    ) -> CompanyAnalysisExecution:
        verified_input = CompanyAnalysisLLMInput.model_validate(analysis_input)
        prompt = build_company_analysis_prompt(verified_input)
        attempt_prompt = prompt
        started_at = time.perf_counter()
        final_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                provider_result = await self.provider.generate(attempt_prompt)
                normalized_output = normalize_evidence_manifest(
                    provider_result.output,
                    task="company_analysis",
                )
                analysis = validate_company_analysis_output(
                    verified_input,
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
                    task="company_analysis",
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
                    attempt_prompt = _with_validation_retry_guidance(
                        prompt,
                        diagnostic,
                    )
                    continue
                break

            duration_ms = max(
                0,
                round((time.perf_counter() - started_at) * 1000),
            )
            return CompanyAnalysisExecution(
                analysis_input=verified_input,
                analysis=analysis,
                prompt=attempt_prompt,
                provider_result=provider_result,
                input_sha256=_model_sha256(verified_input),
                output_sha256=_model_sha256(analysis),
                attempts=attempt,
                duration_ms=duration_ms,
                generated_at=provider_result.created_at
                or datetime.now(timezone.utc),
            )

        raise CompanyAnalysisInvalidOutputError(
            "The provider output failed JSON, schema, or safety validation."
        ) from final_error


def _with_validation_retry_guidance(
    prompt: PromptPackage,
    diagnostic: LLMValidationDiagnostic,
) -> PromptPackage:
    """Add only sanitized validation metadata to a retry prompt.

    The rejected provider output and exception text are deliberately excluded.
    The original verified input remains unchanged inside its data boundary.
    """

    system_message, user_message = prompt.messages
    feedback = json.dumps(
        {
            "stage": diagnostic.stage,
            "fields": list(diagnostic.fields),
            "reasons": list(diagnostic.reasons),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    retry_instruction = (
        "\n<validation_retry_feedback>\n"
        "前次輸出未通過驗證。不要沿用或猜測前次內容，請依原始 verified input "
        "重新產生完整 JSON。下列資料只包含安全化的錯誤欄位與原因：\n"
        f"{feedback}\n"
        "</validation_retry_feedback>"
    )
    return prompt.model_copy(
        update={
            "messages": [
                system_message,
                user_message.model_copy(
                    update={"content": user_message.content + retry_instruction}
                ),
            ]
        }
    )


def _model_sha256(
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
