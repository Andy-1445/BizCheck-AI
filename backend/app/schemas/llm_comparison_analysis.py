from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.schemas.common import APIModel
from app.schemas.company_comparison import (
    COMPARISON_VERSION,
    CompanyComparisonResponse,
)


COMPARISON_LLM_INPUT_SCHEMA_ID = (
    "https://bizcheck.local/schemas/llm-company-comparison-input-v1.schema.json"
)
COMPARISON_LLM_OUTPUT_SCHEMA_ID = (
    "https://bizcheck.local/schemas/llm-company-comparison-output-v1.schema.json"
)
AI_COMPARISON_DISCLAIMER = (
    "AI 比較說明由模型依政府公開登記資料與 BizScore v1 產生，"
    "僅供合作前查核問題整理；順序不是排名，不代表信用、付款或履約能力，"
    "也不是合作、投資、授信或法律建議。"
)

ComparisonEvidencePath = Annotated[
    str,
    Field(
        min_length=18,
        max_length=220,
        pattern=(
            r"^/comparison/(data|meta)"
            r"(/[A-Za-z0-9_~-]+)*$"
        ),
        description=(
            "JSON Pointer into the verified CompanyComparisonResponse under "
            "the comparison input root."
        ),
        examples=[
            "/comparison/data/items/0/bizscore/score",
            "/comparison/data/context/peer_comparison_scope",
        ],
    ),
]


class CompanyComparisonLLMInput(APIModel):
    """Verified 2–3 company comparison supplied to the comparison LLM."""

    schema_version: Literal["1.0"] = "1.0"
    task: Literal["company_comparison_analysis"] = "company_comparison_analysis"
    language: Literal["zh-TW"] = "zh-TW"
    disclaimer_version: Literal["1.0"] = "1.0"
    comparison: CompanyComparisonResponse

    @model_validator(mode="after")
    def validate_comparison_input(self) -> "CompanyComparisonLLMInput":
        tax_ids = [item.company.tax_id for item in self.comparison.data.items]
        if tax_ids != self.comparison.meta.requested_tax_ids:
            raise ValueError("comparison items must preserve requested tax-ID order.")
        if not 2 <= len(tax_ids) <= 3:
            raise ValueError("comparison input must contain two or three companies.")
        if self.comparison.data.version != COMPARISON_VERSION:
            raise ValueError("unsupported deterministic comparison version.")
        return self


ComparisonCompanyTopic = Literal[
    "registration_status",
    "company_age",
    "registered_capital_scale",
    "registration_change_recency",
    "peer_relative_position",
    "data_completeness",
]
ComparisonSharedTopic = Literal[
    "bizscore_context",
    "peer_scope",
    "data_completeness",
]


class LLMCompanyObservation(APIModel):
    tax_id: str = Field(pattern=r"^[0-9]{8}$")
    topic: ComparisonCompanyTopic
    title: str = Field(min_length=1, max_length=48)
    observation: str = Field(min_length=1, max_length=260)
    evidence_paths: list[ComparisonEvidencePath] = Field(
        min_length=1,
        max_length=6,
    )
    caveat: str = Field(min_length=1, max_length=180)

    @model_validator(mode="after")
    def validate_evidence_paths(self) -> "LLMCompanyObservation":
        _ensure_unique_paths(self.evidence_paths, "company_observation.evidence_paths")
        return self


class LLMComparisonObservation(APIModel):
    topic: ComparisonSharedTopic
    title: str = Field(min_length=1, max_length=48)
    observation: str = Field(min_length=1, max_length=300)
    evidence_paths: list[ComparisonEvidencePath] = Field(
        min_length=1,
        max_length=8,
    )
    caveat: str = Field(min_length=1, max_length=180)

    @model_validator(mode="after")
    def validate_evidence_paths(self) -> "LLMComparisonObservation":
        _ensure_unique_paths(
            self.evidence_paths,
            "comparison_observation.evidence_paths",
        )
        return self


class LLMComparisonVerificationItem(APIModel):
    priority: Literal["優先", "一般", "補充"]
    question: str = Field(min_length=1, max_length=140)
    reason: str = Field(min_length=1, max_length=200)
    related_evidence_paths: list[ComparisonEvidencePath] = Field(
        default_factory=list,
        max_length=8,
    )

    @model_validator(mode="after")
    def validate_related_paths(self) -> "LLMComparisonVerificationItem":
        _ensure_unique_paths(
            self.related_evidence_paths,
            "verification_item.related_evidence_paths",
        )
        return self


ComparisonLimitationCode = Literal[
    "public_data_only",
    "ai_generated",
    "not_ranked",
    "partial_source_data",
    "provisional_score",
    "no_numeric_score",
    "cross_industry_comparison",
    "benchmark_unavailable",
]


class LLMComparisonLimitation(APIModel):
    code: ComparisonLimitationCode
    message: str = Field(min_length=1, max_length=200)
    related_evidence_paths: list[ComparisonEvidencePath] = Field(
        default_factory=list,
        max_length=8,
    )

    @model_validator(mode="after")
    def validate_related_paths(self) -> "LLMComparisonLimitation":
        _ensure_unique_paths(
            self.related_evidence_paths,
            "limitation.related_evidence_paths",
        )
        return self


class LLMComparisonProvenance(APIModel):
    input_schema_version: Literal["1.0"] = "1.0"
    prompt_version: Literal["1.0"] = "1.0"
    requested_tax_ids: list[str] = Field(min_length=2, max_length=3)
    comparison_version: Literal["1.0"] = COMPARISON_VERSION
    bizscore_version: Literal["1.0"] = "1.0"
    benchmark_catalog_version: str = Field(min_length=1, max_length=100)
    data_as_of: date
    disclaimer_version: Literal["1.0"] = "1.0"
    evidence_paths_used: list[ComparisonEvidencePath] = Field(
        min_length=1,
        max_length=50,
    )

    @model_validator(mode="after")
    def validate_provenance(self) -> "LLMComparisonProvenance":
        if any(
            len(tax_id) != 8 or not tax_id.isdigit()
            for tax_id in self.requested_tax_ids
        ):
            raise ValueError("requested_tax_ids must contain eight-digit values.")
        if len(self.requested_tax_ids) != len(set(self.requested_tax_ids)):
            raise ValueError("requested_tax_ids must not contain duplicates.")
        _ensure_unique_paths(
            self.evidence_paths_used,
            "provenance.evidence_paths_used",
        )
        return self


class CompanyComparisonLLMOutput(APIModel):
    """Grounded comparison explanation; it never ranks or chooses a company."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["completed", "insufficient_data"]
    headline: str = Field(min_length=1, max_length=90)
    overall_observation: str = Field(min_length=1, max_length=420)
    company_observations: list[LLMCompanyObservation] = Field(
        min_length=2,
        max_length=9,
    )
    comparison_observations: list[LLMComparisonObservation] = Field(
        min_length=1,
        max_length=4,
    )
    verification_items: list[LLMComparisonVerificationItem] = Field(
        min_length=1,
        max_length=10,
    )
    limitations: list[LLMComparisonLimitation] = Field(
        min_length=3,
        max_length=10,
    )
    provenance: LLMComparisonProvenance
    disclaimer: Literal[AI_COMPARISON_DISCLAIMER] = AI_COMPARISON_DISCLAIMER

    @model_validator(mode="after")
    def validate_output_shape(self) -> "CompanyComparisonLLMOutput":
        company_topics = [
            (item.tax_id, item.topic) for item in self.company_observations
        ]
        if len(company_topics) != len(set(company_topics)):
            raise ValueError("company observations must not repeat tax ID and topic.")
        comparison_topics = [item.topic for item in self.comparison_observations]
        if len(comparison_topics) != len(set(comparison_topics)):
            raise ValueError("comparison observations must not repeat a topic.")

        limitation_codes = [item.code for item in self.limitations]
        if len(limitation_codes) != len(set(limitation_codes)):
            raise ValueError("limitations must not repeat the same code.")
        required = {"public_data_only", "ai_generated", "not_ranked"}
        if not required.issubset(limitation_codes):
            raise ValueError(
                "limitations must include public_data_only, ai_generated, and not_ranked."
            )

        referenced_paths = {
            path
            for item in self.company_observations
            for path in item.evidence_paths
        }
        referenced_paths.update(
            path
            for item in self.comparison_observations
            for path in item.evidence_paths
        )
        referenced_paths.update(
            path
            for item in self.verification_items
            for path in item.related_evidence_paths
        )
        referenced_paths.update(
            path
            for item in self.limitations
            for path in item.related_evidence_paths
        )
        if referenced_paths != set(self.provenance.evidence_paths_used):
            raise ValueError(
                "provenance.evidence_paths_used must exactly match referenced paths."
            )
        return self


def build_comparison_llm_input_json_schema() -> dict[str, object]:
    return _schema_document(
        CompanyComparisonLLMInput,
        COMPARISON_LLM_INPUT_SCHEMA_ID,
    )


def build_comparison_llm_output_json_schema() -> dict[str, object]:
    return _schema_document(
        CompanyComparisonLLMOutput,
        COMPARISON_LLM_OUTPUT_SCHEMA_ID,
    )


def _schema_document(model: type[APIModel], schema_id: str) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id,
        **model.model_json_schema(),
    }


def _ensure_unique_paths(paths: list[str], field_name: str) -> None:
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field_name} must not contain duplicates.")
