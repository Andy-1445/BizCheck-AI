from datetime import date
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.schemas.bizscore import BizScoreResult
from app.schemas.common import APIModel
from app.schemas.company import CompanyData, ResponseMeta


LLM_INPUT_SCHEMA_ID = (
    "https://bizcheck.local/schemas/llm-company-analysis-input-v1.schema.json"
)
LLM_OUTPUT_SCHEMA_ID = (
    "https://bizcheck.local/schemas/llm-company-analysis-output-v1.schema.json"
)
AI_ANALYSIS_DISCLAIMER = (
    "AI 內容僅解釋已提供資料與計算結果，可能有錯誤或遺漏，"
    "不應取代獨立查核與專業意見。"
)

EXPECTED_DIMENSION_KEYS = frozenset(
    {
        "registration_status",
        "company_age",
        "registered_capital_scale",
        "registration_change_recency",
        "peer_relative_position",
    }
)

EvidencePath = Annotated[
    str,
    Field(
        min_length=9,
        max_length=180,
        pattern=r"^/(company|bizscore|source_meta)(/[A-Za-z0-9_~-]+)*$",
        description=(
            "指向本次 LLM 輸入的 JSON Pointer；只能引用 company、bizscore "
            "或 source_meta。"
        ),
        examples=["/bizscore/dimensions/0/evidence/0"],
    ),
]


class CompanyAnalysisLLMInput(APIModel):
    """Verified facts supplied to the company-analysis LLM."""

    schema_version: Literal["1.0"] = "1.0"
    task: Literal["company_analysis"] = "company_analysis"
    language: Literal["zh-TW"] = "zh-TW"
    disclaimer_version: Literal["1.0"] = "1.0"
    company: CompanyData
    bizscore: BizScoreResult
    source_meta: ResponseMeta

    @model_validator(mode="after")
    def validate_analysis_input(self) -> "CompanyAnalysisLLMInput":
        if len(self.company.tax_id) != 8 or not self.company.tax_id.isdigit():
            raise ValueError("company.tax_id must contain exactly 8 digits.")

        dimension_keys = [dimension.key for dimension in self.bizscore.dimensions]
        if len(dimension_keys) != len(set(dimension_keys)):
            raise ValueError("bizscore.dimensions contains duplicate keys.")
        if set(dimension_keys) != EXPECTED_DIMENSION_KEYS:
            raise ValueError(
                "bizscore.dimensions must contain exactly the five BizScore v1 keys."
            )
        if self.bizscore.peer_benchmark.dimension.key != "peer_relative_position":
            raise ValueError(
                "bizscore.peer_benchmark.dimension must be peer_relative_position."
            )
        return self


FindingTopic = Literal[
    "registration_status",
    "company_age",
    "registered_capital_scale",
    "registration_change_recency",
    "peer_relative_position",
    "data_completeness",
]


class LLMAnalysisFinding(APIModel):
    topic: FindingTopic
    title: str = Field(min_length=1, max_length=40)
    observation: str = Field(
        min_length=1,
        max_length=240,
        description="僅描述輸入資料可支持的觀察，不得新增分數或未提供的事實。",
    )
    evidence_paths: list[EvidencePath] = Field(min_length=1, max_length=6)
    caveat: str | None = Field(default=None, min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_evidence_paths(self) -> "LLMAnalysisFinding":
        _ensure_unique_paths(self.evidence_paths, "finding.evidence_paths")
        return self


class LLMVerificationItem(APIModel):
    priority: Literal["優先", "一般", "補充"]
    question: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=180)
    related_evidence_paths: list[EvidencePath] = Field(
        default_factory=list,
        max_length=6,
    )

    @model_validator(mode="after")
    def validate_related_paths(self) -> "LLMVerificationItem":
        _ensure_unique_paths(
            self.related_evidence_paths,
            "verification_item.related_evidence_paths",
        )
        return self


LimitationCode = Literal[
    "public_data_only",
    "ai_generated",
    "partial_source_data",
    "missing_dimension",
    "provisional_score",
    "no_numeric_score",
    "benchmark_unavailable",
]


class LLMAnalysisLimitation(APIModel):
    code: LimitationCode
    message: str = Field(min_length=1, max_length=180)
    related_evidence_paths: list[EvidencePath] = Field(
        default_factory=list,
        max_length=6,
    )

    @model_validator(mode="after")
    def validate_related_paths(self) -> "LLMAnalysisLimitation":
        _ensure_unique_paths(
            self.related_evidence_paths,
            "limitation.related_evidence_paths",
        )
        return self


class LLMAnalysisProvenance(APIModel):
    input_schema_version: Literal["1.0"] = "1.0"
    prompt_version: Literal["1.0"] = "1.0"
    company_tax_id: str = Field(pattern=r"^\d{8}$")
    bizscore_version: Literal["1.0"] = "1.0"
    benchmark_catalog_version: str = Field(min_length=1, max_length=100)
    data_as_of: date
    disclaimer_version: Literal["1.0"] = "1.0"
    evidence_paths_used: list[EvidencePath] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def validate_evidence_paths(self) -> "LLMAnalysisProvenance":
        _ensure_unique_paths(
            self.evidence_paths_used,
            "provenance.evidence_paths_used",
        )
        return self


class CompanyAnalysisLLMOutput(APIModel):
    """Natural-language analysis that never owns deterministic BizScore fields."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["completed", "insufficient_data"]
    headline: str = Field(min_length=1, max_length=80)
    overall_observation: str = Field(min_length=1, max_length=360)
    findings: list[LLMAnalysisFinding] = Field(default_factory=list, max_length=6)
    verification_items: list[LLMVerificationItem] = Field(min_length=1, max_length=8)
    limitations: list[LLMAnalysisLimitation] = Field(min_length=2, max_length=8)
    provenance: LLMAnalysisProvenance
    disclaimer: Literal[AI_ANALYSIS_DISCLAIMER] = AI_ANALYSIS_DISCLAIMER

    @model_validator(mode="after")
    def validate_analysis_output(self) -> "CompanyAnalysisLLMOutput":
        if self.status == "completed" and not self.findings:
            raise ValueError("completed output must contain at least one finding.")
        if self.status == "insufficient_data" and self.findings:
            raise ValueError("insufficient_data output must not contain findings.")

        topics = [finding.topic for finding in self.findings]
        if len(topics) != len(set(topics)):
            raise ValueError("findings must not repeat the same topic.")

        limitation_codes = [limitation.code for limitation in self.limitations]
        if len(limitation_codes) != len(set(limitation_codes)):
            raise ValueError("limitations must not repeat the same code.")
        required_limitations = {"public_data_only", "ai_generated"}
        if not required_limitations.issubset(limitation_codes):
            raise ValueError(
                "limitations must include public_data_only and ai_generated."
            )

        referenced_paths = {
            path
            for finding in self.findings
            for path in finding.evidence_paths
        }
        referenced_paths.update(
            path
            for item in self.verification_items
            for path in item.related_evidence_paths
        )
        referenced_paths.update(
            path
            for limitation in self.limitations
            for path in limitation.related_evidence_paths
        )
        if referenced_paths != set(self.provenance.evidence_paths_used):
            raise ValueError(
                "provenance.evidence_paths_used must exactly match all referenced paths."
            )
        return self


def build_llm_input_json_schema() -> dict[str, object]:
    return _schema_document(
        CompanyAnalysisLLMInput,
        LLM_INPUT_SCHEMA_ID,
    )


def build_llm_output_json_schema() -> dict[str, object]:
    return _schema_document(
        CompanyAnalysisLLMOutput,
        LLM_OUTPUT_SCHEMA_ID,
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
