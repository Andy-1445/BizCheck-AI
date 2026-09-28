from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.schemas.bizscore import BizScoreResult
from app.schemas.common import APIModel
from app.schemas.company import CompanyData, ResponseMeta


COMPARISON_VERSION = "1.0"
COMPANY_COMPARISON_DISCLAIMER = (
    "公司比較僅並列政府公開登記資料與 BizScore v1 指標；順序依使用者選取，"
    "不代表信用、付款或履約能力、投資價值或合作建議。"
)
EXPECTED_BIZSCORE_DIMENSION_KEYS = (
    "registration_status",
    "company_age",
    "registered_capital_scale",
    "registration_change_recency",
    "peer_relative_position",
)
COMPANY_WARNING_CODES = {
    "STALE_SOURCE_DATA",
    "PARTIAL_SOURCE_DATA",
    "SOURCE_WARNING",
    "PROVISIONAL_SCORE",
    "NO_NUMERIC_SCORE",
}
GLOBAL_WARNING_CODES = {
    "DIFFERENT_INDUSTRY_BENCHMARKS",
    "PEER_COMPARISON_UNAVAILABLE",
}

TaxId = Annotated[str, Field(pattern=r"^[0-9]{8}$", strict=True)]
ComparisonMetricKey = Literal[
    "registration_status",
    "company_age",
    "registered_capital",
    "last_changed_at",
    "bizscore",
    "peer_index",
    "industry",
]
PeerComparisonScope = Literal[
    "same_industry_snapshot",
    "different_industry_snapshots",
    "unavailable",
]


class CompanyComparisonRequest(APIModel):
    tax_ids: list[TaxId] = Field(
        min_length=2,
        max_length=3,
        description="Two or three unique tax IDs in the requested display order.",
        json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def validate_unique_tax_ids(self) -> "CompanyComparisonRequest":
        if len(self.tax_ids) != len(set(self.tax_ids)):
            raise ValueError("tax_ids must not contain duplicates.")
        return self


class CompanyComparisonMetrics(APIModel):
    registration_status: str | None = None
    company_age_years: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Company age calculated at the shared Benchmark as-of date.",
    )
    registered_capital: int | None = Field(default=None, ge=0)
    last_changed_at: date | None = None
    bizscore: int | None = Field(default=None, ge=0, le=100)
    bizscore_coverage: float = Field(ge=0, le=1)
    bizscore_provisional: bool
    bizscore_status: Literal["complete", "provisional", "unavailable"]
    peer_index: float | None = Field(default=None, ge=0, le=100)
    industry_code: str | None = Field(default=None, min_length=1, max_length=10)
    missing: list[ComparisonMetricKey] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_and_missing(self) -> "CompanyComparisonMetrics":
        expected_status = (
            "unavailable"
            if self.bizscore is None
            else "provisional"
            if self.bizscore_provisional
            else "complete"
        )
        if self.bizscore_status != expected_status:
            raise ValueError("bizscore_status does not match score availability.")

        values = {
            "registration_status": self.registration_status,
            "company_age": self.company_age_years,
            "registered_capital": self.registered_capital,
            "last_changed_at": self.last_changed_at,
            "bizscore": self.bizscore,
            "peer_index": self.peer_index,
            "industry": self.industry_code,
        }
        expected_missing = {key for key, value in values.items() if value is None}
        if set(self.missing) != expected_missing:
            raise ValueError("missing must exactly identify unavailable comparison metrics.")
        if len(self.missing) != len(set(self.missing)):
            raise ValueError("missing must not contain duplicates.")
        return self


class CompanyComparisonItem(APIModel):
    input_index: int = Field(ge=0, le=2)
    company: CompanyData
    bizscore: BizScoreResult
    metrics: CompanyComparisonMetrics
    source_meta: ResponseMeta

    @model_validator(mode="after")
    def validate_nested_consistency(self) -> "CompanyComparisonItem":
        company = self.company
        bizscore = self.bizscore
        metrics = self.metrics
        expected_age = _calculate_company_age_years(
            company.established_at,
            as_of=bizscore.as_of,
        )
        expected_values = {
            "registration_status": (
                company.status.description or company.status.code
            ),
            "company_age_years": expected_age,
            "registered_capital": company.capital.registered,
            "last_changed_at": company.last_changed_at,
            "bizscore": bizscore.score,
            "bizscore_coverage": bizscore.coverage,
            "bizscore_provisional": bizscore.provisional,
            "peer_index": bizscore.peer_benchmark.peer_index,
            "industry_code": bizscore.benchmark.industry_code,
        }
        actual_values = {
            "registration_status": metrics.registration_status,
            "company_age_years": metrics.company_age_years,
            "registered_capital": metrics.registered_capital,
            "last_changed_at": metrics.last_changed_at,
            "bizscore": metrics.bizscore,
            "bizscore_coverage": metrics.bizscore_coverage,
            "bizscore_provisional": metrics.bizscore_provisional,
            "peer_index": metrics.peer_index,
            "industry_code": metrics.industry_code,
        }
        if actual_values != expected_values:
            raise ValueError("comparison metrics do not match nested source data.")

        dimensions = bizscore.dimensions
        dimension_keys = [dimension.key for dimension in dimensions]
        if dimension_keys != list(EXPECTED_BIZSCORE_DIMENSION_KEYS):
            raise ValueError("BizScore v1 dimensions are missing or out of order.")
        if any(
            dimension.score is not None
            and dimension.score > dimension.max_score
            for dimension in dimensions
        ):
            raise ValueError("a BizScore dimension exceeds its maximum score.")

        expected_missing_dimensions = [
            dimension.key
            for dimension in dimensions
            if not dimension.available or dimension.score is None
        ]
        if bizscore.missing_dimensions != expected_missing_dimensions:
            raise ValueError("BizScore missing_dimensions are inconsistent.")
        available_weight = sum(
            dimension.max_score
            for dimension in dimensions
            if dimension.available and dimension.score is not None
        )
        if abs(bizscore.coverage - available_weight / 100) > 1e-9:
            raise ValueError("BizScore coverage is inconsistent with its dimensions.")
        if (bizscore.score is None) != (bizscore.band is None):
            raise ValueError("BizScore score and band availability are inconsistent.")
        if bizscore.provisional and bizscore.score is None:
            raise ValueError("an unavailable BizScore cannot be provisional.")
        if bizscore.score is not None and bizscore.coverage < 0.8:
            raise ValueError("a numeric BizScore requires at least 80% coverage.")
        if (
            bizscore.score is not None
            and bizscore.coverage < 1
            and not bizscore.provisional
        ):
            raise ValueError("an incomplete numeric BizScore must be provisional.")
        if (
            bizscore.score is not None
            and bizscore.status_cap is not None
            and bizscore.score > bizscore.status_cap
        ):
            raise ValueError("BizScore exceeds its registration-status cap.")

        benchmark = bizscore.benchmark
        industry = bizscore.industry
        peer = bizscore.peer_benchmark
        if industry.version != benchmark.classification_version:
            raise ValueError("industry classification versions are inconsistent.")
        primary_industry_code = (
            industry.primary_group.category_code
            if industry.primary_group is not None
            else None
        )
        if industry.available != (industry.primary_group is not None):
            raise ValueError("industry availability and primary group are inconsistent.")
        if primary_industry_code != benchmark.industry_code:
            raise ValueError("industry and Benchmark codes are inconsistent.")
        if (
            peer.industry_code != benchmark.industry_code
            or peer.benchmark_version != benchmark.snapshot_version
        ):
            raise ValueError("peer and Benchmark references are inconsistent.")
        if peer.dimension != dimensions[-1]:
            raise ValueError("peer dimension is inconsistent with BizScore dimensions.")
        if peer.dimension.available != (peer.peer_index is not None):
            raise ValueError("peer availability and peer_index are inconsistent.")
        return self


class CompanyComparisonContext(APIModel):
    ordering: Literal["request_order"] = "request_order"
    tie_handling: Literal["not_ranked"] = "not_ranked"
    benchmark_catalog_version: str = Field(min_length=1, max_length=100)
    benchmark_as_of: date
    peer_comparison_scope: PeerComparisonScope
    same_primary_industry: bool


class CompanyComparisonData(APIModel):
    version: Literal["1.0"] = COMPARISON_VERSION
    items: list[CompanyComparisonItem] = Field(min_length=2, max_length=3)
    context: CompanyComparisonContext
    disclaimer: Literal[COMPANY_COMPARISON_DISCLAIMER] = (
        COMPANY_COMPARISON_DISCLAIMER
    )

    @model_validator(mode="after")
    def validate_positions(self) -> "CompanyComparisonData":
        expected = list(range(len(self.items)))
        input_indexes = [item.input_index for item in self.items]
        if input_indexes != expected:
            raise ValueError(
                "comparison items must remain in contiguous request order."
            )
        tax_ids = [item.company.tax_id for item in self.items]
        if len(tax_ids) != len(set(tax_ids)):
            raise ValueError("comparison items must contain unique companies.")
        return self


class CompanyComparisonWarning(APIModel):
    code: str = Field(min_length=1, max_length=60)
    tax_id: TaxId | None = None
    message: str = Field(min_length=1, max_length=240)


class CompanyComparisonMeta(APIModel):
    comparison_version: Literal["1.0"] = COMPARISON_VERSION
    generated_at: datetime
    requested_tax_ids: list[TaxId] = Field(min_length=2, max_length=3)
    requested_count: int = Field(ge=2, le=3)
    returned_count: int = Field(ge=2, le=3)
    has_partial_source_data: bool
    has_provisional_scores: bool
    has_unscored_companies: bool
    warnings: list[CompanyComparisonWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "CompanyComparisonMeta":
        if len(self.requested_tax_ids) != len(set(self.requested_tax_ids)):
            raise ValueError("requested_tax_ids must not contain duplicates.")
        if self.requested_count != len(self.requested_tax_ids):
            raise ValueError("requested_count does not match requested_tax_ids.")
        if self.returned_count != self.requested_count:
            raise ValueError("comparison responses must be atomic and complete.")
        if (
            self.generated_at.tzinfo is None
            or self.generated_at.utcoffset() is None
        ):
            raise ValueError("generated_at must include a UTC offset.")
        return self


class CompanyComparisonResponse(APIModel):
    data: CompanyComparisonData
    meta: CompanyComparisonMeta

    @model_validator(mode="after")
    def validate_response_consistency(self) -> "CompanyComparisonResponse":
        request_order = [item.company.tax_id for item in self.data.items]
        if request_order != self.meta.requested_tax_ids:
            raise ValueError("comparison items do not match requested_tax_ids.")
        if len(self.data.items) != self.meta.returned_count:
            raise ValueError("returned_count does not match comparison items.")
        expected_partial = any(item.source_meta.partial for item in self.data.items)
        expected_provisional = any(
            item.bizscore.provisional for item in self.data.items
        )
        expected_unscored = any(
            item.bizscore.score is None for item in self.data.items
        )
        if self.meta.has_partial_source_data != expected_partial:
            raise ValueError("has_partial_source_data is inconsistent.")
        if self.meta.has_provisional_scores != expected_provisional:
            raise ValueError("has_provisional_scores is inconsistent.")
        if self.meta.has_unscored_companies != expected_unscored:
            raise ValueError("has_unscored_companies is inconsistent.")

        industry_codes = [
            item.bizscore.benchmark.industry_code for item in self.data.items
        ]
        expected_same_industry = (
            all(code is not None for code in industry_codes)
            and len(set(industry_codes)) == 1
        )
        if self.data.context.same_primary_industry != expected_same_industry:
            raise ValueError("same_primary_industry is inconsistent.")

        peer_scopes = [
            (
                item.bizscore.peer_benchmark.dimension.available,
                item.bizscore.peer_benchmark.peer_index,
                item.bizscore.benchmark.industry_code,
                item.bizscore.benchmark.snapshot_version,
            )
            for item in self.data.items
        ]
        if not all(
            available
            and peer_index is not None
            and industry_code is not None
            and snapshot_version is not None
            for available, peer_index, industry_code, snapshot_version in peer_scopes
        ):
            expected_scope = "unavailable"
        elif len(
            {
                (industry_code, snapshot_version)
                for _, _, industry_code, snapshot_version in peer_scopes
            }
        ) == 1:
            expected_scope = "same_industry_snapshot"
        else:
            expected_scope = "different_industry_snapshots"
        if self.data.context.peer_comparison_scope != expected_scope:
            raise ValueError("peer_comparison_scope is inconsistent.")

        catalog_mapping_versions = {
            item.bizscore.benchmark.catalog_industry_mapping_version
            for item in self.data.items
        }
        classification_versions = {
            item.bizscore.benchmark.classification_version
            for item in self.data.items
        }
        if len(catalog_mapping_versions) != 1 or len(classification_versions) != 1:
            raise ValueError("comparison items use inconsistent mapping versions.")

        snapshot_provenance: dict[
            tuple[str | None, str | None],
            tuple[str | None, str | None, int],
        ] = {}
        for item in self.data.items:
            if (
                item.bizscore.benchmark.catalog_version
                != self.data.context.benchmark_catalog_version
            ):
                raise ValueError("comparison items use different Benchmark catalogs.")
            if (
                item.bizscore.as_of != self.data.context.benchmark_as_of
                or item.bizscore.benchmark.catalog_as_of
                != self.data.context.benchmark_as_of
            ):
                raise ValueError("comparison items use different BizScore dates.")
            benchmark = item.bizscore.benchmark
            snapshot_key = (benchmark.industry_code, benchmark.snapshot_version)
            provenance = (
                benchmark.snapshot_source_version,
                benchmark.snapshot_checksum_sha256,
                benchmark.sample_count,
            )
            prior = snapshot_provenance.setdefault(snapshot_key, provenance)
            if prior != provenance:
                raise ValueError("comparison items use inconsistent snapshot metadata.")

        requested_tax_ids = set(self.meta.requested_tax_ids)
        for warning in self.meta.warnings:
            if warning.tax_id is not None and warning.tax_id not in requested_tax_ids:
                raise ValueError("comparison warning references an unknown tax ID.")
            if warning.code in COMPANY_WARNING_CODES and warning.tax_id is None:
                raise ValueError("company warning must identify its tax ID.")
            if warning.code in GLOBAL_WARNING_CODES and warning.tax_id is not None:
                raise ValueError("global comparison warning cannot identify one company.")
        return self


def _calculate_company_age_years(
    established_at: date | None,
    *,
    as_of: date,
) -> float | None:
    if established_at is None or established_at > as_of:
        return None
    return round((as_of - established_at).days / 365.2425, 1)
