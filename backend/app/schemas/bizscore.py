from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import APIModel
from app.schemas.company import CompanyData, ResponseMeta


class BizScoreDimension(APIModel):
    key: str
    label: str
    score: int | None = Field(default=None, ge=0)
    max_score: int = Field(gt=0)
    available: bool
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RegistrationStatusScore(APIModel):
    dimension: BizScoreDimension
    total_score_eligible: bool
    status_cap: int | None = Field(default=None, ge=0, le=100)
    outcome: Literal[
        "scored",
        "review_required",
        "non_current",
        "insufficient_data",
    ]


class IndustryGroup(APIModel):
    category_code: str
    category_name: str
    source_sequence: str
    source_business_item_code: str
    source_business_item_name: str | None = None
    benchmark_eligible: bool = True


class IndustryClassification(APIModel):
    version: str
    available: bool
    primary_group: IndustryGroup | None = None
    groups: list[IndustryGroup] = Field(default_factory=list)
    excluded_business_item_codes: list[str] = Field(default_factory=list)
    unmapped_business_item_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PeerBenchmarkSample(APIModel):
    company_age_years: Decimal | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    registered_capital: int | None = Field(default=None, ge=0)


class PeerBenchmarkRankCounts(APIModel):
    sample_size: int = Field(ge=0)
    age_lower_count: int = Field(ge=0)
    age_equal_count: int = Field(ge=0)
    capital_lower_count: int = Field(ge=0)
    capital_equal_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "PeerBenchmarkRankCounts":
        if self.age_lower_count + self.age_equal_count > self.sample_size:
            raise ValueError("Age rank counts cannot exceed sample_size.")
        if self.capital_lower_count + self.capital_equal_count > self.sample_size:
            raise ValueError("Capital rank counts cannot exceed sample_size.")
        return self


class PeerBenchmarkMedians(APIModel):
    sample_size: int = Field(ge=0)
    company_age_median: Decimal | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    registered_capital_median: Decimal | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )


class PeerRelativePositionScore(APIModel):
    dimension: BizScoreDimension
    industry_code: str | None = None
    benchmark_version: str | None = None
    sample_size: int = Field(ge=0)
    excluded_sample_count: int = Field(default=0, ge=0)
    minimum_sample_size: int = Field(default=30, ge=1)
    age_percentile: float | None = Field(default=None, ge=0, le=100)
    capital_percentile: float | None = Field(default=None, ge=0, le=100)
    peer_index: float | None = Field(default=None, ge=0, le=100)
    company_age_median: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    registered_capital_median: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )


class BenchmarkSnapshotMetadata(APIModel):
    version: str
    as_of: date
    created_at: datetime
    source: str
    source_version: str
    industry_mapping_version: str
    input_record_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    excluded_record_count: int = Field(ge=0)
    exclusion_counts: dict[str, int] = Field(default_factory=dict)
    group_sample_sizes: dict[str, int] = Field(default_factory=dict)
    checksum_sha256: str


class BenchmarkCatalogEntry(APIModel):
    snapshot_version: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkCatalog(APIModel):
    catalog_version: str = Field(min_length=1)
    as_of: date
    industry_mapping_version: str = Field(min_length=1)
    membership_sample_count: int = Field(ge=0)
    membership_count_note: str = Field(min_length=1)
    sqlite_integrity_check: Literal["ok"]
    categories: dict[str, BenchmarkCatalogEntry]

    @model_validator(mode="after")
    def validate_categories(self) -> "BenchmarkCatalog":
        expected_categories = set("ABCDEFGHIJ")
        actual_categories = set(self.categories)
        if actual_categories != expected_categories:
            missing = sorted(expected_categories - actual_categories)
            unexpected = sorted(actual_categories - expected_categories)
            raise ValueError(
                "Catalog categories must be exactly A-J; "
                f"missing={missing}, unexpected={unexpected}."
            )
        total = sum(entry.sample_count for entry in self.categories.values())
        if total != self.membership_sample_count:
            raise ValueError(
                "membership_sample_count does not equal the A-J category total."
            )
        snapshot_versions = [
            entry.snapshot_version for entry in self.categories.values()
        ]
        if len(set(snapshot_versions)) != len(snapshot_versions):
            raise ValueError(
                "Each A-J category must select a distinct snapshot_version."
            )
        return self


class BenchmarkReference(APIModel):
    catalog_version: str
    catalog_as_of: date
    catalog_industry_mapping_version: str
    classification_version: str
    industry_code: str | None = None
    snapshot_version: str | None = None
    snapshot_source_version: str | None = None
    snapshot_checksum_sha256: str | None = None
    sample_count: int = Field(default=0, ge=0)


BizScoreBand = Literal[
    "公開資料呈現較穩健",
    "公開資料呈現一般",
    "建議進一步查核",
    "需優先查核",
]


class BizScoreResult(APIModel):
    version: Literal["1.0"] = "1.0"
    score: int | None = Field(default=None, ge=0, le=100)
    band: BizScoreBand | None = None
    coverage: float = Field(ge=0, le=1)
    provisional: bool
    as_of: date
    status_cap: int | None = Field(default=None, ge=0, le=100)
    dimensions: list[BizScoreDimension]
    missing_dimensions: list[str] = Field(default_factory=list)
    industry: IndustryClassification
    benchmark: BenchmarkReference
    peer_benchmark: PeerRelativePositionScore
    warnings: list[str] = Field(default_factory=list)


class CompanyBizScoreData(APIModel):
    company: CompanyData
    bizscore: BizScoreResult


class CompanyBizScoreResponse(APIModel):
    data: CompanyBizScoreData
    meta: ResponseMeta
