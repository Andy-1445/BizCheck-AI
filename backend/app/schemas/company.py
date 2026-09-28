from datetime import date, datetime
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import APIModel


class CompanyStatus(APIModel):
    code: str | None = None
    description: str | None = None


class CompanyCapital(APIModel):
    registered: int | None = None
    paid_in: int | None = None
    currency: str = "TWD"


class BusinessItem(APIModel):
    sequence: str
    code: str
    name: str | None = None


class CompanyData(APIModel):
    tax_id: str
    name: str
    status: CompanyStatus
    capital: CompanyCapital
    established_at: date | None = None
    company_age_years: float | None = None
    last_changed_at: date | None = None
    responsible_name: str | None = None
    address: str | None = None
    registration_authority: str | None = None
    business_items: list[BusinessItem] = Field(default_factory=list)


class ResponseMeta(APIModel):
    source: str = "GCIS"
    provider: str = "經濟部商業發展署"
    fetched_at: datetime
    partial: bool = False
    warnings: list[str] = Field(default_factory=list)
    data_freshness: Literal["live", "fresh_cache", "stale_cache"] = "live"
    fallback_reason: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def validate_freshness(self) -> "ResponseMeta":
        if self.fetched_at.tzinfo is None or self.fetched_at.utcoffset() is None:
            raise ValueError("fetched_at must include a UTC offset.")
        if self.data_freshness == "stale_cache" and self.fallback_reason is None:
            raise ValueError("stale_cache metadata requires fallback_reason.")
        if self.data_freshness != "stale_cache" and self.fallback_reason is not None:
            raise ValueError("fallback_reason is only valid for stale_cache data.")
        return self


class CompanyResponse(APIModel):
    data: CompanyData
    meta: ResponseMeta


class CompanySearchItem(APIModel):
    tax_id: str
    name: str
    status: CompanyStatus
    registered_capital: int | None = None
    established_at: date | None = None
    last_changed_at: date | None = None


class CompanySearchResponse(APIModel):
    data: list[CompanySearchItem]
    meta: ResponseMeta
