from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel


class HealthResponse(APIModel):
    status: Literal["ok"]
    service: str
    version: str


class ReadinessCheck(APIModel):
    status: Literal["ok", "unavailable", "invalid"]
    code: str
    message: str
    latency_ms: int = Field(ge=0)
    retryable: bool


class ReadinessChecks(APIModel):
    backend: ReadinessCheck
    gcis: ReadinessCheck
    benchmark: ReadinessCheck


class ReadinessResponse(APIModel):
    status: Literal["ready", "not_ready"]
    service: str
    version: str
    checked_at: datetime
    checks: ReadinessChecks
