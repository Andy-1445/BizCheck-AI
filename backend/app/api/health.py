from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_benchmark_catalog_service, get_gcis_client
from app.schemas.health import (
    HealthResponse,
    ReadinessCheck,
    ReadinessChecks,
    ReadinessResponse,
)
from app.services.benchmark_catalog import (
    BenchmarkCatalogInvalidError,
    BenchmarkCatalogService,
    BenchmarkCatalogUnavailableError,
)
from app.services.gcis import (
    GCISError,
    GCISErrorCategory,
    GCISClient,
)
from app.services.llm_validation_diagnostics import (
    get_latest_llm_validation_diagnostic,
)


router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
async def liveness(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="bizcheck-api",
        version=request.app.state.settings.app_version,
    )


@router.get("/diagnostics/llm", include_in_schema=False)
async def latest_llm_diagnostic(request: Request) -> dict[str, object]:
    """Expose sanitized validation state only to a local non-production client."""

    client_host = request.client.host if request.client is not None else ""
    if (
        request.app.state.settings.environment == "production"
        or client_host not in {"127.0.0.1", "::1", "testclient"}
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    snapshot = get_latest_llm_validation_diagnostic()
    if snapshot is None:
        return {"available": False, "sequence": 0, "diagnostic": None}
    sequence, diagnostic = snapshot
    return {
        "available": True,
        "sequence": sequence,
        "diagnostic": asdict(diagnostic),
    }


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": "One or more required dependencies are unavailable.",
        }
    },
)
async def readiness(
    request: Request,
    gcis_client: Annotated[GCISClient, Depends(get_gcis_client)],
    benchmark_service: Annotated[
        BenchmarkCatalogService,
        Depends(get_benchmark_catalog_service),
    ],
) -> ReadinessResponse | JSONResponse:
    gcis_check, benchmark_check = await asyncio.gather(
        _check_gcis(gcis_client),
        _check_benchmark(benchmark_service),
    )
    checks = ReadinessChecks(
        backend=ReadinessCheck(
            status="ok",
            code="BACKEND_OK",
            message="BizCheck API process is accepting requests.",
            latency_ms=0,
            retryable=False,
        ),
        gcis=gcis_check,
        benchmark=benchmark_check,
    )
    is_ready = all(
        check.status == "ok"
        for check in (checks.backend, checks.gcis, checks.benchmark)
    )
    response = ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        service="bizcheck-api",
        version=request.app.state.settings.app_version,
        checked_at=datetime.now(timezone.utc),
        checks=checks,
    )
    if is_ready:
        return response
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response.model_dump(mode="json"),
    )


async def _check_gcis(gcis_client: GCISClient) -> ReadinessCheck:
    started_at = perf_counter()
    try:
        await gcis_client.check_availability()
    except GCISError as error:
        is_invalid = error.category in {
            GCISErrorCategory.INVALID_RESPONSE,
            GCISErrorCategory.REQUEST_REJECTED,
            GCISErrorCategory.VALIDATION,
        }
        return ReadinessCheck(
            status="invalid" if is_invalid else "unavailable",
            code=f"GCIS_{error.category.value.upper()}",
            message=_gcis_readiness_message(error.category),
            latency_ms=_elapsed_ms(started_at),
            retryable=error.retryable,
        )
    except Exception:
        return ReadinessCheck(
            status="unavailable",
            code="GCIS_CHECK_FAILED",
            message="GCIS availability check failed unexpectedly.",
            latency_ms=_elapsed_ms(started_at),
            retryable=False,
        )
    return ReadinessCheck(
        status="ok",
        code="GCIS_OK",
        message="GCIS responded with a valid JSON payload.",
        latency_ms=_elapsed_ms(started_at),
        retryable=False,
    )


async def _check_benchmark(
    benchmark_service: BenchmarkCatalogService,
) -> ReadinessCheck:
    started_at = perf_counter()
    try:
        catalog = await asyncio.to_thread(benchmark_service.validate_all_snapshots)
    except BenchmarkCatalogUnavailableError:
        return ReadinessCheck(
            status="unavailable",
            code="BENCHMARK_UNAVAILABLE",
            message="Benchmark catalog or snapshot storage is unavailable.",
            latency_ms=_elapsed_ms(started_at),
            retryable=True,
        )
    except BenchmarkCatalogInvalidError:
        return ReadinessCheck(
            status="invalid",
            code="BENCHMARK_INVALID",
            message="Benchmark catalog and snapshot metadata are inconsistent.",
            latency_ms=_elapsed_ms(started_at),
            retryable=False,
        )
    except Exception:
        return ReadinessCheck(
            status="unavailable",
            code="BENCHMARK_CHECK_FAILED",
            message="Benchmark availability check failed unexpectedly.",
            latency_ms=_elapsed_ms(started_at),
            retryable=False,
        )
    return ReadinessCheck(
        status="ok",
        code="BENCHMARK_OK",
        message=(
            f"Benchmark catalog {catalog.catalog_version} and all A-J snapshots "
            "are valid."
        ),
        latency_ms=_elapsed_ms(started_at),
        retryable=False,
    )


def _gcis_readiness_message(category: GCISErrorCategory) -> str:
    messages = {
        GCISErrorCategory.TIMEOUT: "GCIS availability check timed out.",
        GCISErrorCategory.CONNECTION: "GCIS could not be reached.",
        GCISErrorCategory.RATE_LIMITED: "GCIS rate-limited the availability check.",
        GCISErrorCategory.SERVICE_UNAVAILABLE: "GCIS is temporarily unavailable.",
        GCISErrorCategory.REQUEST_REJECTED: "GCIS rejected the probe request.",
        GCISErrorCategory.INVALID_RESPONSE: "GCIS returned an invalid probe response.",
        GCISErrorCategory.VALIDATION: "The GCIS probe configuration is invalid.",
    }
    return messages.get(category, "GCIS availability check failed.")


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))
