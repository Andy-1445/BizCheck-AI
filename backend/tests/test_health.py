from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_benchmark_catalog_service, get_gcis_client
from app.main import create_app
from app.services.benchmark_catalog import (
    BenchmarkCatalogInvalidError,
    BenchmarkCatalogUnavailableError,
)
from app.services.gcis import GCISTimeoutError
from app.services.llm_validation_diagnostics import (
    clear_latest_llm_validation_diagnostic,
)
from app.settings import Settings


pytestmark = pytest.mark.anyio


async def test_root_returns_api_metadata(client: AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "bizcheck-api",
        "version": "0.1.0",
        "environment": "test",
        "docs": "/docs",
    }


async def test_liveness(client: AsyncClient) -> None:
    response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "bizcheck-api",
        "version": "0.1.0",
    }


async def test_local_development_llm_diagnostic_starts_empty(
    client: AsyncClient,
) -> None:
    clear_latest_llm_validation_diagnostic()

    response = await client.get("/health/diagnostics/llm")

    assert response.status_code == 200
    assert response.json() == {
        "available": False,
        "sequence": 0,
        "diagnostic": None,
    }


async def test_production_hides_llm_diagnostic_endpoint() -> None:
    production_app = create_app(
        Settings(
            environment="production",
            cors_origins=("https://bizcheck.example",),
        )
    )
    async with production_app.router.lifespan_context(production_app):
        transport = ASGITransport(app=production_app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as production_client:
            response = await production_client.get("/health/diagnostics/llm")

    assert response.status_code == 404


class FakeGCISReadinessClient:
    def __init__(self, outcome: Exception | None = None) -> None:
        self.outcome = outcome
        self.calls = 0

    async def check_availability(self) -> None:
        self.calls += 1
        if self.outcome is not None:
            raise self.outcome


class FakeBenchmarkReadinessService:
    def __init__(self, outcome: Exception | None = None) -> None:
        self.outcome = outcome
        self.calls = 0

    def validate_all_snapshots(self):
        self.calls += 1
        if self.outcome is not None:
            raise self.outcome
        return SimpleNamespace(
            catalog_version="benchmark-catalog-2026-08-01-v1"
        )


def _override_readiness_dependencies(
    app: FastAPI,
    gcis_client: FakeGCISReadinessClient,
    benchmark_service: FakeBenchmarkReadinessService,
) -> None:
    app.dependency_overrides[get_gcis_client] = lambda: gcis_client
    app.dependency_overrides[get_benchmark_catalog_service] = (
        lambda: benchmark_service
    )


async def test_readiness_reports_backend_gcis_and_benchmark(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    gcis_client = FakeGCISReadinessClient()
    benchmark_service = FakeBenchmarkReadinessService()
    _override_readiness_dependencies(app, gcis_client, benchmark_service)

    response = await client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["service"] == "bizcheck-api"
    assert payload["version"] == "0.1.0"
    datetime.fromisoformat(payload["checked_at"].replace("Z", "+00:00"))
    assert payload["checks"] == {
        "backend": {
            "status": "ok",
            "code": "BACKEND_OK",
            "message": "BizCheck API process is accepting requests.",
            "latency_ms": 0,
            "retryable": False,
        },
        "gcis": {
            "status": "ok",
            "code": "GCIS_OK",
            "message": "GCIS responded with a valid JSON payload.",
            "latency_ms": payload["checks"]["gcis"]["latency_ms"],
            "retryable": False,
        },
        "benchmark": {
            "status": "ok",
            "code": "BENCHMARK_OK",
            "message": (
                "Benchmark catalog benchmark-catalog-2026-08-01-v1 "
                "and all A-J snapshots are valid."
            ),
            "latency_ms": payload["checks"]["benchmark"]["latency_ms"],
            "retryable": False,
        },
    }
    assert payload["checks"]["gcis"]["latency_ms"] >= 0
    assert payload["checks"]["benchmark"]["latency_ms"] >= 0
    assert gcis_client.calls == 1
    assert benchmark_service.calls == 1


async def test_readiness_returns_503_and_classifies_gcis_timeout(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    gcis_client = FakeGCISReadinessClient(
        GCISTimeoutError("probe timed out", attempts=1)
    )
    benchmark_service = FakeBenchmarkReadinessService()
    _override_readiness_dependencies(app, gcis_client, benchmark_service)

    response = await client.get("/health/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["backend"]["status"] == "ok"
    assert payload["checks"]["gcis"] == {
        "status": "unavailable",
        "code": "GCIS_TIMEOUT",
        "message": "GCIS availability check timed out.",
        "latency_ms": payload["checks"]["gcis"]["latency_ms"],
        "retryable": True,
    }
    assert payload["checks"]["benchmark"]["status"] == "ok"


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code", "retryable"),
    (
        (
            BenchmarkCatalogUnavailableError("missing"),
            "unavailable",
            "BENCHMARK_UNAVAILABLE",
            True,
        ),
        (
            BenchmarkCatalogInvalidError("mismatch"),
            "invalid",
            "BENCHMARK_INVALID",
            False,
        ),
    ),
)
async def test_readiness_classifies_benchmark_failures(
    app: FastAPI,
    client: AsyncClient,
    error: Exception,
    expected_status: str,
    expected_code: str,
    retryable: bool,
) -> None:
    gcis_client = FakeGCISReadinessClient()
    benchmark_service = FakeBenchmarkReadinessService(error)
    _override_readiness_dependencies(app, gcis_client, benchmark_service)

    response = await client.get("/health/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["gcis"]["status"] == "ok"
    assert payload["checks"]["benchmark"]["status"] == expected_status
    assert payload["checks"]["benchmark"]["code"] == expected_code
    assert payload["checks"]["benchmark"]["retryable"] is retryable


async def test_readiness_openapi_documents_structured_503(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")

    readiness = response.json()["paths"]["/health/ready"]["get"]
    assert readiness["responses"]["200"]["content"]["application/json"]
    assert readiness["responses"]["503"]["content"]["application/json"]
