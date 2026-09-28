from fastapi import Request

from app.services.benchmark_catalog import BenchmarkCatalogService
from app.services.gcis import GCISClient
from app.services.gcis_response_cache import GCISResponseCache
from app.services.llm_provider import CompanyAnalysisProvider
from app.services.company_comparison_analysis import (
    CompanyComparisonAnalysisExecution,
)
from app.services.llm_cache import AsyncLRUTTLCache


def get_gcis_client(request: Request) -> GCISClient:
    """Return the application-scoped GCIS client created at startup."""

    client = getattr(request.app.state, "gcis_client", None)
    if not isinstance(client, GCISClient):
        raise RuntimeError("GCIS client is not available before application startup.")
    return client


def get_gcis_response_cache(request: Request) -> GCISResponseCache:
    """Return the bounded application-scoped successful GCIS response cache."""

    cache = getattr(request.app.state, "gcis_response_cache", None)
    if not isinstance(cache, GCISResponseCache):
        raise RuntimeError("GCIS response cache is unavailable before startup.")
    return cache


def get_benchmark_catalog_service(request: Request) -> BenchmarkCatalogService:
    """Return the configured production Benchmark catalog service."""

    service = getattr(request.app.state, "benchmark_catalog_service", None)
    if isinstance(service, BenchmarkCatalogService):
        return service

    settings = request.app.state.settings
    service = BenchmarkCatalogService(
        settings.benchmark_catalog_path,
        settings.benchmark_database_path,
        expected_catalog_version=settings.benchmark_catalog_version,
    )
    request.app.state.benchmark_catalog_service = service
    return service


def get_company_analysis_provider(request: Request) -> CompanyAnalysisProvider:
    """Return the application-scoped provider selected at startup."""

    provider = getattr(request.app.state, "company_analysis_provider", None)
    if provider is None:
        raise RuntimeError(
            "Company-analysis provider is not available before application startup."
        )
    return provider


def get_company_comparison_analysis_cache(
    request: Request,
) -> AsyncLRUTTLCache[CompanyComparisonAnalysisExecution]:
    """Return the process-local, application-scoped AI comparison cache."""

    cache = getattr(request.app.state, "company_comparison_analysis_cache", None)
    if not isinstance(cache, AsyncLRUTTLCache):
        raise RuntimeError(
            "Company-comparison analysis cache is unavailable before startup."
        )
    return cache
