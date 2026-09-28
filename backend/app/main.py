from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import validation_exception_handler
from app.api.health import router as health_router
from app.api.router import api_router
from app.services.gcis import GCISClient, GCISClientConfig
from app.services.gcis_response_cache import GCISResponseCache
from app.services.llm_cache import AsyncLRUTTLCache
from app.services.llm_provider import create_company_analysis_provider
from app.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = GCISClientConfig(
            base_url=app_settings.gcis_base_url,
            timeout_seconds=app_settings.gcis_timeout_seconds,
            max_retries=app_settings.gcis_max_retries,
            retry_backoff_seconds=app_settings.gcis_retry_backoff_seconds,
            probe_timeout_seconds=app_settings.gcis_probe_timeout_seconds,
        )
        async with GCISClient(config) as gcis_client:
            analysis_provider = create_company_analysis_provider(app_settings)
            app.state.gcis_client = gcis_client
            app.state.gcis_response_cache = GCISResponseCache(
                max_entries=app_settings.gcis_cache_max_entries,
                ttl_seconds=app_settings.gcis_cache_ttl_seconds,
                stale_if_error_seconds=(
                    app_settings.gcis_cache_stale_if_error_seconds
                ),
            )
            app.state.company_analysis_provider = analysis_provider
            app.state.company_comparison_analysis_cache = AsyncLRUTTLCache(
                max_entries=app_settings.llm_comparison_cache_max_entries,
                ttl_seconds=app_settings.llm_comparison_cache_ttl_seconds,
                stale_if_error_seconds=(
                    app_settings.llm_comparison_cache_stale_if_error_seconds
                ),
            )
            try:
                yield
            finally:
                await analysis_provider.aclose()

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=(
            "Transforms GCIS company registration data into a stable API "
            "and produces schema-validated BizCheck AI company analysis."
        ),
        lifespan=lifespan,
    )

    app.state.settings = app_settings
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        max_age=600,
    )

    app.include_router(health_router)
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def api_metadata() -> dict[str, str]:
        return {
            "service": "bizcheck-api",
            "version": app_settings.app_version,
            "environment": app_settings.environment,
            "docs": "/docs",
        }

    return app


app = create_app()
