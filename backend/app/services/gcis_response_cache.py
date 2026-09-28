from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from datetime import datetime

from app.schemas.company import (
    CompanyResponse,
    CompanySearchResponse,
    ResponseMeta,
)
from app.services.company_normalizer import CompanyNormalizationError
from app.services.gcis import GCISError, GCISErrorCategory
from app.services.llm_cache import AsyncLRUTTLCache, CacheResolution


CachedGCISResponse = CompanyResponse | CompanySearchResponse
FALLBACK_ERROR_CATEGORIES = frozenset(
    {
        GCISErrorCategory.TIMEOUT,
        GCISErrorCategory.CONNECTION,
        GCISErrorCategory.RATE_LIMITED,
        GCISErrorCategory.SERVICE_UNAVAILABLE,
        GCISErrorCategory.INVALID_RESPONSE,
    }
)


class GCISResponseCache:
    """Bounded process-local cache for normalized successful GCIS responses.

    Fresh entries reduce repeated calls. Once the fresh TTL expires, the cache
    attempts a live refresh and only serves the previous snapshot when the
    refresh failed for an explicitly transient upstream reason.
    """

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: int,
        stale_if_error_seconds: int,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] | None = None,
    ) -> None:
        self._cache = AsyncLRUTTLCache[CachedGCISResponse](
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            stale_if_error_seconds=stale_if_error_seconds,
            monotonic=monotonic,
            utcnow=utcnow,
        )

    @property
    def max_entries(self) -> int:
        return self._cache.max_entries

    @property
    def ttl_seconds(self) -> int:
        return self._cache.ttl_seconds

    @property
    def stale_if_error_seconds(self) -> int:
        return self._cache.stale_if_error_seconds

    async def resolve_company(
        self,
        tax_id: str,
        factory: Callable[[], Awaitable[CompanyResponse]],
    ) -> CompanyResponse:
        resolution = await self._cache.resolve(
            f"company:{tax_id}",
            factory,
            should_use_stale=_is_fallbackable_error,
        )
        response = _annotate_resolution(resolution)
        if not isinstance(response, CompanyResponse):
            raise RuntimeError("GCIS company cache returned an unexpected value type.")
        return response

    async def resolve_search(
        self,
        keyword: str,
        company_status: str,
        limit: int,
        factory: Callable[[], Awaitable[CompanySearchResponse]],
    ) -> CompanySearchResponse:
        normalized_keyword = " ".join(keyword.strip().split())
        key = "search:" + json.dumps(
            [normalized_keyword, company_status.strip(), limit],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        resolution = await self._cache.resolve(
            key,
            factory,
            should_use_stale=_is_fallbackable_error,
        )
        response = _annotate_resolution(resolution)
        if not isinstance(response, CompanySearchResponse):
            raise RuntimeError("GCIS search cache returned an unexpected value type.")
        return response


def _is_fallbackable_error(error: Exception) -> bool:
    if isinstance(error, CompanyNormalizationError):
        return True
    return (
        isinstance(error, GCISError)
        and error.category in FALLBACK_ERROR_CATEGORIES
    )


def _annotate_resolution(
    resolution: CacheResolution[CachedGCISResponse],
) -> CachedGCISResponse:
    if resolution.status == "stale":
        data_freshness = "stale_cache"
        fallback_reason = _fallback_reason(resolution.refresh_error)
    elif resolution.status == "hit":
        data_freshness = "fresh_cache"
        fallback_reason = None
    else:
        data_freshness = "live"
        fallback_reason = None

    meta = ResponseMeta.model_validate(
        {
            **resolution.value.meta.model_dump(),
            "data_freshness": data_freshness,
            "fallback_reason": fallback_reason,
        }
    )
    return resolution.value.model_copy(update={"meta": meta})


def _fallback_reason(error: Exception | None) -> str:
    if isinstance(error, CompanyNormalizationError):
        return "GCIS_INVALID_RESPONSE"
    if isinstance(error, GCISError):
        return f"GCIS_{error.category.value.upper()}"
    return "GCIS_REFRESH_FAILED"
