from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.schemas.company import (
    CompanyCapital,
    CompanyData,
    CompanyResponse,
    CompanySearchResponse,
    CompanyStatus,
    ResponseMeta,
)
from app.services.gcis import (
    GCISConnectionError,
    GCISNotFoundError,
    GCISTimeoutError,
)
from app.services.gcis_response_cache import GCISResponseCache


pytestmark = pytest.mark.anyio


class MutableClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.epoch = datetime(2026, 8, 26, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self.value

    def utcnow(self) -> datetime:
        return self.epoch + timedelta(seconds=self.value)

    def advance(self, seconds: float) -> None:
        self.value += seconds


def build_cache(clock: MutableClock) -> GCISResponseCache:
    return GCISResponseCache(
        max_entries=4,
        ttl_seconds=10,
        stale_if_error_seconds=20,
        monotonic=clock.monotonic,
        utcnow=clock.utcnow,
    )


def company_response(snapshot_at: datetime) -> CompanyResponse:
    return CompanyResponse(
        data=CompanyData(
            tax_id="20828393",
            name="宏碁股份有限公司",
            status=CompanyStatus(code="01", description="核准設立"),
            capital=CompanyCapital(registered=40_000_000_000),
        ),
        meta=ResponseMeta(fetched_at=snapshot_at),
    )


async def test_fresh_hit_avoids_live_factory_and_preserves_snapshot_time() -> None:
    clock = MutableClock()
    cache = build_cache(clock)
    calls = 0
    snapshot = company_response(clock.utcnow())

    async def factory() -> CompanyResponse:
        nonlocal calls
        calls += 1
        return snapshot

    live = await cache.resolve_company("20828393", factory)
    clock.advance(5)
    cached = await cache.resolve_company("20828393", factory)

    assert calls == 1
    assert live.meta.data_freshness == "live"
    assert cached.meta.data_freshness == "fresh_cache"
    assert cached.meta.fallback_reason is None
    assert cached.meta.fetched_at == live.meta.fetched_at


async def test_transient_refresh_failure_serves_recent_successful_snapshot() -> None:
    clock = MutableClock()
    cache = build_cache(clock)
    snapshot = company_response(clock.utcnow())

    async def initial_factory() -> CompanyResponse:
        return snapshot

    await cache.resolve_company("20828393", initial_factory)
    clock.advance(11)

    async def failing_factory() -> CompanyResponse:
        raise GCISConnectionError("connection refused", attempts=2)

    fallback = await cache.resolve_company("20828393", failing_factory)

    assert fallback.data == snapshot.data
    assert fallback.meta.fetched_at == snapshot.meta.fetched_at
    assert fallback.meta.data_freshness == "stale_cache"
    assert fallback.meta.fallback_reason == "GCIS_CONNECTION"


async def test_not_found_does_not_resurrect_a_cached_company() -> None:
    clock = MutableClock()
    cache = build_cache(clock)

    async def initial_factory() -> CompanyResponse:
        return company_response(clock.utcnow())

    await cache.resolve_company("20828393", initial_factory)
    clock.advance(11)

    async def not_found_factory() -> CompanyResponse:
        raise GCISNotFoundError("company no longer exists")

    with pytest.raises(GCISNotFoundError):
        await cache.resolve_company("20828393", not_found_factory)


async def test_expired_stale_window_does_not_serve_old_snapshot() -> None:
    clock = MutableClock()
    cache = build_cache(clock)

    async def initial_factory() -> CompanyResponse:
        return company_response(clock.utcnow())

    await cache.resolve_company("20828393", initial_factory)
    clock.advance(31)

    async def failing_factory() -> CompanyResponse:
        raise GCISConnectionError("connection refused")

    with pytest.raises(GCISConnectionError):
        await cache.resolve_company("20828393", failing_factory)


async def test_successful_refresh_resets_ttl_for_same_object_instance() -> None:
    clock = MutableClock()
    cache = build_cache(clock)
    calls = 0
    snapshot = company_response(clock.utcnow())

    async def factory() -> CompanyResponse:
        nonlocal calls
        calls += 1
        return snapshot

    await cache.resolve_company("20828393", factory)
    clock.advance(11)
    refreshed = await cache.resolve_company("20828393", factory)
    clock.advance(1)
    cached = await cache.resolve_company("20828393", factory)

    assert calls == 2
    assert refreshed.meta.data_freshness == "live"
    assert cached.meta.data_freshness == "fresh_cache"


async def test_search_cache_normalizes_equivalent_keyword_keys() -> None:
    clock = MutableClock()
    cache = build_cache(clock)
    calls = 0
    response = CompanySearchResponse(
        data=[],
        meta=ResponseMeta(fetched_at=clock.utcnow()),
    )

    async def factory() -> CompanySearchResponse:
        nonlocal calls
        calls += 1
        return response

    live = await cache.resolve_search("  宏碁  電腦 ", "01", 10, factory)
    cached = await cache.resolve_search("宏碁 電腦", "01", 10, factory)

    assert calls == 1
    assert live.meta.data_freshness == "live"
    assert cached.meta.data_freshness == "fresh_cache"


async def test_search_uses_snapshot_only_for_transient_refresh_failure() -> None:
    clock = MutableClock()
    cache = build_cache(clock)
    response = CompanySearchResponse(
        data=[],
        meta=ResponseMeta(fetched_at=clock.utcnow()),
    )

    async def initial_factory() -> CompanySearchResponse:
        return response

    await cache.resolve_search("宏碁", "01", 10, initial_factory)
    clock.advance(11)

    async def failing_factory() -> CompanySearchResponse:
        raise GCISTimeoutError("search timed out", attempts=2)

    fallback = await cache.resolve_search("宏碁", "01", 10, failing_factory)

    assert fallback.meta.data_freshness == "stale_cache"
    assert fallback.meta.fallback_reason == "GCIS_TIMEOUT"
    assert fallback.meta.fetched_at == response.meta.fetched_at
