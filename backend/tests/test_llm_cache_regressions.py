from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.services.llm_cache import AsyncLRUTTLCache


class _MutableClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.epoch = datetime(2026, 8, 24, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self.value

    def utcnow(self) -> datetime:
        return self.epoch + timedelta(seconds=self.value)

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _cache(clock: _MutableClock) -> AsyncLRUTTLCache[str]:
    return AsyncLRUTTLCache(
        max_entries=4,
        ttl_seconds=10,
        stale_if_error_seconds=20,
        monotonic=clock.monotonic,
        utcnow=clock.utcnow,
    )


async def _wait_for_inflight_cleanup(
    cache: AsyncLRUTTLCache[str],
) -> None:
    for _ in range(20):
        if not cache._inflight:
            return
        await asyncio.sleep(0)
    pytest.fail("completed cache task remained in _inflight")


@pytest.mark.anyio
async def test_cancelled_leader_does_not_strand_completed_refresh_task() -> None:
    clock = _MutableClock()
    cache = _cache(clock)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def factory() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
        return f"value-{calls}"

    leader = asyncio.create_task(cache.resolve("company", factory))
    await started.wait()
    leader.cancel()

    with pytest.raises(asyncio.CancelledError):
        await leader

    release.set()
    await _wait_for_inflight_cleanup(cache)
    assert calls == 1

    clock.advance(31)
    refreshed = await cache.resolve(
        "company",
        factory,
        force_refresh=True,
    )

    assert refreshed.status == "bypass"
    assert refreshed.value == "value-2"
    assert calls == 2
    assert not cache._inflight


@pytest.mark.anyio
async def test_cancelled_follower_does_not_cancel_shared_factory() -> None:
    clock = _MutableClock()
    cache = _cache(clock)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    factory_cancelled = False

    async def factory() -> str:
        nonlocal calls, factory_cancelled
        calls += 1
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            factory_cancelled = True
            raise
        return "shared-value"

    leader = asyncio.create_task(cache.resolve("company", factory))
    await started.wait()
    follower = asyncio.create_task(cache.resolve("company", factory))
    await asyncio.sleep(0)
    follower.cancel()

    with pytest.raises(asyncio.CancelledError):
        await follower

    release.set()
    leader_result = await leader
    hit_result = await cache.resolve("company", factory)

    assert leader_result.status == "miss"
    assert hit_result.status == "hit"
    assert hit_result.value == "shared-value"
    assert calls == 1
    assert factory_cancelled is False
    assert not cache._inflight


@pytest.mark.anyio
async def test_refresh_failure_keeps_valid_stale_entry_and_clears_inflight() -> None:
    clock = _MutableClock()
    cache = _cache(clock)

    async def initial_factory() -> str:
        return "initial"

    await cache.resolve("company", initial_factory)
    clock.advance(11)

    async def failing_factory() -> str:
        raise RuntimeError("provider unavailable")

    stale = await cache.resolve("company", failing_factory)

    assert stale.status == "stale"
    assert stale.value == "initial"
    assert isinstance(stale.refresh_error, RuntimeError)
    assert not cache._inflight

    async def replacement_factory() -> str:
        return "replacement"

    replacement = await cache.resolve("company", replacement_factory)
    assert replacement.status == "miss"
    assert replacement.value == "replacement"
    assert not cache._inflight


@pytest.mark.anyio
async def test_refresh_failure_after_stale_window_is_not_served_as_stale() -> None:
    clock = _MutableClock()
    cache = _cache(clock)

    async def initial_factory() -> str:
        return "initial"

    await cache.resolve("company", initial_factory)
    clock.advance(31)

    async def failing_factory() -> str:
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await cache.resolve("company", failing_factory)

    assert not cache._inflight


@pytest.mark.anyio
async def test_stale_predicate_can_reject_non_fallbackable_error() -> None:
    clock = _MutableClock()
    cache = _cache(clock)

    async def initial_factory() -> str:
        return "initial"

    await cache.resolve("company", initial_factory)
    clock.advance(11)

    async def failing_factory() -> str:
        raise ValueError("request is invalid")

    with pytest.raises(ValueError, match="request is invalid"):
        await cache.resolve(
            "company",
            failing_factory,
            should_use_stale=lambda error: isinstance(error, RuntimeError),
        )

    assert not cache._inflight
