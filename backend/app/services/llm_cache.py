from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Generic, Literal, TypeVar


T = TypeVar("T")
CacheResolutionStatus = Literal["hit", "miss", "bypass", "stale"]


@dataclass(frozen=True, slots=True)
class CacheWindow:
    cached_at: datetime
    expires_at: datetime
    stale_expires_at: datetime


@dataclass(frozen=True, slots=True)
class CacheResolution(Generic[T]):
    value: T
    status: CacheResolutionStatus
    window: CacheWindow
    refresh_error: Exception | None = None


@dataclass(slots=True)
class _CacheEntry(Generic[T]):
    value: T
    cached_at: datetime
    fresh_until: float
    stale_until: float


class AsyncLRUTTLCache(Generic[T]):
    """Bounded process-local fresh/stale cache with per-key single-flight.

    Only values returned successfully by ``factory`` are stored. Provider errors,
    deterministic fallbacks, and raw model output therefore never enter the cache.
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
        if max_entries < 1:
            raise ValueError("cache max_entries must be at least one.")
        if ttl_seconds < 1:
            raise ValueError("cache ttl_seconds must be at least one.")
        if stale_if_error_seconds < 0:
            raise ValueError("cache stale_if_error_seconds cannot be negative.")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.stale_if_error_seconds = stale_if_error_seconds
        self._monotonic = monotonic
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._entries: OrderedDict[str, _CacheEntry[T]] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[_CacheEntry[T]]] = {}
        self._lock = asyncio.Lock()

    async def resolve(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        *,
        force_refresh: bool = False,
        should_use_stale: Callable[[Exception], bool] | None = None,
    ) -> CacheResolution[T]:
        now = self._monotonic()
        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and now >= entry.stale_until:
                del self._entries[key]
                entry = None
            if entry is not None:
                self._entries.move_to_end(key)
                if not force_refresh and now < entry.fresh_until:
                    return CacheResolution(
                        value=entry.value,
                        status="hit",
                        window=self._window(entry),
                    )

            task = self._inflight.get(key)
            leader = task is None
            if task is None:
                task = asyncio.create_task(self._produce_and_store(key, factory))
                self._inflight[key] = task
            stale_entry = entry

        try:
            resolved_entry = await asyncio.shield(task)
        except Exception as error:
            await self._clear_inflight(key, task)
            if stale_entry is not None and (
                should_use_stale is None or should_use_stale(error)
            ):
                return CacheResolution(
                    value=stale_entry.value,
                    status="stale",
                    window=self._window(stale_entry),
                    refresh_error=error,
                )
            raise
        await self._clear_inflight(key, task)
        status: CacheResolutionStatus = (
            "bypass" if leader and force_refresh
            else "miss" if leader
            else "hit"
        )
        return CacheResolution(
            value=resolved_entry.value,
            status=status,
            window=self._window(resolved_entry),
        )

    async def _produce_and_store(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
    ) -> _CacheEntry[T]:
        task = asyncio.current_task()
        try:
            value = await factory()
            return await self._store(key, value)
        finally:
            # A resolve caller awaits this task through ``asyncio.shield`` so
            # cancelling that caller does not cancel work shared by other
            # callers.  The cancelled waiter cannot clean up a task that is
            # still running, though.  Make the producer own its lifecycle so a
            # completed task can never remain in ``_inflight`` and be mistaken
            # for a valid refresh after the cached value's stale window.
            async with self._lock:
                if task is not None and self._inflight.get(key) is task:
                    del self._inflight[key]

    async def _clear_inflight(
        self,
        key: str,
        task: asyncio.Task[_CacheEntry[T]],
    ) -> None:
        async with self._lock:
            if self._inflight.get(key) is task and task.done():
                del self._inflight[key]

    async def _store(self, key: str, value: T) -> _CacheEntry[T]:
        now_monotonic = self._monotonic()
        cached_at = self._utcnow()
        if cached_at.tzinfo is None or cached_at.utcoffset() is None:
            raise ValueError("cache utcnow must return a timezone-aware datetime.")
        entry = _CacheEntry(
            value=value,
            cached_at=cached_at,
            fresh_until=now_monotonic + self.ttl_seconds,
            stale_until=(
                now_monotonic
                + self.ttl_seconds
                + self.stale_if_error_seconds
            ),
        )
        async with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return entry

    def _window(self, entry: _CacheEntry[T]) -> CacheWindow:
        return CacheWindow(
            cached_at=entry.cached_at,
            expires_at=entry.cached_at + timedelta(seconds=self.ttl_seconds),
            stale_expires_at=entry.cached_at
            + timedelta(
                seconds=self.ttl_seconds + self.stale_if_error_seconds
            ),
        )
