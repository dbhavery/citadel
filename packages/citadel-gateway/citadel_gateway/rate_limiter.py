"""Token-bucket rate limiter with per-(key, model) buckets."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """A classic token-bucket that refills at a constant rate."""

    capacity: float
    refill_rate: float  # tokens added per second
    tokens: float = -1.0  # -1 sentinel → initialised to capacity on first use
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.tokens < 0:
            self.tokens = self.capacity

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def try_acquire(self, amount: float = 1.0) -> bool:
        """Try to consume *amount* tokens. Returns True if allowed."""
        self._refill()
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class RateLimiter:
    """Maps ``(api_key, model)`` pairs to individual token buckets.

    Thread-safe for asyncio via an ``asyncio.Lock`` per bucket.
    """

    def __init__(
        self,
        default_capacity: float = 60.0,
        default_refill_rate: float = 1.0,
        per_model_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._default_capacity = default_capacity
        self._default_refill_rate = default_refill_rate
        # model -> (capacity, refill_rate) overrides
        self._per_model: dict[str, tuple[float, float]] = per_model_limits or {}
        self._buckets: dict[tuple[str, str], TokenBucket] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    def _get_bucket(self, key: str, model: str) -> TokenBucket:
        """Return (or create) the bucket for a given key+model pair."""
        pair = (key, model)
        if pair not in self._buckets:
            cap, rate = self._per_model.get(
                model, (self._default_capacity, self._default_refill_rate)
            )
            self._buckets[pair] = TokenBucket(capacity=cap, refill_rate=rate)
        return self._buckets[pair]

    def _get_lock(self, key: str, model: str) -> asyncio.Lock:
        pair = (key, model)
        if pair not in self._locks:
            self._locks[pair] = asyncio.Lock()
        return self._locks[pair]

    async def acquire(self, key: str, model: str, tokens: float = 1.0) -> bool:
        """Attempt to acquire *tokens* for the given key+model.

        Returns ``True`` if the request is allowed, ``False`` otherwise.
        """
        lock = self._get_lock(key, model)
        async with lock:
            bucket = self._get_bucket(key, model)
            return bucket.try_acquire(tokens)

    def acquire_sync(self, key: str, model: str, tokens: float = 1.0) -> bool:
        """Synchronous variant (no lock) — for testing or non-async code."""
        bucket = self._get_bucket(key, model)
        return bucket.try_acquire(tokens)
