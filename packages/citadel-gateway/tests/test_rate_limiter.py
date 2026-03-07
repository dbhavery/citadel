"""Tests for citadel_gateway.rate_limiter."""

import time

from citadel_gateway.rate_limiter import RateLimiter, TokenBucket


class TestTokenBucket:
    """Low-level bucket behaviour."""

    def test_allows_up_to_capacity(self) -> None:
        bucket = TokenBucket(capacity=3, refill_rate=0.0)
        assert bucket.try_acquire(1) is True
        assert bucket.try_acquire(1) is True
        assert bucket.try_acquire(1) is True
        assert bucket.try_acquire(1) is False

    def test_refills_over_time(self) -> None:
        bucket = TokenBucket(capacity=2, refill_rate=100.0)  # 100/sec
        bucket.try_acquire(2)  # drain
        assert bucket.try_acquire(1) is False
        time.sleep(0.05)  # ~5 tokens refilled at 100/sec
        assert bucket.try_acquire(1) is True


class TestRateLimiter:
    """Per-(key, model) rate limiting."""

    def test_allow_under_limit(self) -> None:
        limiter = RateLimiter(default_capacity=5, default_refill_rate=0.0)
        assert limiter.acquire_sync("user1", "gpt-4", 1) is True

    def test_block_over_limit(self) -> None:
        limiter = RateLimiter(default_capacity=2, default_refill_rate=0.0)
        assert limiter.acquire_sync("user1", "gpt-4") is True
        assert limiter.acquire_sync("user1", "gpt-4") is True
        assert limiter.acquire_sync("user1", "gpt-4") is False

    def test_per_key_isolation(self) -> None:
        limiter = RateLimiter(default_capacity=1, default_refill_rate=0.0)
        assert limiter.acquire_sync("userA", "gpt-4") is True
        assert limiter.acquire_sync("userA", "gpt-4") is False
        # Different user should still have capacity
        assert limiter.acquire_sync("userB", "gpt-4") is True

    def test_per_model_isolation(self) -> None:
        limiter = RateLimiter(default_capacity=1, default_refill_rate=0.0)
        assert limiter.acquire_sync("user1", "model-a") is True
        assert limiter.acquire_sync("user1", "model-a") is False
        # Different model should still have capacity
        assert limiter.acquire_sync("user1", "model-b") is True
