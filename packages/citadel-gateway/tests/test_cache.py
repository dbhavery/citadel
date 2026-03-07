"""Tests for citadel_gateway.cache."""

import os
import tempfile
import time

from citadel_gateway.cache import ResponseCache


def _make_cache() -> ResponseCache:
    """Create a cache backed by a temporary file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return ResponseCache(db_path=path)


class TestCacheMissHit:
    """Basic get/put behaviour."""

    def test_miss_returns_none(self) -> None:
        cache = _make_cache()
        assert cache.get("nonexistent-key") is None
        cache.close()

    def test_hit_returns_stored_response(self) -> None:
        cache = _make_cache()
        data = {"choices": [{"message": {"content": "hello"}}]}
        cache.put("key1", data, ttl=600)
        result = cache.get("key1")
        assert result == data
        cache.close()

    def test_different_keys_are_independent(self) -> None:
        cache = _make_cache()
        cache.put("a", {"v": 1}, ttl=600)
        cache.put("b", {"v": 2}, ttl=600)
        assert cache.get("a") == {"v": 1}
        assert cache.get("b") == {"v": 2}
        cache.close()


class TestCacheExpiry:
    """TTL expiry behaviour."""

    def test_expired_entry_returns_none(self) -> None:
        cache = _make_cache()
        cache.put("expire-me", {"v": "old"}, ttl=0)
        # TTL=0 means it expired immediately
        time.sleep(0.05)
        assert cache.get("expire-me") is None
        cache.close()


class TestCacheStats:
    """Stats tracking."""

    def test_stats_track_hits_and_misses(self) -> None:
        cache = _make_cache()
        cache.put("s1", {"ok": True}, ttl=600)

        cache.get("s1")  # hit
        cache.get("s1")  # hit
        cache.get("nope")  # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == round(2 / 3, 4)
        assert stats["total_entries"] == 1
        cache.close()


class TestCacheKeyGeneration:
    """make_key determinism."""

    def test_same_input_same_key(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        k1 = ResponseCache.make_key("gpt-4", msgs)
        k2 = ResponseCache.make_key("gpt-4", msgs)
        assert k1 == k2

    def test_different_model_different_key(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        k1 = ResponseCache.make_key("gpt-4", msgs)
        k2 = ResponseCache.make_key("gpt-3.5", msgs)
        assert k1 != k2
