"""Tests for citadel_gateway.circuit_breaker."""

import time

from citadel_gateway.circuit_breaker import CircuitBreaker, State


class TestCircuitBreakerBasic:
    """Core state-machine behaviour."""

    def test_starts_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == State.CLOSED
        assert cb.is_available() is True

    def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.CLOSED  # not yet
        cb.record_failure()
        assert cb.state == State.OPEN
        assert cb.is_available() is False

    def test_rejects_when_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999)
        cb.record_failure()
        assert cb.is_available() is False

    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        assert cb.state == State.OPEN
        time.sleep(0.1)
        assert cb.state == State.HALF_OPEN
        assert cb.is_available() is True

    def test_closes_on_success_in_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == State.HALF_OPEN
        cb.record_success()
        assert cb.state == State.CLOSED
        assert cb.is_available() is True

    def test_reopens_on_failure_in_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == State.HALF_OPEN
        cb.record_failure()
        assert cb.state == State.OPEN

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # resets count
        cb.record_failure()
        cb.record_failure()
        # Only 2 failures after reset, should still be closed
        assert cb.state == State.CLOSED
