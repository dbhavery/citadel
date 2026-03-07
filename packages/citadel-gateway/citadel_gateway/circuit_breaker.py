"""Circuit breaker for provider failover."""

from __future__ import annotations

import time
from enum import Enum


class State(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal — requests flow through
    OPEN = "open"  # Failing — reject requests immediately
    HALF_OPEN = "half_open"  # Testing — allow one request to probe recovery


class CircuitBreaker:
    """Per-provider circuit breaker.

    Tracks consecutive failures and opens the circuit after
    *failure_threshold* is reached.  After *recovery_timeout* seconds
    in the OPEN state, transitions to HALF_OPEN and allows a single
    probe request.  A success in HALF_OPEN closes the circuit;
    a failure re-opens it.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        name: str = "",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name

        self._state: State = State.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> State:
        """Current state, with automatic OPEN -> HALF_OPEN transition."""
        if self._state == State.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = State.HALF_OPEN
        return self._state

    def is_available(self) -> bool:
        """Return True if requests should be allowed through."""
        return self.state != State.OPEN

    def record_success(self) -> None:
        """Record a successful request."""
        self._failure_count = 0
        self._state = State.CLOSED

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = State.OPEN

    def reset(self) -> None:
        """Force-reset to CLOSED."""
        self._failure_count = 0
        self._state = State.CLOSED
        self._last_failure_time = 0.0
