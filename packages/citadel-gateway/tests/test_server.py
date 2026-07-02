"""End-to-end tests for the assembled gateway request path.

These exercise the full route -> circuit-breaker -> cache -> provider pipeline
through the real FastAPI app via ``TestClient``. No network and no API keys are
involved: providers are replaced with in-process fakes injected on
``app.state``, which is exactly the seam ``create_app`` was built around.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from citadel_gateway.circuit_breaker import CircuitBreaker
from citadel_gateway.config import GatewayConfig
from citadel_gateway.providers.base import CompletionResponse, Provider
from citadel_gateway.router import Router, RoutingRule
from citadel_gateway.server import create_app


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class FakeProvider(Provider):
    """In-memory provider that either returns a canned response or raises.

    Records every ``complete`` call so tests can assert which providers were
    actually hit (e.g. verifying an open circuit was skipped, or a cache hit
    avoided a second call).
    """

    def __init__(
        self,
        *,
        content: str = "hello from fake",
        fail: bool = False,
        models: list[str] | None = None,
    ) -> None:
        self.content = content
        self.fail = fail
        self.models = models or []
        self.calls: list[tuple[list[dict[str, str]], str]] = []

    async def complete(
        self, messages: list[dict[str, str]], model: str, **kwargs: Any
    ) -> CompletionResponse:
        self.calls.append((messages, model))
        if self.fail:
            raise RuntimeError("simulated provider outage")
        return CompletionResponse(
            content=self.content,
            model=model,
            prompt_tokens=7,
            completion_tokens=5,
            finish_reason="stop",
        )

    async def list_models(self) -> list[str]:
        return self.models


def _open_breaker(breaker: CircuitBreaker) -> None:
    """Drive a breaker into the OPEN state by tripping its threshold."""
    for _ in range(breaker.failure_threshold):
        breaker.record_failure()
    assert not breaker.is_available()


def _two_provider_router() -> Router:
    """Router that maps ``test-*`` models to primary then backup."""
    return Router(
        rules=[
            RoutingRule(pattern=r"test-.*", provider="primary", model="{model}", priority=10),
            RoutingRule(pattern=r"test-.*", provider="backup", model="{model}", priority=5),
        ]
    )


def build_app(
    providers: dict[str, Provider],
    router: Router,
    *,
    breakers: dict[str, CircuitBreaker] | None = None,
    cache_path: str | None = None,
):
    """Construct the real app then inject fake subsystems on ``app.state``."""
    config = GatewayConfig(
        providers={},
        rate_limit_enabled=False,
        cache_enabled=cache_path is not None,
        cache_db_path=cache_path or "./unused_cache.db",
    )
    app = create_app(config)
    app.state.providers = providers
    app.state.router = router
    app.state.breakers = breakers or {
        name: CircuitBreaker(name=name) for name in providers
    }
    return app


def _chat_body(model: str = "test-model", content: str = "ping") -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": content}]}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_successful_completion_returns_200_and_openai_shape() -> None:
    primary = FakeProvider(content="pong")
    app = build_app({"primary": primary}, _two_provider_router())
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 200

    data = resp.json()
    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert data["model"] == "test-model"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "pong"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] == 12
    assert len(primary.calls) == 1


def test_models_endpoint_lists_configured_providers() -> None:
    providers = {
        "primary": FakeProvider(models=["test-a", "test-b"]),
        "backup": FakeProvider(models=["test-c"]),
    }
    app = build_app(providers, _two_provider_router())
    client = TestClient(app)

    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert ids == {"test-a", "test-b", "test-c"}


def test_health_endpoint_reports_provider_breaker_state() -> None:
    app = build_app({"primary": FakeProvider()}, _two_provider_router())
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["providers"]["primary"]["state"] == "closed"
    assert data["providers"]["primary"]["available"] is True


# ---------------------------------------------------------------------------
# Failover behaviour (Option A)
# ---------------------------------------------------------------------------


def test_failover_when_primary_errors_backup_serves_request() -> None:
    primary = FakeProvider(fail=True)
    backup = FakeProvider(content="served by backup")
    breakers = {
        "primary": CircuitBreaker(name="primary"),
        "backup": CircuitBreaker(name="backup"),
    }
    app = build_app(
        {"primary": primary, "backup": backup},
        _two_provider_router(),
        breakers=breakers,
    )
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "served by backup"
    # Primary was tried (and its failure recorded), backup then served it.
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1
    assert breakers["primary"]._failure_count == 1
    assert breakers["backup"].state.value == "closed"


def test_failover_skips_provider_with_open_circuit() -> None:
    primary = FakeProvider(content="should not be reached")
    backup = FakeProvider(content="backup wins")
    breakers = {
        "primary": CircuitBreaker(name="primary"),
        "backup": CircuitBreaker(name="backup"),
    }
    _open_breaker(breakers["primary"])
    app = build_app(
        {"primary": primary, "backup": backup},
        _two_provider_router(),
        breakers=breakers,
    )
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "backup wins"
    # Open circuit means primary is never even called.
    assert primary.calls == []
    assert len(backup.calls) == 1


def test_all_providers_failing_returns_502() -> None:
    primary = FakeProvider(fail=True)
    backup = FakeProvider(fail=True)
    app = build_app(
        {"primary": primary, "backup": backup},
        _two_provider_router(),
    )
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 502
    assert "failed" in resp.json()["detail"].lower()
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1


def test_all_circuits_open_returns_503_without_calling_providers() -> None:
    primary = FakeProvider()
    backup = FakeProvider()
    breakers = {
        "primary": CircuitBreaker(name="primary"),
        "backup": CircuitBreaker(name="backup"),
    }
    _open_breaker(breakers["primary"])
    _open_breaker(breakers["backup"])
    app = build_app(
        {"primary": primary, "backup": backup},
        _two_provider_router(),
        breakers=breakers,
    )
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 503
    assert "circuit open" in resp.json()["detail"].lower()
    assert primary.calls == []
    assert backup.calls == []


def test_breaker_opens_after_repeated_failures() -> None:
    provider = FakeProvider(fail=True)
    breaker = CircuitBreaker(name="primary", failure_threshold=2)
    router = Router(
        rules=[RoutingRule(pattern=r"test-.*", provider="primary", model="{model}", priority=10)]
    )
    app = build_app({"primary": provider}, router, breakers={"primary": breaker})
    client = TestClient(app)

    # First two failures return 502 and trip the breaker.
    assert client.post("/v1/chat/completions", json=_chat_body()).status_code == 502
    assert client.post("/v1/chat/completions", json=_chat_body()).status_code == 502
    assert not breaker.is_available()

    # With the sole provider's circuit now open, the next call fails fast (503)
    # and does not reach the provider.
    calls_before = len(provider.calls)
    resp = client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 503
    assert len(provider.calls) == calls_before


# ---------------------------------------------------------------------------
# Cache path
# ---------------------------------------------------------------------------


def test_cache_hit_returns_cached_response_without_second_provider_call(
    tmp_path,
) -> None:
    primary = FakeProvider(content="cache me")
    router = Router(
        rules=[RoutingRule(pattern=r"test-.*", provider="primary", model="{model}", priority=10)]
    )
    cache_db = str(tmp_path / "cache.db")
    app = build_app({"primary": primary}, router, cache_path=cache_db)
    client = TestClient(app)

    first = client.post("/v1/chat/completions", json=_chat_body())
    assert first.status_code == 200
    assert len(primary.calls) == 1

    second = client.post("/v1/chat/completions", json=_chat_body())
    assert second.status_code == 200
    # Identical request served from cache — provider was not called again.
    assert len(primary.calls) == 1
    assert second.json()["choices"][0]["message"]["content"] == "cache me"

    # Distinct request is a cache miss and does hit the provider.
    other = client.post("/v1/chat/completions", json=_chat_body(content="different"))
    assert other.status_code == 200
    assert len(primary.calls) == 2


# ---------------------------------------------------------------------------
# Routing edge cases
# ---------------------------------------------------------------------------


def test_no_matching_route_returns_400() -> None:
    router = Router(
        rules=[RoutingRule(pattern=r"only-this", provider="primary", model="{model}", priority=1)]
    )
    app = build_app({"primary": FakeProvider()}, router)
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json=_chat_body(model="unmatched-model"))
    assert resp.status_code == 400


def test_matching_route_but_no_configured_provider_returns_502() -> None:
    # Router points test-* at "ghost", which is not in the providers dict.
    router = Router(
        rules=[RoutingRule(pattern=r"test-.*", provider="ghost", model="{model}", priority=10)]
    )
    app = build_app({"primary": FakeProvider()}, router)
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 502
    assert "no configured provider" in resp.json()["detail"].lower()
