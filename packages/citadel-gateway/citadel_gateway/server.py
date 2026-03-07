"""FastAPI server — OpenAI-compatible LLM gateway."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from citadel_gateway import __version__
from citadel_gateway.cache import ResponseCache
from citadel_gateway.circuit_breaker import CircuitBreaker
from citadel_gateway.config import GatewayConfig
from citadel_gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ModelInfo,
    ModelList,
    Usage,
)
from citadel_gateway.providers.base import Provider
from citadel_gateway.rate_limiter import RateLimiter
from citadel_gateway.router import Router

logger = logging.getLogger("citadel_gateway")

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    if config is None:
        config = GatewayConfig.from_env()

    app = FastAPI(
        title="Citadel Gateway",
        version=__version__,
        description="OpenAI-compatible LLM reverse proxy",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Subsystems --------------------------------------------------------
    router = Router()
    providers: dict[str, Provider] = _build_providers(config)
    breakers: dict[str, CircuitBreaker] = {
        name: CircuitBreaker(name=name) for name in providers
    }

    cache: ResponseCache | None = None
    if config.cache_enabled:
        cache = ResponseCache(db_path=config.cache_db_path)

    limiter: RateLimiter | None = None
    if config.rate_limit_enabled:
        limiter = RateLimiter(
            default_capacity=float(config.rate_limit_rpm),
            default_refill_rate=config.rate_limit_rpm / 60.0,
        )

    # Store on app.state for access in tests / middleware
    app.state.config = config
    app.state.router = router
    app.state.providers = providers
    app.state.breakers = breakers
    app.state.cache = cache
    app.state.limiter = limiter

    # --- Routes ------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        provider_status = {}
        for name, breaker in breakers.items():
            provider_status[name] = {
                "state": breaker.state.value,
                "available": breaker.is_available(),
            }
        result: dict[str, Any] = {
            "status": "ok",
            "version": __version__,
            "providers": provider_status,
        }
        if cache is not None:
            result["cache"] = cache.stats()
        return result

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """List models from all configured providers."""
        all_models: list[ModelInfo] = []
        for pname, provider in providers.items():
            try:
                model_ids = await provider.list_models()
                for mid in model_ids:
                    all_models.append(ModelInfo(id=mid, owned_by=pname))
            except Exception:
                logger.warning("Failed to list models from %s", pname, exc_info=True)
        return ModelList(data=all_models).model_dump()

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest, request: Request
    ) -> JSONResponse:
        """OpenAI-compatible chat completions endpoint."""
        # --- Rate limiting -------------------------------------------------
        if limiter is not None:
            api_key = _extract_api_key(request)
            allowed = await limiter.acquire(api_key, body.model)
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Try again later.",
                )

        # --- Routing -------------------------------------------------------
        try:
            route = router.resolve(body.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        provider = providers.get(route.provider)
        if provider is None:
            raise HTTPException(
                status_code=502,
                detail=f"Provider '{route.provider}' is not configured.",
            )

        # --- Circuit breaker -----------------------------------------------
        breaker = breakers.get(route.provider)
        if breaker and not breaker.is_available():
            raise HTTPException(
                status_code=503,
                detail=f"Provider '{route.provider}' is temporarily unavailable (circuit open).",
            )

        # --- Cache check ---------------------------------------------------
        messages_dicts = [m.model_dump(exclude_none=True) for m in body.messages]
        cache_key: str | None = None
        if cache is not None and not body.stream:
            cache_key = ResponseCache.make_key(route.model, messages_dicts)
            cached = cache.get(cache_key)
            if cached is not None:
                return JSONResponse(content=cached)

        # --- Provider call -------------------------------------------------
        kwargs: dict[str, Any] = {}
        if body.temperature is not None:
            kwargs["temperature"] = body.temperature
        if body.max_tokens is not None:
            kwargs["max_tokens"] = body.max_tokens
        if body.top_p is not None:
            kwargs["top_p"] = body.top_p
        if body.stop is not None:
            kwargs["stop"] = body.stop

        try:
            result = await provider.complete(messages_dicts, route.model, **kwargs)
            if breaker:
                breaker.record_success()
        except Exception as exc:
            if breaker:
                breaker.record_failure()
            logger.error(
                "Provider %s failed for model %s: %s",
                route.provider,
                route.model,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Provider '{route.provider}' error: {exc}",
            ) from exc

        # --- Build response ------------------------------------------------
        response = ChatCompletionResponse(
            model=result.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=result.content),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            ),
        )

        response_dict = response.to_openai_dict()

        # --- Cache store ---------------------------------------------------
        if cache is not None and cache_key is not None:
            cache.put(cache_key, response_dict, ttl=config.cache_ttl)

        return JSONResponse(content=response_dict)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_providers(config: GatewayConfig) -> dict[str, Provider]:
    """Instantiate provider objects from config."""
    from citadel_gateway.providers.ollama import OllamaProvider
    from citadel_gateway.providers.openai_compat import OpenAICompatProvider

    result: dict[str, Provider] = {}

    if "ollama" in config.providers:
        ollama_cfg = config.providers["ollama"]
        result["ollama"] = OllamaProvider(
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
        )

    if "openai" in config.providers:
        openai_cfg = config.providers["openai"]
        result["openai"] = OpenAICompatProvider(
            api_key=openai_cfg.get("api_key", ""),
            base_url=openai_cfg.get("base_url", "https://api.openai.com/v1"),
        )

    if "anthropic" in config.providers:
        anthropic_cfg = config.providers["anthropic"]
        try:
            from citadel_gateway.providers.anthropic import AnthropicProvider

            result["anthropic"] = AnthropicProvider(
                api_key=anthropic_cfg.get("api_key", ""),
                base_url=anthropic_cfg.get("base_url", "https://api.anthropic.com"),
            )
        except ImportError:
            logger.warning(
                "anthropic package not installed — Anthropic provider disabled. "
                "Install with: pip install citadel-gateway[anthropic]"
            )

    if "google" in config.providers:
        # Google/Gemini uses an OpenAI-compatible endpoint via the generativelanguage API
        google_cfg = config.providers["google"]
        api_key = google_cfg.get("api_key", "")
        result["google"] = OpenAICompatProvider(
            api_key=api_key,
            base_url=google_cfg.get(
                "base_url",
                f"https://generativelanguage.googleapis.com/v1beta/openai",
            ),
        )

    return result


def _extract_api_key(request: Request) -> str:
    """Pull the API key from the Authorization header, or return 'anonymous'."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return "anonymous"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the gateway server via uvicorn."""
    import uvicorn

    config = GatewayConfig.from_env()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
