"""Ollama provider — talks to a local Ollama instance via its HTTP API."""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from citadel_gateway.providers.base import CompletionResponse, Provider


class OllamaProvider(Provider):
    """Adapter for the Ollama ``/api/chat`` endpoint."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send messages to Ollama and return a normalised response."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if "temperature" in kwargs and kwargs["temperature"] is not None:
            payload.setdefault("options", {})["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            payload.setdefault("options", {})["num_predict"] = kwargs["max_tokens"]

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0

        return CompletionResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
            raw=data,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama using its native streaming."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        if "temperature" in kwargs and kwargs["temperature"] is not None:
            payload.setdefault("options", {})["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            payload.setdefault("options", {})["num_predict"] = kwargs["max_tokens"]

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{self._base_url}/api/chat", json=payload
            ) as resp:
                resp.raise_for_status()
                import json

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token

    async def list_models(self) -> list[str]:
        """Fetch model list from Ollama's /api/tags endpoint."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except (httpx.HTTPError, KeyError):
            return []
