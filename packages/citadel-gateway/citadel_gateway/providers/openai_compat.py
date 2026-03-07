"""Generic OpenAI-compatible provider — works with any API that follows the OpenAI spec."""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from citadel_gateway.providers.base import CompletionResponse, Provider


class OpenAICompatProvider(Provider):
    """Adapter for any OpenAI-compatible ``/chat/completions`` endpoint.

    Works with OpenAI, Azure OpenAI, vLLM, LM Studio, etc.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a chat completion to the OpenAI-compatible endpoint."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        for opt in ("temperature", "max_tokens", "top_p", "stop", "presence_penalty", "frequency_penalty"):
            if opt in kwargs and kwargs[opt] is not None:
                payload[opt] = kwargs[opt]

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return CompletionResponse(
            content=choice["message"]["content"],
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from an OpenAI-compatible endpoint (SSE)."""
        import json as _json

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        for opt in ("temperature", "max_tokens", "top_p", "stop"):
            if opt in kwargs and kwargs[opt] is not None:
                payload[opt] = kwargs[opt]

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = _json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token

    async def list_models(self) -> list[str]:
        """Fetch model list from the /models endpoint."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                return [m["id"] for m in data.get("data", [])]
        except (httpx.HTTPError, KeyError):
            return []
