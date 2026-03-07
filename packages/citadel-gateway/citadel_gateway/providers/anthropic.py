"""Anthropic (Claude) provider — uses the anthropic SDK with lazy import."""

from __future__ import annotations

from typing import Any, AsyncIterator

from citadel_gateway.providers.base import CompletionResponse, Provider


class AnthropicProvider(Provider):
    """Adapter for the Anthropic Messages API."""

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com") -> None:
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self) -> Any:
        """Lazy-import and instantiate the Anthropic async client."""
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with: pip install citadel-gateway[anthropic]"
            ) from exc
        return anthropic.AsyncAnthropic(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Split an OpenAI-style message list into (system, messages).

        Anthropic's API takes the system prompt as a top-level parameter
        rather than inside the messages array.
        """
        system_text: str | None = None
        filtered: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                filtered.append({"role": msg["role"], "content": msg.get("content", "")})
        return system_text, filtered

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a chat completion to the Anthropic Messages API."""
        client = self._get_client()
        system_text, api_messages = self._convert_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens") or 4096,
        }
        if system_text:
            create_kwargs["system"] = system_text
        if kwargs.get("temperature") is not None:
            create_kwargs["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None:
            create_kwargs["top_p"] = kwargs["top_p"]
        if kwargs.get("stop"):
            create_kwargs["stop_sequences"] = (
                kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            )

        response = await client.messages.create(**create_kwargs)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return CompletionResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            finish_reason="stop" if response.stop_reason == "end_turn" else (response.stop_reason or "stop"),
            raw={"id": response.id, "type": response.type},
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from the Anthropic Messages API."""
        client = self._get_client()
        system_text, api_messages = self._convert_messages(messages)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens") or 4096,
        }
        if system_text:
            create_kwargs["system"] = system_text
        if kwargs.get("temperature") is not None:
            create_kwargs["temperature"] = kwargs["temperature"]

        async with client.messages.stream(**create_kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def list_models(self) -> list[str]:
        """Return a static list of known Anthropic model IDs."""
        return [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250414",
            "claude-opus-4-20250514",
        ]
