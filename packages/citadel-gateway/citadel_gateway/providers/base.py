"""Abstract base class for LLM providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class CompletionResponse:
    """Normalised response from any provider."""

    content: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider(abc.ABC):
    """Abstract LLM provider.

    Every concrete provider must implement ``complete`` (and optionally
    ``stream``) plus ``list_models``.
    """

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a chat completion request and return a normalised response."""

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens. Default falls back to non-streaming."""
        response = await self.complete(messages, model, **kwargs)
        yield response.content

    async def list_models(self) -> list[str]:
        """Return model IDs available from this provider."""
        return []
