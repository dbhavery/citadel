"""LLM provider adapters — unified interface over multiple backends."""

from citadel_gateway.providers.base import CompletionResponse, Provider

__all__ = ["CompletionResponse", "Provider"]
