"""LLM client abstraction.

Provides a unified interface to multiple LLM providers:
- Ollama (local, via httpx)
- Anthropic Claude (via anthropic SDK, lazy import)
- Google Gemini (via google-genai SDK, lazy import)
- citadel-gateway (if running, routes through it)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Unified LLM client that routes to the appropriate provider.

    Model string format determines the provider:
    - "ollama/<model>" -> Ollama at localhost:11434
    - "claude-*" -> Anthropic API
    - "gemini-*" -> Google Gemini API
    """

    def __init__(self, gateway_url: str | None = None) -> None:
        """Initialize the LLM client.

        Args:
            gateway_url: Optional citadel-gateway URL to route all requests through.
        """
        self.gateway_url = gateway_url
        self._anthropic_client: Any = None
        self._genai_client: Any = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "ollama/qwen3:8b",
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to the appropriate provider.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Model identifier string.
            tools: Optional list of tool schemas for function calling.

        Returns:
            Unified LLMResponse.

        Raises:
            ValueError: If the model provider is not recognized.
        """
        if self.gateway_url:
            return await self._chat_gateway(messages, model, tools)
        elif model.startswith("ollama/"):
            return await self._chat_ollama(messages, model, tools)
        elif model.startswith("claude"):
            return await self._chat_anthropic(messages, model, tools)
        elif model.startswith("gemini"):
            return await self._chat_gemini(messages, model, tools)
        else:
            raise ValueError(
                f"Unknown model provider for '{model}'. "
                "Use 'ollama/<model>', 'claude-*', or 'gemini-*'."
            )

    async def _chat_ollama(
        self,
        messages: list[dict[str, str]],
        model: str,
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        """Route to Ollama at localhost:11434."""
        import httpx

        model_name = model.removeprefix("ollama/")
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        message = data.get("message", {})
        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls", [])

        tool_calls = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            tool_calls.append({
                "name": func.get("name", ""),
                "arguments": func.get("arguments", {}),
            })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

    async def _chat_anthropic(
        self,
        messages: list[dict[str, str]],
        model: str,
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        """Route to Anthropic Claude API (lazy import)."""
        if self._anthropic_client is None:
            try:
                import anthropic
                self._anthropic_client = anthropic.AsyncAnthropic()
            except ImportError as e:
                raise ImportError(
                    "anthropic package required for Claude models. "
                    "Install with: pip install citadel-agents[anthropic]"
                ) from e

        # Separate system message from conversation
        system_prompt = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": chat_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            # Convert from OpenAI-style to Anthropic tool format
            anthropic_tools = []
            for t in tools:
                func = t.get("function", {})
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            kwargs["tools"] = anthropic_tools

        response = await self._anthropic_client.messages.create(**kwargs)

        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "arguments": block.input,
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
            },
        )

    async def _chat_gemini(
        self,
        messages: list[dict[str, str]],
        model: str,
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        """Route to Google Gemini API (lazy import)."""
        if self._genai_client is None:
            try:
                import google.generativeai as genai
                self._genai_client = genai
            except ImportError as e:
                raise ImportError(
                    "google-generativeai package required for Gemini models. "
                    "Install with: pip install google-generativeai"
                ) from e

        genai = self._genai_client
        gen_model = genai.GenerativeModel(model)

        # Build conversation content
        history = []
        for msg in messages:
            role = "user" if msg["role"] in ("user", "system") else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = gen_model.start_chat(history=history[:-1])
        last_msg = history[-1]["parts"][0] if history else ""

        response = chat.send_message(last_msg)

        return LLMResponse(
            content=response.text,
            tool_calls=[],
            usage={},
        )

    async def _chat_gateway(
        self,
        messages: list[dict[str, str]],
        model: str,
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        """Route through citadel-gateway."""
        import httpx

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.gateway_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        raw_tool_calls = message.get("tool_calls", [])

        tool_calls = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append({
                "name": func.get("name", ""),
                "arguments": args,
            })

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
        )
