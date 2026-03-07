"""Auto-instrumentation for LLM SDKs.

Monkey-patches anthropic, openai, and httpx (for Ollama) to
automatically capture traces for every LLM call.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable, Optional

from .collector import TraceCollector

logger = logging.getLogger(__name__)


class Instrumentor:
    """Monkey-patch LLM SDKs to auto-capture traces."""

    def __init__(self, collector: TraceCollector) -> None:
        self.collector = collector
        self._originals: dict[str, Any] = {}

    def instrument_all(self) -> None:
        """Instrument all supported SDKs. Skips any that aren't installed."""
        self.instrument_anthropic()
        self.instrument_openai()
        self.instrument_httpx()

    def uninstrument_all(self) -> None:
        """Restore all original methods."""
        if "anthropic_sync" in self._originals:
            try:
                import anthropic  # type: ignore[import-untyped]
                anthropic.resources.messages.Messages.create = self._originals["anthropic_sync"]
            except (ImportError, AttributeError):
                pass

        if "anthropic_async" in self._originals:
            try:
                import anthropic  # type: ignore[import-untyped]
                anthropic.resources.messages.AsyncMessages.create = self._originals["anthropic_async"]
            except (ImportError, AttributeError):
                pass

        if "openai_sync" in self._originals:
            try:
                import openai  # type: ignore[import-untyped]
                openai.resources.chat.completions.Completions.create = self._originals["openai_sync"]
            except (ImportError, AttributeError):
                pass

        if "openai_async" in self._originals:
            try:
                import openai  # type: ignore[import-untyped]
                openai.resources.chat.completions.AsyncCompletions.create = self._originals["openai_async"]
            except (ImportError, AttributeError):
                pass

        if "httpx_sync" in self._originals:
            try:
                import httpx  # type: ignore[import-untyped]
                httpx.Client.send = self._originals["httpx_sync"]
            except (ImportError, AttributeError):
                pass

        if "httpx_async" in self._originals:
            try:
                import httpx  # type: ignore[import-untyped]
                httpx.AsyncClient.send = self._originals["httpx_async"]
            except (ImportError, AttributeError):
                pass

        self._originals.clear()

    # --- Anthropic ---

    def instrument_anthropic(self) -> None:
        """Patch anthropic.Anthropic.messages.create and async variant."""
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("anthropic SDK not installed, skipping instrumentation")
            return

        collector = self.collector

        # Sync
        try:
            original_create = anthropic.resources.messages.Messages.create
            if getattr(original_create, "_citadel_instrumented", False):
                return
            self._originals["anthropic_sync"] = original_create

            @functools.wraps(original_create)
            def wrapped_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", args[0] if args else "unknown")
                messages = kwargs.get("messages", args[1] if len(args) > 1 else [])

                span = collector.start_span(
                    name="anthropic.messages.create",
                    kind="llm",
                    model=str(model),
                    provider="anthropic",
                    input_messages=_safe_serialize_messages(messages),
                )

                try:
                    result = original_create(self_inner, *args, **kwargs)

                    output_text = _extract_anthropic_output(result)
                    tokens = _extract_anthropic_tokens(result)

                    collector.end_span(span, output=output_text, tokens=tokens)
                    return result

                except Exception as exc:
                    collector.end_span(span, error=str(exc))
                    raise

            wrapped_create._citadel_instrumented = True  # type: ignore[attr-defined]
            anthropic.resources.messages.Messages.create = wrapped_create  # type: ignore[assignment]
        except AttributeError:
            logger.debug("Could not patch anthropic sync Messages.create")

        # Async
        try:
            original_async_create = anthropic.resources.messages.AsyncMessages.create
            if getattr(original_async_create, "_citadel_instrumented", False):
                return
            self._originals["anthropic_async"] = original_async_create

            @functools.wraps(original_async_create)
            async def wrapped_async_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", args[0] if args else "unknown")
                messages = kwargs.get("messages", args[1] if len(args) > 1 else [])

                span = collector.start_span(
                    name="anthropic.messages.create",
                    kind="llm",
                    model=str(model),
                    provider="anthropic",
                    input_messages=_safe_serialize_messages(messages),
                )

                try:
                    result = await original_async_create(self_inner, *args, **kwargs)

                    output_text = _extract_anthropic_output(result)
                    tokens = _extract_anthropic_tokens(result)

                    collector.end_span(span, output=output_text, tokens=tokens)
                    return result

                except Exception as exc:
                    collector.end_span(span, error=str(exc))
                    raise

            wrapped_async_create._citadel_instrumented = True  # type: ignore[attr-defined]
            anthropic.resources.messages.AsyncMessages.create = wrapped_async_create  # type: ignore[assignment]
        except AttributeError:
            logger.debug("Could not patch anthropic async Messages.create")

    # --- OpenAI ---

    def instrument_openai(self) -> None:
        """Patch openai.OpenAI.chat.completions.create and async variant."""
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("openai SDK not installed, skipping instrumentation")
            return

        collector = self.collector

        # Sync
        try:
            original_create = openai.resources.chat.completions.Completions.create
            if getattr(original_create, "_citadel_instrumented", False):
                return
            self._originals["openai_sync"] = original_create

            @functools.wraps(original_create)
            def wrapped_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "unknown")
                messages = kwargs.get("messages", [])

                span = collector.start_span(
                    name="openai.chat.completions.create",
                    kind="llm",
                    model=str(model),
                    provider="openai",
                    input_messages=_safe_serialize_messages(messages),
                )

                try:
                    result = original_create(self_inner, *args, **kwargs)

                    output_text = _extract_openai_output(result)
                    tokens = _extract_openai_tokens(result)

                    collector.end_span(span, output=output_text, tokens=tokens)
                    return result

                except Exception as exc:
                    collector.end_span(span, error=str(exc))
                    raise

            wrapped_create._citadel_instrumented = True  # type: ignore[attr-defined]
            openai.resources.chat.completions.Completions.create = wrapped_create  # type: ignore[assignment]
        except AttributeError:
            logger.debug("Could not patch openai sync Completions.create")

        # Async
        try:
            original_async_create = openai.resources.chat.completions.AsyncCompletions.create
            if getattr(original_async_create, "_citadel_instrumented", False):
                return
            self._originals["openai_async"] = original_async_create

            @functools.wraps(original_async_create)
            async def wrapped_async_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "unknown")
                messages = kwargs.get("messages", [])

                span = collector.start_span(
                    name="openai.chat.completions.create",
                    kind="llm",
                    model=str(model),
                    provider="openai",
                    input_messages=_safe_serialize_messages(messages),
                )

                try:
                    result = await original_async_create(self_inner, *args, **kwargs)

                    output_text = _extract_openai_output(result)
                    tokens = _extract_openai_tokens(result)

                    collector.end_span(span, output=output_text, tokens=tokens)
                    return result

                except Exception as exc:
                    collector.end_span(span, error=str(exc))
                    raise

            wrapped_async_create._citadel_instrumented = True  # type: ignore[attr-defined]
            openai.resources.chat.completions.AsyncCompletions.create = wrapped_async_create  # type: ignore[assignment]
        except AttributeError:
            logger.debug("Could not patch openai async Completions.create")

    # --- httpx (for Ollama) ---

    def instrument_httpx(self) -> None:
        """Patch httpx to capture Ollama API calls."""
        try:
            import httpx  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("httpx not installed, skipping instrumentation")
            return

        collector = self.collector

        # Sync
        try:
            original_send = httpx.Client.send
            if getattr(original_send, "_citadel_instrumented", False):
                return
            self._originals["httpx_sync"] = original_send

            @functools.wraps(original_send)
            def wrapped_send(self_inner: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                url = str(request.url)

                if not _is_ollama_request(url):
                    return original_send(self_inner, request, *args, **kwargs)

                model = "unknown"
                input_messages: list[dict] = []
                try:
                    body = json.loads(request.content)
                    model = body.get("model", "unknown")
                    input_messages = body.get("messages", [])
                except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
                    pass

                span = collector.start_span(
                    name="ollama.chat",
                    kind="llm",
                    model=f"ollama/{model}" if not model.startswith("ollama/") else model,
                    provider="ollama",
                    input_messages=_safe_serialize_messages(input_messages),
                )

                try:
                    response = original_send(self_inner, request, *args, **kwargs)

                    output_text, tokens = _extract_ollama_response(response)
                    collector.end_span(span, output=output_text, tokens=tokens)
                    return response

                except Exception as exc:
                    collector.end_span(span, error=str(exc))
                    raise

            wrapped_send._citadel_instrumented = True  # type: ignore[attr-defined]
            httpx.Client.send = wrapped_send  # type: ignore[assignment]
        except AttributeError:
            logger.debug("Could not patch httpx.Client.send")

        # Async
        try:
            original_async_send = httpx.AsyncClient.send
            if getattr(original_async_send, "_citadel_instrumented", False):
                return
            self._originals["httpx_async"] = original_async_send

            @functools.wraps(original_async_send)
            async def wrapped_async_send(self_inner: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                url = str(request.url)

                if not _is_ollama_request(url):
                    return await original_async_send(self_inner, request, *args, **kwargs)

                model = "unknown"
                input_messages: list[dict] = []
                try:
                    body = json.loads(request.content)
                    model = body.get("model", "unknown")
                    input_messages = body.get("messages", [])
                except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
                    pass

                span = collector.start_span(
                    name="ollama.chat",
                    kind="llm",
                    model=f"ollama/{model}" if not model.startswith("ollama/") else model,
                    provider="ollama",
                    input_messages=_safe_serialize_messages(input_messages),
                )

                try:
                    response = await original_async_send(self_inner, request, *args, **kwargs)

                    output_text, tokens = _extract_ollama_response(response)
                    collector.end_span(span, output=output_text, tokens=tokens)
                    return response

                except Exception as exc:
                    collector.end_span(span, error=str(exc))
                    raise

            wrapped_async_send._citadel_instrumented = True  # type: ignore[attr-defined]
            httpx.AsyncClient.send = wrapped_async_send  # type: ignore[assignment]
        except AttributeError:
            logger.debug("Could not patch httpx.AsyncClient.send")


# --- Helper functions ---


def _safe_serialize_messages(messages: Any) -> list[dict]:
    """Safely convert messages to a list of dicts for storage."""
    if not messages:
        return []
    try:
        if isinstance(messages, list):
            result = []
            for m in messages:
                if isinstance(m, dict):
                    result.append(m)
                elif hasattr(m, "__dict__"):
                    result.append({"role": getattr(m, "role", "unknown"), "content": str(getattr(m, "content", ""))})
                else:
                    result.append({"content": str(m)})
            return result
        return [{"content": str(messages)}]
    except Exception:
        return [{"content": "[serialization error]"}]


def _is_ollama_request(url: str) -> bool:
    """Check if a request URL targets an Ollama API endpoint."""
    return "11434" in url and ("/api/chat" in url or "/api/generate" in url)


def _extract_anthropic_output(result: Any) -> Optional[str]:
    """Extract text output from an Anthropic response."""
    try:
        if hasattr(result, "content") and result.content:
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts) if parts else None
    except Exception:
        pass
    return None


def _extract_anthropic_tokens(result: Any) -> Optional[dict]:
    """Extract token counts from an Anthropic response."""
    try:
        if hasattr(result, "usage") and result.usage:
            return {
                "input": getattr(result.usage, "input_tokens", None),
                "output": getattr(result.usage, "output_tokens", None),
            }
    except Exception:
        pass
    return None


def _extract_openai_output(result: Any) -> Optional[str]:
    """Extract text output from an OpenAI response."""
    try:
        if hasattr(result, "choices") and result.choices:
            msg = result.choices[0].message
            if hasattr(msg, "content"):
                return msg.content
    except Exception:
        pass
    return None


def _extract_openai_tokens(result: Any) -> Optional[dict]:
    """Extract token counts from an OpenAI response."""
    try:
        if hasattr(result, "usage") and result.usage:
            return {
                "input": getattr(result.usage, "prompt_tokens", None),
                "output": getattr(result.usage, "completion_tokens", None),
            }
    except Exception:
        pass
    return None


def _extract_ollama_response(response: Any) -> tuple[Optional[str], Optional[dict]]:
    """Extract output and token counts from an Ollama HTTP response."""
    try:
        body = json.loads(response.content)
        output_text = None
        tokens = None

        # Chat endpoint
        if "message" in body:
            output_text = body["message"].get("content")

        # Generate endpoint
        elif "response" in body:
            output_text = body["response"]

        if "prompt_eval_count" in body or "eval_count" in body:
            tokens = {
                "input": body.get("prompt_eval_count"),
                "output": body.get("eval_count"),
            }

        return output_text, tokens
    except Exception:
        return None, None
