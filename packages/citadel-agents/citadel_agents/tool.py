"""Tool registry for agent tool use.

Tools are functions that agents can call during their ReAct loop.
They can be registered manually or via the @tool decorator.
Parameter schemas are auto-generated from type hints.
"""

from __future__ import annotations

import asyncio
import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolSpec:
    """Specification for a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable  # async or sync function

    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool handler, wrapping sync functions automatically."""
        if asyncio.iscoroutinefunction(self.handler):
            result = await self.handler(**kwargs)
        else:
            result = self.handler(**kwargs)
        return str(result)


# Global registry for the @tool decorator
_GLOBAL_REGISTRY: Optional["ToolRegistry"] = None


def _type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema type."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is list or (hasattr(annotation, "__origin__") and annotation.__origin__ is list):
        return {"type": "array"}
    if annotation is dict or (hasattr(annotation, "__origin__") and annotation.__origin__ is dict):
        return {"type": "object"}
    return {"type": "string"}


def _generate_schema(func: Callable) -> dict[str, Any]:
    """Auto-generate a JSON Schema for function parameters from type hints."""
    sig = inspect.signature(func)

    # Resolve string annotations from `from __future__ import annotations`
    try:
        resolved_hints = typing.get_type_hints(func)
    except Exception:
        resolved_hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = resolved_hints.get(name, param.annotation)
        prop = _type_to_json_schema(annotation)
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


class ToolRegistry:
    """Registry of tools available to an agent."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        handler: Callable | None = None,
    ) -> None:
        """Register a tool manually.

        Args:
            name: Unique tool name.
            description: Human-readable description of what the tool does.
            parameters: JSON Schema for parameters. Auto-generated from handler if None.
            handler: The function to call when the tool is invoked.
        """
        if handler is None:
            raise ValueError(f"handler is required when registering tool '{name}'")
        if parameters is None:
            parameters = _generate_schema(handler)
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def get(self, name: str) -> ToolSpec:
        """Get a tool by name.

        Args:
            name: The tool name.

        Returns:
            The ToolSpec for the named tool.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def list_tools(self) -> list[ToolSpec]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_schema(self) -> list[dict[str, Any]]:
        """Convert all tools to LLM-compatible tool schema format.

        Returns a list of tool definitions suitable for passing to LLM APIs.
        """
        schemas = []
        for spec in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            })
        return schemas


def tool(name: str | None = None, description: str = "") -> Callable:
    """Decorator to register a function as a tool.

    Usage:
        @tool(name="my_tool", description="Does something useful")
        def my_tool(x: int, y: str) -> str:
            return f"{x}: {y}"

    Or with defaults:
        @tool()
        def my_tool(x: int) -> str:
            return str(x)

    Args:
        name: Tool name. Defaults to the function name.
        description: Tool description. Defaults to the function's docstring.

    Returns:
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip() or f"Tool: {tool_name}"
        parameters = _generate_schema(func)

        spec = ToolSpec(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            handler=func,
        )

        # Attach spec to function for later retrieval
        func._tool_spec = spec  # type: ignore[attr-defined]

        # Register in global registry if one exists
        global _GLOBAL_REGISTRY
        if _GLOBAL_REGISTRY is not None:
            _GLOBAL_REGISTRY._tools[tool_name] = spec

        return func

    return decorator


def set_global_registry(registry: ToolRegistry) -> None:
    """Set the global registry for the @tool decorator."""
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = registry


def get_global_registry() -> ToolRegistry | None:
    """Get the current global registry."""
    return _GLOBAL_REGISTRY
