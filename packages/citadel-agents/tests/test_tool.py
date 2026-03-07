"""Tests for citadel_agents.tool — ToolRegistry and @tool decorator."""

from __future__ import annotations

import pytest

from citadel_agents.tool import ToolRegistry, ToolSpec, tool, set_global_registry


class TestToolRegistry:
    """Tests for manual tool registration and retrieval."""

    def test_register_tool_manually(self) -> None:
        """Register a tool manually and retrieve it by name."""
        registry = ToolRegistry()

        def my_handler(x: int, y: int) -> int:
            return x + y

        registry.register(
            name="adder",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
            handler=my_handler,
        )

        spec = registry.get("adder")
        assert spec.name == "adder"
        assert spec.description == "Add two numbers"
        assert spec.handler is my_handler

    def test_register_tool_with_decorator(self) -> None:
        """Register a tool using the @tool decorator."""
        registry = ToolRegistry()
        set_global_registry(registry)

        @tool(name="multiplier", description="Multiply two numbers")
        def multiply(a: int, b: int) -> int:
            return a * b

        # The decorator attaches _tool_spec
        assert hasattr(multiply, "_tool_spec")
        assert multiply._tool_spec.name == "multiplier"

        # And it's registered in the global registry
        spec = registry.get("multiplier")
        assert spec.name == "multiplier"
        assert spec.description == "Multiply two numbers"

        # Cleanup global registry
        set_global_registry(None)

    def test_auto_generate_parameter_schema(self) -> None:
        """Parameter schema is auto-generated from type hints."""
        registry = ToolRegistry()

        def greet(name: str, times: int) -> str:
            return name * times

        registry.register(
            name="greet",
            description="Greet someone",
            handler=greet,
        )

        spec = registry.get("greet")
        props = spec.parameters["properties"]
        assert "name" in props
        assert props["name"]["type"] == "string"
        assert "times" in props
        assert props["times"]["type"] == "integer"
        assert "name" in spec.parameters["required"]
        assert "times" in spec.parameters["required"]

    def test_list_tools_returns_all(self) -> None:
        """list_tools returns all registered tools."""
        registry = ToolRegistry()

        def tool_a() -> str:
            return "a"

        def tool_b() -> str:
            return "b"

        registry.register(name="tool_a", description="Tool A", handler=tool_a)
        registry.register(name="tool_b", description="Tool B", handler=tool_b)

        tools = registry.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_b"}

    def test_get_nonexistent_tool_raises_error(self) -> None:
        """Getting a non-existent tool raises KeyError."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not_real"):
            registry.get("not_real")
