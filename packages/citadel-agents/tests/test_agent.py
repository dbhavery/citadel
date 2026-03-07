"""Tests for citadel_agents.agent — Agent ReAct loop."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from citadel_agents.agent import Agent
from citadel_agents.builtin_tools import calculator
from citadel_agents.llm import LLMClient, LLMResponse
from citadel_agents.tool import ToolRegistry


def _make_mock_client(*responses: LLMResponse) -> LLMClient:
    """Create a mock LLMClient that returns canned responses in sequence."""
    client = LLMClient()
    mock = AsyncMock(side_effect=list(responses))
    client.chat = mock
    return client


class TestAgent:
    """Tests for the core Agent ReAct loop."""

    @pytest.mark.asyncio
    async def test_no_tools_direct_answer(self) -> None:
        """Agent with no tools returns a direct answer from the LLM."""
        mock_client = _make_mock_client(
            LLMResponse(content="The answer is 42.", tool_calls=[], usage={}),
        )

        agent = Agent(
            name="simple",
            system_prompt="You are helpful.",
            model="ollama/test",
            llm_client=mock_client,
        )

        response = await agent.run("What is the answer?")

        assert response.answer == "The answer is 42."
        assert response.iterations == 1
        assert response.tool_calls == []
        assert len(response.trace) == 1
        assert response.trace[0].type == "think"

    @pytest.mark.asyncio
    async def test_tool_execution(self) -> None:
        """Agent executes a tool call and then returns the final answer."""
        # First response: LLM wants to call a tool
        # Second response: LLM gives final answer after seeing tool result
        mock_client = _make_mock_client(
            LLMResponse(
                content="",
                tool_calls=[{"name": "calculator", "arguments": {"expression": "2+2"}}],
                usage={},
            ),
            LLMResponse(
                content="The result of 2+2 is 4.",
                tool_calls=[],
                usage={},
            ),
        )

        registry = ToolRegistry()
        registry.register(
            name="calculator",
            description="Calculate math",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
            handler=lambda expression: calculator(expression),
        )

        agent = Agent(
            name="calc_agent",
            system_prompt="You are a calculator assistant.",
            model="ollama/test",
            tools=registry,
            llm_client=mock_client,
        )

        response = await agent.run("What is 2+2?")

        assert response.answer == "The result of 2+2 is 4."
        assert response.iterations == 2
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool == "calculator"
        assert response.tool_calls[0].result == "4"

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self) -> None:
        """Agent stops after max_iterations even if LLM keeps calling tools."""
        # Create a client that always returns tool calls
        infinite_tool_response = LLMResponse(
            content="",
            tool_calls=[{"name": "noop", "arguments": {}}],
            usage={},
        )
        mock_client = _make_mock_client(
            *[infinite_tool_response for _ in range(5)]
        )

        registry = ToolRegistry()
        registry.register(
            name="noop",
            description="Does nothing",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "ok",
        )

        agent = Agent(
            name="looper",
            system_prompt="You loop forever.",
            model="ollama/test",
            tools=registry,
            max_iterations=3,
            llm_client=mock_client,
        )

        response = await agent.run("Loop please")

        assert response.iterations == 3
        assert "maximum iterations" in response.answer.lower()

    @pytest.mark.asyncio
    async def test_trace_captures_all_steps(self) -> None:
        """Agent trace captures think, act, and observe steps."""
        mock_client = _make_mock_client(
            LLMResponse(
                content="",
                tool_calls=[{"name": "echo", "arguments": {"msg": "hello"}}],
                usage={},
            ),
            LLMResponse(
                content="Done!",
                tool_calls=[],
                usage={},
            ),
        )

        registry = ToolRegistry()
        registry.register(
            name="echo",
            description="Echo a message",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
            handler=lambda msg: f"Echo: {msg}",
        )

        agent = Agent(
            name="tracer",
            system_prompt="You trace things.",
            model="ollama/test",
            tools=registry,
            llm_client=mock_client,
        )

        response = await agent.run("Say hello")

        # First iteration: think + act + observe
        # Second iteration: think (final)
        assert len(response.trace) == 4
        assert response.trace[0].type == "think"
        assert response.trace[1].type == "act"
        assert response.trace[2].type == "observe"
        assert response.trace[3].type == "think"
        assert "Echo: hello" in response.trace[2].content
