"""Tests for citadel_agents.orchestrator — Multi-agent coordination."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from citadel_agents.agent import Agent, AgentResponse
from citadel_agents.llm import LLMClient, LLMResponse
from citadel_agents.orchestrator import Orchestrator
from citadel_agents.tool import ToolRegistry


def _make_mock_client(*responses: LLMResponse) -> LLMClient:
    """Create a mock LLMClient that returns canned responses in sequence."""
    client = LLMClient()
    mock = AsyncMock(side_effect=list(responses))
    client.chat = mock
    return client


class TestOrchestrator:
    """Tests for multi-agent orchestration."""

    @pytest.mark.asyncio
    async def test_keyword_routing(self) -> None:
        """Orchestrator routes to the correct agent based on keyword match."""
        math_client = _make_mock_client(
            LLMResponse(content="Math answer: 42", tool_calls=[], usage={}),
        )
        writing_client = _make_mock_client(
            LLMResponse(content="Here is a poem.", tool_calls=[], usage={}),
        )

        math_agent = Agent(
            name="math",
            system_prompt="You do math.",
            model="ollama/test",
            llm_client=math_client,
        )
        writing_agent = Agent(
            name="writing",
            system_prompt="You write creatively.",
            model="ollama/test",
            llm_client=writing_client,
        )

        orchestrator = Orchestrator(agents={"math": math_agent, "writing": writing_agent})

        # "math" keyword should route to math agent
        response = await orchestrator.run("Help me with math homework")
        assert response.answer == "Math answer: 42"

    @pytest.mark.asyncio
    async def test_agent_as_tool_delegation(self) -> None:
        """An agent can delegate to another agent via the auto-registered tool."""
        # The helper agent will be called as a tool
        helper_client = _make_mock_client(
            LLMResponse(content="Helper says: the capital of France is Paris", tool_calls=[], usage={}),
        )

        # The main agent first calls ask_helper, then answers
        main_client = _make_mock_client(
            LLMResponse(
                content="",
                tool_calls=[{"name": "ask_helper", "arguments": {"message": "What is the capital of France?"}}],
                usage={},
            ),
            LLMResponse(
                content="According to my helper, the capital of France is Paris.",
                tool_calls=[],
                usage={},
            ),
        )

        helper_agent = Agent(
            name="helper",
            system_prompt="You answer geography questions.",
            model="ollama/test",
            llm_client=helper_client,
        )
        main_agent = Agent(
            name="main",
            system_prompt="You coordinate with other agents.",
            model="ollama/test",
            llm_client=main_client,
        )

        orchestrator = Orchestrator(agents={"main": main_agent, "helper": helper_agent})

        # The main agent should have ask_helper registered
        assert "ask_helper" in [t.name for t in main_agent.tools.list_tools()]

        # Route to main (it appears first and "main" is in input)
        response = await orchestrator.run("main: What is the capital of France?")
        assert "Paris" in response.answer
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool == "ask_helper"
