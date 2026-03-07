"""Core agent implementation using a ReAct-style loop.

The agent reasons about user input, decides whether to call a tool or
provide a final answer, observes tool results, and repeats until done.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from citadel_agents.llm import LLMClient, LLMResponse
from citadel_agents.memory import ConversationMemory
from citadel_agents.tool import ToolRegistry, ToolSpec


@dataclass
class Step:
    """A single step in the agent's reasoning trace."""

    type: str  # "think", "act", or "observe"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCall:
    """Record of a tool invocation."""

    tool: str
    args: dict[str, Any]
    result: str


@dataclass
class AgentResponse:
    """Final response from an agent run."""

    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    trace: list[Step] = field(default_factory=list)


@dataclass
class ThinkResult:
    """Result of a single think step — either a final answer or a tool call."""

    is_final: bool
    content: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)


class Agent:
    """ReAct-style agent that reasons, acts via tools, and observes results.

    The agent loop:
    1. Think — send conversation to LLM, get response
    2. If LLM returns a final answer, stop
    3. If LLM returns a tool call, execute it (Act)
    4. Add tool result to conversation (Observe)
    5. Repeat from step 1

    Args:
        name: Agent name for identification.
        system_prompt: System prompt defining the agent's behavior.
        model: Model identifier string (e.g., "ollama/qwen3:8b").
        tools: Optional ToolRegistry with available tools.
        memory: Optional ConversationMemory instance.
        max_iterations: Maximum number of think-act-observe cycles.
        llm_client: Optional LLMClient instance (created automatically if None).
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "ollama/qwen3:8b",
        tools: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
        max_iterations: int = 10,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools or ToolRegistry()
        self.memory = memory or ConversationMemory()
        self.max_iterations = max_iterations
        self.llm_client = llm_client or LLMClient()

    async def run(self, user_input: str) -> AgentResponse:
        """Run the agent on user input through the ReAct loop.

        Args:
            user_input: The user's message or query.

        Returns:
            AgentResponse with the final answer, tool call records, and trace.
        """
        trace: list[Step] = []
        tool_call_records: list[ToolCall] = []

        # Add user input to memory
        self.memory.add("user", user_input)

        for iteration in range(1, self.max_iterations + 1):
            # Build messages from memory
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.memory.get_messages(),
            ]

            # Think
            think_result = await self._think(messages)
            trace.append(Step(type="think", content=think_result.content or think_result.tool_name))

            if think_result.is_final:
                # LLM gave a final answer
                self.memory.add("assistant", think_result.content)
                return AgentResponse(
                    answer=think_result.content,
                    tool_calls=tool_call_records,
                    iterations=iteration,
                    trace=trace,
                )

            # Act — execute the tool
            trace.append(Step(
                type="act",
                content=f"Calling {think_result.tool_name}({think_result.tool_args})",
            ))

            tool_result = await self._execute_tool(think_result.tool_name, think_result.tool_args)
            tool_call_records.append(ToolCall(
                tool=think_result.tool_name,
                args=think_result.tool_args,
                result=tool_result,
            ))

            # Observe — add tool result to memory
            trace.append(Step(type="observe", content=tool_result))
            self.memory.add(
                "system",
                f"Tool '{think_result.tool_name}' returned: {tool_result}",
            )

        # Hit max iterations — return whatever we have
        final_answer = (
            f"Reached maximum iterations ({self.max_iterations}). "
            "Unable to provide a complete answer."
        )
        self.memory.add("assistant", final_answer)
        return AgentResponse(
            answer=final_answer,
            tool_calls=tool_call_records,
            iterations=self.max_iterations,
            trace=trace,
        )

    async def _think(self, messages: list[dict[str, str]]) -> ThinkResult:
        """Call the LLM and interpret its response.

        Args:
            messages: Current conversation messages.

        Returns:
            ThinkResult indicating either a final answer or a tool call.
        """
        tool_schemas = self.tools.to_schema() if self.tools.list_tools() else None

        response: LLMResponse = await self.llm_client.chat(
            messages=messages,
            model=self.model,
            tools=tool_schemas,
        )

        if response.tool_calls:
            tc = response.tool_calls[0]  # Handle one tool call at a time
            return ThinkResult(
                is_final=False,
                content=response.content,
                tool_name=tc["name"],
                tool_args=tc.get("arguments", {}),
            )

        return ThinkResult(is_final=True, content=response.content)

    async def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a registered tool and return its result.

        Args:
            tool_name: Name of the tool to execute.
            args: Arguments to pass to the tool.

        Returns:
            String result from the tool, or error message.
        """
        try:
            tool_spec = self.tools.get(tool_name)
        except KeyError:
            return f"Error: Tool '{tool_name}' is not registered"

        try:
            result = await tool_spec.execute(**args)
            return result
        except Exception as e:
            return f"Error executing tool '{tool_name}': {type(e).__name__}: {e}"
