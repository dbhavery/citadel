"""Multi-agent orchestration.

Coordinates multiple agents, routing user input to the appropriate agent
based on a custom router function or simple keyword matching.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from citadel_agents.agent import Agent, AgentResponse


class Orchestrator:
    """Coordinate multiple agents.

    Routes incoming requests to the appropriate agent based on a router
    function. If no router is provided, uses simple keyword matching
    against agent names.

    Agents can also call other agents as tools — each agent is
    automatically registered as a callable tool in every other agent's
    tool registry.

    Args:
        agents: Dictionary mapping agent names to Agent instances.
        router: Optional function that takes user_input and agent names,
                returns the name of the agent to route to.
    """

    def __init__(
        self,
        agents: dict[str, Agent] | None = None,
        router: Callable[[str, list[str]], str] | None = None,
    ) -> None:
        self.agents: dict[str, Agent] = agents or {}
        self.router = router
        self._register_agents_as_tools()

    def add_agent(self, name: str, agent: Agent) -> None:
        """Add an agent to the orchestrator.

        Args:
            name: Name to register the agent under.
            agent: The Agent instance.
        """
        self.agents[name] = agent
        self._register_agents_as_tools()

    async def run(self, user_input: str) -> AgentResponse:
        """Route user input to the appropriate agent and run it.

        Args:
            user_input: The user's message.

        Returns:
            AgentResponse from the selected agent.

        Raises:
            ValueError: If no agents are registered.
        """
        if not self.agents:
            raise ValueError("No agents registered in orchestrator")

        agent_name = self._route(user_input)
        agent = self.agents[agent_name]
        return await agent.run(user_input)

    def _route(self, user_input: str) -> str:
        """Determine which agent should handle the input.

        Args:
            user_input: The user's message.

        Returns:
            Name of the selected agent.
        """
        agent_names = list(self.agents.keys())

        if self.router is not None:
            return self.router(user_input, agent_names)

        return self._keyword_route(user_input, agent_names)

    def _keyword_route(self, user_input: str, agent_names: list[str]) -> str:
        """Simple keyword-based routing.

        Checks if any agent name appears in the user input (case-insensitive).
        Falls back to the first registered agent.

        Args:
            user_input: The user's message.
            agent_names: List of available agent names.

        Returns:
            Name of the matched agent, or the first agent as fallback.
        """
        input_lower = user_input.lower()
        for name in agent_names:
            if name.lower() in input_lower:
                return name
        return agent_names[0]

    def _register_agents_as_tools(self) -> None:
        """Register each agent as a callable tool in every other agent."""
        for owner_name, owner_agent in self.agents.items():
            for target_name, target_agent in self.agents.items():
                if owner_name == target_name:
                    continue

                tool_name = f"ask_{target_name}"
                # Check if already registered
                try:
                    owner_agent.tools.get(tool_name)
                    continue  # Already registered
                except KeyError:
                    pass

                # Create a closure to capture the target agent
                def make_handler(agent: Agent) -> Callable:
                    async def handler(message: str) -> str:
                        """Delegate a question to another agent."""
                        response = await agent.run(message)
                        return response.answer
                    return handler

                owner_agent.tools.register(
                    name=tool_name,
                    description=f"Ask the '{target_name}' agent a question. "
                                f"Agent description: {target_agent.system_prompt[:100]}",
                    parameters={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The question or message to send to the agent",
                            }
                        },
                        "required": ["message"],
                    },
                    handler=make_handler(target_agent),
                )
