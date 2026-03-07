"""YAML agent definition loader.

Load agent configurations from YAML files and instantiate Agent objects.

Example YAML format:
    name: researcher
    model: claude-sonnet
    system_prompt: |
      You are a research assistant. Use tools to find information.
    tools:
      - web_search
      - calculator
    max_iterations: 5
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from citadel_agents.agent import Agent
from citadel_agents.builtin_tools import register_builtin_tools
from citadel_agents.memory import ConversationMemory
from citadel_agents.tool import ToolRegistry


def load_agent(
    path: str | Path,
    tool_registry: ToolRegistry | None = None,
) -> Agent:
    """Load an agent definition from a YAML file.

    Args:
        path: Path to the YAML file.
        tool_registry: Optional pre-populated ToolRegistry. If not provided,
                       a new one is created with built-in tools.

    Returns:
        Configured Agent instance.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValueError: If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Agent definition not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Empty agent definition: {path}")

    name = config.get("name")
    if not name:
        raise ValueError(f"Agent definition missing 'name': {path}")

    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    model = config.get("model", "ollama/qwen3:8b")
    max_iterations = config.get("max_iterations", 10)
    max_turns = config.get("max_turns", 50)

    # Build tool registry
    if tool_registry is None:
        tool_registry = ToolRegistry()
        register_builtin_tools(tool_registry)

    # Filter tools if specified in YAML
    requested_tools = config.get("tools")
    if requested_tools is not None:
        filtered_registry = ToolRegistry()
        for tool_name in requested_tools:
            try:
                tool_spec = tool_registry.get(tool_name)
                filtered_registry.register(
                    name=tool_spec.name,
                    description=tool_spec.description,
                    parameters=tool_spec.parameters,
                    handler=tool_spec.handler,
                )
            except KeyError:
                # Tool not found in registry — skip with warning
                pass
        tool_registry = filtered_registry

    memory = ConversationMemory(max_turns=max_turns)

    return Agent(
        name=name,
        system_prompt=system_prompt,
        model=model,
        tools=tool_registry,
        memory=memory,
        max_iterations=max_iterations,
    )


def load_agents(
    directory: str | Path,
    tool_registry: ToolRegistry | None = None,
) -> dict[str, Agent]:
    """Load all agent definitions from YAML files in a directory.

    Args:
        directory: Path to directory containing .yaml/.yml agent files.
        tool_registry: Optional shared ToolRegistry.

    Returns:
        Dictionary mapping agent names to Agent instances.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    agents: dict[str, Agent] = {}
    for yaml_file in sorted(directory.iterdir()):
        if yaml_file.suffix in (".yaml", ".yml"):
            agent = load_agent(yaml_file, tool_registry=tool_registry)
            agents[agent.name] = agent

    return agents
