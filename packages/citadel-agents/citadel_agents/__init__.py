"""citadel-agents — Agent runtime framework.

Provides tool use, memory, planning, and multi-agent orchestration.
Agents can be defined in Python or YAML.
"""

__version__ = "0.1.0"

from citadel_agents.agent import Agent, AgentResponse, Step, ToolCall
from citadel_agents.loader import load_agent, load_agents
from citadel_agents.llm import LLMClient, LLMResponse
from citadel_agents.memory import ConversationMemory, VectorMemory
from citadel_agents.orchestrator import Orchestrator
from citadel_agents.tool import ToolRegistry, ToolSpec, tool

__all__ = [
    "__version__",
    "Agent",
    "AgentResponse",
    "ConversationMemory",
    "LLMClient",
    "LLMResponse",
    "load_agent",
    "load_agents",
    "Orchestrator",
    "Step",
    "tool",
    "ToolCall",
    "ToolRegistry",
    "ToolSpec",
    "VectorMemory",
]
