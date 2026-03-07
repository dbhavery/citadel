# citadel-agents

Agent runtime framework for building AI agents with tool use, memory, planning, and multi-agent orchestration. Part of the Citadel monorepo.

## Quick Start

### Define an agent in Python

```python
from citadel_agents import Agent, ToolRegistry, tool

# Register tools
registry = ToolRegistry()

@tool(name="calculator", description="Evaluate a math expression")
def calculator(expression: str) -> str:
    return str(eval(expression))

registry.register(
    name="calculator",
    description="Evaluate a math expression",
    handler=calculator,
)

# Create an agent
agent = Agent(
    name="math_helper",
    system_prompt="You are a helpful math assistant. Use the calculator tool when needed.",
    model="ollama/qwen3:8b",
    tools=registry,
)

# Run it
import asyncio
response = asyncio.run(agent.run("What is 2 + 2?"))
print(response.answer)
```

### Define an agent in YAML

```yaml
# agents/researcher.yaml
name: researcher
model: ollama/qwen3:8b
system_prompt: |
  You are a research assistant. Use tools to find information.
tools:
  - calculator
  - current_time
  - http_get
max_iterations: 5
```

```python
from citadel_agents import load_agent

agent = load_agent("agents/researcher.yaml")
```

### Multi-agent orchestration

```python
from citadel_agents import Agent, Orchestrator

math_agent = Agent(name="math", system_prompt="You solve math problems.", model="ollama/qwen3:8b")
writer_agent = Agent(name="writer", system_prompt="You write content.", model="ollama/qwen3:8b")

orchestrator = Orchestrator(agents={"math": math_agent, "writer": writer_agent})
response = asyncio.run(orchestrator.run("Help me with math"))
```

## Installation

```bash
pip install citadel-agents
```

With optional dependencies:

```bash
pip install citadel-agents[anthropic]  # Claude support
pip install citadel-agents[vector]     # Vector memory
pip install citadel-agents[dev]        # Testing
```

## Architecture

- **Agent** -- ReAct loop (Reason, Act, Observe) with configurable LLM backend
- **ToolRegistry** -- Register tools via decorator or manually; auto-generates JSON Schema from type hints
- **ConversationMemory** -- Short-term message history with max_turns truncation
- **VectorMemory** -- Long-term memory with optional vector search (falls back to TF-IDF)
- **LLMClient** -- Unified interface for Ollama, Claude, and Gemini
- **Orchestrator** -- Route requests to multiple agents; agents can call each other as tools
- **Loader** -- Instantiate agents from YAML definitions

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```
