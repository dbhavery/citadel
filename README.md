# Citadel

**Open-source AI Operations Platform**

A modular monorepo of 6 independently pip-installable packages for building, routing, observing, and orchestrating LLM-powered applications.

```
citadel serve          # Start gateway + trace server + dashboard
citadel ingest ./docs  # Parse, chunk, embed, and store documents
citadel search "query" # Semantic search over ingested content
citadel agent def.yaml # Run a ReAct agent from a YAML definition
citadel traces         # View recent LLM traces with cost/latency
citadel cost           # Cost breakdown by model and day
citadel status         # Health check all services
```

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              citadel CLI                  │
                    │     (unified command-line interface)      │
                    └──────┬────┬────┬────┬────┬────┬─────────┘
                           │    │    │    │    │    │
            ┌──────────────┘    │    │    │    │    └──────────────┐
            ▼                   ▼    │    ▼    ▼                  ▼
    ┌───────────────┐  ┌────────┴──┐ │ ┌──┴────────┐  ┌─────────────────┐
    │ citadel-gateway│  │  citadel- │ │ │  citadel- │  │ citadel-dashboard│
    │               │  │  vector   │ │ │  agents   │  │                 │
    │ LLM proxy     │  │          │ │ │           │  │ Static HTML SPA │
    │ + routing     │  │ HNSW     │ │ │ ReAct loop│  │ dark-theme UI   │
    │ + caching     │  │ from     │ │ │ + tools   │  │ zero build step │
    │ + rate limit  │  │ scratch  │ │ │ + memory  │  │                 │
    │ + circuit     │  │ + REST   │ │ │ + YAML    │  │                 │
    │   breaker     │  │   API    │ │ │   defs    │  │                 │
    └───────────────┘  └──────────┘ │ └───────────┘  └─────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                   ┌──────┴──────┐   ┌────────┴────────┐
                   │ citadel-    │   │ citadel-trace    │
                   │ ingest      │   │                  │
                   │             │   │ Auto-instrument  │
                   │ Parse →     │   │ SDKs, collect    │
                   │ Chunk →     │   │ spans, track     │
                   │ Embed →     │   │ cost, latency    │
                   │ Store       │   │ percentiles,     │
                   │             │   │ alerts           │
                   └─────────────┘   └──────────────────┘
```

## Packages

| Package | Description | Key Features |
|---------|-------------|-------------|
| **citadel-gateway** | OpenAI-compatible LLM proxy | Regex-based model→provider routing, SQLite response cache, token bucket rate limiter, circuit breaker failover, Ollama/Anthropic/OpenAI adapters |
| **citadel-vector** | HNSW vector search engine | Implemented from the Malkov & Yashunin 2018 paper in Python+NumPy, persistent storage, metadata filtering, REST API |
| **citadel-agents** | Agent runtime framework | ReAct-style reasoning loop, `@tool` decorator with auto-schema from type hints, conversation + vector memory, multi-agent orchestration, YAML agent definitions |
| **citadel-ingest** | Document ingestion pipeline | 4 chunking strategies (fixed, sentence, semantic, code), file parsers (MD, TXT, Python, PDF, DOCX), embedding, SHA-256 deduplication |
| **citadel-trace** | LLM observability | Span/Trace data model, SQLite collector, model pricing database (15+ models), auto-instrumentation via monkey-patching (Anthropic/OpenAI/httpx), metrics (cost/latency/tokens/errors), alert rules |
| **citadel-dashboard** | Operations dashboard | Single-file HTML SPA, dark theme, zero build step, traces/cost/latency/model views |

## Installation

```bash
# Install everything
pip install citadel-ai

# Or install individual packages
pip install citadel-gateway
pip install citadel-vector
pip install citadel-agents
pip install citadel-ingest
pip install citadel-trace
```

## Quick Start

```bash
# Start the platform
citadel serve

# Ingest documents
citadel ingest ./my-docs --chunk-strategy sentence

# Search
citadel search "how does authentication work?"

# Run an agent
citadel agent agents/researcher.yaml -i "summarize the codebase"

# Check costs
citadel cost --days 30
```

## Docker

```bash
docker compose up -d
```

Services:
- Gateway: `http://localhost:8080`
- Trace Server: `http://localhost:8081`
- Dashboard: `http://localhost:3000`

## Development

```bash
# Run all tests
cd packages/citadel-gateway && python -m pytest tests/ -v
cd packages/citadel-vector  && python -m pytest tests/ -v
cd packages/citadel-agents  && python -m pytest tests/ -v
cd packages/citadel-ingest  && python -m pytest tests/ -v
cd packages/citadel-trace   && python -m pytest tests/ -v
```

## Tech Stack

- **Python 3.10+** — all packages
- **FastAPI + Uvicorn** — REST APIs (gateway, vector, trace)
- **SQLite** — response cache, trace storage (zero-config)
- **NumPy** — HNSW vector math
- **Click + Rich** — CLI
- **Docker Compose** — deployment

## License

MIT
