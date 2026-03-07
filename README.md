# Citadel

**Open-Source AI Operations Platform**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Packages](https://img.shields.io/badge/packages-6-green.svg)](#packages)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Citadel is a modular, self-hosted platform for running LLM-powered applications in production. It provides a unified gateway for multiple LLM providers, a zero-dependency vector engine built from the HNSW paper, an agent framework with tool use, a document ingestion pipeline, cost/latency tracing with alerts, and a real-time operations dashboard -- all as independent, composable Python packages that work standalone or together.

Built for developers who want full control over their AI infrastructure without vendor lock-in.

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
                            +------------------+
                            |    Dashboard     |
                            |   (port 3000)    |
                            |  Static HTML SPA |
                            +--------+---------+
                                     |
                          REST API   |   fetch()
                                     |
+-------------------+       +--------+---------+       +------------------+
|                   |       |                  |       |                  |
|  citadel-ingest   +------>+  citadel-gateway +------>+  citadel-trace   |
|                   |       |    (port 8080)   |       |   (port 8081)    |
|  PDF, MD, HTML,   |       |                  |       |                  |
|  TXT, DOCX, CSV,  |       |  Model routing   |       |  Span/trace      |
|  Python -> chunks  |       |  Response cache  |       |  collection,     |
|                   |       |  Rate limiting   |       |  cost tracking,  |
+--------+----------+       |  Circuit breaker |       |  latency p50/99, |
         |                  |  Provider adapt. |       |  alert rules     |
         v                  +----+--------+----+       +------------------+
+--------+----------+            |        |
|                   |            |        |
|  citadel-vector   |            v        v          +----+-----+----+
|                   |       +---------+  +--------+  |         |    |
|  HNSW index       |       |Anthropic|  | OpenAI |  | Google  |Ollama
|  (Malkov 2018)    |       +---------+  +--------+  +---------+----+
|  Persistent store |
|  Metadata filter  |       +--------------------------------------------+
|  REST API         |       |              citadel-agents                 |
|                   |       |  YAML-defined agents, @tool decorator,     |
+-------------------+       |  ReAct loop, conversation + vector memory, |
                            |  multi-agent orchestration                 |
                            +--------------------------------------------+
```

---

## Packages

| Package | Description | Key Features | Standalone? |
|---------|-------------|-------------|-------------|
| **citadel-gateway** | OpenAI-compatible LLM proxy | Regex model-to-provider routing, SQLite response cache, token-bucket rate limiter, circuit breaker failover, Ollama/Anthropic/OpenAI adapters | Yes |
| **citadel-vector** | HNSW vector search engine | Implemented from the Malkov & Yashunin 2018 paper in Python+NumPy, persistent storage, metadata filtering, REST API | Yes |
| **citadel-agents** | Agent runtime framework | ReAct-style reasoning loop, `@tool` decorator with auto-schema from type hints, conversation + vector memory, multi-agent orchestration, YAML definitions | Yes |
| **citadel-ingest** | Document ingestion pipeline | 4 chunking strategies (fixed, sentence, semantic, code), file parsers (MD, TXT, Python, PDF, DOCX), embedding, SHA-256 deduplication | Yes |
| **citadel-trace** | LLM observability | Span/Trace data model, SQLite collector, pricing database (15+ models), auto-instrumentation via monkey-patching, cost/latency/token/error metrics, alert rules | Yes |
| **citadel-dashboard** | Operations dashboard | Single-file HTML SPA, dark theme, zero build step, traces/cost/latency/model views, auto-refresh, demo mode | Yes |

Every package works independently. Use one, use all, or use any combination.

---

## Quick Start

### Full Platform (Docker)

```bash
git clone https://github.com/dbhavery/citadel.git
cd citadel
cp .env.example .env
# Edit .env with your API keys

docker compose up
```

- **Gateway:** http://localhost:8080
- **Dashboard:** http://localhost:3000
- **Trace Server:** http://localhost:8081

### Full Platform (CLI)

```bash
pip install citadel-ai
citadel serve --port 8080
```

### Just the Gateway

```bash
pip install citadel-gateway
```

```python
from citadel_gateway import GatewayClient

client = GatewayClient()
response = client.complete(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello, Citadel."}],
)
print(response.text)
print(f"Cost: ${response.cost:.4f} | Latency: {response.latency_ms}ms")
```

### Just Vector Search

```bash
pip install citadel-vector
```

```python
from citadel_vector import VectorStore

store = VectorStore(path="./my_vectors", dim=384)
store.add("doc_1", embedding=[0.1, 0.2, ...], metadata={"source": "readme.md"})

results = store.search(query_embedding=[0.1, 0.2, ...], k=5)
for result in results:
    print(f"{result.id}: score={result.score:.4f}")
```

### Just Ingestion

```bash
pip install citadel-ingest
citadel ingest ./docs/ --chunk-strategy sentence --chunk-size 500
```

```python
from citadel_ingest import Ingestor

ingestor = Ingestor(chunk_strategy="sentence", chunk_size=500)
chunks = ingestor.ingest("./docs/architecture.md")
print(f"Created {len(chunks)} chunks")
```

### Just Agents

```bash
pip install citadel-agents
```

```yaml
# researcher.yaml
name: researcher
model: claude-sonnet-4-20250514
system: You are a research assistant. Use tools to find and synthesize information.
tools:
  - web_search
  - read_file
  - write_file
max_steps: 10
```

```bash
citadel agent researcher.yaml -i "Summarize the latest advances in RLHF"
```

---

## Dashboard

The dashboard is a single HTML file with zero dependencies -- no build step, no Node.js required, no framework. Open it in any browser or serve it with any static file server.

```
+------------------------------------------------------------------+
| Citadel                          http://localhost:8080  Connected |
+----------+-------------------------------------------------------+
|          |  Overview                                              |
| Overview |  +------------+ +----------+ +----------+ +---------+ |
| Traces   |  | Requests   | | Cost     | | Cache    | | Errors  | |
| Agents   |  | 1,243      | | $18.42   | | 34.2%    | | 0.8%    | |
| Data     |  +------------+ +----------+ +----------+ +---------+ |
| Models   |                                                       |
| Alerts   |  Cost (Last 7 Days)                                   |
| Settings |  [####  ][######][###   ][########][#####][##  ][####] |
|          |   Mon     Tue    Wed     Thu       Fri    Sat   Sun    |
|          |                                                       |
|          |  Recent Traces                                        |
|          |  Time     | Model          | Tokens | Cost   | Status |
|          |  14:23:01 | claude-sonnet  | 2,341  | $0.042 | OK     |
|          |  14:22:45 | gpt-4o         | 1,892  | $0.031 | OK     |
|          |  14:22:12 | qwen3:8b       | 956    | $0.000 | OK     |
|          |  14:21:58 | claude-haiku   | 3,102  | $0.005 | ERROR  |
|          |                                                       |
|          |  Model Usage                                          |
|          |  qwen3:8b        [================] 40.2%  3,200 reqs |
|          |  claude-haiku    [==============]   27.0%  2,150 reqs |
|          |  claude-sonnet   [========]         15.6%  1,243 reqs |
|          |  gpt-4o          [======]           11.0%    876 reqs |
|          |  gemini-2.5      [====]              6.8%    540 reqs |
+----------+-------------------------------------------------------+
```

**Pages:**
- **Overview** -- stat cards, 7-day cost chart, recent traces, model usage breakdown
- **Traces** -- searchable/filterable trace list with click-to-expand span trees showing parent-child relationships
- **Agents** -- agent monitoring (placeholder for runtime agent tracking)
- **Data** -- vector store monitoring (placeholder for collection stats)
- **Models** -- per-model request count, average latency, total cost, error rate with visual comparisons
- **Alerts** -- configured alert rules with status badges and triggered alert history
- **Settings** -- provider API keys (masked), cache toggles, rate limit configuration, alert thresholds

**Features:**
- Auto-refreshes every 15 seconds
- Configurable API base URL in the top bar
- Connection status indicator
- Demo mode with realistic sample data when the API is unreachable
- Fully responsive (sidebar collapses on mobile)
- Keyboard shortcut: Escape to close trace detail panel

---

## CLI Reference

```
Usage: citadel [OPTIONS] COMMAND [ARGS]...

  Citadel -- AI Operations Platform.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  serve    Start all Citadel services.
  ingest   Ingest documents into the vector store.
  search   Search ingested documents.
  agent    Run an agent from a YAML definition.
  traces   View recent traces.
  cost     Show cost summary.
  status   Check Citadel service status.
```

### Examples

```bash
# Start everything on custom port
citadel serve --port 9090 --host 0.0.0.0

# Ingest with paragraph-level chunking
citadel ingest ./docs/ --chunk-strategy paragraph --chunk-size 800 --overlap 100

# Search with more results
citadel search "how does the caching layer work" --k 10

# Run an agent with verbose output
citadel agent agents/researcher.yaml -i "Compare HNSW vs IVF indexing" -v

# View only error traces
citadel traces --limit 50 --status error

# 30-day cost report
citadel cost --days 30

# Service health check against custom URL
citadel status --url http://staging:8080
```

---

## Configuration

All configuration is done via environment variables (or a `.env` file at the repo root). Copy `.env.example` to `.env` and fill in your values.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | -- | Anthropic/Claude API key |
| `OPENAI_API_KEY` | -- | OpenAI API key |
| `GOOGLE_API_KEY` | -- | Google/Gemini API key |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `GATEWAY_HOST` | `0.0.0.0` | Gateway bind address |
| `GATEWAY_PORT` | `8080` | Gateway listen port |
| `GATEWAY_CACHE_ENABLED` | `true` | Enable response caching |
| `GATEWAY_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `TRACE_DB_PATH` | `./data/traces.db` | SQLite path for trace storage |
| `VECTOR_STORAGE_PATH` | `./data/vectors` | Directory for vector indices |
| `RATE_LIMIT_RPM` | `60` | Requests per minute limit |
| `RATE_LIMIT_TPM` | `100000` | Tokens per minute limit |
| `ALERT_WEBHOOK_URL` | -- | Webhook for alert notifications (Slack, Discord, etc.) |
| `ALERT_DAILY_COST_LIMIT` | `10.00` | Daily cost alert threshold ($) |
| `ALERT_ERROR_RATE_THRESHOLD` | `5.0` | Error rate alert threshold (%) |

---

## Modular Design Philosophy

Citadel is built on one principle: **every package works alone**.

You should never have to install the entire platform to use one piece of it. Need a vector store? Install `citadel-vector` -- it has zero required dependencies beyond Python and NumPy. Need an LLM gateway? Install `citadel-gateway` -- it handles provider routing, caching, and rate limiting without requiring the rest of Citadel.

When you do use multiple packages together, they compose naturally through the CLI and shared configuration. The gateway emits traces that the trace server collects. The ingest pipeline writes to the vector store. The agent framework routes through the gateway. But none of these connections are mandatory.

This matters because:
- **Incremental adoption.** Start with one package. Add more when you need them.
- **Independent testing.** Each package has its own test suite that runs without the others.
- **Clear boundaries.** No circular dependencies. No hidden coupling. Each package owns its domain.
- **Deployment flexibility.** Run everything in one process, split across containers, or deploy individual packages as microservices.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Gateway server | FastAPI + Uvicorn |
| Vector indexing | Custom HNSW implementation (Malkov & Yashunin 2018) + NumPy |
| Agent framework | YAML definitions + ReAct loop |
| Document parsing | Built-in parsers (PDF, MD, HTML, DOCX, CSV, TXT, Python) |
| Trace storage | SQLite (zero-config) |
| Dashboard | Vanilla HTML/CSS/JS (single file, no build step) |
| CLI | Click + Rich |
| Packaging | pyproject.toml (PEP 621) |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
citadel/
  packages/
    citadel-gateway/      # LLM provider gateway
    citadel-vector/       # Vector storage engine
    citadel-agents/       # Agent framework
    citadel-ingest/       # Document ingestion
    citadel-trace/        # Tracing and cost tracking
    citadel-dashboard/    # Web dashboard (static HTML)
  citadel_cli/            # CLI meta-package
  docker-compose.yml      # Full-stack deployment
  pyproject.toml          # Root package config
  .env.example            # Environment template
```

## Development

```bash
# Run individual package tests
cd packages/citadel-gateway && python -m pytest tests/ -v
cd packages/citadel-vector  && python -m pytest tests/ -v
cd packages/citadel-agents  && python -m pytest tests/ -v
cd packages/citadel-ingest  && python -m pytest tests/ -v
cd packages/citadel-trace   && python -m pytest tests/ -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
