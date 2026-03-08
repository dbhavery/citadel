# Citadel

A self-hosted AI operations platform that replaces managed LLM infrastructure with a single unified gateway -- multi-provider routing, semantic caching, vector search, agent runtime, and cost observability, all as independent pip-installable packages.

[![CI](https://github.com/dbhavery/citadel/actions/workflows/ci.yml/badge.svg)](https://github.com/dbhavery/citadel/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Packages](https://img.shields.io/badge/packages-6-green.svg)](#packages)
[![Tests](https://img.shields.io/badge/tests-100-brightgreen.svg)](#tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## Why I Built This

Running LLM applications in production means stitching together separate services for routing, caching, vector search, observability, and agent orchestration. Each one adds a dependency, a bill, and a failure mode. Managed vector databases alone run $25-70+/month (Weaviate, Pinecone). Provider lock-in means rewriting integration code when you switch models.

I wanted a single platform where every component is independent, composable, and self-hosted -- with zero mandatory cloud dependencies. So I built one.

## What It Does

- **Unified LLM gateway** with automatic routing across Claude, Gemini, OpenAI, and Ollama -- provider failover via circuit breakers eliminates single-provider lock-in
- **HNSW vector search engine** implemented from the Malkov & Yashunin 2018 paper in pure Python+NumPy -- O(log n) approximate nearest neighbor search with zero external vector DB dependency
- **Semantic response caching** deduplicates LLM API calls, saving $0.01-0.10 per cached hit depending on model and token count
- **ReAct agent runtime** with YAML-defined agents, auto-schema `@tool` decorator, and multi-agent orchestration
- **Auto-instrumented observability** tracking cost, latency, tokens, and error rates across every LLM call, with configurable alert rules
- **Document ingestion pipeline** with 4 chunking strategies (fixed, sentence, semantic, code) and 7 file format parsers

## Architecture

```
                        +------------------+
                        |    Dashboard     |  Single-file HTML SPA
                        |   (port 3000)    |  Zero build step
                        +--------+---------+
                                 |
                      REST API   |   fetch()
                                 |
+-------------------+   +--------+---------+   +------------------+
|                   |   |                  |   |                  |
|  citadel-ingest   +-->+  citadel-gateway +-->+  citadel-trace   |
|                   |   |    (port 8080)   |   |   (port 8081)    |
|  PDF, MD, HTML,   |   |                  |   |                  |
|  TXT, DOCX, CSV,  |   |  Model routing   |   |  Span collection |
|  Python -> chunks  |   |  Response cache  |   |  Cost tracking   |
|                   |   |  Rate limiting   |   |  p50/p99 latency |
+--------+----------+   |  Circuit breaker |   |  Alert rules     |
         |              +----+--------+----+   +------------------+
         v                   |        |
+--------+----------+        |        |
|                   |        v        v        +----+-----+----+
|  citadel-vector   |   +---------+  +------+  |         |    |
|                   |   |Anthropic|  |OpenAI|  | Google  |Ollama
|  HNSW index       |   +---------+  +------+  +---------+----+
|  (Malkov 2018)    |
|  Persistent store |   +-------------------------------------------+
|  Metadata filter  |   |             citadel-agents                 |
|  REST API         |   |  YAML-defined, @tool decorator,           |
|                   |   |  ReAct loop, conversation + vector memory  |
+-------------------+   +-------------------------------------------+
```

Every package works independently. Use one, use all, or any combination.

## Packages

| Package | What It Does | Standalone? |
|---------|-------------|-------------|
| **citadel-gateway** | OpenAI-compatible LLM proxy with regex model routing, SQLite response cache, token-bucket rate limiter, circuit breaker failover | Yes |
| **citadel-vector** | HNSW vector search engine (Malkov & Yashunin 2018), persistent storage, metadata filtering, REST API | Yes |
| **citadel-agents** | ReAct agent runtime with `@tool` auto-schema, conversation + vector memory, multi-agent orchestration, YAML definitions | Yes |
| **citadel-ingest** | Document pipeline with 4 chunking strategies, 7 format parsers, SHA-256 deduplication | Yes |
| **citadel-trace** | LLM observability with span/trace model, pricing DB (15+ models), auto-instrumentation, cost/latency/token metrics, alert rules | Yes |
| **citadel-dashboard** | Operations dashboard -- single HTML file, dark theme, zero build step, auto-refresh, demo mode | Yes |

## Key Technical Decisions

- **Built HNSW from scratch instead of using FAISS/pgvector.** Needed a dependency-free, pip-installable vector index that works on any platform without compiled binaries. FAISS requires platform-specific C++ builds. pgvector requires PostgreSQL. This implementation is pure Python+NumPy -- `pip install` and it works everywhere.

- **FastAPI over Flask for the gateway.** Concurrent LLM requests are the default workload -- async support is mandatory, not optional. FastAPI also generates OpenAPI docs automatically, which doubles as the gateway's API reference.

- **YAML-defined agents over code-defined.** Agent behavior (model, system prompt, tools, constraints) is configuration, not logic. YAML definitions let you iterate on agent behavior without touching code and version-control agent configs separately from runtime code.

- **Package-per-concern architecture.** Each of the 6 packages has its own `pyproject.toml`, test suite, and dependency list. You can install `citadel-vector` without pulling in FastAPI, or `citadel-trace` without pulling in NumPy. This eliminates the "install the world to use one feature" problem.

- **SQLite for trace storage over Postgres/ClickHouse.** For a self-hosted platform, zero-config storage matters more than write throughput. SQLite handles the trace volumes of a single-team deployment without requiring a database server.

## Results & Metrics

| Metric | Value |
|--------|-------|
| HNSW search complexity | O(log n) approximate nearest neighbor |
| Vector DB infrastructure cost | $0/month (vs. $25-70+/month for managed alternatives) |
| Provider lock-in | None -- automatic failover across 4 providers |
| Cache savings per hit | $0.01-0.10 depending on model |
| Codebase | 10K+ lines across 6 packages |
| Test coverage | 100 tests across all packages |

## Live Demo

[Citadel Explorer on HuggingFace Spaces](https://huggingface.co/spaces/dbhavery/citadel-explorer) -- interactive demo of the gateway, vector search, and dashboard.

## Quick Start

```bash
# Install the full platform
pip install citadel-ai
citadel serve --port 8080

# Or install only what you need
pip install citadel-vector    # Just vector search
pip install citadel-gateway   # Just the LLM gateway
pip install citadel-agents    # Just the agent runtime
```

```python
# Vector search -- zero external dependencies
from citadel_vector import VectorStore

store = VectorStore(path="./my_vectors", dim=384)
store.add("doc_1", embedding=[0.1, 0.2, ...], metadata={"source": "readme.md"})
results = store.search(query_embedding=[0.1, 0.2, ...], k=5)

# LLM gateway -- unified interface across providers
from citadel_gateway import GatewayClient

client = GatewayClient()
response = client.complete(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello, Citadel."}],
)
print(f"Cost: ${response.cost:.4f} | Latency: {response.latency_ms}ms")
```

## Lessons Learned

1. **HNSW tuning is more art than science.** The `ef_construction` and `M` parameters dramatically affect the recall vs. speed tradeoff. Small changes to `M` (e.g., 12 vs. 16) shifted recall by 5-8% on the same dataset. I settled on M=16, ef=200 after benchmarking against multiple embedding distributions.

2. **Semantic cache invalidation is harder than it looks.** My first approach used cosine similarity thresholds to decide if a cached response was "close enough" to a new query. This produced subtle bugs where semantically different questions with similar embeddings returned wrong cached answers. Switched to TTL-based expiry -- simpler, predictable, and avoids serving stale results.

3. **Circuit breaker timing needs real-world calibration.** My initial 30-second timeout was too aggressive for Claude's longer responses on complex prompts. Production-grade timeouts need to account for the tail latency of the slowest provider, not the average case.

## Tests

```bash
# Run all tests
cd packages/citadel-gateway && python -m pytest tests/ -v
cd packages/citadel-vector  && python -m pytest tests/ -v
cd packages/citadel-agents  && python -m pytest tests/ -v
cd packages/citadel-ingest  && python -m pytest tests/ -v
cd packages/citadel-trace   && python -m pytest tests/ -v
```

100 tests across 5 packages covering: HNSW index operations and recall accuracy, gateway routing and failover logic, agent ReAct loop execution, document chunking strategies, trace collection and cost calculation, rate limiter and circuit breaker state transitions.

## License

MIT License. See [LICENSE](LICENSE) for details.
