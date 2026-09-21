# Citadel

A self-hosted AI operations platform that replaces managed LLM infrastructure with a single unified gateway -- multi-provider routing, semantic caching, vector search, agent runtime, and cost observability, all as independent pip-installable packages.

[![CI](https://github.com/dbhavery/citadel/actions/workflows/ci.yml/badge.svg)](https://github.com/dbhavery/citadel/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Packages](https://img.shields.io/badge/packages-6-green.svg)](#packages)
[![Tests](https://img.shields.io/badge/tests-118-brightgreen.svg)](#tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## Why I Built This

Running LLM applications in production means stitching together separate services for routing, caching, vector search, observability, and agent orchestration. Each one adds a dependency, a bill, and a failure mode. Managed vector databases alone run $25-70+/month (Weaviate, Pinecone). Provider lock-in means rewriting integration code when you switch models.

I wanted a single platform where every component is independent, composable, and self-hosted -- with zero mandatory cloud dependencies. So I built one.

## What It Does

- **Unified LLM gateway** with automatic routing across Claude, Gemini, OpenAI, and Ollama -- when a provider errors or its circuit breaker is open, the gateway fails over to the next eligible provider in priority order, eliminating single-provider lock-in
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
| **citadel-gateway** | OpenAI-compatible LLM proxy with regex model routing, SQLite response cache, token-bucket rate limiter, and per-provider circuit breakers with automatic cross-provider failover | Yes |
| **citadel-vector** | HNSW vector search engine (Malkov & Yashunin 2018), persistent storage, metadata filtering, REST API | Yes |
| **citadel-agents** | ReAct agent runtime with `@tool` auto-schema, conversation + vector memory, multi-agent orchestration, YAML definitions | Yes |
| **citadel-ingest** | Document pipeline with 4 chunking strategies, 7 format parsers, SHA-256 deduplication | Yes |
| **citadel-trace** | LLM observability with span/trace model, pricing DB (12 cloud models + local Ollama), auto-instrumentation, cost/latency/token metrics, alert rules | Yes |
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
| Provider lock-in | None -- automatic failover to the next eligible provider |
| Cache savings per hit | $0.01-0.10 depending on model |
| Codebase | 10K+ lines across 6 packages |
| Test coverage | 118 tests across all packages |

## Live Demo

[Citadel Explorer on HuggingFace Spaces](https://huggingface.co/spaces/dbhavery/citadel-explorer) -- interactive demo of the gateway, vector search, and dashboard.

## Quick Start

None of `citadel-ai`, `citadel-vector`, `citadel-gateway`, or `citadel-agents` are published on PyPI. Install a package straight from a clone, in editable mode.

```bash
git clone https://github.com/dbhavery/citadel.git
cd citadel
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e packages/citadel-vector
```

Vector search runs standalone, no server, no external vector database:

```python
import numpy as np
from citadel_vector import VectorStore

store = VectorStore(path="./my_vectors", dim=4)
store.add(np.array([0.1, 0.2, 0.3, 0.4]), "doc_1", metadata={"source": "readme.md"})
store.add(np.array([0.9, 0.1, 0.0, 0.2]), "doc_2", metadata={"source": "other.md"})

results = store.search(np.array([0.1, 0.2, 0.3, 0.4]), k=2)
for doc_id, distance, metadata in results:
    print(f"{doc_id}: distance={distance:.4f} metadata={metadata}")
```

Output, from a run of this exact code against this repo:

```
doc_1: distance=0.0000 metadata={'source': 'readme.md'}
doc_2: distance=0.6259 metadata={'source': 'other.md'}
```

There is no `GatewayClient` class. `citadel-gateway` is a FastAPI app (`citadel_gateway.server.create_app`), reached over its OpenAI-compatible HTTP routes, not imported as a client.

`citadel serve` reads its providers and keys from the environment through `GatewayConfig.from_env()`, so it will make real calls to whatever providers you have configured. The example below instead exercises the same request path with an in-process fake provider, which needs no key and makes no network call.

The example below exercises the real gateway request path, route, circuit breaker, cache, provider call, using an in-process fake provider in place of a real one. It needs no API key and makes no network call:

```bash
pip install -e packages/citadel-gateway
```

```python
from typing import Any

from fastapi.testclient import TestClient

from citadel_gateway.config import GatewayConfig
from citadel_gateway.providers.base import CompletionResponse, Provider
from citadel_gateway.router import Router, RoutingRule
from citadel_gateway.server import create_app


class LocalProvider(Provider):
    async def complete(self, messages: list[dict[str, str]], model: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(content="Hello from Citadel.", model=model, prompt_tokens=6, completion_tokens=4)


config = GatewayConfig(providers={}, cache_enabled=False, rate_limit_enabled=False)
app = create_app(config)
app.state.router = Router(rules=[RoutingRule(pattern=r"local-.*", provider="local", model="{model}", priority=10)])
app.state.providers = {"local": LocalProvider()}

client = TestClient(app)
response = client.post(
    "/v1/chat/completions",
    json={"model": "local-demo", "messages": [{"role": "user", "content": "Hello, Citadel."}]},
)
print(response.status_code)
print(response.json())
```

Output, from a run of this exact code against this repo (the response id and created timestamp differ on every run):

```
200
{'id': 'chatcmpl-3945f97270f0', 'object': 'chat.completion', 'created': 1789951451, 'model': 'local-demo', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Hello from Citadel.'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 6, 'completion_tokens': 4, 'total_tokens': 10}}
```

To route to a real provider instead of `LocalProvider`, use one of the concrete providers in `citadel_gateway/providers/` (Anthropic, OpenAI-compatible, Ollama) and set its API key through `GatewayConfig`.

`citadel-agents` needs `citadel-vector` installed first, for its vector-memory backend:

```bash
pip install -e packages/citadel-vector
pip install -e packages/citadel-agents
```

## Lessons Learned

1. **HNSW tuning is more art than science.** The `ef_construction` and `M` parameters dramatically affect the recall vs. speed tradeoff. Small changes to `M` (e.g., 12 vs. 16) shifted recall by 5-8% on the same dataset. I settled on M=16, ef=200 after benchmarking against multiple embedding distributions.

2. **Semantic cache invalidation is harder than it looks.** My first approach used cosine similarity thresholds to decide if a cached response was "close enough" to a new query. This produced subtle bugs where semantically different questions with similar embeddings returned wrong cached answers. Switched to TTL-based expiry -- simpler, predictable, and avoids serving stale results.

3. **Circuit breaker timing needs real-world calibration.** My initial 30-second timeout was too aggressive for Claude's longer responses on complex prompts. Production-grade timeouts need to account for the tail latency of the slowest provider, not the average case.

## Tests

Each package's test dependencies live in its own `dev` extra. Install that extra from the clone before running its tests. `citadel-agents` needs `citadel-vector` installed too, for its vector-memory backend.

```bash
pip install -e "packages/citadel-vector[dev]"
cd packages/citadel-vector && python -m pytest tests/ -v && cd ../..

pip install -e "packages/citadel-gateway[dev]"
cd packages/citadel-gateway && python -m pytest tests/ -v && cd ../..

pip install -e "packages/citadel-vector[dev]"
pip install -e "packages/citadel-agents[dev]"
cd packages/citadel-agents && python -m pytest tests/ -v && cd ../..

pip install -e "packages/citadel-ingest[dev]"
cd packages/citadel-ingest && python -m pytest tests/ -v && cd ../..

pip install -e "packages/citadel-trace[dev]"
cd packages/citadel-trace && python -m pytest tests/ -v && cd ../..
```

Observed on a fresh clone:

| Package | Tests |
|---|---|
| citadel-gateway | 49 passed |
| citadel-vector | 18 passed |
| citadel-agents | 18 passed |
| citadel-ingest | 15 passed |
| citadel-trace | 18 passed |
| Total | 118 passed |

Coverage: HNSW index operations and recall accuracy, gateway routing and cross-provider failover logic, the end-to-end gateway request path (route -> circuit breaker -> cache -> provider) via `test_server.py`, agent ReAct loop execution, document chunking strategies, trace collection and cost calculation, rate limiter and circuit breaker state transitions.

## License

MIT License. See [LICENSE](LICENSE) for details.
