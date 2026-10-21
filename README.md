# Citadel

A self-hosted gateway and toolkit for apps that call large language models: one OpenAI-compatible endpoint with provider failover, an exact-match response cache and rate limits, plus separate packages for vector search, document ingest, agents and cost tracing.

[![CI](https://github.com/dbhavery/citadel/actions/workflows/ci.yml/badge.svg)](https://github.com/dbhavery/citadel/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## The problem

An app that calls a language model usually needs more than the model. It needs a fallback when the provider has an outage, a cap on how fast each client can send requests, a way to avoid paying twice for the same request, somewhere to keep document embeddings, and a record of what each call cost. Teams often set these up as separate hosted services, each billed and run apart from the others, and the app still stops working when its one provider goes down.

## How it solves it

Citadel puts those pieces in one repo as separate Python packages that can be installed one at a time.

- **Gateway.** Apps send OpenAI-style chat requests to `POST /v1/chat/completions`. Regex rules map a model name to providers (by default `claude-*` to Anthropic, `gemini-*` to Google, `gpt-*` to OpenAI, anything else to Ollama). If a provider raises an error, or its circuit breaker is open, the gateway tries the next configured provider for that model in priority order.
- **Response cache.** The gateway stores responses in SQLite under a SHA-256 hash of the model name and the exact message list. A later request with the same model and the same messages gets the stored response until its time limit runs out (one hour by default). A request that differs by a single character is a miss. There is no similarity matching. Temperature, `max_tokens`, `top_p` and `stop` are not part of the key, and streaming requests skip the cache.
- **Rate limits.** A token bucket per API key (from the `Authorization: Bearer` header) and model, 60 requests per minute by default.
- **Vector search.** An HNSW index written in Python and NumPy, with cosine, Euclidean and dot-product distance, metadata stored in SQLite, and an optional REST API.
- **Document ingest.** Parsers for Markdown, plain text, Python, JavaScript, TypeScript, Go and Rust files, PDF (needs PyPDF2) and Word `.docx` (needs python-docx). Four chunkers: fixed size, sentence, paragraph (the class is named `SemanticChunker`; it splits on blank lines and does not use embeddings) and code (splits on function and class lines). Duplicate chunks are dropped by SHA-256 hash. Embeddings come from Ollama.
- **Agents.** A ReAct loop with tools declared by a `@tool` decorator that builds the schema from the function signature, agents defined in YAML, conversation memory, and an orchestrator that picks an agent by name match or a custom router function. Vector memory uses a hashed bag-of-words vector stored in the HNSW index, not a neural embedding model.
- **Tracing.** Wraps the Anthropic and OpenAI Python clients to record spans with tokens, latency, errors and cost from a built-in price table. Alert rules on daily cost, error rate and latency can post to a webhook.
- **Dashboard.** One static HTML file that reads the gateway and trace APIs and shows sample data when they cannot be reached.

## Who it helps

Small teams that want to run model routing, caching and cost tracking on machines they run, across Claude, OpenAI, Gemini and local Ollama models, and developers who want to read a small HNSW, circuit breaker or ReAct implementation in plain Python.

## Technical details

| Package | Contents |
|---|---|
| `citadel-gateway` | FastAPI app with regex routing, per-provider circuit breakers (open after 5 failures, retry after 30 seconds), token-bucket rate limiter and SQLite response cache. Providers: Anthropic SDK, any OpenAI-compatible API (used for OpenAI and Google), Ollama. |
| `citadel-vector` | HNSW index (M=16, ef_construction=200 by default), persistent storage, metadata filter, optional REST API. |
| `citadel-agents` | ReAct agent runtime, `@tool` decorator, YAML loader, conversation and vector memory, orchestrator. |
| `citadel-ingest` | Parsers, four chunkers, SHA-256 dedup, Ollama embedder. |
| `citadel-trace` | Span and trace model, client instrumentation, price table, cost and latency metrics, alert rules, FastAPI server. |
| `citadel-dashboard` | Single HTML file with inline CSS and JavaScript, no build step. |

The root package `citadel-ai` installs a `citadel` command with `serve`, `ingest`, `search`, `agent`, `traces`, `cost` and `status` subcommands. `docker-compose.yml` runs the gateway (port 8080) and trace server (port 8081) with health checks, and serves the dashboard through nginx on port 3000. The gateway service reads provider keys from a `.env` file; see `.env.example`.

Python 3.10 or newer. None of the packages are published on PyPI; install them from a clone.

### Quick start

```bash
git clone https://github.com/dbhavery/citadel.git
cd citadel
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e packages/citadel-vector
```

Vector search works without a server or an external database:

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

Output from running this code against this repo:

```
doc_1: distance=0.0000 metadata={'source': 'readme.md'}
doc_2: distance=0.6259 metadata={'source': 'other.md'}
```

`citadel-gateway` is a FastAPI app (`citadel_gateway.server.create_app`) that clients reach over HTTP. There is no client class to import. `citadel serve` reads providers and keys from the environment through `GatewayConfig.from_env()` and makes real calls to the providers you configure.

The example below sends one request through the real gateway request path (route, circuit breaker, provider call) using an in-process fake provider. It needs no API key and makes no network call:

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

Output from running this code against this repo (the id and timestamp change on every run):

```
200
{'id': 'chatcmpl-10da83751a7c', 'object': 'chat.completion', 'created': 1792591852, 'model': 'local-demo', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Hello from Citadel.'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 6, 'completion_tokens': 4, 'total_tokens': 10}}
```

To use a real provider in place of `LocalProvider`, use one of the providers in `citadel_gateway/providers/` (Anthropic, OpenAI-compatible, Ollama) and set its key through `GatewayConfig`.

`citadel-agents` needs `citadel-vector` installed first for its vector memory:

```bash
pip install -e packages/citadel-vector
pip install -e packages/citadel-agents
```

### Demo

[Citadel Explorer on Hugging Face Spaces](https://huggingface.co/spaces/dbhavery/citadel-explorer)

### Tests

Each package has a separate test suite and a `dev` extra. CI runs every package's suite on Python 3.13 on each push and pull request to `master`. To run them locally:

```bash
pip install -e "packages/citadel-vector[dev]"
cd packages/citadel-vector && python -m pytest -q && cd ../..

pip install -e "packages/citadel-gateway[dev]"
cd packages/citadel-gateway && python -m pytest -q && cd ../..

pip install -e "packages/citadel-agents[dev]"
cd packages/citadel-agents && python -m pytest -q && cd ../..

pip install -e "packages/citadel-ingest[dev]"
cd packages/citadel-ingest && python -m pytest -q && cd ../..

pip install -e "packages/citadel-trace[dev]"
cd packages/citadel-trace && python -m pytest -q && cd ../..
```

What the tests check:

- **Gateway:** routing rules, failover to the next provider on error and when a circuit is open, circuit breaker state changes, rate limiter, cache hit, miss and expiry, and the full request path through `create_app`.
- **Vector:** distance functions, index add, search and delete, a filter function, save and load, metadata, and a recall test that builds an index of 1,000 random 64-dimension vectors and requires average recall@10 above 0.9 against brute-force search.
- **Agents:** the ReAct loop and its iteration limit, tool registration and schema generation, conversation and vector memory, keyword routing and agent-as-tool delegation. YAML loading has no test.
- **Ingest:** each chunker, Markdown, Python and text parsing, deduplication, directory ingest and search. The PDF and Word parsers have no test.
- **Trace:** spans and traces, cost calculation, latency percentiles, model comparison and alert thresholds.

## License

MIT License. See [LICENSE](LICENSE).
