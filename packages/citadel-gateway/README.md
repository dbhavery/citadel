# Citadel Gateway

An OpenAI-compatible reverse proxy that routes LLM requests to multiple providers (Claude, Gemini, Ollama, OpenAI) with smart routing, response caching, token-bucket rate limiting, and circuit-breaker failover. Drop it in front of any OpenAI SDK client and it just works.

## Quick Start

```bash
pip install citadel-gateway
citadel-gateway
```

The server starts on `http://0.0.0.0:8080` and is immediately usable with any OpenAI-compatible client.

## Configuration

All settings come from environment variables:

| Variable | Default | Description |
|---|---|---|
| `GATEWAY_HOST` | `0.0.0.0` | Bind address |
| `GATEWAY_PORT` | `8080` | Bind port |
| `ANTHROPIC_API_KEY` | — | Enables the Anthropic/Claude provider |
| `OPENAI_API_KEY` | — | Enables the OpenAI provider |
| `GOOGLE_API_KEY` | — | Enables the Google/Gemini provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (always enabled) |
| `GATEWAY_CACHE_ENABLED` | `true` | Enable/disable response caching |
| `GATEWAY_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `GATEWAY_RATE_LIMIT_ENABLED` | `true` | Enable/disable rate limiting |
| `GATEWAY_RATE_LIMIT_RPM` | `60` | Requests per minute per key |

## Routing Rules

Model names are matched by regex to providers:

| Pattern | Provider | Example |
|---|---|---|
| `claude-*` | Anthropic | `claude-sonnet-4-20250514` |
| `gemini-*` | Google | `gemini-2.5-flash` |
| `gpt-*` | OpenAI | `gpt-4o` |
| `*` (fallback) | Ollama | `qwen3:8b`, `llama3` |

Custom rules can be loaded from a YAML file via `GATEWAY_ROUTING_RULES_PATH`.

## API Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible chat completions
- `GET /v1/models` — List available models across all providers
- `GET /health` — Health check with provider and cache status

## API Compatibility

The gateway is a drop-in replacement for the OpenAI API. Point any OpenAI SDK client at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```
