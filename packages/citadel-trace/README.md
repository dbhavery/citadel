# citadel-trace

LLM observability library. Auto-instruments the Anthropic, OpenAI, and Google GenAI SDKs to capture traces, track costs, measure latency, and alert on anomalies.

## Install

```bash
pip install citadel-trace
# With optional REST API server:
pip install citadel-trace[server]
# With webhook alerts:
pip install citadel-trace[alerts]
```

## Quick Start

### Auto-instrumentation

Drop two lines into your application to start capturing every LLM call automatically:

```python
from citadel_trace import TraceCollector, Instrumentor

collector = TraceCollector(db_path="./traces.db")
instrumentor = Instrumentor(collector)
instrumentor.instrument_all()

# Now every call to anthropic, openai, or ollama is traced automatically.
# Use the SDKs as normal -- citadel-trace captures everything in the background.
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)

# When done, restore original methods:
instrumentor.uninstrument_all()
```

### Manual Spans

```python
from citadel_trace import TraceCollector

collector = TraceCollector(db_path="./traces.db")

span = collector.start_span(
    name="chat.completion",
    kind="llm",
    model="claude-sonnet-4-6",
    provider="anthropic",
)

# ... do work ...

collector.end_span(
    span,
    output="The response text",
    tokens={"input": 150, "output": 42},
)
```

### Cost Tracking

```python
from citadel_trace import calculate_cost

cost = calculate_cost("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
print(f"Cost: ${cost:.4f}")  # Cost: $0.0105
```

### Metrics

```python
from citadel_trace import TraceCollector, MetricsCalculator

collector = TraceCollector(db_path="./traces.db")
metrics = MetricsCalculator(collector)

print(metrics.cost_summary(days=7))
print(metrics.latency_percentiles())
print(metrics.token_usage(days=7))
print(metrics.error_rate(days=7))
print(metrics.model_comparison())
```

### Alerts

```python
from citadel_trace import TraceCollector, AlertManager, AlertRule

collector = TraceCollector(db_path="./traces.db")
alerts = AlertManager(collector)

alerts.add_rule(AlertRule(
    name="High daily cost",
    metric="daily_cost",
    threshold=5.0,
))

alerts.add_rule(AlertRule(
    name="High error rate",
    metric="error_rate",
    threshold=0.05,
))

triggered = alerts.check_rules()
for alert in triggered:
    print(f"ALERT: {alert['rule_name']} -- current value: {alert['current_value']}")
```

### REST API Server

```bash
citadel-trace serve --port 8100 --db ./traces.db
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/traces` | List recent traces |
| GET | `/traces/{trace_id}` | Get full trace with spans |
| GET | `/metrics/cost` | Cost summary |
| GET | `/metrics/latency` | Latency percentiles |
| GET | `/metrics/tokens` | Token usage |
| GET | `/metrics/models` | Model comparison |
| GET | `/alerts` | Check alert rules |

## Supported Providers

| Provider | Auto-instrumented | Method |
|----------|-------------------|--------|
| Anthropic | Yes | Patches `messages.create` |
| OpenAI | Yes | Patches `chat.completions.create` |
| Ollama | Yes | Patches httpx to intercept `:11434` calls |
| Google GenAI | Manual | Use `start_span` / `end_span` |

## Pricing

Built-in pricing for current models (per 1M tokens):

| Model | Input | Output |
|-------|-------|--------|
| claude-sonnet-4-6 | $3.00 | $15.00 |
| claude-opus-4-6 | $15.00 | $75.00 |
| claude-haiku-4-5 | $0.80 | $4.00 |
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gemini-2.5-flash | $0.15 | $0.60 |
| gemini-2.5-pro | $1.25 | $10.00 |
| ollama/* | $0.00 | $0.00 |

Unknown models default to $0.00. Add custom pricing by updating `MODEL_PRICING`.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
