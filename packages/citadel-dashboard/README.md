# Citadel Dashboard

Single-page dashboard for the Citadel AI Operations Platform. Built as a static HTML file with inline CSS and vanilla JavaScript -- no build step required.

## Quick Start

### Option 1: npx serve (recommended)

```bash
cd packages/citadel-dashboard
npx serve . -l 3000
```

Then open http://localhost:3000

### Option 2: Python

```bash
cd packages/citadel-dashboard
python -m http.server 3000
```

### Option 3: Docker (via root docker-compose)

```bash
# From the repo root
docker compose up dashboard
```

The dashboard is served via nginx on port 3000.

### Option 4: Any static file server

Drop `index.html` into any static hosting (nginx, Apache, S3, Vercel, Netlify). It has zero external dependencies.

## Configuration

The dashboard connects to the Citadel Gateway API. By default it targets `http://localhost:8080`. You can change this in the top bar's URL input field at runtime.

### API Endpoints Used

| Endpoint | Description |
|---|---|
| `GET /health` | Connection health check |
| `GET /traces` | Trace list |
| `GET /traces/{id}` | Trace detail with span tree |
| `GET /metrics/cost` | Daily cost breakdown |
| `GET /metrics/latency` | Latency percentiles |
| `GET /metrics/tokens` | Token usage data |
| `GET /metrics/models` | Per-model comparison |
| `GET /alerts` | Alert rules and history |

## Demo Mode

When the API is unreachable, the dashboard automatically renders demo data so you can see what the UI looks like. Once a real Citadel Gateway is running, it switches to live data.

## Pages

- **Overview** -- stat cards, cost chart, recent traces, model usage breakdown
- **Traces** -- searchable/filterable trace list with span tree detail panel
- **Agents** -- agent monitoring (coming soon)
- **Data** -- vector store monitoring (coming soon)
- **Models** -- per-model request count, latency, cost, and error rate comparison
- **Alerts** -- configured alert rules and triggered alert history
- **Settings** -- provider API keys, cache, rate limiting, alert thresholds
