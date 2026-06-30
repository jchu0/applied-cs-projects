# Model Routing Layer

Intelligent model routing gateway with load balancing, rate limiting, and cost optimization for LLM APIs.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Cohere, self-hosted models
- **Load Balancing**: Round-robin, least-connections, weighted strategies
- **Rate Limiting**: Per-user, per-API-key, global limits
- **Cost Tracking**: Real-time cost monitoring and budgets
- **Caching**: Semantic caching for repeated queries
- **Failover**: Automatic fallback between providers
- **Observability**: Prometheus metrics, Jaeger tracing

## Installation

```bash
pip install -e ".[full]"
```

## Quick Start

```python
from modelrouter import Router, ProviderConfig

# Configure providers
router = Router(
    providers=[
        ProviderConfig("openai", api_key="..."),
        ProviderConfig("anthropic", api_key="..."),
    ],
    strategy="least_cost"
)

# Route request to optimal provider
response = router.complete(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

## API Server

```bash
# Start the routing gateway
uvicorn modelrouter.api:app --host 0.0.0.0 --port 8000

# Use as drop-in replacement for OpenAI API
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"model": "auto", "messages": [...]}'
```

## Infrastructure

```bash
docker-compose up -d
# Redis (rate limiting): localhost:6379
# PostgreSQL (quotas): localhost:5432
```

## Configuration

See `.env.example` for all configuration options including:
- Provider API keys
- Rate limits and quotas
- Cost controls
- Caching settings

## Testing

```bash
pytest tests/ -v  # 106 tests
```
