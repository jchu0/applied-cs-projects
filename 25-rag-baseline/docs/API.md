# RAG Baseline API Documentation

## Overview

The RAG Baseline API is a FastAPI application (`ragbaseline.api`) that exposes
document ingestion, retrieval, and Retrieval-Augmented Generation over a
multi-tenant index. It is created with `create_app(...)` and defaults to a mock
LLM provider, so it runs without any external credentials.

Interactive OpenAPI docs are served by FastAPI at `/docs` (Swagger UI) and
`/redoc`, with the schema at `/openapi.json`.

## Base URL

```
http://localhost:8000
```

There is no `/api/v1` prefix — routes are mounted at the root.

## Authentication

Authentication is **opt-in** and disabled by default. It activates only when the
`API_KEYS` environment variable is set to a comma-separated list of valid keys.
When enabled, every request must present a key using either header form:

```http
Authorization: Bearer <your-api-key>
```

or

```http
X-API-Key: <your-api-key>
```

A missing or invalid key returns `401` with a `WWW-Authenticate: Bearer` header.
The following paths are always open (no key required, exempt from rate limiting):
`/`, `/health`, `/ready`, `/readiness`, `/docs`, `/redoc`, `/openapi.json`.

### Rate limiting and timeouts

Both are opt-in, in-process, and stdlib-only:

- `RATE_LIMIT_PER_MINUTE` — sliding-window limit per API key (or client IP when no
  key is present). Default `120`; set `0` to disable. Exceeding it returns `429`
  with a `Retry-After` header.
- `REQUEST_TIMEOUT_SECONDS` — per-request timeout (default `30`).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | RAG query (retrieve + generate) |
| POST | `/query/stream` | RAG query with SSE streaming |
| POST | `/search` | Retrieval only, no generation |
| POST | `/ingest` | Ingest a document from a file path |
| POST | `/ingest/text` | Ingest raw text content |
| GET | `/tenants` | List all tenant IDs |
| GET | `/tenants/{tenant_id}` | Get tenant information |
| DELETE | `/tenants/{tenant_id}` | Delete a tenant and all its data |
| GET | `/analytics` | Aggregate retrieval analytics |
| GET | `/usage/{tenant_id}` | Usage metrics for a tenant |

---

### Health

#### `GET /health`

Liveness check.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### RAG Query

#### `POST /query`

Retrieve relevant chunks for the given tenant and generate an answer with the
configured LLM provider.

**Request Body:**

```json
{
  "question": "What is the capital of France?",
  "tenant_id": "default",
  "top_k": 5,
  "filter": null,
  "filter_string": null,
  "rerank": false,
  "stream": false
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `question` | string | (required) | Question to answer |
| `tenant_id` | string | `"default"` | Tenant whose index to query |
| `top_k` | int | `5` | Results to retrieve (1–20) |
| `filter` | object \| null | `null` | Metadata filter dict |
| `filter_string` | string \| null | `null` | Filter expression parsed by `MetadataFilter`; overrides `filter` when set |
| `rerank` | bool | `false` | Enable reranking |
| `stream` | bool | `false` | Present for parity; use `/query/stream` for streaming |

**Response:**

```json
{
  "answer": "The capital of France is Paris.",
  "sources": [
    {
      "id": "chunk-abc-123",
      "content": "France's capital is Paris...",
      "score": 0.92,
      "metadata": { "source": "geography.txt" }
    }
  ],
  "query": "What is the capital of France?",
  "latency_ms": 12.4
}
```

`sources` entries are the serialized retrieved chunks (`to_dict()`); exact keys
depend on the chunk/document schema.

**Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is the capital of France?", "top_k": 3},
)
data = response.json()
print(data["answer"])
```

#### `POST /query/stream`

Same request body as `/query`. Returns a `text/event-stream` (Server-Sent
Events) response. Each event carries a JSON `content` fragment, and the stream
terminates with a `[DONE]` sentinel:

```
data: {"content": "The capital "}

data: {"content": "of France is Paris."}

data: [DONE]
```

```python
import requests

with requests.post(
    "http://localhost:8000/query/stream",
    json={"question": "Explain RAG in one sentence."},
    stream=True,
) as r:
    for line in r.iter_lines():
        if line:
            print(line.decode())
```

---

### Search

#### `POST /search`

Retrieval only — returns matching chunks without LLM generation.

**Request Body:**

```json
{
  "query": "artificial intelligence",
  "tenant_id": "default",
  "top_k": 5,
  "filter": null,
  "hybrid": false,
  "alpha": 0.5
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `query` | string | (required) | Search query |
| `tenant_id` | string | `"default"` | Tenant whose index to search |
| `top_k` | int | `5` | Results to return (1–50) |
| `filter` | object \| null | `null` | Metadata filter dict |
| `hybrid` | bool | `false` | Request hybrid (vector + keyword) search |
| `alpha` | float | `0.5` | Vector weight for hybrid search (0–1) |

**Response:**

```json
{
  "results": [
    {
      "id": "chunk-abc-123",
      "content": "Artificial Intelligence (AI) is...",
      "score": 0.92,
      "metadata": { "source": "research_paper" }
    }
  ],
  "latency_ms": 8.1
}
```

---

### Document Ingestion

#### `POST /ingest`

Ingest a document from a file path readable by the server. Supported formats
include plain text, and PDF/HTML/Markdown when their optional dependencies are
installed.

**Request Body:**

```json
{
  "tenant_id": "default",
  "file_path": "/data/docs/report.pdf",
  "metadata": { "source": "upload" }
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `tenant_id` | string | `"default"` | Target tenant |
| `file_path` | string | (required) | Server-side path to the file |
| `metadata` | object \| null | `null` | Extra metadata merged into the document |

**Response:**

```json
{
  "status": "success",
  "document_id": "doc-abc-123",
  "chunks": 12
}
```

Returns `404` if the file does not exist, `500` if parsing/indexing fails.

#### `POST /ingest/text`

Ingest raw text content directly, without a file.

**Request Body:**

```json
{
  "tenant_id": "default",
  "content": "Artificial Intelligence is transforming industries...",
  "metadata": { "topic": "ai" },
  "source": "manual"
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `tenant_id` | string | `"default"` | Target tenant |
| `content` | string | (required) | Text to ingest |
| `metadata` | object \| null | `null` | Document metadata |
| `source` | string | `""` | Source identifier |

**Response:**

```json
{
  "status": "success",
  "document_id": "doc-abc-123",
  "chunks": 3
}
```

---

### Tenants

#### `GET /tenants`

List all tenant IDs.

**Response:**

```json
["default", "acme", "globex"]
```

#### `GET /tenants/{tenant_id}`

Get information about a single tenant.

**Response:**

```json
{
  "tenant_id": "acme",
  "document_count": 42,
  "created_at": "2026-01-15T10:30:00Z"
}
```

#### `DELETE /tenants/{tenant_id}`

Permanently delete a tenant and all of its data.

**Response:**

```json
{
  "status": "deleted",
  "tenant_id": "acme"
}
```

---

### Analytics and Usage

#### `GET /analytics`

Aggregate retrieval analytics across tenants, derived from the query logs.

**Response:**

```json
{
  "total_queries": 1523,
  "tenants": 3,
  "latency": {
    "p50": 12.4,
    "p95": 48.9
  }
}
```

`latency` may be `null` when no queries have been logged yet.

#### `GET /usage/{tenant_id}`

Usage metrics tracked by `UsageTracker` for a single tenant (e.g. query and
document counts). The exact shape is whatever the tracker records for the tenant.

```json
{
  "queries": 128,
  "documents": 42
}
```

---

## Error Handling

Errors use FastAPI's standard response shape:

```json
{
  "detail": "File not found: /data/docs/missing.pdf"
}
```

Validation errors (malformed request bodies) return `422` with FastAPI's
per-field `detail` array.

### Common status codes

| HTTP Status | When |
|-------------|------|
| `401` | Auth enabled and API key missing/invalid |
| `404` | Resource not found (e.g. ingest file path) |
| `422` | Request body failed validation |
| `429` | Rate limit exceeded (when enabled) |
| `500` | Ingestion or internal error |

## cURL Examples

```bash
# Ingest raw text
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"content": "AI is transforming industries...", "source": "manual"}'

# RAG query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?", "top_k": 3}'

# With auth enabled (API_KEYS set on the server)
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?"}'
```
