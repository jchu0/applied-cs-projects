# RAG Baseline API Documentation

## Overview

The RAG Baseline API provides a comprehensive interface for document indexing, retrieval, and question answering using Retrieval-Augmented Generation. The API is built with FastAPI and supports both REST and WebSocket protocols.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

All API requests require authentication using API keys.

```http
Authorization: Bearer <your-api-key>
```

## API Endpoints

### Document Management

#### Index Documents

```http
POST /index/documents
```

Indexes one or more documents into the RAG system.

**Request Body:**

```json
{
  "documents": [
    {
      "id": "doc-123",
      "content": "Document text content...",
      "metadata": {
        "source": "manual_upload",
        "title": "Document Title",
        "author": "John Doe",
        "tags": ["ai", "machine-learning"]
      }
    }
  ],
  "chunking_strategy": "semantic",
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

**Response:**

```json
{
  "status": "success",
  "indexed_count": 1,
  "documents": [
    {
      "id": "doc-123",
      "chunks_created": 5,
      "embedding_time": 0.234,
      "status": "indexed"
    }
  ],
  "total_time": 0.456
}
```

**Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/index/documents",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "documents": [
            {
                "id": "doc-001",
                "content": "Artificial Intelligence is transforming industries...",
                "metadata": {"source": "research_paper"}
            }
        ]
    }
)
```

#### Upload Document File

```http
POST /index/upload
```

Upload and index a document file (PDF, TXT, MD, HTML).

**Request:** Multipart form data

```python
files = {"file": open("document.pdf", "rb")}
data = {"metadata": json.dumps({"source": "upload"})}
```

**Response:**

```json
{
  "status": "success",
  "document_id": "doc-abc-123",
  "filename": "document.pdf",
  "chunks_created": 12,
  "processing_time": 1.234
}
```

#### Get Document

```http
GET /index/documents/{document_id}
```

Retrieve a specific document's information.

**Response:**

```json
{
  "id": "doc-123",
  "metadata": {
    "source": "manual_upload",
    "title": "Document Title",
    "indexed_at": "2024-01-15T10:30:00Z"
  },
  "chunk_count": 5,
  "status": "active"
}
```

#### Delete Document

```http
DELETE /index/documents/{document_id}
```

Remove a document from the index.

**Response:**

```json
{
  "status": "success",
  "message": "Document deleted successfully",
  "chunks_removed": 5
}
```

#### List Documents

```http
GET /index/documents
```

List all indexed documents with pagination.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page (default: 20)
- `filter` (json): Metadata filter

**Response:**

```json
{
  "documents": [
    {
      "id": "doc-123",
      "metadata": {...},
      "chunk_count": 5,
      "indexed_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 20
}
```

### Search and Retrieval

#### Search Documents

```http
POST /index/search
```

Search for relevant documents using vector similarity.

**Request Body:**

```json
{
  "query": "What is artificial intelligence?",
  "k": 5,
  "score_threshold": 0.7,
  "filter": {
    "source": "research_paper"
  }
}
```

**Response:**

```json
{
  "results": [
    {
      "id": "chunk-abc-123",
      "document_id": "doc-123",
      "content": "Artificial Intelligence (AI) is...",
      "score": 0.92,
      "metadata": {
        "source": "research_paper",
        "page": 5
      }
    }
  ],
  "query": "What is artificial intelligence?",
  "search_time": 0.145
}
```

#### Hybrid Search

```http
POST /index/hybrid-search
```

Perform hybrid search combining vector and keyword search.

**Request Body:**

```json
{
  "query": "machine learning algorithms",
  "vector_weight": 0.7,
  "keyword_weight": 0.3,
  "k": 10,
  "rerank": true
}
```

**Response:**

```json
{
  "results": [...],
  "vector_results_count": 8,
  "keyword_results_count": 6,
  "combined_count": 10,
  "search_time": 0.234
}
```

### RAG Query

#### Standard Query

```http
POST /rag/query
```

Perform a RAG query to get an AI-generated answer.

**Request Body:**

```json
{
  "query": "What are the main applications of AI?",
  "max_tokens": 500,
  "temperature": 0.7,
  "retrieval_config": {
    "k": 5,
    "score_threshold": 0.7
  },
  "include_sources": true
}
```

**Response:**

```json
{
  "query": "What are the main applications of AI?",
  "answer": "The main applications of AI include:\n\n1. Healthcare: AI is used for disease diagnosis, drug discovery...",
  "sources": [
    {
      "id": "chunk-123",
      "document_id": "doc-456",
      "content": "AI applications in healthcare...",
      "score": 0.89,
      "metadata": {...}
    }
  ],
  "metadata": {
    "retrieval_time": 0.145,
    "generation_time": 1.234,
    "total_time": 1.379,
    "tokens_used": 245,
    "model": "gpt-3.5-turbo"
  }
}
```

**Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/rag/query",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "query": "What is deep learning?",
        "max_tokens": 300,
        "include_sources": True
    }
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Sources: {len(data['sources'])} documents")
```

#### Streaming Query

```http
GET /rag/stream
```

Get streaming responses for real-time generation.

**WebSocket Connection:**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/rag/stream');

ws.onopen = () => {
    ws.send(JSON.stringify({
        query: "Explain quantum computing",
        max_tokens: 500
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'chunk') {
        // Append to response
        console.log(data.content);
    } else if (data.type === 'sources') {
        // Handle sources
        console.log('Sources:', data.sources);
    } else if (data.type === 'done') {
        // Complete
        console.log('Generation complete');
    }
};
```

#### Batch Query

```http
POST /rag/batch
```

Process multiple queries in batch.

**Request Body:**

```json
{
  "queries": [
    "What is AI?",
    "How does machine learning work?",
    "What are neural networks?"
  ],
  "max_tokens": 300,
  "parallel": true
}
```

**Response:**

```json
{
  "results": [
    {
      "query": "What is AI?",
      "answer": "...",
      "sources": [...]
    },
    {
      "query": "How does machine learning work?",
      "answer": "...",
      "sources": [...]
    }
  ],
  "total_time": 2.456,
  "parallel_execution": true
}
```

### Feedback and Analytics

#### Submit Feedback

```http
POST /feedback
```

Submit feedback on a RAG response.

**Request Body:**

```json
{
  "query_id": "query-123",
  "rating": 4,
  "relevant": true,
  "feedback": "The answer was helpful but could be more detailed",
  "selected_sources": ["chunk-123", "chunk-456"]
}
```

**Response:**

```json
{
  "status": "success",
  "feedback_id": "feedback-789",
  "message": "Feedback recorded successfully"
}
```

#### Get Analytics

```http
GET /analytics/summary
```

Get usage analytics and performance metrics.

**Query Parameters:**
- `start_date`: Start date (ISO format)
- `end_date`: End date (ISO format)
- `granularity`: hour|day|week|month

**Response:**

```json
{
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "metrics": {
    "total_queries": 1523,
    "unique_users": 87,
    "avg_response_time": 1.234,
    "avg_tokens_per_query": 267,
    "retrieval_accuracy": 0.85,
    "user_satisfaction": 4.2
  },
  "top_queries": [...],
  "error_rate": 0.02
}
```

### Configuration

#### Get Configuration

```http
GET /config
```

Get current system configuration.

**Response:**

```json
{
  "embedding_model": "text-embedding-ada-002",
  "llm_model": "gpt-3.5-turbo",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "retrieval_k": 5,
  "vector_store": "chroma",
  "cache_enabled": true
}
```

#### Update Configuration

```http
PUT /config
```

Update system configuration (admin only).

**Request Body:**

```json
{
  "chunk_size": 1024,
  "retrieval_k": 10,
  "cache_enabled": false
}
```

**Response:**

```json
{
  "status": "success",
  "message": "Configuration updated",
  "updated_fields": ["chunk_size", "retrieval_k", "cache_enabled"]
}
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request body is invalid",
    "details": {
      "field": "query",
      "issue": "Query cannot be empty"
    }
  },
  "request_id": "req-123-456"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Missing or invalid API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Rate Limiting

API requests are rate-limited per API key:

- **Standard tier**: 100 requests/minute
- **Premium tier**: 1000 requests/minute
- **Enterprise tier**: Unlimited

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642521600
```

## Pagination

List endpoints support pagination using standard parameters:

```http
GET /api/v1/index/documents?page=2&page_size=50
```

Paginated responses include:

```json
{
  "data": [...],
  "pagination": {
    "total": 523,
    "page": 2,
    "page_size": 50,
    "total_pages": 11,
    "has_next": true,
    "has_prev": true
  }
}
```

## Webhooks

Configure webhooks to receive real-time notifications:

```http
POST /webhooks
```

**Request Body:**

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["document.indexed", "query.completed"],
  "secret": "webhook-secret-key"
}
```

### Webhook Events

- `document.indexed`: Document successfully indexed
- `document.deleted`: Document removed from index
- `query.completed`: RAG query completed
- `error.occurred`: Error during processing

## SDK Examples

### Python SDK

```python
from ragbaseline import RAGClient

# Initialize client
client = RAGClient(api_key="your-api-key")

# Index document
doc_id = client.index_document(
    content="Document content...",
    metadata={"source": "upload"}
)

# Perform query
response = client.query(
    "What is AI?",
    k=5,
    include_sources=True
)

print(response.answer)
for source in response.sources:
    print(f"- {source.content[:100]}...")
```

### JavaScript SDK

```javascript
import { RAGClient } from 'ragbaseline-js';

// Initialize client
const client = new RAGClient({ apiKey: 'your-api-key' });

// Index document
const docId = await client.indexDocument({
    content: 'Document content...',
    metadata: { source: 'upload' }
});

// Perform query
const response = await client.query('What is AI?', {
    k: 5,
    includeSources: true
});

console.log(response.answer);
```

### cURL Examples

```bash
# Index a document
curl -X POST http://localhost:8000/api/v1/index/documents \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "content": "AI is transforming industries...",
      "metadata": {"source": "manual"}
    }]
  }'

# Perform RAG query
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "max_tokens": 300
  }'
```

## Best Practices

1. **Batch Operations**: Use batch endpoints for multiple documents/queries
2. **Caching**: Enable caching for frequently asked questions
3. **Filtering**: Use metadata filters to improve retrieval precision
4. **Monitoring**: Track API usage and performance metrics
5. **Error Handling**: Implement retry logic with exponential backoff
6. **Security**: Rotate API keys regularly and use HTTPS