# Distributed Job Queue - API Documentation

## Base URL

```
http://localhost:8000
```

The API server runs on port 8000 by default. Interactive documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Authentication

API key authentication can be configured for production deployments:

```http
Authorization: Bearer <api_key>
```

**Note:** Authentication is optional in development mode.

## Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks` | POST | Submit a new task |
| `/tasks/{task_id}` | GET | Get task status |
| `/tasks/{task_id}/result` | GET | Get task result |
| `/tasks/{task_id}` | DELETE | Cancel a task |
| `/tasks/{task_id}/retry` | POST | Retry a failed task |
| `/tasks/bulk` | POST | Submit multiple tasks |
| `/queues` | GET | List all queues |
| `/queues/stats` | GET | Get queue statistics |
| `/queues/{queue_name}/pause` | POST | Pause a queue |
| `/queues/{queue_name}/resume` | POST | Resume a queue |
| `/queues/{queue_name}/purge` | POST | Purge queue tasks |
| `/workers` | GET | List active workers |
| `/workers/{worker_id}` | GET | Get worker details |
| `/workers/{worker_id}/stop` | POST | Stop a worker |
| `/schedule` | GET/POST | Manage scheduled tasks |
| `/schedule/{schedule_id}` | PUT/DELETE | Update/delete schedule |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/dlq` | GET | List dead letter queue tasks |
| `/dlq/{task_id}/resubmit` | POST | Resubmit DLQ task |
| `/dlq/purge` | POST | Purge DLQ |

## Endpoints

### Task Management

#### Submit Task

Create a new task for asynchronous processing.

**Endpoint:** `POST /tasks`

**Request Body:**
```json
{
  "name": "send_email",
  "queue": "default",
  "payload": {
    "to": "user@example.com",
    "subject": "Hello",
    "body": "Task processed successfully"
  },
  "priority": 5,
  "timeout": 30,
  "max_retries": 3,
  "retry_delay": 5.0
}
```

**Parameters:**
- `name` (required): Task type/handler name
- `queue` (optional): Target queue name (default: "default")
- `payload` (optional): Task data as JSON object
- `priority` (optional): Priority level 0-10 (default: 5)
- `timeout` (optional): Task timeout in seconds (default: 30)
- `max_retries` (optional): Maximum retry attempts (default: 3)
- `retry_delay` (optional): Delay between retries in seconds (default: 5.0)

**Response:**
```json
{
  "id": "task-7f3a4b5c",
  "name": "send_email",
  "queue": "default",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z",
  "priority": 5
}
```

**Status Codes:**
- `201`: Task created successfully
- `400`: Invalid request parameters
- `401`: Unauthorized
- `422`: Validation error

**Example:**
```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "process_image",
    "payload": {"image_url": "https://example.com/image.jpg"},
    "priority": 2
  }'
```

**Note:** Priority uses the TaskPriority enum values: CRITICAL (0), HIGH (1), NORMAL (2), LOW (3).

#### Get Task Status

Retrieve detailed information about a specific task.

**Endpoint:** `GET /tasks/{task_id}`

**Response:**
```json
{
  "id": "task-7f3a4b5c",
  "name": "send_email",
  "queue": "default",
  "status": "completed",
  "payload": {
    "to": "user@example.com",
    "subject": "Hello"
  },
  "result": {
    "message_id": "msg-123456",
    "sent_at": "2024-01-15T10:31:00Z"
  },
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:30Z",
  "completed_at": "2024-01-15T10:31:00Z",
  "worker_id": "worker-abc123",
  "priority": 5,
  "retry_count": 0,
  "execution_time": 0.5
}
```

**Status Values:**
- `pending`: Task waiting in queue
- `running`: Currently being processed
- `completed`: Successfully completed
- `failed`: Failed after all retries
- `cancelled`: Manually cancelled
- `retrying`: Failed but will retry

**Example:**
```bash
curl http://localhost:8000/tasks/task-7f3a4b5c
```

#### List Tasks

Query tasks with filtering and pagination.

**Endpoint:** `GET /tasks`

**Query Parameters:**
- `queue`: Filter by queue name
- `status`: Filter by task status
- `name`: Filter by task name
- `worker_id`: Filter by worker
- `created_after`: Tasks created after timestamp
- `created_before`: Tasks created before timestamp
- `limit`: Maximum results (default: 20, max: 100)
- `offset`: Skip first N results
- `sort`: Sort field (created_at, priority, status)
- `order`: Sort order (asc, desc)

**Response:**
```json
{
  "tasks": [
    {
      "id": "task-7f3a4b5c",
      "name": "send_email",
      "queue": "default",
      "status": "completed",
      "created_at": "2024-01-15T10:30:00Z",
      "priority": 5
    }
  ],
  "total": 150,
  "limit": 20,
  "offset": 0,
  "has_more": true
}
```

**Example:**
```bash
curl "http://localhost:8000/tasks?queue=priority&status=pending&limit=10"
```

#### Cancel Task

Cancel a pending or running task.

**Endpoint:** `POST /tasks/{task_id}/cancel`

**Response:**
```json
{
  "task_id": "task-7f3a4b5c",
  "message": "Task cancelled successfully",
  "previous_status": "pending",
  "cancelled_at": "2024-01-15T10:35:00Z"
}
```

**Status Codes:**
- `200`: Task cancelled successfully
- `400`: Task cannot be cancelled (already completed/failed)
- `404`: Task not found

#### Retry Failed Task

Retry a failed task with the same parameters.

**Endpoint:** `POST /tasks/{task_id}/retry`

**Request Body (optional):**
```json
{
  "priority": 10,
  "delay": 0
}
```

**Response:**
```json
{
  "original_task_id": "task-7f3a4b5c",
  "new_task_id": "task-8g4b5d6e",
  "message": "Task resubmitted for retry"
}
```

#### Bulk Submit Tasks

Submit multiple tasks in a single request.

**Endpoint:** `POST /tasks/bulk`

**Request Body:**
```json
{
  "tasks": [
    {
      "name": "send_email",
      "payload": {"to": "user1@example.com"},
      "priority": 5
    },
    {
      "name": "send_email",
      "payload": {"to": "user2@example.com"},
      "priority": 5
    }
  ]
}
```

**Response:**
```json
{
  "submitted": 2,
  "task_ids": ["task-abc123", "task-def456"],
  "errors": []
}
```

### Queue Management

#### Get Queue Statistics

Retrieve statistics for all queues or specific queue.

**Endpoint:** `GET /queues/stats`

**Query Parameters:**
- `queue`: Specific queue name (optional)

**Response:**
```json
{
  "default": {
    "pending": 45,
    "running": 10,
    "completed": 1523,
    "failed": 12,
    "avg_wait_time": 2.5,
    "avg_execution_time": 5.3,
    "throughput": 120.5
  },
  "priority": {
    "pending": 5,
    "running": 3,
    "completed": 234,
    "failed": 2,
    "avg_wait_time": 0.8,
    "avg_execution_time": 3.2,
    "throughput": 45.2
  }
}
```

#### Pause Queue

Temporarily stop processing tasks from a queue.

**Endpoint:** `POST /queues/{queue_name}/pause`

**Response:**
```json
{
  "queue": "default",
  "message": "Queue paused successfully",
  "pending_tasks": 45
}
```

#### Resume Queue

Resume processing tasks from a paused queue.

**Endpoint:** `POST /queues/{queue_name}/resume`

**Response:**
```json
{
  "queue": "default",
  "message": "Queue resumed successfully",
  "pending_tasks": 45
}
```

#### Purge Queue

Remove all pending tasks from a queue.

**Endpoint:** `POST /queues/{queue_name}/purge`

**Request Body (optional):**
```json
{
  "status": ["pending", "failed"],
  "older_than": "2024-01-01T00:00:00Z"
}
```

**Response:**
```json
{
  "queue": "default",
  "message": "Queue purged successfully",
  "tasks_removed": 45
}
```

### Worker Management

#### List Workers

Get information about active workers.

**Endpoint:** `GET /workers`

**Query Parameters:**
- `status`: Filter by status (active, idle, offline)
- `queue`: Filter by queue subscription

**Response:**
```json
{
  "workers": [
    {
      "id": "worker-abc123",
      "hostname": "worker-01.example.com",
      "pid": 12345,
      "status": "active",
      "queues": ["default", "priority"],
      "started_at": "2024-01-15T08:00:00Z",
      "last_heartbeat": "2024-01-15T10:40:00Z",
      "current_task_id": "task-xyz789",
      "tasks_processed": 523,
      "tasks_failed": 5,
      "cpu_percent": 45.2,
      "memory_mb": 256
    }
  ],
  "total": 5,
  "active": 3,
  "idle": 2
}
```

#### Get Worker Details

Get detailed information about a specific worker.

**Endpoint:** `GET /workers/{worker_id}`

**Response:**
```json
{
  "id": "worker-abc123",
  "hostname": "worker-01.example.com",
  "pid": 12345,
  "status": "active",
  "queues": ["default", "priority"],
  "configuration": {
    "concurrency": 10,
    "poll_interval": 1.0,
    "heartbeat_interval": 5.0,
    "circuit_breaker_enabled": true
  },
  "started_at": "2024-01-15T08:00:00Z",
  "last_heartbeat": "2024-01-15T10:40:00Z",
  "current_tasks": [
    {
      "task_id": "task-xyz789",
      "name": "process_image",
      "started_at": "2024-01-15T10:39:30Z"
    }
  ],
  "statistics": {
    "tasks_processed": 523,
    "tasks_failed": 5,
    "average_execution_time": 3.2,
    "uptime_seconds": 9600
  },
  "resources": {
    "cpu_percent": 45.2,
    "memory_mb": 256,
    "threads": 15
  }
}
```

#### Stop Worker

Gracefully stop a worker.

**Endpoint:** `POST /workers/{worker_id}/stop`

**Request Body (optional):**
```json
{
  "wait_for_tasks": true,
  "timeout": 30
}
```

**Response:**
```json
{
  "worker_id": "worker-abc123",
  "message": "Worker stop signal sent",
  "pending_tasks": 2
}
```

### Scheduling

#### Create Schedule

Create a recurring task schedule.

**Endpoint:** `POST /schedule`

**Request Body:**
```json
{
  "name": "daily_report",
  "task_name": "generate_report",
  "queue": "scheduled",
  "payload": {
    "report_type": "daily"
  },
  "cron": "0 9 * * *",
  "timezone": "America/New_York",
  "enabled": true,
  "max_instances": 1
}
```

**Alternative with interval:**
```json
{
  "name": "health_check",
  "task_name": "check_system_health",
  "interval": 300,
  "enabled": true
}
```

**Response:**
```json
{
  "schedule_id": "schedule-def789",
  "name": "daily_report",
  "cron": "0 9 * * *",
  "next_run": "2024-01-16T09:00:00Z",
  "created_at": "2024-01-15T10:45:00Z"
}
```

#### List Schedules

Get all scheduled tasks.

**Endpoint:** `GET /schedule`

**Response:**
```json
{
  "schedules": [
    {
      "id": "schedule-def789",
      "name": "daily_report",
      "task_name": "generate_report",
      "cron": "0 9 * * *",
      "enabled": true,
      "next_run": "2024-01-16T09:00:00Z",
      "last_run": "2024-01-15T09:00:00Z",
      "total_runs": 30,
      "failed_runs": 1
    }
  ],
  "total": 5
}
```

#### Update Schedule

Modify an existing schedule.

**Endpoint:** `PUT /schedule/{schedule_id}`

**Request Body:**
```json
{
  "enabled": false,
  "cron": "0 10 * * *"
}
```

#### Delete Schedule

Remove a scheduled task.

**Endpoint:** `DELETE /schedule/{schedule_id}`

**Response:**
```json
{
  "schedule_id": "schedule-def789",
  "message": "Schedule deleted successfully"
}
```

### Monitoring

#### Health Check

Check API service health.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:50:00Z",
  "version": "1.0.0",
  "checks": {
    "redis": "connected",
    "workers": 5
  }
}
```

#### Metrics (Prometheus Format)

Export metrics in Prometheus format.

**Endpoint:** `GET /metrics`

**Response:**
```
# HELP tasks_submitted_total Total number of submitted tasks
# TYPE tasks_submitted_total counter
tasks_submitted_total{queue="default"} 1523
tasks_submitted_total{queue="priority"} 234

# HELP tasks_completed_total Total number of completed tasks
# TYPE tasks_completed_total counter
tasks_completed_total{queue="default"} 1456
tasks_completed_total{queue="priority"} 220

# HELP tasks_failed_total Total number of failed tasks
# TYPE tasks_failed_total counter
tasks_failed_total{queue="default"} 12
tasks_failed_total{queue="priority"} 2

# HELP queue_depth Current number of pending tasks
# TYPE queue_depth gauge
queue_depth{queue="default"} 45
queue_depth{queue="priority"} 5

# HELP worker_count Number of active workers
# TYPE worker_count gauge
worker_count{status="active"} 3
worker_count{status="idle"} 2

# HELP task_execution_duration_seconds Task execution time
# TYPE task_execution_duration_seconds histogram
task_execution_duration_seconds_bucket{le="0.1"} 234
task_execution_duration_seconds_bucket{le="0.5"} 567
task_execution_duration_seconds_bucket{le="1.0"} 890
task_execution_duration_seconds_bucket{le="5.0"} 1234
task_execution_duration_seconds_bucket{le="+Inf"} 1456
```

### Dead Letter Queue

#### List DLQ Tasks

Get tasks that failed permanently.

**Endpoint:** `GET /dlq`

**Query Parameters:**
- `queue`: Filter by original queue
- `limit`: Maximum results
- `offset`: Skip first N results

**Response:**
```json
{
  "tasks": [
    {
      "id": "task-failed-123",
      "name": "process_payment",
      "original_queue": "priority",
      "payload": {},
      "error": "Payment gateway timeout",
      "failed_at": "2024-01-15T10:00:00Z",
      "retry_count": 3
    }
  ],
  "total": 12
}
```

#### Resubmit DLQ Task

Move task from DLQ back to queue.

**Endpoint:** `POST /dlq/{task_id}/resubmit`

**Request Body (optional):**
```json
{
  "queue": "default",
  "priority": 10,
  "max_retries": 5
}
```

**Response:**
```json
{
  "original_task_id": "task-failed-123",
  "new_task_id": "task-retry-456",
  "message": "Task resubmitted from DLQ"
}
```

#### Purge DLQ

Remove tasks from Dead Letter Queue.

**Endpoint:** `POST /dlq/purge`

**Request Body (optional):**
```json
{
  "older_than": "2024-01-01T00:00:00Z",
  "task_names": ["failed_task_type"]
}
```

## Error Responses

All error responses follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "priority",
        "message": "Priority must be between 0 and 10"
      }
    ],
    "request_id": "req-abc123",
    "timestamp": "2024-01-15T10:55:00Z"
  }
}
```

**Error Codes:**
- `VALIDATION_ERROR`: Invalid request parameters
- `NOT_FOUND`: Resource not found
- `UNAUTHORIZED`: Invalid or missing API key
- `FORBIDDEN`: Insufficient permissions
- `RATE_LIMITED`: Too many requests
- `INTERNAL_ERROR`: Server error
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

## Rate Limiting

API requests are rate limited per API key:

- **Default limit**: 1000 requests per minute
- **Bulk operations**: 100 requests per minute
- **Headers returned**:
  - `X-RateLimit-Limit`: Maximum requests
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

## Webhooks

Configure webhooks for task events:

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["task.completed", "task.failed"],
  "secret": "webhook-secret-key"
}
```

**Event Types:**
- `task.submitted`: Task created
- `task.started`: Task processing started
- `task.completed`: Task completed successfully
- `task.failed`: Task failed permanently
- `task.retrying`: Task failed but retrying
- `worker.online`: Worker came online
- `worker.offline`: Worker went offline

**Webhook Payload:**
```json
{
  "event": "task.completed",
  "timestamp": "2024-01-15T11:00:00Z",
  "data": {
    "task_id": "task-abc123",
    "name": "send_email",
    "queue": "default",
    "result": {}
  },
  "signature": "sha256=..."
}
```

## Client Libraries

### Python Client

```python
from jobqueue import Client

client = Client(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Submit task
task = client.submit_task(
    name="process_data",
    payload={"data": "value"},
    priority=8
)

# Check status
status = client.get_task(task.id)

# Wait for completion
result = client.wait_for_result(task.id, timeout=30)
```

### JavaScript Client

```javascript
const JobQueue = require('jobqueue-client');

const client = new JobQueue({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Submit task
const task = await client.submitTask({
  name: 'process_data',
  payload: { data: 'value' },
  priority: 8
});

// Check status
const status = await client.getTask(task.id);

// Wait for completion
const result = await client.waitForResult(task.id, { timeout: 30000 });
```