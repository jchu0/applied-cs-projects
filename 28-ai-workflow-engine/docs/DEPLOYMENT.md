# AI Workflow Engine - Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment Options](#deployment-options)
5. [Production Setup](#production-setup)
6. [Monitoring & Observability](#monitoring--observability)
7. [Scaling](#scaling)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended for production)
- **CPU**: 2+ cores recommended
- **Storage**: 10GB+ available space for logs and checkpoints

### Dependencies

```bash
# Core dependencies
python>=3.8
asyncio
pydantic>=2.0
pyyaml>=6.0
networkx>=3.0

# Optional dependencies
redis>=4.0           # For distributed caching
celery>=5.0          # For distributed execution
prometheus-client    # For metrics
boto3               # For S3 storage
```

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/your-org/ai-workflow-engine.git
cd ai-workflow-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Using pip

```bash
pip install ai-workflow-engine
```

### Using Docker

```bash
# Pull official image
docker pull ai-workflow-engine:latest

# Or build from source
docker build -t ai-workflow-engine .
```

## Configuration

### Configuration File

Create `config.yaml`:

```yaml
engine:
  max_parallel: 10
  enable_versioning: true
  enable_optimization: true
  checkpoint_interval: 5
  checkpoint_dir: /var/lib/aiworkflow/checkpoints

storage:
  backend: s3  # Options: local, s3, gcs
  bucket: workflow-storage
  prefix: workflows/
  region: us-west-2

database:
  type: postgresql
  host: localhost
  port: 5432
  database: aiworkflow
  username: ${DB_USER}
  password: ${DB_PASS}

redis:
  host: localhost
  port: 6379
  db: 0
  password: ${REDIS_PASS}

monitoring:
  enabled: true
  metrics_port: 9090
  log_level: INFO
  log_file: /var/log/aiworkflow/engine.log

security:
  enable_auth: true
  auth_provider: jwt
  jwt_secret: ${JWT_SECRET}
  enable_tls: true
  cert_file: /etc/ssl/certs/aiworkflow.crt
  key_file: /etc/ssl/private/aiworkflow.key
```

### Environment Variables

Create `.env` file:

```bash
# Database
DB_USER=aiworkflow_user
DB_PASS=secure_password
DATABASE_URL=postgresql://user:pass@localhost:5432/aiworkflow

# Redis
REDIS_PASS=redis_password
REDIS_URL=redis://:password@localhost:6379/0

# Storage (S3)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-west-2

# Security
JWT_SECRET=your_jwt_secret_key
API_KEY=your_api_key

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
```

## Deployment Options

### 1. Standalone Server

```bash
# Start the engine server
python -m aiworkflow.server \
  --config config.yaml \
  --host 0.0.0.0 \
  --port 8080
```

### 2. Docker Deployment

#### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  engine:
    image: ai-workflow-engine:latest
    ports:
      - "8080:8080"
      - "9090:9090"  # Metrics
    volumes:
      - ./config.yaml:/app/config.yaml
      - workflow-data:/var/lib/aiworkflow
      - workflow-logs:/var/log/aiworkflow
    environment:
      - CONFIG_FILE=/app/config.yaml
      - LOG_LEVEL=INFO
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: aiworkflow
      POSTGRES_USER: aiworkflow_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7
    command: redis-server --requirepass redis_password
    volumes:
      - redis-data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

volumes:
  workflow-data:
  workflow-logs:
  postgres-data:
  redis-data:
  prometheus-data:
```

Start services:

```bash
docker-compose up -d
```

### 3. Kubernetes Deployment

#### Deployment Manifest

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-workflow-engine
  labels:
    app: ai-workflow-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-workflow-engine
  template:
    metadata:
      labels:
        app: ai-workflow-engine
    spec:
      containers:
      - name: engine
        image: ai-workflow-engine:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: CONFIG_FILE
          value: /config/config.yaml
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASS
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        volumeMounts:
        - name: config
          mountPath: /config
        - name: storage
          mountPath: /var/lib/aiworkflow
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: engine-config
      - name: storage
        persistentVolumeClaim:
          claimName: workflow-storage

---
apiVersion: v1
kind: Service
metadata:
  name: ai-workflow-engine
spec:
  selector:
    app: ai-workflow-engine
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

Deploy to Kubernetes:

```bash
# Create namespace
kubectl create namespace aiworkflow

# Create secrets
kubectl create secret generic db-credentials \
  --from-literal=username=aiworkflow_user \
  --from-literal=password=secure_password \
  -n aiworkflow

# Create configmap
kubectl create configmap engine-config \
  --from-file=config.yaml \
  -n aiworkflow

# Apply deployment
kubectl apply -f deployment.yaml -n aiworkflow

# Check status
kubectl get pods -n aiworkflow
```

### 4. AWS Deployment

#### ECS with Fargate

```json
{
  "family": "ai-workflow-engine",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "engine",
      "image": "your-ecr-repo/ai-workflow-engine:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "CONFIG_FILE",
          "value": "/config/config.yaml"
        }
      ],
      "secrets": [
        {
          "name": "DB_PASS",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:db-password"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-workflow-engine",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "engine"
        }
      }
    }
  ]
}
```

## Production Setup

### Database Setup

```sql
-- Create database
CREATE DATABASE aiworkflow;

-- Create user
CREATE USER aiworkflow_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE aiworkflow TO aiworkflow_user;

-- Create tables (run migrations)
python -m aiworkflow.db.migrate
```

### Load Balancer Configuration

#### Nginx

```nginx
upstream workflow_engine {
    server engine1:8080 weight=5;
    server engine2:8080 weight=5;
    server engine3:8080 weight=5;
}

server {
    listen 80;
    server_name workflow.example.com;

    location / {
        proxy_pass http://workflow_engine;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /metrics {
        proxy_pass http://workflow_engine:9090/metrics;
    }
}
```

### SSL/TLS Configuration

```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/aiworkflow.key \
  -out /etc/ssl/certs/aiworkflow.crt

# Or use Let's Encrypt
certbot certonly --standalone -d workflow.example.com
```

## Monitoring & Observability

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-workflow-engine'
    static_configs:
      - targets: ['engine:9090']
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Grafana Dashboard

Import dashboard JSON:

```json
{
  "dashboard": {
    "title": "AI Workflow Engine",
    "panels": [
      {
        "title": "Workflow Execution Rate",
        "targets": [
          {
            "expr": "rate(workflow_executions_total[5m])"
          }
        ]
      },
      {
        "title": "Node Execution Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, workflow_node_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(workflow_errors_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### Logging

#### Structured Logging

```python
import logging
import json

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if hasattr(record, 'workflow_id'):
            log_obj['workflow_id'] = record.workflow_id
        if hasattr(record, 'node_id'):
            log_obj['node_id'] = record.node_id
        return json.dumps(log_obj)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logging.getLogger('aiworkflow').addHandler(handler)
```

#### Log Aggregation with ELK

```yaml
# Filebeat configuration
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/aiworkflow/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "aiworkflow-%{+yyyy.MM.dd}"
```

## Scaling

### Horizontal Scaling

#### Auto-scaling Configuration

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: workflow-engine-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-workflow-engine
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Scaling

```bash
# Increase resources for single instance
docker run -d \
  --name workflow-engine \
  --memory="8g" \
  --cpus="4" \
  -p 8080:8080 \
  ai-workflow-engine:latest
```

### Database Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

## Troubleshooting

### Common Issues

#### 1. High Memory Usage

```bash
# Check memory usage
docker stats workflow-engine

# Analyze memory profile
python -m memory_profiler aiworkflow.server

# Solution: Increase memory limits or optimize workflow
```

#### 2. Slow Workflow Execution

```python
# Enable profiling
import cProfile
profiler = cProfile.Profile()
profiler.enable()

# Run workflow
result = await engine.run_flow(flow)

profiler.disable()
profiler.dump_stats('workflow_profile.stats')

# Analyze profile
python -m pstats workflow_profile.stats
```

#### 3. Database Connection Issues

```bash
# Test database connection
python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@localhost/aiworkflow')
conn = engine.connect()
print('Connection successful')
conn.close()
"

# Check connection pool status
SELECT * FROM pg_stat_activity WHERE datname = 'aiworkflow';
```

#### 4. Node Failures

```python
# Enable debug logging for specific node
logging.getLogger('aiworkflow.nodes.specific_node').setLevel(logging.DEBUG)

# Check node execution history
SELECT * FROM node_executions
WHERE node_id = 'failing_node'
ORDER BY executed_at DESC
LIMIT 10;
```

### Health Checks

```python
# Health check endpoint implementation
async def health_check():
    checks = {
        "database": check_database_connection(),
        "redis": check_redis_connection(),
        "storage": check_storage_access(),
        "memory": check_memory_usage() < 80,
        "disk": check_disk_usage() < 80
    }

    if all(checks.values()):
        return {"status": "healthy", "checks": checks}
    else:
        return {"status": "unhealthy", "checks": checks}, 503
```

### Performance Tuning

```yaml
# Optimization settings
optimization:
  cache_enabled: true
  cache_ttl: 3600
  batch_size: 100
  prefetch_count: 5
  connection_pool_size: 20
  thread_pool_size: 10
  async_io_threads: 4
```

### Backup and Recovery

```bash
# Backup database
pg_dump -U aiworkflow_user -d aiworkflow > backup_$(date +%Y%m%d).sql

# Backup workflow definitions
aws s3 sync /var/lib/aiworkflow/workflows s3://backup-bucket/workflows/

# Restore database
psql -U aiworkflow_user -d aiworkflow < backup_20240101.sql

# Restore workflows
aws s3 sync s3://backup-bucket/workflows/ /var/lib/aiworkflow/workflows
```