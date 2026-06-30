# Distributed Job Queue - Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Production Configuration](#production-configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Backup and Recovery](#backup-and-recovery)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB
- Network: 100 Mbps

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- Network: 1 Gbps

### Software Dependencies

- Python 3.10+
- Redis 7.0+
- Docker 20.10+ (for containerized deployment)
- Kubernetes 1.25+ (for K8s deployment)

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourorg/distributed-job-queue.git
cd distributed-job-queue
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Configure Environment

Create `.env` file:

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SSL=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_KEY=your-secure-api-key

# Worker Configuration
WORKER_CONCURRENCY=10
WORKER_QUEUES=default,priority
WORKER_POLL_INTERVAL=1.0
WORKER_HEARTBEAT_INTERVAL=5.0

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

### 5. Start Services

```bash
# Start Redis
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Start API server
python -m jobqueue.api

# Start worker (in another terminal)
python -m jobqueue.worker

# Start scheduler (optional)
python -m jobqueue.scheduler
```

## Docker Deployment

### 1. Build Images

```bash
# Build base image
docker build -t jobqueue:latest .

# Build specific components
docker build -f Dockerfile.api -t jobqueue-api:latest .
docker build -f Dockerfile.worker -t jobqueue-worker:latest .
```

### 2. Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: jobqueue-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    image: jobqueue-api:latest
    container_name: jobqueue-api
    ports:
      - "8000:8000"
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
      API_HOST: 0.0.0.0
      API_PORT: 8000
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    image: jobqueue-worker:latest
    container_name: jobqueue-worker
    deploy:
      replicas: 3
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
      WORKER_CONCURRENCY: 10
      WORKER_QUEUES: default,priority
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  scheduler:
    image: jobqueue-api:latest
    container_name: jobqueue-scheduler
    command: python -m jobqueue.scheduler
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
    depends_on:
      redis:
        condition: service_healthy

  prometheus:
    image: prom/prometheus:latest
    container_name: jobqueue-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    container_name: jobqueue-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: false

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Scale workers
docker-compose up -d --scale worker=5

# View logs
docker-compose logs -f worker

# Stop services
docker-compose down
```

## Kubernetes Deployment

### 1. Create Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: jobqueue
```

### 2. Redis Deployment

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: jobqueue
spec:
  serviceName: redis
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
  volumeClaimTemplates:
  - metadata:
      name: redis-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi

---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: jobqueue
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

### 3. API Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: jobqueue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: jobqueue-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: redis
        - name: REDIS_PORT
          value: "6379"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: api
  namespace: jobqueue
spec:
  selector:
    app: api
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### 4. Worker Deployment with HPA

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: worker
  namespace: jobqueue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: worker
  template:
    metadata:
      labels:
        app: worker
    spec:
      containers:
      - name: worker
        image: jobqueue-worker:latest
        env:
        - name: REDIS_HOST
          value: redis
        - name: REDIS_PORT
          value: "6379"
        - name: WORKER_CONCURRENCY
          value: "10"
        - name: WORKER_QUEUES
          value: "default,priority"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: worker-hpa
  namespace: jobqueue
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: worker
  minReplicas: 3
  maxReplicas: 20
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

### 5. ConfigMap for Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: jobqueue-config
  namespace: jobqueue
data:
  config.yaml: |
    redis:
      host: redis
      port: 6379
      db: 0
      max_connections: 50

    worker:
      concurrency: 10
      prefetch_limit: 20
      heartbeat_interval: 5
      circuit_breaker:
        enabled: true
        failure_threshold: 5
        reset_timeout: 60

    api:
      host: 0.0.0.0
      port: 8000
      workers: 4
      cors_origins:
        - http://localhost:3000
        - https://app.example.com
```

### 6. Secrets Management

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: jobqueue-secrets
  namespace: jobqueue
type: Opaque
data:
  redis-password: <base64-encoded-password>
  api-key: <base64-encoded-api-key>
  jwt-secret: <base64-encoded-jwt-secret>
```

### 7. Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: jobqueue-ingress
  namespace: jobqueue
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.jobqueue.example.com
    secretName: jobqueue-tls
  rules:
  - host: api.jobqueue.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api
            port:
              number: 8000
```

### 8. Apply Kubernetes Resources

```bash
# Apply all resources
kubectl apply -f k8s/

# Check deployment status
kubectl get all -n jobqueue

# View logs
kubectl logs -f deployment/worker -n jobqueue

# Scale workers
kubectl scale deployment/worker --replicas=10 -n jobqueue
```

## Production Configuration

### 1. Environment Variables

```bash
# Production settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Redis Configuration
REDIS_HOST=redis-cluster.internal
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_SSL=true
REDIS_CONNECTION_POOL_SIZE=50
REDIS_SOCKET_KEEPALIVE=true
REDIS_SOCKET_KEEPALIVE_OPTIONS=1:1:3

# Security
API_KEY=${API_KEY}
JWT_SECRET=${JWT_SECRET}
ALLOWED_HOSTS=api.jobqueue.example.com
CORS_ORIGINS=https://app.example.com

# Performance
WORKER_CONCURRENCY=20
WORKER_PREFETCH_LIMIT=40
TASK_TIME_LIMIT=300
TASK_SOFT_TIME_LIMIT=240

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
ENABLE_TRACING=true
JAEGER_AGENT_HOST=jaeger.monitoring.svc.cluster.local
JAEGER_AGENT_PORT=6831

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=1000
RATE_LIMIT_BURST=100
```

### 2. Redis High Availability

**Redis Sentinel Configuration:**

```conf
# sentinel.conf
port 26379
sentinel monitor mymaster redis-master 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 10000
```

**Redis Cluster Configuration:**

```bash
# Create Redis cluster
redis-cli --cluster create \
  redis-1:6379 redis-2:6379 redis-3:6379 \
  redis-4:6379 redis-5:6379 redis-6:6379 \
  --cluster-replicas 1
```

### 3. Load Balancing

**HAProxy Configuration:**

```conf
global
    maxconn 4096
    log stdout local0

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

backend api_servers
    balance roundrobin
    server api1 api-1:8000 check
    server api2 api-2:8000 check
    server api3 api-3:8000 check
```

### 4. SSL/TLS Configuration

```python
# SSL context for Redis
import ssl

ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_REQUIRED
ssl_context.load_cert_chain(
    certfile='/etc/ssl/certs/client.crt',
    keyfile='/etc/ssl/private/client.key'
)

redis_client = redis.Redis(
    host='redis.example.com',
    port=6379,
    ssl=True,
    ssl_context=ssl_context
)
```

## Monitoring Setup

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'jobqueue-api'
    static_configs:
      - targets: ['api:9090']

  - job_name: 'jobqueue-workers'
    static_configs:
      - targets: ['worker-1:9090', 'worker-2:9090', 'worker-3:9090']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']
```

### 2. Grafana Dashboards

Import provided dashboards:
- `grafana/dashboards/overview.json` - System overview
- `grafana/dashboards/workers.json` - Worker metrics
- `grafana/dashboards/tasks.json` - Task processing metrics
- `grafana/dashboards/queues.json` - Queue depth and latency

### 3. Alerting Rules

```yaml
# alerts.yml
groups:
- name: jobqueue
  rules:
  - alert: HighQueueDepth
    expr: queue_depth > 1000
    for: 5m
    annotations:
      summary: "Queue {{ $labels.queue }} has high depth"
      description: "Queue {{ $labels.queue }} has {{ $value }} pending tasks"

  - alert: WorkerDown
    expr: up{job="jobqueue-workers"} == 0
    for: 1m
    annotations:
      summary: "Worker {{ $labels.instance }} is down"

  - alert: HighErrorRate
    expr: rate(tasks_failed_total[5m]) > 0.1
    for: 5m
    annotations:
      summary: "High task failure rate"
      description: "Failure rate is {{ $value }} per second"
```

### 4. Logging Configuration

```python
# logging_config.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/jobqueue/app.log',
            'maxBytes': 104857600,  # 100MB
            'backupCount': 10,
            'formatter': 'json'
        },
        'elasticsearch': {
            'class': 'CMRESHandler.CMRESHandler',
            'hosts': [{'host': 'elasticsearch', 'port': 9200}],
            'es_index_name': 'jobqueue-logs',
            'es_doc_type': 'log',
            'formatter': 'json'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file', 'elasticsearch']
    }
}
```

## Backup and Recovery

### 1. Redis Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/redis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup
redis-cli --rdb $BACKUP_DIR/dump_$TIMESTAMP.rdb

# Compress
gzip $BACKUP_DIR/dump_$TIMESTAMP.rdb

# Upload to S3
aws s3 cp $BACKUP_DIR/dump_$TIMESTAMP.rdb.gz s3://backup-bucket/redis/

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -name "*.rdb.gz" -mtime +30 -delete
```

### 2. Restore Procedure

```bash
# Stop Redis
systemctl stop redis

# Restore backup
gunzip -c dump_20240115_120000.rdb.gz > /var/lib/redis/dump.rdb

# Set permissions
chown redis:redis /var/lib/redis/dump.rdb

# Start Redis
systemctl start redis

# Verify
redis-cli ping
```

### 3. Disaster Recovery Plan

1. **RPO (Recovery Point Objective)**: 1 hour
2. **RTO (Recovery Time Objective)**: 30 minutes

**Recovery Steps:**
1. Provision new infrastructure
2. Restore Redis from latest backup
3. Deploy application containers
4. Verify system health
5. Update DNS/load balancer
6. Monitor for anomalies

## Troubleshooting

### Common Issues

#### 1. Workers Not Processing Tasks

**Symptoms:** Tasks remain in pending state

**Check:**
```bash
# Check worker status
curl http://api:8000/workers

# Check Redis connectivity
redis-cli ping

# Check worker logs
docker logs jobqueue-worker

# Verify queue status
redis-cli llen queue:default:pending
```

**Solutions:**
- Restart workers
- Check Redis connection
- Verify queue names match
- Check for circuit breaker activation

#### 2. High Memory Usage

**Symptoms:** OOM errors, slow performance

**Check:**
```bash
# Check Redis memory
redis-cli info memory

# Check container memory
docker stats

# Analyze large keys
redis-cli --bigkeys
```

**Solutions:**
- Increase memory limits
- Enable Redis eviction policy
- Reduce task payload size
- Implement result expiration

#### 3. Task Timeouts

**Symptoms:** Tasks failing with timeout errors

**Check:**
```bash
# Check task execution time
grep "task_execution_time" /var/log/jobqueue/app.log

# Monitor slow queries
redis-cli slowlog get 10
```

**Solutions:**
- Increase task timeout
- Optimize task handler code
- Break large tasks into smaller ones
- Add more workers

### Performance Tuning

#### 1. Redis Optimization

```conf
# redis.conf
maxmemory 8gb
maxmemory-policy allkeys-lru
tcp-backlog 511
tcp-keepalive 60
```

#### 2. Worker Tuning

```python
# Optimal worker configuration
WORKER_CONCURRENCY = CPU_CORES * 2
PREFETCH_LIMIT = WORKER_CONCURRENCY * 2
CONNECTION_POOL_SIZE = WORKER_CONCURRENCY * 3
```

#### 3. Network Optimization

```bash
# Increase system limits
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 8192" >> /etc/sysctl.conf
sysctl -p
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment
export LOG_LEVEL=DEBUG
export DEBUG=true
```

### Health Checks

```bash
#!/bin/bash
# healthcheck.sh

# Check API
curl -f http://localhost:8000/health || exit 1

# Check Redis
redis-cli ping || exit 1

# Check worker count
WORKERS=$(curl -s http://localhost:8000/workers | jq '.active')
if [ "$WORKERS" -lt 1 ]; then
    echo "No active workers!"
    exit 1
fi

echo "All systems operational"
```