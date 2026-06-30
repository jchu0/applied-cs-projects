# Multi-Tenant GPU Scheduler - Deployment Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment Scenarios](#deployment-scenarios)
5. [Production Setup](#production-setup)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements

**Minimum Requirements:**
- CPU: 8 cores
- RAM: 16 GB
- Storage: 100 GB SSD
- Network: 1 Gbps

**Recommended Production Setup:**
- CPU: 16+ cores
- RAM: 64+ GB
- Storage: 500 GB+ NVMe SSD
- Network: 10+ Gbps

### GPU Requirements

**Supported GPU Types:**
- NVIDIA A100 (40GB/80GB)
- NVIDIA H100 (80GB)
- NVIDIA V100 (16GB/32GB)
- NVIDIA T4 (16GB)
- NVIDIA A10G (24GB)
- NVIDIA L4 (24GB)

**GPU Driver Requirements:**
- NVIDIA Driver: >= 470.x
- CUDA: >= 11.4
- NVIDIA Container Toolkit (for containerized deployments)

### Software Requirements

**Python Environment:**
- Python 3.8+
- pip or conda package manager

**System Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    nvidia-driver-470 \
    nvidia-container-toolkit

# RHEL/CentOS
sudo yum install -y \
    python3-devel \
    python3-pip \
    gcc \
    nvidia-driver-470 \
    nvidia-container-toolkit
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/multi-tenant-gpu-scheduler.git
cd multi-tenant-gpu-scheduler
```

### 2. Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate

# Using conda
conda create -n gpu-scheduler python=3.9
conda activate gpu-scheduler
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

### 4. Verify Installation

```bash
# Run tests
python -m pytest tests/

# Check GPU availability
nvidia-smi

# Import check
python -c "from gpusched import __version__; print(__version__)"
```

---

## Configuration

### Basic Configuration

Create `config.yaml`:

```yaml
# Cluster configuration
cluster:
  id: "production-cluster"
  name: "ML Training Cluster"
  region: "us-west-1"

# Scheduler configuration
scheduler:
  interval: 10  # seconds
  batch_size: 100  # max pods per cycle
  mode: "queue"  # standard|queue|preemption
  plugins:
    - name: "NodeAffinity"
      weight: 1.0
      enabled: true
    - name: "GPUResource"
      weight: 2.0
      enabled: true
    - name: "BinPacking"
      weight: 1.5
      enabled: true
    - name: "FairShare"
      weight: 1.0
      enabled: true

# Allocator configuration
allocator:
  mode: "auto"  # auto|exclusive|shared|mig
  max_sharing_factor: 4
  mig_enabled: true
  mig_profiles:
    - "1g.5gb"
    - "2g.10gb"
    - "3g.20gb"
    - "7g.40gb"

# Monitor configuration
monitor:
  enabled: true
  interval: 60  # seconds
  retention_days: 7
  metrics:
    collect_gpu_metrics: true
    collect_node_metrics: true
    collect_job_metrics: true
  alerts:
    enabled: true
    gpu_temperature_threshold: 85  # Celsius
    gpu_utilization_threshold: 90  # Percentage
    memory_pressure_threshold: 95  # Percentage

# Queue configuration
queues:
  default:
    priority_weight: 1.0
    gpu_quota: 100
    max_jobs: 50
    preemptible: true

# Tenant configuration
tenants:
  default:
    name: "Default Tenant"
    gpu_quota: 100
    priority_class: "NORMAL"
    fairshare_weight: 1.0
```

### Advanced Configuration

```yaml
# High Availability
ha:
  enabled: true
  leader_election:
    enabled: true
    lease_duration: 15s
    renew_deadline: 10s
    retry_period: 2s

# Security
security:
  authentication:
    enabled: true
    type: "token"  # token|oauth|ldap
  authorization:
    enabled: true
    rbac_config: "/etc/scheduler/rbac.yaml"
  tls:
    enabled: true
    cert_file: "/etc/scheduler/tls/cert.pem"
    key_file: "/etc/scheduler/tls/key.pem"

# Logging
logging:
  level: "INFO"  # DEBUG|INFO|WARNING|ERROR
  format: "json"  # json|text
  output: "stdout"  # stdout|file
  file_path: "/var/log/gpu-scheduler/scheduler.log"
  max_size: "100MB"
  max_age: "7d"
  max_backups: 5

# Metrics Export
metrics:
  prometheus:
    enabled: true
    port: 9090
    path: "/metrics"
  grafana:
    enabled: true
    dashboard_config: "/etc/scheduler/grafana-dashboard.json"
```

---

## Deployment Scenarios

### 1. Standalone Deployment

**For development and testing:**

```bash
# Start scheduler service
python -m gpusched.scheduler.service \
  --config config.yaml \
  --port 8080 \
  --workers 4

# In another terminal, start monitor
python -m gpusched.monitor.service \
  --config config.yaml \
  --port 9090
```

### 2. Docker Deployment

**Dockerfile:**

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config.yaml .

# Run scheduler
CMD ["python3", "-m", "gpusched.scheduler.service"]
```

**Docker Compose:**

```yaml
version: '3.8'

services:
  scheduler:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=all
    ports:
      - "8080:8080"
    volumes:
      - ./config.yaml:/app/config.yaml
      - /var/run/docker.sock:/var/run/docker.sock
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  monitor:
    build: .
    command: ["python3", "-m", "gpusched.monitor.service"]
    ports:
      - "9090:9090"
    volumes:
      - ./config.yaml:/app/config.yaml
    depends_on:
      - scheduler

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

### 3. Kubernetes Deployment

**Deployment manifest:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-scheduler
  namespace: gpu-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gpu-scheduler
  template:
    metadata:
      labels:
        app: gpu-scheduler
    spec:
      serviceAccountName: gpu-scheduler
      containers:
      - name: scheduler
        image: your-registry/gpu-scheduler:latest
        ports:
        - containerPort: 8080
        env:
        - name: CONFIG_PATH
          value: /etc/scheduler/config.yaml
        volumeMounts:
        - name: config
          mountPath: /etc/scheduler
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
      volumes:
      - name: config
        configMap:
          name: scheduler-config
---
apiVersion: v1
kind: Service
metadata:
  name: gpu-scheduler
  namespace: gpu-system
spec:
  selector:
    app: gpu-scheduler
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

---

## Production Setup

### 1. High Availability Setup

**Multiple Scheduler Instances:**

```bash
# Primary scheduler
python -m gpusched.scheduler.service \
  --config ha-config.yaml \
  --role primary \
  --peers scheduler-2:8080,scheduler-3:8080

# Secondary schedulers
python -m gpusched.scheduler.service \
  --config ha-config.yaml \
  --role secondary \
  --primary scheduler-1:8080
```

### 2. Database Backend

**PostgreSQL Setup:**

```sql
-- Create database
CREATE DATABASE gpu_scheduler;

-- Create tables
CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    tenant_id VARCHAR(255),
    state VARCHAR(50),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE allocations (
    id UUID PRIMARY KEY,
    pod_id UUID,
    node_id VARCHAR(255),
    gpu_id VARCHAR(255),
    allocated_at TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_jobs_tenant ON jobs(tenant_id);
CREATE INDEX idx_jobs_state ON jobs(state);
CREATE INDEX idx_allocations_node ON allocations(node_id);
```

### 3. Load Balancer Configuration

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

backend gpu_schedulers
    balance roundrobin
    option httpchk GET /health
    server scheduler1 10.0.1.10:8080 check
    server scheduler2 10.0.1.11:8080 check
    server scheduler3 10.0.1.12:8080 check

frontend scheduler_frontend
    bind *:80
    default_backend gpu_schedulers
```

### 4. Security Hardening

```bash
# Create service user
sudo useradd -r -s /bin/false gpu-scheduler

# Set file permissions
sudo chown -R gpu-scheduler:gpu-scheduler /opt/gpu-scheduler
sudo chmod 750 /opt/gpu-scheduler
sudo chmod 640 /opt/gpu-scheduler/config.yaml

# Configure firewall
sudo ufw allow 8080/tcp  # Scheduler API
sudo ufw allow 9090/tcp  # Metrics
sudo ufw enable

# Enable SELinux/AppArmor profiles
sudo semanage fcontext -a -t gpu_scheduler_t /opt/gpu-scheduler
sudo restorecon -Rv /opt/gpu-scheduler
```

---

## Monitoring Setup

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'gpu-scheduler'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9400']
```

### 2. Grafana Dashboard

Import the provided dashboard JSON:

```bash
# Import dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana-dashboard.json
```

### 3. Alert Rules

```yaml
# alerts.yml
groups:
  - name: gpu_alerts
    rules:
      - alert: HighGPUTemperature
        expr: gpu_temperature > 85
        for: 5m
        annotations:
          summary: "GPU temperature critical"

      - alert: HighGPUUtilization
        expr: gpu_utilization > 95
        for: 10m
        annotations:
          summary: "GPU utilization sustained high"

      - alert: SchedulerDown
        expr: up{job="gpu-scheduler"} == 0
        for: 1m
        annotations:
          summary: "GPU Scheduler is down"
```

---

## Troubleshooting

### Common Issues

#### 1. Scheduler Not Starting

```bash
# Check logs
journalctl -u gpu-scheduler -f

# Verify configuration
python -m gpusched.config.validate --config config.yaml

# Check port availability
sudo netstat -tlnp | grep 8080
```

#### 2. GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify GPU visibility
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. High Memory Usage

```bash
# Check memory consumption
ps aux | grep gpu-scheduler
htop

# Enable memory profiling
python -m gpusched.scheduler.service \
  --config config.yaml \
  --profile-memory
```

#### 4. Scheduling Delays

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check scheduler queue
from gpusched.scheduler import get_queue_status
status = get_queue_status()
print(f"Pending pods: {status['pending']}")
print(f"Queue depth: {status['depth']}")
```

### Performance Tuning

```yaml
# Optimize scheduler performance
performance:
  scheduler:
    parallel_evaluations: true
    cache_node_info: true
    cache_ttl: 30  # seconds
    batch_size: 200

  database:
    connection_pool_size: 20
    query_timeout: 5000  # ms

  api:
    rate_limit: 1000  # requests per minute
    request_timeout: 30  # seconds
```

### Health Checks

```bash
# API health check
curl http://localhost:8080/health

# Detailed status
curl http://localhost:8080/status

# Metrics endpoint
curl http://localhost:9090/metrics
```

---

## Backup and Recovery

### Backup Strategy

```bash
# Backup configuration
cp -r /etc/gpu-scheduler /backup/gpu-scheduler-$(date +%Y%m%d)

# Backup database
pg_dump gpu_scheduler > backup-$(date +%Y%m%d).sql

# Backup state
python -m gpusched.tools.backup \
  --state-dir /var/lib/gpu-scheduler \
  --backup-dir /backup/state
```

### Recovery Procedure

```bash
# Restore configuration
cp -r /backup/gpu-scheduler-20240101 /etc/gpu-scheduler

# Restore database
psql gpu_scheduler < backup-20240101.sql

# Restore state
python -m gpusched.tools.restore \
  --backup-dir /backup/state \
  --state-dir /var/lib/gpu-scheduler

# Restart services
systemctl restart gpu-scheduler
systemctl restart gpu-monitor
```