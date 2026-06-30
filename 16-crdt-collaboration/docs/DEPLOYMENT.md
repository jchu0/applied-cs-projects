# CRDT Collaboration Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Rust**: 1.70+ (with Cargo)
- **Memory**: Minimum 2GB RAM (4GB+ recommended for production)
- **Storage**: 10GB+ for operation logs and snapshots
- **Network**: Low-latency connection for real-time collaboration

### Required Dependencies

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version

# Install additional tools
cargo install cargo-watch
cargo install cargo-audit
```

## Development Setup

### 1. Clone and Build

```bash
# Clone repository
git clone https://github.com/yourusername/crdt-collaboration.git
cd crdt-collaboration

# Build the project
cargo build --release

# Run tests
cargo test

# Run with hot reload
cargo watch -x run
```

### 2. Development Configuration

Create `config/development.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8080
workers = 4

[storage]
path = "./data/dev"
max_log_size = "100MB"
snapshot_interval = 1000

[logging]
level = "debug"
format = "pretty"

[limits]
max_connections = 100
max_document_size = "10MB"
operation_rate_limit = 100
```

### 3. Run Development Server

```bash
# Set environment
export RUST_ENV=development

# Run server
cargo run --bin server

# Or with custom config
cargo run --bin server -- --config config/development.toml
```

## Production Deployment

### 1. Optimization Build

```bash
# Build with optimizations
cargo build --release

# Strip debug symbols
strip target/release/crdt-collaboration-server

# Run benchmarks
cargo bench
```

### 2. Production Configuration

Create `config/production.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8080
workers = 16
max_connections = 10000

[storage]
path = "/var/lib/crdt-collaboration"
max_log_size = "1GB"
snapshot_interval = 5000
compression = true

[security]
tls_enabled = true
cert_path = "/etc/ssl/certs/server.crt"
key_path = "/etc/ssl/private/server.key"
auth_required = true

[logging]
level = "info"
format = "json"
output = "/var/log/crdt-collaboration/server.log"

[monitoring]
metrics_enabled = true
metrics_port = 9090
health_check_path = "/health"
```

### 3. SystemD Service

Create `/etc/systemd/system/crdt-collaboration.service`:

```ini
[Unit]
Description=CRDT Collaboration Server
After=network.target

[Service]
Type=simple
User=crdt-service
Group=crdt-service
WorkingDirectory=/opt/crdt-collaboration
Environment="RUST_ENV=production"
ExecStart=/opt/crdt-collaboration/bin/server --config /opt/crdt-collaboration/config/production.toml
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/crdt-collaboration /var/log/crdt-collaboration

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable crdt-collaboration
sudo systemctl start crdt-collaboration
sudo systemctl status crdt-collaboration
```

## Docker Deployment

### 1. Dockerfile

```dockerfile
# Build stage
FROM rust:1.70 AS builder

WORKDIR /usr/src/app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/app/target/release/crdt-collaboration-server /usr/local/bin/

# Create user
RUN useradd -m -U crdt-service

# Create data directories
RUN mkdir -p /var/lib/crdt-collaboration && \
    mkdir -p /var/log/crdt-collaboration && \
    chown -R crdt-service:crdt-service /var/lib/crdt-collaboration /var/log/crdt-collaboration

USER crdt-service

EXPOSE 8080 9090

VOLUME ["/var/lib/crdt-collaboration", "/var/log/crdt-collaboration"]

CMD ["crdt-collaboration-server"]
```

### 2. Docker Compose

```yaml
version: '3.8'

services:
  crdt-server:
    build: .
    image: crdt-collaboration:latest
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - crdt-data:/var/lib/crdt-collaboration
      - crdt-logs:/var/log/crdt-collaboration
      - ./config:/etc/crdt-collaboration:ro
    environment:
      - RUST_ENV=production
      - CONFIG_PATH=/etc/crdt-collaboration/production.toml
    restart: unless-stopped
    networks:
      - crdt-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9090/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - crdt-server
    networks:
      - crdt-network

volumes:
  crdt-data:
  crdt-logs:

networks:
  crdt-network:
    driver: bridge
```

### 3. Build and Run

```bash
# Build image
docker build -t crdt-collaboration:latest .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f crdt-server

# Scale horizontally
docker-compose up -d --scale crdt-server=3
```

## Kubernetes Deployment

### 1. Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crdt-collaboration
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crdt-collaboration
  template:
    metadata:
      labels:
        app: crdt-collaboration
    spec:
      containers:
      - name: crdt-server
        image: crdt-collaboration:latest
        ports:
        - containerPort: 8080
          name: websocket
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_ENV
          value: "production"
        - name: CONFIG_PATH
          value: "/etc/crdt/config.toml"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 9090
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /etc/crdt
        - name: data
          mountPath: /var/lib/crdt-collaboration
      volumes:
      - name: config
        configMap:
          name: crdt-config
      - name: data
        persistentVolumeClaim:
          claimName: crdt-data-pvc
```

### 2. Service and Ingress

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: crdt-collaboration-service
spec:
  selector:
    app: crdt-collaboration
  ports:
  - port: 8080
    targetPort: 8080
    name: websocket
  - port: 9090
    targetPort: 9090
    name: metrics
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: crdt-collaboration-ingress
  annotations:
    nginx.ingress.kubernetes.io/websocket-services: "crdt-collaboration-service"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
spec:
  tls:
  - hosts:
    - crdt.example.com
    secretName: crdt-tls
  rules:
  - host: crdt.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: crdt-collaboration-service
            port:
              number: 8080
```

### 3. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace crdt-system

# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check status
kubectl get pods -n crdt-system
kubectl get svc -n crdt-system

# Scale deployment
kubectl scale deployment crdt-collaboration --replicas=5 -n crdt-system

# View logs
kubectl logs -f deployment/crdt-collaboration -n crdt-system
```

## Configuration

### Environment Variables

```bash
# Server configuration
CRDT_HOST=0.0.0.0
CRDT_PORT=8080
CRDT_WORKERS=16

# Storage
CRDT_STORAGE_PATH=/var/lib/crdt
CRDT_MAX_LOG_SIZE=1GB
CRDT_SNAPSHOT_INTERVAL=5000

# Security
CRDT_TLS_ENABLED=true
CRDT_TLS_CERT=/path/to/cert.pem
CRDT_TLS_KEY=/path/to/key.pem

# Logging
CRDT_LOG_LEVEL=info
CRDT_LOG_FORMAT=json

# Limits
CRDT_MAX_CONNECTIONS=10000
CRDT_MAX_DOCUMENT_SIZE=10MB
```

### NGINX Configuration

```nginx
upstream crdt_backend {
    least_conn;
    server crdt-server-1:8080 max_fails=3 fail_timeout=30s;
    server crdt-server-2:8080 max_fails=3 fail_timeout=30s;
    server crdt-server-3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    listen [::]:80;
    server_name crdt.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name crdt.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location /ws {
        proxy_pass http://crdt_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }

    location /health {
        proxy_pass http://crdt_backend:9090/health;
    }
}
```

## Monitoring

### Prometheus Configuration

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'crdt-collaboration'
    static_configs:
      - targets: ['crdt-server-1:9090', 'crdt-server-2:9090', 'crdt-server-3:9090']
```

### Grafana Dashboard

Import dashboard JSON from `monitoring/grafana-dashboard.json` with panels for:

- Active connections
- Operations per second
- Document count
- Storage usage
- Error rate
- Latency percentiles

### Health Checks

```bash
# Basic health check
curl http://localhost:9090/health

# Detailed health check
curl http://localhost:9090/health/detailed

# Readiness check
curl http://localhost:9090/ready
```

## Troubleshooting

### Common Issues

#### 1. Connection Issues

```bash
# Check server is running
systemctl status crdt-collaboration

# Check port is listening
netstat -tlnp | grep 8080

# Test WebSocket connection
wscat -c ws://localhost:8080/ws
```

#### 2. Performance Issues

```bash
# Check resource usage
htop

# Profile server
RUST_LOG=debug cargo run --release

# Enable flame graph
cargo flamegraph --bin server
```

#### 3. Storage Issues

```bash
# Check disk space
df -h /var/lib/crdt-collaboration

# Compact logs
./scripts/compact-logs.sh

# Clean old snapshots
./scripts/cleanup-snapshots.sh
```

### Debug Mode

```bash
# Enable debug logging
export RUST_LOG=crdt_collaboration=debug

# Enable backtrace
export RUST_BACKTRACE=1

# Run with verbose output
cargo run --bin server -- -vvv
```

### Recovery Procedures

#### Restore from Backup

```bash
# Stop server
systemctl stop crdt-collaboration

# Restore data
tar -xzf backup-2024-01-01.tar.gz -C /var/lib/crdt-collaboration

# Start server
systemctl start crdt-collaboration
```

#### Rebuild from Operation Log

```bash
# Run recovery tool
./tools/recover --from-log /var/lib/crdt-collaboration/operations.log

# Verify integrity
./tools/verify --data-dir /var/lib/crdt-collaboration
```

## Security Considerations

### TLS Configuration

```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Use Let's Encrypt (production)
certbot certonly --standalone -d crdt.example.com
```

### Firewall Rules

```bash
# Allow WebSocket port
ufw allow 8080/tcp

# Allow metrics port (internal only)
ufw allow from 10.0.0.0/8 to any port 9090
```

### Rate Limiting

Configure in `production.toml`:

```toml
[limits]
max_operations_per_second = 100
max_connections_per_ip = 10
max_document_size = "10MB"
```

## Backup and Recovery

### Automated Backups

```bash
# Backup script (add to crontab)
#!/bin/bash
BACKUP_DIR="/backup/crdt"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup data
tar -czf "${BACKUP_DIR}/data_${DATE}.tar.gz" /var/lib/crdt-collaboration

# Backup config
cp -r /etc/crdt-collaboration "${BACKUP_DIR}/config_${DATE}"

# Rotate old backups
find ${BACKUP_DIR} -name "*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery

1. **Regular Backups**: Daily automated backups to off-site storage
2. **Replication**: Multi-region deployment with data replication
3. **Point-in-Time Recovery**: Use operation log for precise recovery
4. **Testing**: Regular disaster recovery drills