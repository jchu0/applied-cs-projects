# Service Mesh Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Standalone Deployment](#standalone-deployment)
7. [Multi-Cluster Setup](#multi-cluster-setup)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Security Hardening](#security-hardening)
10. [Troubleshooting](#troubleshooting)
11. [Migration Guide](#migration-guide)

## Prerequisites

### System Requirements

- **CPU**: Minimum 2 cores, recommended 4+ cores
- **Memory**: Minimum 2GB RAM, recommended 4GB+ RAM
- **Disk**: 10GB available space
- **Network**: Low-latency network connectivity between services

### Software Requirements

- Rust 1.70+ (for building from source)
- Docker 20.10+ (for containerized deployment)
- Kubernetes 1.24+ (for K8s deployment)
- OpenSSL 1.1+ (for certificate operations)

### Platform Support

- Linux (x86_64, ARM64)
- macOS (x86_64, Apple Silicon)
- Windows (WSL2 recommended)

## Installation

### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-org/service-mesh.git
cd service-mesh

# Build in release mode
cargo build --release

# Run tests
cargo test

# Install binary
cargo install --path .
```

### Using Pre-built Binaries

```bash
# Download latest release
curl -L https://github.com/your-org/service-mesh/releases/latest/download/service-mesh-linux-amd64.tar.gz | tar xz

# Move to PATH
sudo mv service-mesh /usr/local/bin/

# Verify installation
service-mesh --version
```

### Docker Installation

```dockerfile
# Dockerfile for service mesh
FROM rust:1.70 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/service-mesh /usr/local/bin/

EXPOSE 15001

ENTRYPOINT ["service-mesh"]
```

Build and push:

```bash
docker build -t your-registry/service-mesh:latest .
docker push your-registry/service-mesh:latest
```

## Configuration

### Configuration File Format

```yaml
# config.yaml
mesh:
  name: production-mesh
  cluster: us-west-2

proxy:
  listen_address: "0.0.0.0:15001"
  upstream_address: "127.0.0.1:8080"
  admin_port: 15000

tls:
  ca_cert: /etc/mesh/ca.crt
  cert: /etc/mesh/cert.pem
  key: /etc/mesh/key.pem

  # Optional: certificate rotation
  rotation:
    enabled: true
    check_interval: 1h
    renewal_threshold: 30d

service_discovery:
  registry_url: "http://registry.mesh.local:8500"
  cache_ttl: 60s
  health_check_interval: 10s

policies:
  authorization:
    enabled: true
    rules_file: /etc/mesh/authz.yaml

  circuit_breaker:
    enabled: true
    failure_threshold: 5
    failure_rate: 0.5
    timeout: 30s
    half_open_timeout: 10s

  retry:
    enabled: true
    max_attempts: 3
    base_delay: 100ms
    max_delay: 10s
    retryable_status_codes: [500, 502, 503]

  timeout:
    request: 5s
    stream: 30s
    connect: 10s

observability:
  metrics:
    enabled: true
    port: 9090
    path: /metrics

  tracing:
    enabled: true
    collector_endpoint: "http://jaeger:14268/api/traces"
    sampling_rate: 0.1

  logging:
    level: info
    format: json
    output: stdout
```

### Environment Variables

```bash
# Core settings
export MESH_NAME=production-mesh
export MESH_CLUSTER=us-west-2
export MESH_NAMESPACE=default

# Proxy settings
export PROXY_LISTEN_ADDR=0.0.0.0:15001
export PROXY_UPSTREAM_ADDR=127.0.0.1:8080
export PROXY_ADMIN_PORT=15000

# TLS settings
export TLS_CA_CERT=/etc/mesh/ca.crt
export TLS_CERT=/etc/mesh/cert.pem
export TLS_KEY=/etc/mesh/key.pem

# Service discovery
export REGISTRY_URL=http://registry:8500
export DISCOVERY_CACHE_TTL=60

# Observability
export METRICS_ENABLED=true
export TRACING_ENABLED=true
export LOG_LEVEL=info
```

## Kubernetes Deployment

### Namespace and RBAC

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: service-mesh

---
# rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: service-mesh
rules:
- apiGroups: [""]
  resources: ["services", "endpoints", "pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: service-mesh
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: service-mesh
subjects:
- kind: ServiceAccount
  name: service-mesh
  namespace: service-mesh
```

### Control Plane Deployment

```yaml
# control-plane.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mesh-control-plane
  namespace: service-mesh
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mesh-control-plane
  template:
    metadata:
      labels:
        app: mesh-control-plane
    spec:
      serviceAccountName: service-mesh
      containers:
      - name: ca
        image: your-registry/mesh-ca:latest
        ports:
        - containerPort: 8443
        env:
        - name: CA_NAME
          value: "production-mesh-ca"
        - name: CA_VALIDITY
          value: "87600h"  # 10 years
        volumeMounts:
        - name: ca-certs
          mountPath: /certs

      - name: registry
        image: your-registry/mesh-registry:latest
        ports:
        - containerPort: 8500
        env:
        - name: REGISTRY_PORT
          value: "8500"

      - name: policy-manager
        image: your-registry/mesh-policy:latest
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: policies
          mountPath: /policies

      volumes:
      - name: ca-certs
        secret:
          secretName: mesh-ca-certs
      - name: policies
        configMap:
          name: mesh-policies
```

### Sidecar Injection

```yaml
# sidecar-injector.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sidecar-injector-webhook
  namespace: service-mesh
data:
  sidecar-config.yaml: |
    containers:
    - name: sidecar-proxy
      image: your-registry/service-mesh:latest
      imagePullPolicy: Always
      ports:
      - containerPort: 15001
        name: proxy
      - containerPort: 15000
        name: admin
      env:
      - name: SERVICE_NAME
        valueFrom:
          fieldRef:
            fieldPath: metadata.labels['app']
      - name: POD_NAMESPACE
        valueFrom:
          fieldRef:
            fieldPath: metadata.namespace
      - name: POD_IP
        valueFrom:
          fieldRef:
            fieldPath: status.podIP
      volumeMounts:
      - name: certs
        mountPath: /etc/mesh/certs
        readOnly: true
      livenessProbe:
        httpGet:
          path: /healthz
          port: admin
        initialDelaySeconds: 10
      readinessProbe:
        httpGet:
          path: /ready
          port: admin
        initialDelaySeconds: 5
```

### Application Deployment with Sidecar

```yaml
# app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: example-app
  namespace: default
  labels:
    app: example-app
    mesh: enabled
spec:
  replicas: 3
  selector:
    matchLabels:
      app: example-app
  template:
    metadata:
      labels:
        app: example-app
      annotations:
        mesh.io/inject: "true"
    spec:
      containers:
      - name: app
        image: your-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: HTTP_PROXY
          value: "http://localhost:15001"
        - name: HTTPS_PROXY
          value: "http://localhost:15001"
```

### Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: example-app
  namespace: default
spec:
  selector:
    app: example-app
  ports:
  - port: 80
    targetPort: 15001  # Route through sidecar
    name: http

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mesh-ingress
  namespace: default
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: example-app
            port:
              number: 80
```

## Docker Deployment

### Docker Compose Setup

```yaml
# docker-compose.yaml
version: '3.8'

services:
  ca:
    image: your-registry/mesh-ca:latest
    environment:
      CA_NAME: local-mesh-ca
      CA_VALIDITY: 87600h
    volumes:
      - ca-data:/data
    networks:
      - mesh

  registry:
    image: your-registry/mesh-registry:latest
    ports:
      - "8500:8500"
    environment:
      REGISTRY_PORT: 8500
    networks:
      - mesh

  app:
    image: your-app:latest
    networks:
      - mesh
    depends_on:
      - app-sidecar

  app-sidecar:
    image: your-registry/service-mesh:latest
    environment:
      SERVICE_NAME: app
      UPSTREAM_ADDR: app:8080
      REGISTRY_URL: http://registry:8500
    volumes:
      - ./config:/etc/mesh
    networks:
      - mesh
    ports:
      - "15001:15001"

volumes:
  ca-data:

networks:
  mesh:
    driver: bridge
```

### Running with Docker

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app-sidecar

# Scale services
docker-compose scale app=3

# Stop services
docker-compose down
```

## Standalone Deployment

### Systemd Service

```ini
# /etc/systemd/system/service-mesh.service
[Unit]
Description=Service Mesh Sidecar Proxy
After=network.target

[Service]
Type=simple
User=mesh
Group=mesh
ExecStart=/usr/local/bin/service-mesh \
  --config /etc/mesh/config.yaml
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Installation Script

```bash
#!/bin/bash
# install.sh

set -e

# Create user
useradd -r -s /bin/false mesh || true

# Create directories
mkdir -p /etc/mesh/certs
mkdir -p /var/lib/mesh
mkdir -p /var/log/mesh

# Set permissions
chown -R mesh:mesh /etc/mesh
chown -R mesh:mesh /var/lib/mesh
chown -R mesh:mesh /var/log/mesh

# Copy binary
cp service-mesh /usr/local/bin/
chmod +x /usr/local/bin/service-mesh

# Copy config
cp config.yaml /etc/mesh/

# Install systemd service
cp service-mesh.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable service-mesh

echo "Service mesh installed successfully"
echo "Start with: systemctl start service-mesh"
```

## Multi-Cluster Setup

### Cluster Configuration

```yaml
# cluster-config.yaml
clusters:
  - name: us-west-2
    endpoints:
      - https://mesh-control.us-west-2.example.com:8443
    ca_cert: /etc/mesh/clusters/us-west-2-ca.crt

  - name: eu-west-1
    endpoints:
      - https://mesh-control.eu-west-1.example.com:8443
    ca_cert: /etc/mesh/clusters/eu-west-1-ca.crt

federation:
  enabled: true
  local_cluster: us-west-2
  sync_interval: 30s
```

### Cross-Cluster Service Discovery

```yaml
# cross-cluster-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: remote-service
  namespace: default
  annotations:
    mesh.io/cluster: eu-west-1
spec:
  type: ExternalName
  externalName: remote-service.eu-west-1.mesh.local
  ports:
  - port: 80
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'service-mesh'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_mesh]
      regex: enabled
      action: keep
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
      regex: (.+)
      target_label: __address__
      replacement: ${1}:${2}
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Service Mesh Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(mesh_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Success Rate",
        "targets": [
          {
            "expr": "rate(mesh_requests_success[5m]) / rate(mesh_requests_total[5m])"
          }
        ]
      },
      {
        "title": "P99 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(mesh_request_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Jaeger Integration

```yaml
# jaeger-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: service-mesh
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:latest
        ports:
        - containerPort: 16686
          name: ui
        - containerPort: 14268
          name: collector
        env:
        - name: COLLECTOR_ZIPKIN_HTTP_PORT
          value: "9411"
```

## Security Hardening

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mesh-ingress
  namespace: default
spec:
  podSelector:
    matchLabels:
      mesh: enabled
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          mesh: enabled
    ports:
    - protocol: TCP
      port: 15001
```

### Certificate Rotation

```bash
#!/bin/bash
# rotate-certs.sh

# Generate new certificates
openssl req -x509 -newkey rsa:4096 -keyout new-key.pem -out new-cert.pem -days 90 -nodes

# Update Kubernetes secret
kubectl create secret tls mesh-certs \
  --cert=new-cert.pem \
  --key=new-key.pem \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new certs
kubectl rollout restart deployment -n default
```

### Security Scanning

```dockerfile
# security-scan.dockerfile
FROM your-registry/service-mesh:latest

# Run security scan
RUN trivy filesystem --severity HIGH,CRITICAL /usr/local/bin/service-mesh

# Run as non-root
USER 1000:1000
```

## Troubleshooting

### Common Issues

#### 1. Certificate Validation Failures

```bash
# Check certificate validity
openssl x509 -in /etc/mesh/cert.pem -text -noout

# Verify certificate chain
openssl verify -CAfile /etc/mesh/ca.crt /etc/mesh/cert.pem

# Test TLS connection
openssl s_client -connect localhost:15001 -cert /etc/mesh/cert.pem -key /etc/mesh/key.pem
```

#### 2. Service Discovery Issues

```bash
# Check registry connectivity
curl -v http://registry:8500/v1/catalog/services

# Debug DNS resolution
nslookup service.namespace.svc.cluster.local

# View registered endpoints
service-mesh registry list
```

#### 3. Circuit Breaker Troubleshooting

```bash
# Check circuit state
curl localhost:15000/stats/circuit_breaker

# Reset circuit breaker
curl -X POST localhost:15000/reset_circuit_breaker

# View circuit breaker metrics
curl localhost:9090/metrics | grep circuit
```

### Debug Mode

```yaml
# Enable debug logging
logging:
  level: debug
  modules:
    - service_mesh::proxy: trace
    - service_mesh::discovery: debug
    - service_mesh::cert: info
```

### Health Checks

```bash
# Liveness check
curl http://localhost:15000/healthz

# Readiness check
curl http://localhost:15000/ready

# Full status
curl http://localhost:15000/status
```

## Migration Guide

### From Istio

```bash
# Export Istio configuration
istioctl proxy-config all deployment/app -o json > istio-config.json

# Convert to service-mesh format
service-mesh migrate from-istio --input istio-config.json --output mesh-config.yaml

# Apply new configuration
kubectl apply -f mesh-config.yaml
```

### From Linkerd

```bash
# Export Linkerd configuration
linkerd viz edges -n default -o json > linkerd-edges.json

# Convert policies
service-mesh migrate from-linkerd --edges linkerd-edges.json --output policies.yaml
```

### Gradual Rollout

```yaml
# canary-rollout.yaml
apiVersion: v1
kind: Service
metadata:
  name: app
spec:
  selector:
    app: app
  ports:
  - port: 80
    targetPort: 8080

---
apiVersion: v1
kind: Service
metadata:
  name: app-mesh
spec:
  selector:
    app: app
    mesh: enabled
  ports:
  - port: 80
    targetPort: 15001
```

## Performance Tuning

### Connection Pool Settings

```yaml
connection_pool:
  max_connections: 1000
  max_pending_requests: 100
  connect_timeout: 10s
  h2:
    max_concurrent_streams: 100
    initial_stream_window_size: 65536
    initial_connection_window_size: 1048576
```

### Resource Limits

```yaml
resources:
  limits:
    cpu: 100m
    memory: 128Mi
  requests:
    cpu: 10m
    memory: 32Mi
```

### JVM Options (if using JVM-based components)

```bash
JAVA_OPTS="-Xmx512m -Xms256m -XX:MaxMetaspaceSize=128m -XX:+UseG1GC"
```

## Backup and Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Backup certificates
tar -czf certs-backup-$(date +%Y%m%d).tar.gz /etc/mesh/certs/

# Backup configuration
cp -r /etc/mesh/ /backup/mesh-config-$(date +%Y%m%d)/

# Backup registry data
curl http://registry:8500/v1/snapshot > registry-snapshot-$(date +%Y%m%d).snap
```

### Recovery Procedure

```bash
# Restore certificates
tar -xzf certs-backup-20240101.tar.gz -C /

# Restore configuration
cp -r /backup/mesh-config-20240101/* /etc/mesh/

# Restore registry
curl -X PUT --data-binary @registry-snapshot-20240101.snap http://registry:8500/v1/snapshot
```

## Maintenance

### Rolling Updates

```bash
# Update deployment with zero downtime
kubectl set image deployment/mesh-control-plane mesh-ca=your-registry/mesh-ca:v2.0.0 -n service-mesh

# Monitor rollout
kubectl rollout status deployment/mesh-control-plane -n service-mesh

# Rollback if needed
kubectl rollout undo deployment/mesh-control-plane -n service-mesh
```

### Cleanup

```bash
# Remove unused certificates
service-mesh certs cleanup --older-than 30d

# Clean up stale endpoints
service-mesh registry cleanup --unhealthy --older-than 1h

# Purge old metrics
curl -X POST http://localhost:15000/metrics/purge
```