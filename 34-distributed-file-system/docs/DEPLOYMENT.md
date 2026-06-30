# HDFS Deployment Guide

## Prerequisites

### System Requirements

#### NameNode (Master)
- **OS**: Linux (Ubuntu 20.04+, CentOS 7+, RHEL 7+)
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum (1GB per million files/blocks)
- **Disk**: 100GB SSD for metadata and logs
- **Network**: 1Gbps minimum, 10Gbps recommended
- **Python**: 3.8+

#### DataNode (Workers)
- **OS**: Linux (Ubuntu 20.04+, CentOS 7+, RHEL 7+)
- **CPU**: 2+ cores
- **RAM**: 4GB minimum
- **Disk**: Large capacity HDDs (1TB+ each)
- **Network**: 1Gbps minimum
- **Python**: 3.8+

### Software Dependencies

```bash
# Install Python and pip
sudo apt-get update
sudo apt-get install python3.8 python3-pip

# Install required Python packages
pip3 install asyncio
pip3 install aiofiles
pip3 install pyyaml
pip3 install psutil
```

## Installation

### 1. Clone the Repository

```bash
# Clone the HDFS project
git clone https://github.com/your-org/hdfs-python.git
cd hdfs-python

# Install the package
pip3 install -e .
```

### 2. Directory Structure

Create the necessary directories:

```bash
# On NameNode
sudo mkdir -p /var/hdfs/namenode
sudo mkdir -p /var/log/hdfs
sudo mkdir -p /etc/hdfs

# On DataNodes
sudo mkdir -p /var/hdfs/datanode
sudo mkdir -p /var/log/hdfs
sudo mkdir -p /etc/hdfs

# Set permissions
sudo chown -R hdfs:hdfs /var/hdfs
sudo chown -R hdfs:hdfs /var/log/hdfs
```

## Configuration

### NameNode Configuration

Create `/etc/hdfs/namenode.yaml`:

```yaml
# NameNode Configuration
namenode:
  # Network settings
  host: 0.0.0.0
  port: 9000

  # File system settings
  default_replication: 3
  default_block_size: 134217728  # 128MB

  # Timing settings
  heartbeat_interval: 3.0
  heartbeat_timeout: 10.0
  checkpoint_interval: 3600.0

  # Safe mode settings
  safe_mode_threshold: 0.999
  safe_mode_extension: 30.0

  # Storage settings
  namespace_dir: /var/hdfs/namenode
  edit_log_dir: /var/hdfs/namenode/edits
  checkpoint_dir: /var/hdfs/namenode/checkpoint

  # Performance tuning
  max_threads: 100
  handler_count: 10

  # Memory settings
  heap_size: 4096  # MB

  # Monitoring
  metrics_port: 9870
  enable_metrics: true

# Logging configuration
logging:
  level: INFO
  file: /var/log/hdfs/namenode.log
  max_size: 100MB
  backup_count: 10
```

### DataNode Configuration

Create `/etc/hdfs/datanode.yaml`:

```yaml
# DataNode Configuration
datanode:
  # Identity
  node_id: ${HOSTNAME}

  # NameNode connection
  namenode_host: namenode.example.com
  namenode_port: 9000

  # Network settings
  host: 0.0.0.0
  port: 50010
  ipc_port: 50020

  # Storage settings
  data_dirs:
    - /var/hdfs/datanode/data1
    - /var/hdfs/datanode/data2

  # Performance settings
  max_threads: 50
  transfer_threads: 10
  max_bandwidth: null  # bytes/sec, null for unlimited

  # Timing settings
  heartbeat_interval: 3.0
  block_report_interval: 3600.0

  # Block scanning
  block_scan_interval: 504  # hours (3 weeks)

  # Memory settings
  heap_size: 2048  # MB

  # Monitoring
  metrics_port: 9864
  enable_metrics: true

# Logging configuration
logging:
  level: INFO
  file: /var/log/hdfs/datanode.log
  max_size: 100MB
  backup_count: 10
```

### Client Configuration

Create `/etc/hdfs/client.yaml`:

```yaml
# Client Configuration
client:
  # NameNode connection
  namenode_host: namenode.example.com
  namenode_port: 9000

  # Default settings
  default_block_size: 134217728  # 128MB
  default_replication: 3

  # Performance settings
  parallel_reads: true
  parallel_writes: true
  connection_pool_size: 10

  # Retry settings
  max_retries: 3
  retry_delay: 1.0
  exponential_backoff: true

  # Caching
  enable_cache: true
  cache_ttl: 60
  cache_size: 1000

  # Timeouts
  connection_timeout: 10.0
  read_timeout: 60.0
  write_timeout: 60.0

# Logging configuration
logging:
  level: WARNING
  file: /var/log/hdfs/client.log
```

## Single Node Deployment

For testing or development:

### 1. Start NameNode

```bash
# Initialize namespace
hdfs-namenode format

# Start NameNode
hdfs-namenode start

# Or run in foreground
hdfs-namenode run
```

### 2. Start DataNode

```bash
# Start DataNode
hdfs-datanode start

# Or run in foreground
hdfs-datanode run
```

### 3. Verify Installation

```bash
# Check cluster status
hdfs dfsadmin -report

# Create test file
echo "Hello HDFS" | hdfs dfs -put - /test.txt

# Read test file
hdfs dfs -cat /test.txt
```

## Multi-Node Cluster Deployment

### 1. Prepare All Nodes

On all nodes:

```bash
# Create hdfs user
sudo useradd -m -s /bin/bash hdfs

# Set up SSH keys for hdfs user
sudo -u hdfs ssh-keygen -t rsa -P ""
```

### 2. Configure SSH Access

On NameNode:

```bash
# Copy SSH key to all DataNodes
for node in datanode1 datanode2 datanode3; do
    ssh-copy-id hdfs@$node
done
```

### 3. Deploy NameNode

On the master node:

```bash
# Format NameNode (only on first setup!)
sudo -u hdfs hdfs-namenode format

# Start NameNode
sudo systemctl start hdfs-namenode

# Enable auto-start
sudo systemctl enable hdfs-namenode
```

### 4. Deploy DataNodes

On each worker node:

```bash
# Start DataNode
sudo systemctl start hdfs-datanode

# Enable auto-start
sudo systemctl enable hdfs-datanode
```

### 5. Verify Cluster

```bash
# Check cluster health
hdfs dfsadmin -report

# Check live nodes
hdfs dfsadmin -printTopology
```

## Systemd Service Configuration

### NameNode Service

Create `/etc/systemd/system/hdfs-namenode.service`:

```ini
[Unit]
Description=HDFS NameNode
After=network.target

[Service]
Type=simple
User=hdfs
Group=hdfs
ExecStart=/usr/local/bin/hdfs-namenode run
ExecStop=/usr/local/bin/hdfs-namenode stop
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

# Environment
Environment="PYTHONPATH=/opt/hdfs/lib"
Environment="HDFS_CONFIG=/etc/hdfs/namenode.yaml"

[Install]
WantedBy=multi-user.target
```

### DataNode Service

Create `/etc/systemd/system/hdfs-datanode.service`:

```ini
[Unit]
Description=HDFS DataNode
After=network.target

[Service]
Type=simple
User=hdfs
Group=hdfs
ExecStart=/usr/local/bin/hdfs-datanode run
ExecStop=/usr/local/bin/hdfs-datanode stop
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

# Environment
Environment="PYTHONPATH=/opt/hdfs/lib"
Environment="HDFS_CONFIG=/etc/hdfs/datanode.yaml"

[Install]
WantedBy=multi-user.target
```

## Docker Deployment

### Dockerfile for NameNode

```dockerfile
FROM python:3.8-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    supervisor \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Create hdfs user
RUN useradd -m -s /bin/bash hdfs

# Copy application
COPY --chown=hdfs:hdfs . /opt/hdfs
WORKDIR /opt/hdfs

# Install Python dependencies
RUN pip install -r requirements.txt

# Create directories
RUN mkdir -p /var/hdfs/namenode /var/log/hdfs /etc/hdfs \
    && chown -R hdfs:hdfs /var/hdfs /var/log/hdfs /etc/hdfs

# Copy configuration
COPY --chown=hdfs:hdfs config/namenode.yaml /etc/hdfs/

# Expose ports
EXPOSE 9000 9870

USER hdfs
CMD ["python", "-m", "hdfs.namenode"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  namenode:
    build:
      context: .
      dockerfile: Dockerfile.namenode
    container_name: hdfs-namenode
    hostname: namenode
    ports:
      - "9000:9000"
      - "9870:9870"
    volumes:
      - namenode_data:/var/hdfs/namenode
      - ./config/namenode.yaml:/etc/hdfs/namenode.yaml
    networks:
      - hdfs-network
    environment:
      - HDFS_CONFIG=/etc/hdfs/namenode.yaml

  datanode1:
    build:
      context: .
      dockerfile: Dockerfile.datanode
    container_name: hdfs-datanode1
    hostname: datanode1
    depends_on:
      - namenode
    ports:
      - "50010:50010"
    volumes:
      - datanode1_data:/var/hdfs/datanode
      - ./config/datanode.yaml:/etc/hdfs/datanode.yaml
    networks:
      - hdfs-network
    environment:
      - HDFS_CONFIG=/etc/hdfs/datanode.yaml
      - NAMENODE_HOST=namenode

  datanode2:
    build:
      context: .
      dockerfile: Dockerfile.datanode
    container_name: hdfs-datanode2
    hostname: datanode2
    depends_on:
      - namenode
    ports:
      - "50011:50010"
    volumes:
      - datanode2_data:/var/hdfs/datanode
      - ./config/datanode.yaml:/etc/hdfs/datanode.yaml
    networks:
      - hdfs-network
    environment:
      - HDFS_CONFIG=/etc/hdfs/datanode.yaml
      - NAMENODE_HOST=namenode

  datanode3:
    build:
      context: .
      dockerfile: Dockerfile.datanode
    container_name: hdfs-datanode3
    hostname: datanode3
    depends_on:
      - namenode
    ports:
      - "50012:50010"
    volumes:
      - datanode3_data:/var/hdfs/datanode
      - ./config/datanode.yaml:/etc/hdfs/datanode.yaml
    networks:
      - hdfs-network
    environment:
      - HDFS_CONFIG=/etc/hdfs/datanode.yaml
      - NAMENODE_HOST=namenode

volumes:
  namenode_data:
  datanode1_data:
  datanode2_data:
  datanode3_data:

networks:
  hdfs-network:
    driver: bridge
```

### Deploy with Docker Compose

```bash
# Build images
docker-compose build

# Start cluster
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f namenode

# Stop cluster
docker-compose down
```

## Kubernetes Deployment

### NameNode StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: hdfs-namenode
spec:
  serviceName: hdfs-namenode
  replicas: 1
  selector:
    matchLabels:
      app: hdfs-namenode
  template:
    metadata:
      labels:
        app: hdfs-namenode
    spec:
      containers:
      - name: namenode
        image: hdfs/namenode:latest
        ports:
        - containerPort: 9000
          name: rpc
        - containerPort: 9870
          name: http
        volumeMounts:
        - name: namenode-data
          mountPath: /var/hdfs/namenode
        - name: config
          mountPath: /etc/hdfs
        env:
        - name: HDFS_CONFIG
          value: /etc/hdfs/namenode.yaml
      volumes:
      - name: config
        configMap:
          name: hdfs-config
  volumeClaimTemplates:
  - metadata:
      name: namenode-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### DataNode DaemonSet

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: hdfs-datanode
spec:
  selector:
    matchLabels:
      app: hdfs-datanode
  template:
    metadata:
      labels:
        app: hdfs-datanode
    spec:
      containers:
      - name: datanode
        image: hdfs/datanode:latest
        ports:
        - containerPort: 50010
          name: data
        - containerPort: 50020
          name: ipc
        volumeMounts:
        - name: datanode-data
          mountPath: /var/hdfs/datanode
        - name: config
          mountPath: /etc/hdfs
        env:
        - name: HDFS_CONFIG
          value: /etc/hdfs/datanode.yaml
        - name: NAMENODE_HOST
          value: hdfs-namenode
      volumes:
      - name: datanode-data
        hostPath:
          path: /var/hdfs/datanode
      - name: config
        configMap:
          name: hdfs-config
```

## Production Best Practices

### 1. High Availability

Configure NameNode HA with ZooKeeper:

```yaml
# HA Configuration
ha:
  enabled: true
  cluster_name: hdfs-cluster
  namenode_ids:
    - nn1
    - nn2
  zookeeper_quorum:
    - zk1.example.com:2181
    - zk2.example.com:2181
    - zk3.example.com:2181
```

### 2. Security

#### Enable Kerberos

```yaml
security:
  authentication: kerberos
  kerberos:
    principal: hdfs/_HOST@EXAMPLE.COM
    keytab: /etc/hdfs/hdfs.keytab
```

#### Enable TLS

```yaml
security:
  encryption: tls
  tls:
    keystore: /etc/hdfs/keystore.jks
    keystore_password: ${KEYSTORE_PASSWORD}
    truststore: /etc/hdfs/truststore.jks
    truststore_password: ${TRUSTSTORE_PASSWORD}
```

### 3. Monitoring

#### Prometheus Integration

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    path: /metrics
```

#### Grafana Dashboards

Import provided dashboards:
- `dashboards/hdfs-overview.json`
- `dashboards/hdfs-namenode.json`
- `dashboards/hdfs-datanode.json`

### 4. Backup and Recovery

#### NameNode Backup

```bash
# Backup script
#!/bin/bash
BACKUP_DIR=/backup/hdfs/namenode/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# Save checkpoint
hdfs dfsadmin -safemode enter
hdfs dfsadmin -saveNamespace
hdfs dfsadmin -safemode leave

# Copy metadata
cp -r /var/hdfs/namenode/current $BACKUP_DIR/
```

#### Disaster Recovery

```bash
# Restore from backup
#!/bin/bash
BACKUP_DIR=/backup/hdfs/namenode/20240101

# Stop NameNode
systemctl stop hdfs-namenode

# Restore metadata
rm -rf /var/hdfs/namenode/current
cp -r $BACKUP_DIR/current /var/hdfs/namenode/

# Start NameNode
systemctl start hdfs-namenode
```

### 5. Performance Tuning

#### JVM Tuning

```bash
# NameNode JVM options
export NAMENODE_OPTS="-Xmx4g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"

# DataNode JVM options
export DATANODE_OPTS="-Xmx2g -XX:+UseG1GC"
```

#### OS Tuning

```bash
# Increase file descriptors
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Network tuning
echo "net.core.somaxconn = 1024" >> /etc/sysctl.conf
echo "net.ipv4.tcp_tw_reuse = 1" >> /etc/sysctl.conf
sysctl -p
```

## Troubleshooting

### Common Issues

#### NameNode Won't Start

```bash
# Check logs
tail -f /var/log/hdfs/namenode.log

# Verify configuration
hdfs-namenode validate-config

# Check namespace
hdfs namenode -recover
```

#### DataNode Not Connecting

```bash
# Check connectivity
telnet namenode.example.com 9000

# Check logs
tail -f /var/log/hdfs/datanode.log

# Verify configuration
hdfs-datanode validate-config
```

#### Slow Performance

```bash
# Check network
iperf3 -c namenode.example.com

# Check disk I/O
iostat -x 1

# Check DataNode health
hdfs dfsadmin -report
```

### Health Checks

```bash
# Cluster health
hdfs dfsadmin -report

# File system check
hdfs fsck /

# Check under-replicated blocks
hdfs fsck / -blocks -locations -racks

# Check DataNode storage
hdfs dfsadmin -getDatanodeInfo datanode1:50010
```

## Maintenance

### Regular Tasks

```bash
# Daily
- Monitor cluster health
- Check logs for errors
- Review metrics

# Weekly
- Run fsck
- Review under-replicated blocks
- Clean old logs

# Monthly
- Review capacity planning
- Update documentation
- Performance analysis

# Quarterly
- Security audit
- Disaster recovery test
- Hardware maintenance
```

### Upgrade Procedure

```bash
# 1. Backup metadata
hdfs dfsadmin -safemode enter
hdfs dfsadmin -saveNamespace

# 2. Stop services
systemctl stop hdfs-datanode
systemctl stop hdfs-namenode

# 3. Upgrade software
pip install --upgrade hdfs-python

# 4. Start NameNode
systemctl start hdfs-namenode

# 5. Verify NameNode
hdfs dfsadmin -report

# 6. Start DataNodes
systemctl start hdfs-datanode

# 7. Exit safe mode
hdfs dfsadmin -safemode leave
```