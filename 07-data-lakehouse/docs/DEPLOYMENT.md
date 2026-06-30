# Data Lakehouse Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployments](#cloud-deployments)
6. [Production Configuration](#production-configuration)
7. [Monitoring & Observability](#monitoring--observability)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

#### Minimum Requirements
- CPU: 4 cores
- RAM: 8 GB
- Storage: 100 GB SSD
- OS: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+ with WSL2

#### Recommended Production Requirements
- CPU: 16+ cores
- RAM: 64 GB+
- Storage: 1 TB+ NVMe SSD
- Network: 10 Gbps+

### Software Dependencies

```bash
# Python 3.8+
python --version

# Java 8 or 11 (for Spark)
java -version

# Apache Spark 3.3+
spark-submit --version

# Docker (optional)
docker --version

# Kubernetes (optional)
kubectl version
```

### Python Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pyspark>=3.3.0
delta-spark>=2.3.0
pandas>=1.5.0
numpy>=1.23.0
pyarrow>=10.0.0
boto3>=1.26.0  # For AWS
azure-storage-blob>=12.14.0  # For Azure
google-cloud-storage>=2.7.0  # For GCP
prometheus-client>=0.15.0
structlog>=22.3.0
pydantic>=1.10.0
```

---

## Local Development

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/data-lakehouse.git
cd data-lakehouse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install package in editable mode
```

### 2. Configuration

Create local configuration file:

```python
# config/local.py
from lakehouse.config import LakehouseConfig

config = LakehouseConfig(
    bronze_path="./data/bronze",
    silver_path="./data/silver",
    gold_path="./data/gold",
    checkpoint_path="./checkpoints",
    enable_cdc=True,
    vacuum_retention_hours=24,
    optimize_interval_hours=4
)
```

### 3. Run Local Tests

```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests
pytest tests/integration -v

# Run with coverage
pytest --cov=lakehouse --cov-report=html
```

### 4. Start Local Development Server

```python
# run_local.py
from pyspark.sql import SparkSession
from lakehouse.processor import LakehouseProcessor
from config.local import config

# Initialize Spark
spark = SparkSession.builder \
    .appName("lakehouse-local") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Initialize processor
processor = LakehouseProcessor(spark, config)

# Run sample pipeline
if __name__ == "__main__":
    processor.bronze_ingestion(
        source_path="./sample_data/*.json",
        bronze_path=config.bronze_path + "/events",
        source_name="local_test"
    )
```

---

## Docker Deployment

### 1. Dockerfile

```dockerfile
# Dockerfile
FROM openjdk:11-jre-slim

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYSPARK_PYTHON=python3

# Download and install Spark
RUN wget -q https://dlcdn.apache.org/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz && \
    tar -xzf spark-3.4.0-bin-hadoop3.tgz && \
    mv spark-3.4.0-bin-hadoop3 $SPARK_HOME && \
    rm spark-3.4.0-bin-hadoop3.tgz

# Download Delta Lake JARs
RUN wget -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/io/delta/delta-core_2.12/2.3.0/delta-core_2.12-2.3.0.jar && \
    wget -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/io/delta/delta-storage/2.3.0/delta-storage-2.3.0.jar

# Expose ports
EXPOSE 4040 8080 7077

# Entry point
ENTRYPOINT ["spark-submit"]
CMD ["--master", "local[*]", "src/main.py"]
```

### 2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  spark-master:
    build: .
    container_name: lakehouse-master
    ports:
      - "7077:7077"
      - "8080:8080"
    environment:
      - SPARK_MODE=master
      - SPARK_MASTER_HOST=spark-master
      - SPARK_MASTER_PORT=7077
      - SPARK_MASTER_WEBUI_PORT=8080
    volumes:
      - ./data:/data
      - ./checkpoints:/checkpoints
    networks:
      - lakehouse-network
    command: ["spark", "start-master"]

  spark-worker:
    build: .
    container_name: lakehouse-worker
    depends_on:
      - spark-master
    ports:
      - "8081:8081"
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER=spark://spark-master:7077
      - SPARK_WORKER_CORES=2
      - SPARK_WORKER_MEMORY=2g
      - SPARK_WORKER_WEBUI_PORT=8081
    volumes:
      - ./data:/data
      - ./checkpoints:/checkpoints
    networks:
      - lakehouse-network
    command: ["spark", "start-worker", "spark://spark-master:7077"]

  lakehouse-app:
    build: .
    container_name: lakehouse-app
    depends_on:
      - spark-master
      - spark-worker
    environment:
      - SPARK_MASTER=spark://spark-master:7077
      - BRONZE_PATH=/data/bronze
      - SILVER_PATH=/data/silver
      - GOLD_PATH=/data/gold
      - CHECKPOINT_PATH=/checkpoints
    volumes:
      - ./data:/data
      - ./checkpoints:/checkpoints
      - ./src:/app/src
    networks:
      - lakehouse-network
    command: ["--master", "spark://spark-master:7077", "/app/src/main.py"]

  # Optional: Jupyter for interactive development
  jupyter:
    image: jupyter/pyspark-notebook:latest
    container_name: lakehouse-jupyter
    ports:
      - "8888:8888"
    environment:
      - SPARK_MASTER=spark://spark-master:7077
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/data
    networks:
      - lakehouse-network

networks:
  lakehouse-network:
    driver: bridge
```

### 3. Build and Run

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f lakehouse-app

# Scale workers
docker-compose up -d --scale spark-worker=3

# Stop services
docker-compose down
```

---

## Kubernetes Deployment

### 1. Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: lakehouse

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: lakehouse-config
  namespace: lakehouse
data:
  spark-defaults.conf: |
    spark.master                     k8s://https://kubernetes.default.svc.cluster.local
    spark.submit.deployMode          cluster
    spark.kubernetes.namespace       lakehouse
    spark.kubernetes.container.image your-registry/lakehouse:latest
    spark.sql.extensions            io.delta.sql.DeltaSparkSessionExtension
    spark.sql.catalog.spark_catalog org.apache.spark.sql.delta.catalog.DeltaCatalog

  lakehouse.yaml: |
    bronze_path: s3a://your-bucket/bronze
    silver_path: s3a://your-bucket/silver
    gold_path: s3a://your-bucket/gold
    checkpoint_path: s3a://your-bucket/checkpoints
```

### 2. Spark Operator Deployment

```yaml
# k8s/spark-operator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spark-operator
  namespace: lakehouse
spec:
  replicas: 1
  selector:
    matchLabels:
      app: spark-operator
  template:
    metadata:
      labels:
        app: spark-operator
    spec:
      serviceAccountName: spark-operator
      containers:
      - name: spark-operator
        image: gcr.io/spark-operator/spark-operator:latest
        args:
        - -namespace=lakehouse
        - -enable-webhook=true
        - -enable-metrics=true
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### 3. SparkApplication CRD

```yaml
# k8s/spark-application.yaml
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: lakehouse-processor
  namespace: lakehouse
spec:
  type: Python
  pythonVersion: "3"
  mode: cluster
  image: your-registry/lakehouse:latest
  imagePullPolicy: Always
  mainApplicationFile: local:///app/src/main.py
  sparkVersion: "3.4.0"

  driver:
    cores: 2
    coreLimit: "2000m"
    memory: "4g"
    labels:
      app: lakehouse-driver
    serviceAccount: spark
    volumeMounts:
    - name: config
      mountPath: /config

  executor:
    cores: 2
    instances: 3
    memory: "4g"
    labels:
      app: lakehouse-executor
    volumeMounts:
    - name: config
      mountPath: /config

  volumes:
  - name: config
    configMap:
      name: lakehouse-config

  deps:
    jars:
    - local:///opt/spark/jars/delta-core_2.12-2.3.0.jar
    - local:///opt/spark/jars/delta-storage-2.3.0.jar

  sparkConf:
    "spark.kubernetes.authenticate.driver.serviceAccountName": spark
    "spark.kubernetes.authenticate.executor.serviceAccountName": spark
    "spark.hadoop.fs.s3a.access.key": "YOUR_ACCESS_KEY"
    "spark.hadoop.fs.s3a.secret.key": "YOUR_SECRET_KEY"
```

### 4. Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/spark-operator.yaml
kubectl apply -f k8s/spark-application.yaml

# Check status
kubectl get sparkapplication -n lakehouse
kubectl logs -f lakehouse-processor-driver -n lakehouse

# Port forward for UI
kubectl port-forward lakehouse-processor-driver 4040:4040 -n lakehouse
```

---

## Cloud Deployments

### AWS EMR

```bash
# Create EMR cluster
aws emr create-cluster \
  --name "lakehouse-cluster" \
  --release-label emr-6.9.0 \
  --applications Name=Spark Name=JupyterHub \
  --ec2-attributes KeyName=your-key,SubnetId=subnet-xxx \
  --instance-groups \
    InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge \
    InstanceGroupType=CORE,InstanceCount=3,InstanceType=m5.2xlarge \
  --configurations file://emr-config.json \
  --bootstrap-actions Path=s3://your-bucket/bootstrap.sh \
  --steps file://steps.json \
  --log-uri s3://your-bucket/logs/

# emr-config.json
[
  {
    "Classification": "spark-defaults",
    "Properties": {
      "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
      "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog"
    }
  }
]

# Submit job
aws emr add-steps \
  --cluster-id j-XXXXX \
  --steps Type=Spark,Name="LakehouseJob",\
    Args=[--deploy-mode,cluster,--master,yarn,\
    s3://your-bucket/scripts/lakehouse_job.py]
```

### Azure Databricks

```python
# Databricks notebook
# Cell 1: Install dependencies
%pip install delta-spark

# Cell 2: Configure
spark.conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
spark.conf.set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

# Cell 3: Mount storage
dbutils.fs.mount(
  source = "wasbs://container@storage.blob.core.windows.net/",
  mount_point = "/mnt/lakehouse",
  extra_configs = {"fs.azure.account.key.storage.blob.core.windows.net": dbutils.secrets.get("lakehouse", "storage-key")}
)

# Cell 4: Run pipeline
from lakehouse.processor import LakehouseProcessor
from lakehouse.config import LakehouseConfig

config = LakehouseConfig(
    bronze_path="/mnt/lakehouse/bronze",
    silver_path="/mnt/lakehouse/silver",
    gold_path="/mnt/lakehouse/gold"
)

processor = LakehouseProcessor(spark, config)
```

### Google Cloud Dataproc

```bash
# Create cluster
gcloud dataproc clusters create lakehouse-cluster \
  --region=us-central1 \
  --zone=us-central1-a \
  --master-machine-type=n1-standard-4 \
  --worker-machine-type=n1-standard-4 \
  --num-workers=3 \
  --image-version=2.0 \
  --optional-components=JUPYTER \
  --initialization-actions=gs://your-bucket/init-scripts/install-delta.sh \
  --properties=spark:spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension

# Submit job
gcloud dataproc jobs submit pyspark \
  gs://your-bucket/scripts/lakehouse_job.py \
  --cluster=lakehouse-cluster \
  --region=us-central1 \
  --jars=gs://your-bucket/jars/delta-core_2.12-2.3.0.jar
```

---

## Production Configuration

### 1. Spark Configuration

```properties
# spark-defaults.conf
spark.master                       yarn
spark.submit.deployMode            cluster
spark.executor.instances           10
spark.executor.cores              4
spark.executor.memory             16g
spark.driver.memory               8g
spark.driver.maxResultSize        4g

# Delta optimization
spark.databricks.delta.optimizeWrite.enabled         true
spark.databricks.delta.autoCompact.enabled          true
spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite  true
spark.databricks.delta.properties.defaults.autoOptimize.autoCompact    true

# Adaptive Query Execution
spark.sql.adaptive.enabled                          true
spark.sql.adaptive.coalescePartitions.enabled       true
spark.sql.adaptive.skewJoin.enabled                true

# Performance tuning
spark.sql.shuffle.partitions                        200
spark.serializer                                   org.apache.spark.serializer.KryoSerializer
spark.sql.files.maxPartitionBytes                   134217728
spark.sql.files.openCostInBytes                    4194304
```

### 2. Security Configuration

```yaml
# security-config.yaml
authentication:
  enabled: true
  type: kerberos
  keytab: /etc/security/keytabs/lakehouse.keytab
  principal: lakehouse@DOMAIN.COM

authorization:
  enabled: true
  provider: ranger
  policy_repository: lakehouse-policies

encryption:
  at_rest:
    enabled: true
    algorithm: AES256
    key_provider: aws_kms
    master_key: arn:aws:kms:region:account:key/xxx

  in_transit:
    enabled: true
    protocol: TLS1.3
    cipher_suites:
      - TLS_AES_256_GCM_SHA384
      - TLS_AES_128_GCM_SHA256

data_masking:
  enabled: true
  rules:
    - column: email
      type: hash
    - column: ssn
      type: redact
    - column: phone
      type: partial
      show_last: 4
```

### 3. High Availability Setup

```yaml
# ha-config.yaml
cluster:
  mode: high_availability

  master:
    instances: 3
    zookeeper:
      servers:
        - zk1.domain.com:2181
        - zk2.domain.com:2181
        - zk3.domain.com:2181
      path: /lakehouse

  metadata_store:
    type: external_metastore
    uri: thrift://metastore.domain.com:9083

  state_store:
    type: redis
    endpoints:
      - redis1.domain.com:6379
      - redis2.domain.com:6379
    cluster_mode: true

  checkpointing:
    enabled: true
    interval: 100
    path: hdfs://namenode:9000/checkpoints
```

---

## Monitoring & Observability

### 1. Metrics Collection

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
job_counter = Counter('lakehouse_jobs_total', 'Total number of jobs', ['layer', 'status'])
job_duration = Histogram('lakehouse_job_duration_seconds', 'Job duration', ['layer'])
active_streams = Gauge('lakehouse_active_streams', 'Number of active streams')
data_quality_score = Gauge('lakehouse_data_quality_score', 'Data quality score', ['table'])

# Instrument code
def monitor_job(layer):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                job_counter.labels(layer=layer, status='success').inc()
                return result
            except Exception as e:
                job_counter.labels(layer=layer, status='failure').inc()
                raise
            finally:
                job_duration.labels(layer=layer).observe(time.time() - start)
        return wrapper
    return decorator

# Start metrics server
start_http_server(8000)
```

### 2. Logging Configuration

```python
# logging_config.py
import structlog
import logging

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Configure for production
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            "/var/log/lakehouse/app.log",
            maxBytes=100_000_000,
            backupCount=10
        )
    ]
)
```

### 3. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Lakehouse Monitoring",
    "panels": [
      {
        "title": "Job Success Rate",
        "targets": [
          {
            "expr": "rate(lakehouse_jobs_total{status='success'}[5m]) / rate(lakehouse_jobs_total[5m])"
          }
        ]
      },
      {
        "title": "Average Job Duration",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, lakehouse_job_duration_seconds)"
          }
        ]
      },
      {
        "title": "Active Streams",
        "targets": [
          {
            "expr": "lakehouse_active_streams"
          }
        ]
      },
      {
        "title": "Data Quality Score",
        "targets": [
          {
            "expr": "lakehouse_data_quality_score"
          }
        ]
      }
    ]
  }
}
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

```bash
# Increase executor memory
spark.executor.memory=32g
spark.executor.memoryOverhead=8g

# Enable off-heap memory
spark.memory.offHeap.enabled=true
spark.memory.offHeap.size=16g
```

#### 2. Slow Queries

```python
# Analyze query plan
df.explain(True)

# Check partition pruning
spark.sql("ANALYZE TABLE my_table COMPUTE STATISTICS")

# Enable adaptive query execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

#### 3. Delta Lake Issues

```python
# Repair corrupted table
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "/path/to/table")
delta_table.restoreToVersion(last_good_version)

# Fix transaction log
spark.sql("FSCK REPAIR TABLE my_table")

# Vacuum with retention check
delta_table.vacuum(168)  # 7 days retention
```

#### 4. Streaming Failures

```python
# Reset checkpoint
import shutil
shutil.rmtree("/checkpoints/failed_stream")

# Restart with new checkpoint
stream = processor.start_bronze_stream(
    source_path=source,
    bronze_path=bronze,
    checkpoint_path="/checkpoints/new_stream"
)
```

### Debug Commands

```bash
# Check Spark UI
kubectl port-forward spark-driver-pod 4040:4040

# View executor logs
kubectl logs -f spark-executor-pod

# SSH into container
kubectl exec -it spark-driver-pod -- /bin/bash

# Check HDFS health
hdfs dfsadmin -report

# Test S3 connectivity
aws s3 ls s3://your-bucket/ --debug
```

### Performance Tuning Checklist

- [ ] Enable adaptive query execution
- [ ] Configure appropriate shuffle partitions
- [ ] Use broadcast joins for small tables
- [ ] Enable dynamic allocation
- [ ] Optimize file sizes (128-256 MB)
- [ ] Use Z-ordering on filter columns
- [ ] Regular OPTIMIZE and VACUUM
- [ ] Monitor and adjust memory settings
- [ ] Use caching for frequently accessed data
- [ ] Enable speculative execution

---

## Support

For additional support:
- Documentation: https://docs.lakehouse.io
- Issues: https://github.com/your-org/lakehouse/issues
- Slack: #lakehouse-support
- Email: lakehouse-team@your-org.com