# ML Compiler Deployment Guide

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment Scenarios](#deployment-scenarios)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

## Requirements

### System Requirements

#### Minimum Requirements
- CPU: x86-64 or ARM64 processor
- RAM: 8GB
- Storage: 10GB free space
- OS: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10+

#### Recommended Requirements
- CPU: Intel Xeon or AMD EPYC with AVX-512
- RAM: 32GB or more
- Storage: 50GB SSD
- GPU: NVIDIA GPU with CUDA 11.0+ (optional)

### Software Dependencies

#### Required
```bash
# Python
python >= 3.8

# Build tools
cmake >= 3.16
gcc >= 7.5 or clang >= 10.0
make or ninja

# Libraries
numpy >= 1.19.0
pybind11 >= 2.6.0
```

#### Optional (for GPU support)
```bash
# NVIDIA CUDA
cuda >= 11.0
cudnn >= 8.0

# AMD ROCm
rocm >= 4.0

# Intel OneAPI
oneapi >= 2021.1
```

## Installation

### From PyPI

```bash
pip install mlcompiler

# With GPU support
pip install mlcompiler[cuda]

# With all optional dependencies
pip install mlcompiler[all]
```

### From Source

```bash
# Clone repository
git clone https://github.com/your-org/ml-compiler.git
cd ml-compiler

# Install dependencies
pip install -r requirements.txt

# Build and install
python setup.py build
python setup.py install

# Or using pip
pip install -e .
```

### Docker Installation

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install ML Compiler
RUN pip3 install mlcompiler[cuda]

# Set working directory
WORKDIR /app

# Copy application
COPY . /app

# Run application
CMD ["python3", "app.py"]
```

Build and run:
```bash
docker build -t mlcompiler-app .
docker run --gpus all -p 8080:8080 mlcompiler-app
```

### Kubernetes Deployment

```yaml
# mlcompiler-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlcompiler-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlcompiler
  template:
    metadata:
      labels:
        app: mlcompiler
    spec:
      containers:
      - name: mlcompiler
        image: your-registry/mlcompiler:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1  # Request 1 GPU
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        env:
        - name: MLCOMPILER_CONFIG
          value: "/config/config.yaml"
        volumeMounts:
        - name: config
          mountPath: /config
        - name: models
          mountPath: /models
      volumes:
      - name: config
        configMap:
          name: mlcompiler-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mlcompiler-service
spec:
  selector:
    app: mlcompiler
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f mlcompiler-deployment.yaml
```

## Configuration

### Configuration File

Create `mlcompiler.yaml`:

```yaml
# mlcompiler.yaml
compiler:
  target: cuda
  optimization_level: O2
  debug: false
  profile: true

runtime:
  num_threads: 8
  memory_limit: 8GB
  batch_timeout: 100ms
  max_batch_size: 32

cache:
  enabled: true
  directory: /var/cache/mlcompiler
  max_size: 10GB
  ttl: 86400  # 24 hours

logging:
  level: INFO
  file: /var/log/mlcompiler.log
  max_size: 100MB
  max_files: 10

monitoring:
  enabled: true
  prometheus_port: 9090
  metrics_interval: 10s

gpu:
  device_ids: [0, 1]
  memory_fraction: 0.9
  allow_growth: true
```

### Environment Variables

```bash
export MLCOMPILER_CONFIG=/path/to/config.yaml
export MLCOMPILER_CACHE_DIR=/var/cache/mlcompiler
export MLCOMPILER_LOG_LEVEL=DEBUG
export MLCOMPILER_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0,1
```

### Python Configuration

```python
from mlcompiler import Config

config = Config(
    compiler={
        'target': 'cuda',
        'optimization_level': 'O2',
        'mixed_precision': True
    },
    runtime={
        'num_threads': 8,
        'batch_size': 32
    },
    cache={
        'enabled': True,
        'directory': '/tmp/mlcompiler_cache'
    }
)

config.save('mlcompiler_config.json')
```

## Deployment Scenarios

### 1. Single Server Deployment

```python
# server.py
from flask import Flask, request, jsonify
from mlcompiler import MLCompiler, CompilerConfig
import numpy as np

app = Flask(__name__)

# Initialize compiler
config = CompilerConfig.from_file('config.yaml')
compiler = MLCompiler(config)

# Load and compile model
model = compiler.compile_from_onnx('model.onnx')

@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json['data'], dtype=np.float32)
    output = model.run(data)
    return jsonify({'predictions': output.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

Run:
```bash
gunicorn -w 4 -b 0.0.0.0:8080 server:app
```

### 2. Multi-GPU Server

```python
# multi_gpu_server.py
from mlcompiler import MLCompiler, DistributedConfig
import multiprocessing as mp

def worker(gpu_id, model_path, queue):
    """Worker process for each GPU."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    compiler = MLCompiler(target='cuda')
    model = compiler.compile_from_path(model_path)

    while True:
        data = queue.get()
        if data is None:
            break
        result = model.run(data)
        # Process result

def main():
    num_gpus = 4
    model_path = 'model.mlc'

    # Create work queue
    queue = mp.Queue()

    # Start workers
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, model_path, queue))
        p.start()
        workers.append(p)

    # Distribute work
    # ...

if __name__ == '__main__':
    main()
```

### 3. Microservices Architecture

```python
# inference_service.py
from mlcompiler import MLCompiler
from kafka import KafkaConsumer, KafkaProducer
import json
import numpy as np

class InferenceService:
    def __init__(self, model_path, kafka_config):
        self.compiler = MLCompiler()
        self.model = self.compiler.compile_from_path(model_path)

        self.consumer = KafkaConsumer(
            'inference_requests',
            **kafka_config
        )
        self.producer = KafkaProducer(
            **kafka_config
        )

    def run(self):
        for message in self.consumer:
            request = json.loads(message.value)

            # Run inference
            input_data = np.array(request['data'])
            output = self.model.run(input_data)

            # Send response
            response = {
                'request_id': request['id'],
                'predictions': output.tolist()
            }
            self.producer.send(
                'inference_responses',
                value=json.dumps(response).encode()
            )

if __name__ == '__main__':
    service = InferenceService(
        model_path='model.mlc',
        kafka_config={
            'bootstrap_servers': ['localhost:9092'],
            'group_id': 'inference_group'
        }
    )
    service.run()
```

### 4. Serverless Deployment (AWS Lambda)

```python
# lambda_handler.py
import json
import boto3
import numpy as np
from mlcompiler import MLCompiler

# Initialize outside handler for reuse
s3 = boto3.client('s3')
compiler = MLCompiler(target='cpu')

# Download and compile model
s3.download_file('models-bucket', 'model.mlc', '/tmp/model.mlc')
model = compiler.load('/tmp/model.mlc')

def lambda_handler(event, context):
    # Parse input
    input_data = np.array(json.loads(event['body'])['data'])

    # Run inference
    output = model.run(input_data)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'predictions': output.tolist()
        })
    }
```

### 5. Edge Deployment

```python
# edge_deployment.py
from mlcompiler import MLCompiler, EdgeConfig
import time

class EdgeInference:
    def __init__(self, model_path):
        # Configure for edge device
        config = EdgeConfig(
            target='arm_neon',  # ARM NEON optimization
            memory_limit='512MB',
            power_mode='low_power'
        )

        self.compiler = MLCompiler(config)
        self.model = self.compiler.compile_from_path(model_path)

    def run_continuous(self, get_input_fn):
        """Run continuous inference on edge device."""
        while True:
            input_data = get_input_fn()  # Get input from sensor

            # Run inference
            output = self.model.run(input_data)

            # Process output
            self.process_output(output)

            time.sleep(0.1)  # 10 Hz

    def process_output(self, output):
        # Send to cloud, trigger action, etc.
        pass
```

## Performance Tuning

### Compiler Optimization

```python
# Aggressive optimization
config = CompilerConfig(
    optimization_level='O3',
    unsafe_math=True,  # Enable fast math
    vectorize=True,
    parallel=True,
    fusion_threshold=0.8
)

# Profile-guided optimization
compiler = MLCompiler(config)
model = compiler.compile(module)

# Run profiling
profile_data = model.profile(sample_inputs, runs=100)

# Recompile with profile data
optimized = compiler.recompile(model, profile_data)
```

### Memory Optimization

```python
# Enable memory optimization
config = CompilerConfig(
    memory_optimization=True,
    memory_pool_size='4GB',
    enable_inplace=True,
    gradient_checkpointing=True  # For training
)

# Monitor memory usage
stats = model.get_memory_stats()
print(f"Peak memory: {stats.peak_memory}MB")
print(f"Current memory: {stats.current_memory}MB")
```

### Batching Configuration

```python
# Dynamic batching
from mlcompiler.serving import BatchingServer

server = BatchingServer(
    model=model,
    max_batch_size=32,
    batch_timeout_ms=50,
    pad_to_max_batch=False  # Dynamic shapes
)

# Adaptive batching
server.enable_adaptive_batching(
    target_latency_ms=100,
    min_batch_size=1,
    max_batch_size=64
)
```

### Hardware-Specific Tuning

#### CPU Optimization
```python
config = CompilerConfig(
    target='cpu',
    cpu_features=['avx512', 'vnni'],  # Intel features
    num_threads='auto',  # Use all cores
    thread_affinity='compact'  # NUMA-aware
)
```

#### GPU Optimization
```python
config = CompilerConfig(
    target='cuda',
    gpu_arch='sm_86',  # Ampere architecture
    use_tensor_cores=True,
    use_cudnn=True,
    cuda_graphs=True  # Enable CUDA graphs
)
```

## Monitoring

### Metrics Collection

```python
from mlcompiler.monitoring import MetricsCollector

collector = MetricsCollector(
    prometheus_port=9090,
    interval_seconds=10
)

# Register model
collector.register_model(model, name='my_model')

# Custom metrics
collector.record_metric('custom_metric', value)
```

### Available Metrics

```yaml
# Prometheus metrics
mlcompiler_inference_latency_ms
mlcompiler_inference_throughput_qps
mlcompiler_batch_size
mlcompiler_queue_size
mlcompiler_memory_usage_bytes
mlcompiler_gpu_utilization_percent
mlcompiler_compilation_time_seconds
mlcompiler_cache_hit_rate
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "ML Compiler Metrics",
    "panels": [
      {
        "title": "Inference Latency",
        "targets": [{
          "expr": "histogram_quantile(0.99, mlcompiler_inference_latency_ms)"
        }]
      },
      {
        "title": "Throughput",
        "targets": [{
          "expr": "rate(mlcompiler_inference_count[1m])"
        }]
      },
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "mlcompiler_gpu_utilization_percent"
        }]
      }
    ]
  }
}
```

### Logging

```python
import logging
from mlcompiler.logging import setup_logging

# Configure logging
setup_logging(
    level='INFO',
    file='/var/log/mlcompiler.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('mlcompiler')
logger.info('Model compiled successfully')
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Symptoms:** OOM errors during compilation or inference

**Solutions:**
```python
# Reduce memory usage
config = CompilerConfig(
    memory_limit='4GB',
    enable_memory_pool=False,
    batch_size=16  # Reduce batch size
)

# Enable gradient checkpointing for training
config.gradient_checkpointing = True
```

#### 2. Slow Compilation

**Symptoms:** Compilation takes too long

**Solutions:**
```python
# Use compilation cache
config = CompilerConfig(
    cache_dir='/var/cache/mlcompiler',
    cache_enabled=True
)

# Reduce optimization level
config.optimization_level = 'O1'

# Parallel compilation
config.parallel_compilation = True
```

#### 3. CUDA Errors

**Symptoms:** CUDA out of memory, version mismatch

**Solutions:**
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Set memory fraction
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

# Clear GPU memory
nvidia-smi --gpu-reset
```

#### 4. Performance Regression

**Symptoms:** Lower than expected performance

**Solutions:**
```python
# Profile to identify bottlenecks
profile = model.profile(input_data)
print(profile.bottlenecks())

# Enable auto-tuning
from mlcompiler import AutoTuner
tuner = AutoTuner(model)
best_config = tuner.tune(sample_inputs)
model.apply_config(best_config)
```

### Debug Mode

```python
# Enable debug mode
config = CompilerConfig(debug=True)
compiler = MLCompiler(config)

# Debug specific pass
compiler.debug_pass('operator_fusion')

# Dump IR at each stage
compiler.dump_ir_stages('/tmp/ir_stages/')

# Visualize computation graph
compiler.visualize_graph('graph.svg')
```

### Performance Profiling

```python
# Detailed profiling
from mlcompiler.profiler import Profiler

profiler = Profiler()
with profiler.profile():
    output = model.run(input_data)

# Get report
report = profiler.get_report()
print(report.summary())
print(report.operation_times)
print(report.memory_timeline)

# Export to Chrome tracing format
profiler.export_trace('trace.json')
```

## Best Practices

### 1. Model Optimization
- Use appropriate data types (FP16 for inference)
- Enable operator fusion
- Optimize for target hardware
- Use quantization when possible

### 2. Deployment
- Use production-grade servers (gunicorn, uvicorn)
- Implement health checks and monitoring
- Use load balancing for multiple instances
- Enable caching for frequently used models

### 3. Security
- Validate all inputs
- Use TLS for network communication
- Implement authentication and authorization
- Regular security updates

### 4. Reliability
- Implement circuit breakers
- Use retry logic with exponential backoff
- Monitor and alert on failures
- Regular backups of models and configurations
