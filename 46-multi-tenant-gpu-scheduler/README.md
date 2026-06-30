# Multi-Tenant GPU Scheduler

A Kubernetes-inspired GPU resource scheduler designed for efficient multi-tenant GPU cluster management. This system provides fair-share scheduling, resource quotas, preemption support, and comprehensive monitoring for GPU workloads.

## Features

- **Multi-Tenant Support**: Resource isolation and fair-share scheduling across tenants
- **Advanced Scheduling**: Multiple scheduling algorithms including gang scheduling and preemption
- **GPU Partitioning**: Support for MIG (Multi-Instance GPU), MPS, and time-sharing
- **Resource Quotas**: Enforce GPU, memory, and job limits per tenant/queue
- **Comprehensive Monitoring**: Real-time GPU metrics, health checks, and alerting
- **High Availability**: Leader election and state replication for production deployments
- **Extensible Architecture**: Plugin-based scheduling and allocation strategies

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/multi-tenant-gpu-scheduler.git
cd multi-tenant-gpu-scheduler

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from gpusched.core.resources import Cluster, create_node, create_training_job
from gpusched.scheduler.scheduler import GPUScheduler

# Create cluster
cluster = Cluster("my-cluster")

# Add nodes with GPUs
node = create_node("gpu-server-01", num_gpus=8, gpu_type=GPUType.A100)
cluster.add_node(node)

# Create scheduler
scheduler = GPUScheduler(cluster)

# Submit a job
job = create_training_job(
    name="bert-training",
    num_gpus=4,
    gpu_memory_gb=60.0,
    priority=PriorityClass.HIGH
)
cluster.submit_job(job)

# Schedule the job
decisions = scheduler.run_scheduling_cycle()
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     API Layer                            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Scheduler Core                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │Queue Manager │  │GPU Scheduler │  │  Preemption  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Resource Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Allocator   │  │   Cluster    │  │   Monitor    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## Core Components

### Resource Management
- **GPU**: Physical GPU device tracking
- **Node**: Compute nodes with multiple GPUs
- **Pod/Container**: Workload units with resource requirements
- **Job**: Collections of pods for training/inference
- **Cluster**: Overall cluster state management

### Scheduling Engine
- **Plugins**: Modular scheduling logic (affinity, resources, bin-packing, fair-share)
- **Queue Management**: Multi-queue scheduling with priorities
- **Gang Scheduling**: All-or-nothing scheduling for distributed jobs
- **Preemption**: High-priority jobs can preempt lower-priority ones

### GPU Allocation
- **Exclusive Mode**: Entire GPU for one workload
- **Shared Mode**: Time-sharing between multiple workloads
- **MIG Mode**: Hardware partitioning (A100/H100)
- **MPS Mode**: CUDA-level sharing

### Monitoring
- **Metrics Collection**: GPU utilization, temperature, power, memory
- **Health Checking**: Automatic detection of unhealthy GPUs/nodes
- **Alerting**: Configurable alerts for various conditions
- **Export**: Prometheus/Grafana integration

## Examples

### Multi-Tenant Setup

```python
from gpusched.core.resources import Tenant, Queue

# Create tenants
ml_tenant = Tenant("ml-team", "Machine Learning Team", total_gpu_quota=32)
research_tenant = Tenant("research", "Research Team", total_gpu_quota=16)

# Create queues
training_queue = Queue(
    name="ml-training",
    tenant_id="ml-team",
    gpu_quota=24,
    priority_weight=2.0
)

# Configure cluster
cluster.tenants["ml-team"] = ml_tenant
cluster.queues["ml-training"] = training_queue
```

### Gang Scheduling

```python
# Create distributed training job
distributed_job = create_training_job(
    name="distributed-bert",
    num_gpus=2,
    parallelism=4,  # 4 pods with 2 GPUs each
)
distributed_job.gang_schedule = True  # All pods must start together

cluster.submit_job(distributed_job)
decisions = scheduler.run_scheduling_cycle()
```

### Custom Scheduling Plugin

```python
from gpusched.scheduler.scheduler import SchedulingPlugin

class LocalityPlugin(SchedulingPlugin):
    def name(self) -> str:
        return "DataLocality"

    def score(self, ctx, node):
        # Prefer nodes with local data
        if node.labels.get("has_dataset") == ctx.pod.labels.get("dataset"):
            return 100.0
        return 0.0

scheduler.add_plugin(LocalityPlugin(), weight=1.5)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=gpusched tests/

# Run specific test module
pytest tests/test_scheduler.py

# Run integration tests
pytest tests/test_integration.py -v
```

The test suite includes:
- Unit tests for all major components
- Integration tests for end-to-end workflows
- Test fixtures and mocking utilities
- 60%+ code coverage target

## Configuration

Create `config.yaml`:

```yaml
scheduler:
  interval: 10  # seconds
  plugins:
    - name: NodeAffinity
      weight: 1.0
    - name: GPUResource
      weight: 2.0
    - name: BinPacking
      weight: 1.5

allocator:
  mode: auto  # auto|exclusive|shared|mig
  max_sharing_factor: 4

monitor:
  interval: 60
  alerts:
    gpu_temperature_threshold: 85
    gpu_utilization_threshold: 90
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [API Reference](docs/API.md) - Complete API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Contributing](docs/CONTRIBUTING.md) - How to contribute to the project

## Performance

- Schedule 1000+ pods per second
- Support 10,000+ GPUs per cluster
- Sub-second scheduling decisions
- Minimal memory overhead (<100MB per 1000 pods)

## Supported GPUs

- NVIDIA A100 (40GB/80GB)
- NVIDIA H100 (80GB)
- NVIDIA V100 (16GB/32GB)
- NVIDIA T4 (16GB)
- NVIDIA A10G (24GB)
- NVIDIA L4 (24GB)

## Requirements

- Python 3.8+
- NVIDIA Driver 470.x+
- CUDA 11.4+
- Linux (Ubuntu 20.04+, RHEL 8+)

## Production Deployment

For production deployments, see the [Deployment Guide](docs/DEPLOYMENT.md) which covers:
- High availability setup
- Kubernetes deployment
- Monitoring integration
- Security hardening
- Backup and recovery

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for:
- Code style guidelines
- Testing requirements
- Pull request process
- Development setup

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: Check our comprehensive docs
- **Email**: gpu-scheduler@example.com

## Roadmap

- [ ] Distributed scheduling across multiple schedulers
- [ ] Advanced preemption with checkpoint/restore
- [ ] Cost-based optimization
- [ ] ML-based workload prediction
- [ ] Multi-cluster federation
- [ ] Enhanced GPU virtualization support

## Acknowledgments

- Inspired by Kubernetes scheduler design
- Built on NVIDIA GPU management technologies
- Community contributions and feedback

---

Built with ❤️ for efficient GPU resource management