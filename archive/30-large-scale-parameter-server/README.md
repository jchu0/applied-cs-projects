# Large-Scale Parameter Server

Distributed parameter server for large-scale machine learning training with sharding, synchronization, and fault tolerance.

## Features

- **Parameter Sharding**: Automatic distribution across nodes
- **Sync Modes**: Synchronous, asynchronous, bounded staleness
- **Aggregation**: Mean, sum, weighted aggregation strategies
- **Compression**: LZ4, Zstd, Blosc for network efficiency
- **Fault Tolerance**: Checkpointing and recovery
- **Coordination**: Etcd-based distributed coordination

## Installation

```bash
pip install -e ".[full]"
```

## Quick Start

### Server

```python
from paramserver import ParameterServer

# Start parameter server
server = ParameterServer(
    num_shards=4,
    sync_mode="async",
    storage_backend="redis"
)
server.start(host="0.0.0.0", port=50051)
```

### Client

```python
from paramserver import ParameterClient

# Connect to parameter server
client = ParameterClient("localhost:50051")

# Push gradients
client.push("layer1.weight", gradients)

# Pull updated parameters
params = client.pull("layer1.weight")
```

## Infrastructure

```bash
docker-compose up -d
# Redis: localhost:6379
# Etcd: localhost:2379
# PostgreSQL: localhost:5432
```

## Architecture

```
┌─────────────┐     ┌─────────────┐
│   Worker 1  │────▶│             │
├─────────────┤     │  Parameter  │
│   Worker 2  │────▶│   Server    │
├─────────────┤     │  (Sharded)  │
│   Worker N  │────▶│             │
└─────────────┘     └─────────────┘
                          │
                    ┌─────┴─────┐
                    ▼           ▼
               ┌───────┐   ┌───────┐
               │ Redis │   │ Etcd  │
               └───────┘   └───────┘
```

## Configuration

See `.env.example` for configuration options.

## Testing

```bash
pytest tests/ -v  # 131 tests
```
