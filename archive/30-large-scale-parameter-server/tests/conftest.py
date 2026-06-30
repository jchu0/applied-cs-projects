"""Pytest fixtures for parameter server tests."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import numpy as np
import asyncio

from paramserver import (
    ParameterServer,
    ParameterStore,
    ShardManager,
    ParameterPartitioner,
    SyncManager,
    StalenessTracker,
    GradientAggregator,
    AsyncAggregator,
    SparsifiedAggregator,
    WorkerClient,
    ConsistencyModel,
    AggregationType,
    AggregationConfig,
    PartitionStrategy,
    Parameter,
    Gradient,
    generate_id,
)


# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============ Parameter Fixtures ============

@pytest.fixture
def sample_parameter():
    """Create a sample parameter."""
    return Parameter(
        name="layer1.weight",
        shape=(100, 50),
        dtype="float32",
        data=np.random.randn(100, 50).astype(np.float32),
        version=0,
        shard_id=0,
    )


@pytest.fixture
def sample_parameters():
    """Create multiple sample parameters."""
    return [
        Parameter(
            name=f"layer{i}.weight",
            shape=(64, 64),
            dtype="float32",
            data=np.random.randn(64, 64).astype(np.float32),
            version=0,
        )
        for i in range(5)
    ]


@pytest.fixture
def large_parameter():
    """Create a large parameter for memory tests."""
    return Parameter(
        name="embedding.weight",
        shape=(10000, 256),
        dtype="float32",
        data=np.random.randn(10000, 256).astype(np.float32),
        version=0,
    )


# ============ Gradient Fixtures ============

@pytest.fixture
def sample_gradient():
    """Create a sample gradient."""
    return Gradient(
        name="layer1.weight",
        data=np.random.randn(100, 50).astype(np.float32) * 0.01,
        worker_id="worker_1",
        iteration=0,
    )


@pytest.fixture
def multiple_gradients():
    """Create gradients from multiple workers."""
    return [
        Gradient(
            name="layer1.weight",
            data=np.random.randn(64, 64).astype(np.float32) * 0.01,
            worker_id=f"worker_{i}",
            iteration=0,
        )
        for i in range(4)
    ]


@pytest.fixture
def gradients_for_aggregation():
    """Create deterministic gradients for testing aggregation."""
    return [
        Gradient(
            name="test_param",
            data=np.full((10, 10), float(i + 1), dtype=np.float32),
            worker_id=f"worker_{i}",
            iteration=0,
        )
        for i in range(4)
    ]


# ============ Storage Fixtures ============

@pytest.fixture
def parameter_store():
    """Create a parameter store instance."""
    return ParameterStore()


@pytest.fixture
def shard_manager():
    """Create a shard manager with 4 shards."""
    return ShardManager(num_shards=4)


@pytest.fixture
def partitioner():
    """Create a parameter partitioner."""
    strategy = PartitionStrategy(num_shards=4, strategy_type="hash")
    return ParameterPartitioner(strategy)


@pytest.fixture
def range_partitioner():
    """Create a range-based partitioner."""
    strategy = PartitionStrategy(num_shards=4, strategy_type="range")
    return ParameterPartitioner(strategy)


@pytest.fixture
def round_robin_partitioner():
    """Create a round-robin partitioner."""
    strategy = PartitionStrategy(num_shards=4, strategy_type="round_robin")
    return ParameterPartitioner(strategy)


# ============ Sync Fixtures ============

@pytest.fixture
def sync_manager_bsp():
    """Create a BSP sync manager."""
    return SyncManager(ConsistencyModel.BSP)


@pytest.fixture
def sync_manager_asp():
    """Create an ASP sync manager."""
    return SyncManager(ConsistencyModel.ASP)


@pytest.fixture
def sync_manager_ssp():
    """Create an SSP sync manager."""
    return SyncManager(ConsistencyModel.SSP)


@pytest.fixture
def staleness_tracker():
    """Create a staleness tracker with max staleness of 3."""
    return StalenessTracker(max_staleness=3)


# ============ Aggregator Fixtures ============

@pytest.fixture
def gradient_aggregator():
    """Create a gradient aggregator with default config."""
    return GradientAggregator()


@pytest.fixture
def sum_aggregator():
    """Create a sum aggregator."""
    config = AggregationConfig(aggregation_type=AggregationType.SUM)
    return GradientAggregator(config)


@pytest.fixture
def mean_aggregator():
    """Create a mean aggregator."""
    config = AggregationConfig(aggregation_type=AggregationType.MEAN)
    return GradientAggregator(config)


@pytest.fixture
def aggregator_with_clipping():
    """Create an aggregator with gradient clipping."""
    config = AggregationConfig(
        aggregation_type=AggregationType.MEAN,
        clip_norm=1.0,
    )
    return GradientAggregator(config)


@pytest.fixture
def aggregator_with_momentum():
    """Create an aggregator with momentum."""
    config = AggregationConfig(
        aggregation_type=AggregationType.MEAN,
        momentum=0.9,
    )
    return GradientAggregator(config)


@pytest.fixture
def async_aggregator():
    """Create an async aggregator."""
    return AsyncAggregator()


@pytest.fixture
def sparsified_aggregator():
    """Create a sparsified aggregator."""
    return SparsifiedAggregator(top_k=0.1)


# ============ Server Fixtures ============

@pytest.fixture
def bsp_server():
    """Create a BSP parameter server."""
    return ParameterServer(
        num_shards=4,
        consistency_model=ConsistencyModel.BSP,
        learning_rate=0.01,
    )


@pytest.fixture
def asp_server():
    """Create an ASP parameter server."""
    return ParameterServer(
        num_shards=4,
        consistency_model=ConsistencyModel.ASP,
        learning_rate=0.01,
    )


@pytest.fixture
def ssp_server():
    """Create an SSP parameter server."""
    return ParameterServer(
        num_shards=4,
        consistency_model=ConsistencyModel.SSP,
        learning_rate=0.01,
    )


@pytest.fixture
def server_with_workers(bsp_server):
    """Create a BSP server with registered workers."""
    for i in range(4):
        bsp_server.register_worker(f"worker_{i}", port=6000 + i)
    return bsp_server


# ============ Worker Client Fixtures ============

@pytest.fixture
def worker_client():
    """Create a worker client."""
    return WorkerClient(
        worker_id="worker_1",
        server_host="localhost",
        server_port=5000,
    )


@pytest.fixture
def worker_clients():
    """Create multiple worker clients."""
    return [
        WorkerClient(worker_id=f"worker_{i}")
        for i in range(4)
    ]


@pytest.fixture
def connected_worker_client(bsp_server):
    """Create a worker client connected to a server."""
    client = WorkerClient(worker_id="worker_1")
    client.set_server(bsp_server)
    bsp_server.register_worker("worker_1")
    return client


# ============ Utility Fixtures ============

@pytest.fixture
def worker_ids():
    """Generate a list of worker IDs."""
    return [f"worker_{i}" for i in range(4)]


@pytest.fixture
def parameter_names():
    """Generate a list of parameter names."""
    return [
        "encoder.layer1.weight",
        "encoder.layer1.bias",
        "encoder.layer2.weight",
        "encoder.layer2.bias",
        "decoder.layer1.weight",
        "decoder.layer1.bias",
    ]
