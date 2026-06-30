"""Pytest fixtures for parameter server tests."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import numpy as np
from typing import Dict

from paramserver.server.parameter_server import ParameterServer
from paramserver.server.cluster import ParameterServerCluster
from paramserver.server.sharding import UniformSharding
from paramserver.optimizer.sgd import SGDEngine
from paramserver.consistency.hogwild import HogwildConsistency
from paramserver.worker.worker import Worker, MockGradientComputer


@pytest.fixture
def sample_params() -> Dict[str, np.ndarray]:
    """Sample model parameters for testing."""
    return {
        "layer1.weight": np.random.randn(100, 64).astype(np.float32),
        "layer1.bias": np.random.randn(100).astype(np.float32),
        "layer2.weight": np.random.randn(50, 100).astype(np.float32),
        "layer2.bias": np.random.randn(50).astype(np.float32),
        "output.weight": np.random.randn(10, 50).astype(np.float32),
        "output.bias": np.random.randn(10).astype(np.float32),
    }


@pytest.fixture
def small_params() -> Dict[str, np.ndarray]:
    """Small parameters for quick tests."""
    return {
        "w1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "w2": np.array([4.0, 5.0], dtype=np.float32),
        "b": np.array([0.1], dtype=np.float32),
    }


@pytest.fixture
def sgd_engine():
    """SGD optimizer with default settings."""
    return SGDEngine(lr=0.01)


@pytest.fixture
def sgd_with_momentum():
    """SGD optimizer with momentum."""
    return SGDEngine(lr=0.01, momentum=0.9)


@pytest.fixture
def hogwild_consistency():
    """Hogwild consistency model."""
    return HogwildConsistency()


@pytest.fixture
def uniform_sharding():
    """Uniform sharding strategy."""
    return UniformSharding()


@pytest.fixture
def parameter_server(sgd_engine, hogwild_consistency):
    """Single parameter server instance."""
    return ParameterServer(
        shard_id=0,
        update_engine=sgd_engine,
        consistency=hogwild_consistency,
    )


@pytest.fixture
def ps_cluster():
    """Parameter server cluster with 3 servers."""
    return ParameterServerCluster.create(num_servers=3)


@pytest.fixture
async def initialized_cluster(ps_cluster, sample_params):
    """Cluster initialized with sample params."""
    await ps_cluster.initialize(sample_params)
    return ps_cluster


@pytest.fixture
def mock_gradient_computer(small_params):
    """Mock gradient computer for testing."""
    shapes = {name: p.shape for name, p in small_params.items()}
    return MockGradientComputer(shapes, gradient_scale=0.01)


@pytest.fixture
async def worker_with_cluster(initialized_cluster, small_params):
    """Worker connected to an initialized cluster."""
    # Re-initialize with small params
    await initialized_cluster.initialize(small_params)

    worker = Worker(
        worker_id=0,
        ps_cluster=initialized_cluster,
        param_names=list(small_params.keys()),
    )

    shapes = {name: p.shape for name, p in small_params.items()}
    worker.set_gradient_fn(MockGradientComputer(shapes))

    return worker
