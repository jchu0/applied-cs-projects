"""Pytest configuration for streaming platform tests."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import importlib.util
import pytest

# Check if confluent_kafka is available
KAFKA_AVAILABLE = importlib.util.find_spec("confluent_kafka") is not None


def pytest_collection_modifyitems(config, items):
    """Skip all tests if confluent_kafka is not installed."""
    if not KAFKA_AVAILABLE:
        skip_kafka = pytest.mark.skip(reason="confluent_kafka not installed")
        for item in items:
            item.add_marker(skip_kafka)
