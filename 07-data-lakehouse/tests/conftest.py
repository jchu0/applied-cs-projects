"""Pytest configuration for data lakehouse tests."""

import importlib.util
import sys
from pathlib import Path

import pytest

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Check if pyspark is available
PYSPARK_AVAILABLE = importlib.util.find_spec("pyspark") is not None

# Test files that don't require pyspark
NO_PYSPARK_REQUIRED = [
    "test_delta_log_comprehensive.py",
    "test_transactions.py",
    "test_time_travel.py",
    "test_optimizer.py",
    "test_processor_comprehensive.py",
    "test_checkpoint.py",
    "test_processor.py",  # Import tests work without pyspark
    "test_query_processing.py",  # Uses mock Spark, tests work without pyspark
]


def pytest_collection_modifyitems(config, items):
    """Skip tests that require pyspark if not installed."""
    if not PYSPARK_AVAILABLE:
        skip_pyspark = pytest.mark.skip(reason="pyspark not installed")
        for item in items:
            # Only skip tests in files that actually require pyspark
            test_file = item.fspath.basename if hasattr(item.fspath, 'basename') else str(item.fspath).split("/")[-1]
            if test_file not in NO_PYSPARK_REQUIRED:
                item.add_marker(skip_pyspark)
