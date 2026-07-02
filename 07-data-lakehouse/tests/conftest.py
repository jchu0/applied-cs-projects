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


# ---------------------------------------------------------------------------
# Spark JVM availability guard
# ---------------------------------------------------------------------------
# pyspark can be importable while the underlying Java gateway is broken or
# missing (e.g. no/incompatible JDK -> JAVA_GATEWAY_EXITED). In that case any
# test that spins up a real SparkSession would ERROR at setup. We probe the JVM
# exactly once per session and skip the Spark-dependent tests with a clear
# reason instead of letting them error out.
_SPARK_PROBE = {"done": False, "session": None, "error": None}


def _probe_spark():
    """Try to create a minimal local SparkSession once; cache the outcome."""
    if _SPARK_PROBE["done"]:
        return _SPARK_PROBE
    _SPARK_PROBE["done"] = True
    if not PYSPARK_AVAILABLE:
        _SPARK_PROBE["error"] = "pyspark not installed"
        return _SPARK_PROBE
    try:
        from pyspark.sql import SparkSession

        session = (
            SparkSession.builder.appName("spark_availability_probe")
            .master("local[1]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
        # Force the JVM gateway to actually do something.
        session.sparkContext.parallelize([1]).count()
        _SPARK_PROBE["session"] = session
    except BaseException as exc:  # noqa: BLE001 - JVM failures can be non-Exception
        _SPARK_PROBE["error"] = f"{type(exc).__name__}: {exc}"
    return _SPARK_PROBE


@pytest.fixture(scope="session")
def spark_available():
    """Session-scoped guard: skip Spark tests if the JVM gateway is unusable.

    Returns a working session-scoped SparkSession when available so callers can
    reuse it. Skips (rather than errors) the requesting test with a clear
    JVM-unavailable reason otherwise.
    """
    probe = _probe_spark()
    if probe["error"] is not None:
        pytest.skip(f"Spark JVM unavailable: {probe['error']}")
    return probe["session"]
