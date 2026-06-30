"""Tests for Delta Lake checkpoint and optimization features."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Import directly from module file to avoid pyspark dependency
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import importlib.util
spec = importlib.util.spec_from_file_location("delta_log", src_path / "lakehouse" / "delta_log.py")
delta_log_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delta_log_module)

AddFile = delta_log_module.AddFile
DeltaLog = delta_log_module.DeltaLog
Metadata = delta_log_module.Metadata
Protocol = delta_log_module.Protocol
RemoveFile = delta_log_module.RemoveFile
CommitInfo = delta_log_module.CommitInfo
SetTransaction = delta_log_module.SetTransaction


@pytest.fixture
def temp_table_path():
    """Create a temporary directory for test tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_table"


@pytest.fixture
def delta_log(temp_table_path):
    """Create a DeltaLog instance for testing."""
    return DeltaLog(str(temp_table_path))


@pytest.fixture
def populated_delta_log(delta_log):
    """Create a DeltaLog with multiple commits."""
    for i in range(10):
        add_file = AddFile(
            path=f"part-{i:05d}.parquet",
            partition_values={"year": "2024", "month": f"{(i % 12) + 1:02d}"},
            size=1000 * (i + 1),
            modification_time=1700000000 + i * 1000,
            data_change=True,
            stats=json.dumps({"numRecords": 100 * (i + 1)}),
        )
        delta_log.commit([add_file])
    return delta_log


class TestCheckpoint:
    """Tests for checkpoint functionality."""

    def test_create_checkpoint(self, populated_delta_log):
        """Test creating a checkpoint."""
        populated_delta_log.create_checkpoint()

        # Check that checkpoint file was created
        checkpoint_parquet = populated_delta_log.log_path / "00000000000000000009.checkpoint.parquet"
        checkpoint_json = populated_delta_log.log_path / "00000000000000000009.checkpoint.json"

        # At least one checkpoint format should exist
        assert checkpoint_parquet.exists() or checkpoint_json.exists()

    def test_read_checkpoint(self, populated_delta_log):
        """Test reading from a checkpoint."""
        populated_delta_log.create_checkpoint()

        # Read the checkpoint
        state = populated_delta_log._read_checkpoint(9)

        # Should have 10 files
        assert len(state.files) == 10

    def test_find_latest_checkpoint(self, populated_delta_log):
        """Test finding the latest checkpoint."""
        # Initially no checkpoint
        assert populated_delta_log._find_latest_checkpoint() is None

        # Create checkpoint
        populated_delta_log.create_checkpoint()

        # Now should find the checkpoint
        assert populated_delta_log._find_latest_checkpoint() == 9


class TestTimeTravel:
    """Tests for time travel functionality."""

    def test_time_travel_to_version(self, populated_delta_log):
        """Test time traveling to a specific version."""
        # Version 0 should have 1 file
        state_v0 = populated_delta_log.time_travel(0)
        assert len(state_v0) == 1

        # Version 4 should have 5 files
        state_v4 = populated_delta_log.time_travel(4)
        assert len(state_v4) == 5

        # Version 9 should have all 10 files
        state_v9 = populated_delta_log.time_travel(9)
        assert len(state_v9) == 10

    def test_time_travel_with_removes(self, delta_log):
        """Test time travel correctly handles removes."""
        # Add 3 files
        for i in range(3):
            add = AddFile(
                path=f"file-{i}.parquet",
                partition_values={},
                size=100,
                modification_time=i,
                data_change=True,
            )
            delta_log.commit([add])

        # Remove middle file
        remove = RemoveFile(
            path="file-1.parquet",
            deletion_timestamp=100,
            data_change=True,
        )
        delta_log.commit([remove])

        # Version 2 (before remove) should have 3 files
        state_v2 = delta_log.time_travel(2)
        assert len(state_v2) == 3

        # Version 3 (after remove) should have 2 files
        state_v3 = delta_log.time_travel(3)
        assert len(state_v3) == 2


class TestActiveFiles:
    """Tests for get_active_files functionality."""

    def test_get_active_files_simple(self, populated_delta_log):
        """Test getting active files."""
        active = populated_delta_log.get_active_files()
        assert len(active) == 10

    def test_get_active_files_after_remove(self, delta_log):
        """Test that removed files are not in active list."""
        # Add 3 files
        for i in range(3):
            add = AddFile(
                path=f"file-{i}.parquet",
                partition_values={},
                size=100,
                modification_time=i,
                data_change=True,
            )
            delta_log.commit([add])

        # Remove one
        remove = RemoveFile(
            path="file-1.parquet",
            deletion_timestamp=100,
            data_change=True,
        )
        delta_log.commit([remove])

        active = delta_log.get_active_files()
        assert len(active) == 2
        paths = [f.path for f in active]
        assert "file-1.parquet" not in paths


class TestTableProperties:
    """Tests for table properties functionality."""

    def test_get_table_properties_empty(self, delta_log):
        """Test getting properties when no metadata set."""
        props = delta_log.get_table_properties()
        assert props == {}

    def test_get_table_properties_with_metadata(self, delta_log):
        """Test getting properties with metadata."""
        metadata = Metadata(
            id="test-id",
            name="test_table",
            description="A test table",
            schema_string='{"type": "struct"}',
            partition_columns=["year", "month"],
            configuration={"delta.enableChangeDataFeed": "true"},
        )
        delta_log.commit([metadata])

        props = delta_log.get_table_properties()
        assert props["name"] == "test_table"
        assert props["partition_columns"] == ["year", "month"]
        assert "delta.enableChangeDataFeed" in props["configuration"]


class TestVacuum:
    """Tests for vacuum functionality."""

    def test_vacuum_removes_old_files(self, delta_log):
        """Test vacuum identifies old deleted files."""
        # Add and remove a file
        add = AddFile(
            path="old-file.parquet",
            partition_values={},
            size=100,
            modification_time=1,
            data_change=True,
        )
        delta_log.commit([add])

        remove = RemoveFile(
            path="old-file.parquet",
            deletion_timestamp=1000,  # Very old timestamp
            data_change=True,
        )
        delta_log.commit([remove])

        # Vacuum with 0 retention should find the file
        removed = delta_log.vacuum(retention_hours=0)
        assert "old-file.parquet" in removed

    def test_vacuum_respects_retention(self, delta_log):
        """Test vacuum respects retention period."""
        import time

        # Add and remove a file with recent timestamp
        add = AddFile(
            path="recent-file.parquet",
            partition_values={},
            size=100,
            modification_time=1,
            data_change=True,
        )
        delta_log.commit([add])

        remove = RemoveFile(
            path="recent-file.parquet",
            deletion_timestamp=int(time.time() * 1000),  # Current time
            data_change=True,
        )
        delta_log.commit([remove])

        # Vacuum with 168 hour retention (7 days) should not find recent files
        removed = delta_log.vacuum(retention_hours=168)
        assert "recent-file.parquet" not in removed


class TestPartitions:
    """Tests for partition functionality."""

    def test_get_partitions(self, populated_delta_log):
        """Test getting partitions."""
        partitions = populated_delta_log.get_partitions()

        # Should have files distributed across partitions
        assert len(partitions) > 0

        # Each partition should have files
        for partition_key, files in partitions.items():
            assert len(files) > 0

    def test_get_partitions_unpartitioned(self, delta_log):
        """Test getting partitions for unpartitioned table."""
        add = AddFile(
            path="data.parquet",
            partition_values={},
            size=100,
            modification_time=1,
            data_change=True,
        )
        delta_log.commit([add])

        partitions = delta_log.get_partitions()
        # Single partition with empty key
        assert len(partitions) == 1


class TestStatistics:
    """Tests for statistics collection."""

    def test_collect_stats(self, populated_delta_log):
        """Test collecting table statistics."""
        stats = populated_delta_log.collect_stats()

        assert stats["total_files"] == 10
        assert stats["total_size"] > 0
        assert stats["total_records"] > 0
        assert stats["version"] == 9

    def test_collect_stats_empty_table(self, delta_log):
        """Test collecting stats on empty table."""
        # Add a file with no stats
        add = AddFile(
            path="data.parquet",
            partition_values={},
            size=100,
            modification_time=1,
            data_change=True,
        )
        delta_log.commit([add])

        stats = delta_log.collect_stats()
        assert stats["total_files"] == 1
        assert stats["total_size"] == 100
        assert stats["total_records"] == 0  # No stats in file


class TestZOrderOptimization:
    """Tests for Z-order optimization."""

    def test_optimize_z_order(self, populated_delta_log):
        """Test Z-order optimization returns files."""
        optimized = populated_delta_log.optimize_z_order(["id", "timestamp"])

        # Should return all active files
        assert len(optimized) == 10


class TestNewActionTypes:
    """Tests for CommitInfo and SetTransaction actions."""

    def test_commit_info(self):
        """Test CommitInfo action creation."""
        commit_info = CommitInfo(
            timestamp=1700000000,
            user_id="user123",
            operation="WRITE",
            operation_parameters={"mode": "append"},
            job_name="daily_etl",
            is_blind_append=True,
        )

        assert commit_info.timestamp == 1700000000
        assert commit_info.user_id == "user123"
        assert commit_info.operation == "WRITE"
        assert commit_info.is_blind_append is True

    def test_set_transaction(self):
        """Test SetTransaction action creation."""
        txn = SetTransaction(
            app_id="app123",
            version=5,
            last_updated=1700000000,
        )

        assert txn.app_id == "app123"
        assert txn.version == 5
        assert txn.last_updated == 1700000000

    def test_set_transaction_defaults(self):
        """Test SetTransaction default values."""
        txn = SetTransaction(
            app_id="app123",
            version=1,
        )

        assert txn.last_updated is None


class TestReadVersion:
    """Tests for read_version functionality."""

    def test_read_version_existing(self, delta_log):
        """Test reading an existing version."""
        add = AddFile(
            path="data.parquet",
            partition_values={},
            size=100,
            modification_time=1,
            data_change=True,
        )
        delta_log.commit([add])

        actions = delta_log.read_version(0)
        assert len(actions) == 1
        assert "add" in actions[0]
        assert actions[0]["add"]["path"] == "data.parquet"

    def test_read_version_nonexistent(self, delta_log):
        """Test reading a non-existent version."""
        actions = delta_log.read_version(999)
        assert actions == []
