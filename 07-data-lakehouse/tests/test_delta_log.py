"""Unit tests for Delta Lake transaction log implementation."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("pyspark")

from lakehouse.delta_log import (
    Action,
    AddFile,
    CommitInfo,
    DeltaLog,
    Metadata,
    Protocol,
    RemoveFile,
    SetTransaction,
)


@pytest.fixture
def temp_table_path():
    """Create a temporary directory for a test table (module-wide)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_table"


class TestActions:
    """Test Delta Lake action classes."""

    def test_add_file_action(self):
        """Test AddFile action creation and serialization."""
        add_file = AddFile(
            path="data/part-001.parquet",
            partition_values={"year": "2024", "month": "01"},
            size=1024000,
            modification_time=1700000000,
            data_change=True,
            stats='{"numRecords": 1000, "minValues": {}, "maxValues": {}}',
            tags={"source": "streaming"}
        )

        assert add_file.path == "data/part-001.parquet"
        assert add_file.partition_values["year"] == "2024"
        assert add_file.size == 1024000
        assert add_file.data_change is True
        assert "numRecords" in add_file.stats

    def test_remove_file_action(self):
        """Test RemoveFile action creation."""
        remove_file = RemoveFile(
            path="data/part-000.parquet",
            deletion_timestamp=1700001000,
            data_change=True,
            partition_values={"year": "2023"}
        )

        assert remove_file.path == "data/part-000.parquet"
        assert remove_file.deletion_timestamp == 1700001000
        assert remove_file.data_change is True
        assert remove_file.partition_values["year"] == "2023"

    def test_metadata_action(self):
        """Test Metadata action with schema and configuration."""
        metadata = Metadata(
            id="test-table-id",
            name="test_table",
            description="Test Delta table",
            schema_string='{"type": "struct", "fields": [{"name": "id", "type": "long"}]}',
            partition_columns=["year", "month"],
            configuration={"delta.autoOptimize.optimizeWrite": "true"},
            format_provider="parquet"
        )

        assert metadata.id == "test-table-id"
        assert metadata.name == "test_table"
        assert len(metadata.partition_columns) == 2
        assert "delta.autoOptimize.optimizeWrite" in metadata.configuration

    def test_protocol_action(self):
        """Test Protocol action for version management."""
        protocol = Protocol(
            min_reader_version=1,
            min_writer_version=2,
            reader_features=["columnMapping"],
            writer_features=["appendOnly", "changeDataFeed"]
        )

        assert protocol.min_reader_version == 1
        assert protocol.min_writer_version == 2
        assert "columnMapping" in protocol.reader_features
        assert "changeDataFeed" in protocol.writer_features

    def test_commit_info_action(self):
        """Test CommitInfo action with operation details."""
        commit_info = CommitInfo(
            timestamp=1700000000,
            user_id="user123",
            operation="MERGE",
            operation_parameters={"predicate": "id = 1"},
            job_name="daily_etl",
            notebook_id="notebook123",
            is_blind_append=False
        )

        assert commit_info.operation == "MERGE"
        assert commit_info.operation_parameters["predicate"] == "id = 1"
        assert commit_info.is_blind_append is False


class TestDeltaLog:
    """Test DeltaLog transaction management."""

    @pytest.fixture
    def delta_log(self, temp_table_path):
        """Create a DeltaLog instance for testing."""
        return DeltaLog(temp_table_path)

    def test_delta_log_initialization(self, delta_log, temp_table_path):
        """Test DeltaLog initialization and directory creation."""
        assert delta_log.table_path == str(temp_table_path)
        assert delta_log.log_path == temp_table_path / "_delta_log"
        assert delta_log.current_version == -1
        assert delta_log.snapshot == []

    def test_commit_actions(self, delta_log):
        """Test committing actions to the transaction log."""
        # Create test actions
        metadata = Metadata(
            id="test-id",
            name="test",
            description="Test table",
            schema_string='{"type": "struct"}',
            partition_columns=[]
        )

        add_file = AddFile(
            path="part-001.parquet",
            partition_values={},
            size=1000,
            modification_time=1700000000,
            data_change=True
        )

        # Commit actions
        version = delta_log.commit([metadata, add_file])

        assert version == 0
        assert delta_log.current_version == 0
        assert len(delta_log.snapshot) == 2

        # Verify commit file was created
        commit_file = delta_log.log_path / "00000000000000000000.json"
        assert commit_file.exists()

    def test_read_version(self, delta_log):
        """Test reading a specific version from the log."""
        # Commit initial version
        add_file1 = AddFile(
            path="part-001.parquet",
            partition_values={},
            size=1000,
            modification_time=1700000000,
            data_change=True
        )
        delta_log.commit([add_file1])

        # Commit second version
        add_file2 = AddFile(
            path="part-002.parquet",
            partition_values={},
            size=2000,
            modification_time=1700001000,
            data_change=True
        )
        delta_log.commit([add_file2])

        # Read specific versions
        actions_v0 = delta_log.read_version(0)
        actions_v1 = delta_log.read_version(1)

        assert len(actions_v0) == 1
        assert actions_v0[0]["add"]["path"] == "part-001.parquet"
        assert len(actions_v1) == 1
        assert actions_v1[0]["add"]["path"] == "part-002.parquet"

    def test_checkpoint_creation(self, delta_log):
        """Test checkpoint creation after multiple commits."""
        # Create multiple commits
        for i in range(10):
            add_file = AddFile(
                path=f"part-{i:03d}.parquet",
                partition_values={},
                size=1000 * (i + 1),
                modification_time=1700000000 + i * 1000,
                data_change=True
            )
            delta_log.commit([add_file])

        # Create checkpoint
        delta_log.create_checkpoint()

        # Verify checkpoint file exists
        checkpoint_file = delta_log.log_path / "00000000000000000009.checkpoint.parquet"
        assert checkpoint_file.exists()

        # Verify snapshot contains all files
        assert len(delta_log.snapshot) == 10

    def test_time_travel(self, delta_log):
        """Test time travel to previous versions."""
        # Create versioned data
        versions = []
        for i in range(5):
            add_file = AddFile(
                path=f"part-{i:03d}.parquet",
                partition_values={},
                size=1000,
                modification_time=1700000000 + i * 1000,
                data_change=True
            )
            version = delta_log.commit([add_file])
            versions.append(version)

        # Test time travel to each version
        for target_version in versions:
            snapshot = delta_log.time_travel(target_version)
            assert len(snapshot) == target_version + 1

    def test_get_active_files(self, delta_log):
        """Test getting active files after adds and removes."""
        # Add files
        add1 = AddFile(
            path="part-001.parquet",
            partition_values={},
            size=1000,
            modification_time=1700000000,
            data_change=True
        )
        add2 = AddFile(
            path="part-002.parquet",
            partition_values={},
            size=2000,
            modification_time=1700001000,
            data_change=True
        )
        delta_log.commit([add1, add2])

        # Remove one file
        remove1 = RemoveFile(
            path="part-001.parquet",
            deletion_timestamp=1700002000,
            data_change=True
        )
        delta_log.commit([remove1])

        # Get active files
        active_files = delta_log.get_active_files()

        assert len(active_files) == 1
        assert active_files[0].path == "part-002.parquet"

    def test_get_table_properties(self, delta_log):
        """Test retrieving table properties from metadata."""
        metadata = Metadata(
            id="test-id",
            name="test_table",
            description="Test table",
            schema_string='{"type": "struct", "fields": []}',
            partition_columns=["date"],
            configuration={"delta.enableChangeDataFeed": "true"}
        )
        delta_log.commit([metadata])

        properties = delta_log.get_table_properties()

        assert properties["name"] == "test_table"
        assert properties["partition_columns"] == ["date"]
        assert properties["configuration"]["delta.enableChangeDataFeed"] == "true"

    def test_concurrent_writes_handling(self, delta_log):
        """Test handling of concurrent write attempts."""
        # Simulate concurrent modification by making atomic_write always fail
        with patch.object(delta_log, '_atomic_write') as mock_write:
            # Always raise FileExistsError to simulate concurrent write conflict
            mock_write.side_effect = FileExistsError("File already exists")

            add_file = AddFile(
                path="my-file.parquet",
                partition_values={},
                size=1000,
                modification_time=1700000000,
                data_change=True
            )

            with pytest.raises(Exception) as exc_info:
                delta_log.commit([add_file], max_retries=3)
            assert "retries" in str(exc_info.value).lower()


class TestDeltaLogOptimization:
    """Test Delta Log optimization features."""

    @pytest.fixture
    def delta_log_with_data(self, temp_table_path):
        """Create a DeltaLog with test data."""
        delta_log = DeltaLog(temp_table_path)

        # Add multiple files
        for i in range(20):
            add_file = AddFile(
                path=f"part-{i:03d}.parquet",
                partition_values={"year": "2024", "month": f"{(i % 12) + 1:02d}"},
                size=100000 * (i % 5 + 1),
                modification_time=1700000000 + i * 1000,
                data_change=True,
                stats=json.dumps({
                    "numRecords": 1000 * (i + 1),
                    "minValues": {"id": i * 1000},
                    "maxValues": {"id": (i + 1) * 1000 - 1}
                })
            )
            delta_log.commit([add_file])

        return delta_log

    def test_vacuum_old_files(self, delta_log_with_data):
        """Test vacuum operation to remove old deleted files."""
        # Mark some files as deleted
        for i in range(5):
            remove = RemoveFile(
                path=f"part-{i:03d}.parquet",
                deletion_timestamp=1700100000,
                data_change=True
            )
            delta_log_with_data.commit([remove])

        # Run vacuum with retention period
        removed_files = delta_log_with_data.vacuum(retention_hours=0)

        assert len(removed_files) == 5
        assert all(f"part-{i:03d}.parquet" in removed_files for i in range(5))

    def test_optimize_small_files(self, delta_log_with_data):
        """Test optimization to compact small files."""
        # Get files by partition
        partitions = delta_log_with_data.get_partitions()

        # Find partitions with multiple small files
        small_file_partitions = []
        for partition, files in partitions.items():
            total_size = sum(f.size for f in files)
            if len(files) > 1 and total_size < 1000000:
                small_file_partitions.append(partition)

        assert len(small_file_partitions) > 0

    def test_z_order_optimization(self, delta_log_with_data):
        """Test Z-order optimization for query performance."""
        # Get current file layout
        files_before = delta_log_with_data.get_active_files()

        # Simulate Z-order optimization
        z_order_columns = ["id", "timestamp"]
        optimized_files = delta_log_with_data.optimize_z_order(z_order_columns)

        assert len(optimized_files) > 0

    def test_stats_collection(self, delta_log_with_data):
        """Test statistics collection for files."""
        stats = delta_log_with_data.collect_stats()

        assert "total_files" in stats
        assert "total_size" in stats
        assert "total_records" in stats
        assert "partitions" in stats
        assert stats["total_files"] == 20
        assert stats["total_size"] > 0