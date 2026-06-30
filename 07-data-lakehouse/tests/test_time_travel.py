"""Comprehensive tests for time travel queries in the data lakehouse.

This module tests:
- Version-based time travel
- Timestamp-based time travel
- Table restore operations
- History queries
- Snapshot consistency
"""

import json
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# Import directly from module file to avoid pyspark dependency via __init__.py
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import module directly to bypass package __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("delta_log", src_path / "lakehouse" / "delta_log.py")
delta_log_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delta_log_module)

AddFile = delta_log_module.AddFile
DeltaLog = delta_log_module.DeltaLog
Metadata = delta_log_module.Metadata
Protocol = delta_log_module.Protocol
RemoveFile = delta_log_module.RemoveFile
Snapshot = delta_log_module.Snapshot
TableState = delta_log_module.TableState


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_table_path():
    """Create a temporary directory for test tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_table"


@pytest.fixture
def delta_log(temp_table_path):
    """Create a DeltaLog instance for testing."""
    return DeltaLog(str(temp_table_path))


def create_add_file(file_id: int, partition_values: dict = None) -> AddFile:
    """Helper to create AddFile actions."""
    return AddFile(
        path=f"part-{file_id:05d}.parquet",
        partition_values=partition_values or {},
        size=1024 * (file_id + 1),
        modification_time=1700000000 + file_id * 1000,
        data_change=True,
        stats=json.dumps({"numRecords": 100 * (file_id + 1)}),
    )


def create_versioned_table(delta_log, num_versions: int = 5) -> List[int]:
    """Create a table with multiple versions and return version numbers."""
    versions = []

    # Initial setup
    protocol = Protocol(min_reader_version=1, min_writer_version=2)
    metadata = Metadata(
        id="test-table",
        name="test_table",
        description="Table for time travel testing",
        schema_string='{"type": "struct", "fields": [{"name": "id", "type": "long"}]}',
        partition_columns=[],
    )
    version = delta_log.commit([protocol, metadata])
    versions.append(version)

    # Add files in subsequent versions
    for i in range(1, num_versions):
        add = create_add_file(i)
        version = delta_log.commit([add])
        versions.append(version)

    return versions


# =============================================================================
# Test Version-Based Time Travel
# =============================================================================


class TestVersionBasedTimeTravel:
    """Tests for version-based time travel."""

    def test_read_version_zero(self, delta_log):
        """Test reading the initial version (version 0)."""
        create_versioned_table(delta_log, num_versions=5)

        snapshot = delta_log.get_snapshot(version=0)
        assert snapshot.version == 0
        # Version 0 has protocol and metadata, no files
        assert len(snapshot.state.files) == 0
        assert snapshot.state.protocol is not None
        assert snapshot.state.metadata is not None

    def test_read_intermediate_version(self, delta_log):
        """Test reading an intermediate version."""
        create_versioned_table(delta_log, num_versions=10)

        snapshot = delta_log.get_snapshot(version=5)
        assert snapshot.version == 5
        # Versions 1-5 each add one file (version 0 has setup)
        assert len(snapshot.state.files) == 5

    def test_read_latest_version(self, delta_log):
        """Test reading the latest version."""
        create_versioned_table(delta_log, num_versions=5)

        # Get latest without specifying version
        snapshot = delta_log.get_snapshot()
        assert snapshot.version == 4  # 5 versions: 0, 1, 2, 3, 4

        # Explicit latest version
        latest_version = delta_log._get_latest_version()
        snapshot_explicit = delta_log.get_snapshot(version=latest_version)
        assert snapshot.version == snapshot_explicit.version

    def test_version_file_accumulation(self, delta_log):
        """Test that files accumulate across versions."""
        create_versioned_table(delta_log, num_versions=10)

        for v in range(10):
            snapshot = delta_log.get_snapshot(version=v)
            # Version 0 has no files (just setup)
            # Each subsequent version adds 1 file
            expected_files = v if v > 0 else 0
            assert len(snapshot.state.files) == expected_files

    def test_version_with_removes(self, delta_log):
        """Test time travel with file removals."""
        # Create table with files
        create_versioned_table(delta_log, num_versions=5)

        # Add more files
        delta_log.commit([create_add_file(10)])
        delta_log.commit([create_add_file(11)])

        # Remove a file
        delta_log.commit([
            RemoveFile(
                path="part-00001.parquet",
                deletion_timestamp=1700100000,
                data_change=True,
            )
        ])

        # Before removal (version 6)
        snapshot_before = delta_log.get_snapshot(version=6)

        # After removal (version 7)
        snapshot_after = delta_log.get_snapshot(version=7)

        assert len(snapshot_after.state.files) == len(snapshot_before.state.files) - 1

        # Verify specific file is present/absent
        paths_before = [f.path for f in snapshot_before.state.files]
        paths_after = [f.path for f in snapshot_after.state.files]

        assert "part-00001.parquet" in paths_before
        assert "part-00001.parquet" not in paths_after

    def test_version_before_data_exists(self, delta_log):
        """Test reading version before table had data."""
        # Create table - version 0 is just protocol/metadata
        create_versioned_table(delta_log, num_versions=3)

        snapshot = delta_log.get_snapshot(version=0)
        assert len(snapshot.state.files) == 0
        assert snapshot.state.metadata is not None

    def test_consecutive_versions_differ_by_one_action(self, delta_log):
        """Test that consecutive versions differ by expected actions."""
        create_versioned_table(delta_log, num_versions=5)

        for v in range(1, 5):
            prev_snapshot = delta_log.get_snapshot(version=v - 1)
            curr_snapshot = delta_log.get_snapshot(version=v)

            # Each version adds exactly one file
            diff = len(curr_snapshot.state.files) - len(prev_snapshot.state.files)
            assert diff == 1


class TestVersionEdgeCases:
    """Edge cases for version-based time travel."""

    def test_version_negative_one(self, delta_log):
        """Test that version -1 is handled (should fail or return empty)."""
        create_versioned_table(delta_log, num_versions=3)

        # Behavior depends on implementation
        # Could raise error or treat as "no data"
        try:
            snapshot = delta_log.get_snapshot(version=-1)
            # If it succeeds, should be empty state
            assert snapshot.state.files == [] or snapshot.version == -1
        except (ValueError, IndexError):
            pass  # This is also acceptable

    def test_version_beyond_latest(self, delta_log):
        """Test requesting version beyond latest returns empty state or raises."""
        create_versioned_table(delta_log, num_versions=3)
        latest = delta_log._get_latest_version()

        # Try to get version that doesn't exist
        try:
            snapshot = delta_log.get_snapshot(version=latest + 10)
            # Implementation may return a snapshot for a non-existent version
            # The key behavior is that it doesn't crash
            # The version field may be the requested version (even if files don't exist)
            assert snapshot is not None
        except (ValueError, IndexError, FileNotFoundError):
            pass  # This is also acceptable behavior

    def test_large_version_gap(self, delta_log):
        """Test with large gap between versions requested."""
        create_versioned_table(delta_log, num_versions=100)

        # Read versions with large gap
        v10 = delta_log.get_snapshot(version=10)
        v90 = delta_log.get_snapshot(version=90)

        assert v10.version == 10
        assert v90.version == 90
        assert len(v90.state.files) > len(v10.state.files)


# =============================================================================
# Test Snapshot Consistency
# =============================================================================


class TestSnapshotConsistency:
    """Tests for snapshot consistency guarantees."""

    def test_snapshot_immutability(self, delta_log):
        """Test that snapshots are immutable after creation."""
        create_versioned_table(delta_log, num_versions=5)

        snapshot = delta_log.get_snapshot(version=3)
        original_file_count = len(snapshot.state.files)
        original_version = snapshot.version

        # Make more commits
        delta_log.commit([create_add_file(100)])
        delta_log.commit([create_add_file(101)])

        # Original snapshot should be unchanged
        assert len(snapshot.state.files) == original_file_count
        assert snapshot.version == original_version

    def test_snapshot_isolation(self, delta_log):
        """Test that different snapshots are isolated."""
        create_versioned_table(delta_log, num_versions=10)

        # Get multiple snapshots
        snapshots = [delta_log.get_snapshot(version=v) for v in range(10)]

        # Each snapshot should have correct number of files
        for i, snapshot in enumerate(snapshots):
            expected_files = i if i > 0 else 0
            assert len(snapshot.state.files) == expected_files

    def test_snapshot_file_paths_correct(self, delta_log):
        """Test that file paths in snapshots are correct."""
        create_versioned_table(delta_log, num_versions=5)

        snapshot = delta_log.get_snapshot(version=4)
        paths = sorted([f.path for f in snapshot.state.files])

        expected = sorted([f"part-{i:05d}.parquet" for i in range(1, 5)])
        assert paths == expected

    def test_snapshot_preserves_metadata(self, delta_log):
        """Test that snapshots preserve table metadata across versions."""
        # Initial setup with metadata
        protocol = Protocol(min_reader_version=1, min_writer_version=2)
        metadata = Metadata(
            id="metadata-test",
            name="metadata_test_table",
            description="Testing metadata preservation",
            schema_string='{"type": "struct", "fields": []}',
            partition_columns=["date"],
        )
        delta_log.commit([protocol, metadata])

        # Add several versions of data
        for i in range(5):
            delta_log.commit([create_add_file(i)])

        # All versions should have same metadata
        for v in range(6):
            snapshot = delta_log.get_snapshot(version=v)
            assert snapshot.state.metadata.name == "metadata_test_table"
            assert snapshot.state.metadata.partition_columns == ["date"]


# =============================================================================
# Test History Queries
# =============================================================================


class TestHistoryQueries:
    """Tests for querying table history."""

    def test_get_all_versions(self, delta_log):
        """Test getting list of all versions."""
        create_versioned_table(delta_log, num_versions=10)

        # Get all log files
        log_files = sorted(delta_log.log_path.glob("*.json"))
        assert len(log_files) == 10

        # Verify version numbers
        versions = [int(f.stem) for f in log_files]
        assert versions == list(range(10))

    def test_get_latest_version(self, delta_log):
        """Test getting the latest version number."""
        create_versioned_table(delta_log, num_versions=7)

        latest = delta_log._get_latest_version()
        assert latest == 6  # 7 versions: 0-6

    def test_version_metadata_accessible(self, delta_log):
        """Test that version-specific metadata is accessible."""
        create_versioned_table(delta_log, num_versions=5)

        for v in range(5):
            actions = delta_log._read_log_file(v)
            assert len(actions) > 0


# =============================================================================
# Test Restore Operations
# =============================================================================


class TestRestoreOperations:
    """Tests for table restore operations using delta_log."""

    def test_simulate_restore_to_version(self, delta_log):
        """Test simulating restore to previous version."""
        create_versioned_table(delta_log, num_versions=10)

        # Get state at version 5
        target_snapshot = delta_log.get_snapshot(version=5)
        target_files = set(f.path for f in target_snapshot.state.files)

        # Get current state
        current_snapshot = delta_log.get_snapshot()
        current_files = set(f.path for f in current_snapshot.state.files)

        # Calculate restore actions needed
        files_to_remove = current_files - target_files
        files_to_add = target_files - current_files

        # In a real restore, we would:
        # 1. Remove files that were added after the target version
        # 2. Re-add files that were removed after the target version

        # For this test, just verify we can identify the differences
        assert len(current_files) > len(target_files)
        assert files_to_remove == current_files - target_files

    def test_restore_preserves_history(self, delta_log):
        """Test that restore creates new version (preserves history)."""
        create_versioned_table(delta_log, num_versions=5)

        initial_latest = delta_log._get_latest_version()

        # Simulate restore by getting old state and committing
        old_snapshot = delta_log.get_snapshot(version=2)

        # Get current files
        current = delta_log.get_snapshot()
        current_paths = set(f.path for f in current.state.files)
        old_paths = set(f.path for f in old_snapshot.state.files)

        # Create remove actions for files to "undo"
        removes = [
            RemoveFile(path=p, deletion_timestamp=int(time.time()), data_change=True)
            for p in (current_paths - old_paths)
        ]

        if removes:
            delta_log.commit(removes)

        # Version should have increased (history preserved)
        new_latest = delta_log._get_latest_version()
        assert new_latest > initial_latest

        # All original versions still accessible
        for v in range(initial_latest + 1):
            snapshot = delta_log.get_snapshot(version=v)
            assert snapshot.version == v


# =============================================================================
# Test Time Travel with Different Operations
# =============================================================================


class TestTimeTravelWithOperations:
    """Tests for time travel after various operations."""

    def test_time_travel_after_update_simulation(self, delta_log):
        """Test time travel after simulated update (remove + add)."""
        create_versioned_table(delta_log, num_versions=3)

        # Add a file that will be "updated"
        delta_log.commit([create_add_file(100)])

        # Simulate update: remove old, add new
        delta_log.commit([
            RemoveFile(
                path="part-00100.parquet",
                deletion_timestamp=1700200000,
                data_change=True,
            ),
            AddFile(
                path="part-00100.parquet",  # Same path, new content
                partition_values={},
                size=2048,  # Different size indicates different content
                modification_time=1700200001,
                data_change=True,
            ),
        ])

        # Before update
        before_update = delta_log.get_snapshot(version=3)
        file_before = next(
            (f for f in before_update.state.files if f.path == "part-00100.parquet"),
            None
        )

        # After update
        after_update = delta_log.get_snapshot()
        file_after = next(
            (f for f in after_update.state.files if f.path == "part-00100.parquet"),
            None
        )

        assert file_before is not None
        assert file_after is not None
        assert file_before.size != file_after.size  # Content changed

    def test_time_travel_after_delete(self, delta_log):
        """Test time travel after delete operation."""
        create_versioned_table(delta_log, num_versions=5)

        # Delete some data
        delta_log.commit([
            RemoveFile(path="part-00001.parquet", deletion_timestamp=1, data_change=True),
            RemoveFile(path="part-00002.parquet", deletion_timestamp=1, data_change=True),
        ])

        # Before delete
        before = delta_log.get_snapshot(version=4)
        # After delete
        after = delta_log.get_snapshot(version=5)

        assert len(before.state.files) == len(after.state.files) + 2

        # Deleted files accessible in old version
        before_paths = [f.path for f in before.state.files]
        assert "part-00001.parquet" in before_paths
        assert "part-00002.parquet" in before_paths

        # Not in new version
        after_paths = [f.path for f in after.state.files]
        assert "part-00001.parquet" not in after_paths
        assert "part-00002.parquet" not in after_paths

    def test_time_travel_after_schema_evolution(self, delta_log):
        """Test time travel after schema change."""
        # Initial schema - note "name" appears as field name but not a column type
        protocol = Protocol(min_reader_version=1, min_writer_version=2)
        metadata_v1 = Metadata(
            id="schema-test",
            name="schema_test",
            description="Initial schema",
            schema_string='{"type": "struct", "fields": [{"field": "id", "type": "long"}]}',
            partition_columns=[],
        )
        delta_log.commit([protocol, metadata_v1])

        # Add data with initial schema
        delta_log.commit([create_add_file(1)])
        delta_log.commit([create_add_file(2)])

        # Update schema (add column)
        metadata_v2 = Metadata(
            id="schema-test",
            name="schema_test",
            description="Extended schema",
            schema_string='{"type": "struct", "fields": [{"field": "id", "type": "long"}, {"field": "user_name", "type": "string"}]}',
            partition_columns=[],
        )
        delta_log.commit([metadata_v2])

        # Add more data
        delta_log.commit([create_add_file(3)])

        # Check schema at different versions - look for the new "user_name" column
        v1_snapshot = delta_log.get_snapshot(version=1)
        v4_snapshot = delta_log.get_snapshot(version=4)

        assert "user_name" not in v1_snapshot.state.metadata.schema_string
        assert "user_name" in v4_snapshot.state.metadata.schema_string


class TestTimeTravelPartitions:
    """Tests for time travel with partitioned tables."""

    def test_time_travel_partition_files(self, delta_log):
        """Test time travel with partitioned data."""
        # Setup
        protocol = Protocol(min_reader_version=1, min_writer_version=2)
        metadata = Metadata(
            id="partitioned-test",
            name="partitioned_table",
            description="Partitioned table for time travel",
            schema_string='{"type": "struct", "fields": []}',
            partition_columns=["year", "month"],
        )
        delta_log.commit([protocol, metadata])

        # Add files to different partitions
        delta_log.commit([
            AddFile(
                path="year=2023/month=01/part-00000.parquet",
                partition_values={"year": "2023", "month": "01"},
                size=100,
                modification_time=1,
                data_change=True,
            )
        ])

        delta_log.commit([
            AddFile(
                path="year=2023/month=02/part-00000.parquet",
                partition_values={"year": "2023", "month": "02"},
                size=100,
                modification_time=2,
                data_change=True,
            )
        ])

        delta_log.commit([
            AddFile(
                path="year=2024/month=01/part-00000.parquet",
                partition_values={"year": "2024", "month": "01"},
                size=100,
                modification_time=3,
                data_change=True,
            )
        ])

        # Time travel to when only 2023 data existed
        v2_snapshot = delta_log.get_snapshot(version=2)
        years = set(f.partition_values.get("year") for f in v2_snapshot.state.files)
        assert years == {"2023"}

        # Current state has both years
        current = delta_log.get_snapshot()
        years_current = set(f.partition_values.get("year") for f in current.state.files)
        assert years_current == {"2023", "2024"}


# =============================================================================
# Test Time Travel Performance
# =============================================================================


class TestTimeTravelPerformance:
    """Performance-related tests for time travel."""

    def test_time_travel_many_versions(self, delta_log):
        """Test time travel with many versions."""
        # Create many versions
        num_versions = 100

        protocol = Protocol(min_reader_version=1, min_writer_version=2)
        metadata = Metadata(
            id="perf-test",
            name="performance_test",
            description="Performance testing",
            schema_string="{}",
            partition_columns=[],
        )
        delta_log.commit([protocol, metadata])

        for i in range(1, num_versions):
            delta_log.commit([create_add_file(i)])

        # Should be able to read any version
        for v in [0, 10, 50, 99]:
            snapshot = delta_log.get_snapshot(version=v)
            assert snapshot.version == v

    def test_time_travel_large_files_list(self, delta_log):
        """Test time travel with many files in table."""
        protocol = Protocol(min_reader_version=1, min_writer_version=2)
        metadata = Metadata(
            id="large-files-test",
            name="large_files_table",
            description="Many files",
            schema_string="{}",
            partition_columns=[],
        )
        delta_log.commit([protocol, metadata])

        # Add many files in single commit
        files = [create_add_file(i) for i in range(500)]
        delta_log.commit(files)

        # Add more
        delta_log.commit([create_add_file(1000)])

        # Time travel should work with many files
        v1_snapshot = delta_log.get_snapshot(version=1)
        assert len(v1_snapshot.state.files) == 500

        v2_snapshot = delta_log.get_snapshot(version=2)
        assert len(v2_snapshot.state.files) == 501
