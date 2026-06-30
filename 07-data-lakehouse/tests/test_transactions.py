"""Comprehensive tests for transaction handling in the data lakehouse.

This module tests:
- Optimistic concurrency control
- Transaction isolation
- Commit retry logic
- Atomic writes
- Transaction log integrity
"""

import json
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


@pytest.fixture
def initialized_table(delta_log):
    """Create a DeltaLog with initial protocol and metadata."""
    protocol = Protocol(min_reader_version=1, min_writer_version=2)
    metadata = Metadata(
        id="test-table-id",
        name="test_table",
        description="Test table for transactions",
        schema_string='{"type": "struct", "fields": [{"name": "id", "type": "long"}]}',
        partition_columns=[],
    )
    delta_log.commit([protocol, metadata])
    return delta_log


def create_add_file(file_id: int, partition_values: dict = None) -> AddFile:
    """Helper to create AddFile actions."""
    return AddFile(
        path=f"part-{file_id:05d}.parquet",
        partition_values=partition_values or {},
        size=1024 * (file_id + 1),
        modification_time=1700000000 + file_id,
        data_change=True,
        stats=json.dumps({"numRecords": 1000 * (file_id + 1)}),
    )


# =============================================================================
# Test Optimistic Concurrency Control
# =============================================================================


class TestOptimisticConcurrencyControl:
    """Tests for optimistic concurrency control mechanisms."""

    def test_sequential_commits_succeed(self, delta_log):
        """Test that sequential commits from same client succeed."""
        for i in range(10):
            add = create_add_file(i)
            version = delta_log.commit([add])
            assert version == i

    def test_concurrent_read_does_not_block_write(self, delta_log):
        """Test that reading does not block writes."""
        # Initial commit
        delta_log.commit([create_add_file(0)])

        # Start reading in thread
        read_results = []

        def reader():
            for _ in range(5):
                snapshot = delta_log.get_snapshot()
                read_results.append(len(snapshot.state.files))
                time.sleep(0.01)

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()

        # Write while reading
        for i in range(1, 5):
            delta_log.commit([create_add_file(i)])

        reader_thread.join()

        # All reads should have succeeded
        assert len(read_results) == 5
        # File count should monotonically increase
        for i in range(1, len(read_results)):
            assert read_results[i] >= read_results[i - 1]

    def test_version_mismatch_triggers_retry(self, temp_table_path):
        """Test that version mismatch causes retry attempts.

        Note: The implementation re-reads latest version before each retry,
        so it will automatically advance past conflict files.
        """
        log = DeltaLog(str(temp_table_path))

        # First commit
        log.commit([create_add_file(0)])

        # Manually create a conflicting version file
        version_file = temp_table_path / "_delta_log" / "00000000000000000001.json"
        with open(version_file, "x") as f:
            f.write(json.dumps({"add": {"path": "conflict.parquet", "size": 100, "modificationTime": 1, "dataChange": True, "partitionValues": {}}}) + "\n")

        # Next commit should succeed by advancing past conflict
        version = log.commit([create_add_file(2)])
        # Should be version 2 (skipping conflict at version 1)
        assert version == 2

    def test_commit_with_zero_retries(self, temp_table_path):
        """Test commit behavior with max_retries=0 (no retries)."""
        log = DeltaLog(str(temp_table_path))
        log.commit([create_add_file(0)])

        # Create conflict
        version_1_file = temp_table_path / "_delta_log" / "00000000000000000001.json"
        version_1_file.touch()

        # Should fail immediately with no retry option
        # Note: Current implementation has max_retries default of 3
        with pytest.raises(Exception):
            log.commit([create_add_file(1)], max_retries=0)


class TestAtomicWrites:
    """Tests for atomic write operations."""

    def test_commit_is_atomic(self, delta_log):
        """Test that commit writes are atomic (all or nothing)."""
        actions = [
            create_add_file(0),
            create_add_file(1),
            create_add_file(2),
        ]

        delta_log.commit(actions)

        # Read back - should have all or none
        snapshot = delta_log.get_snapshot()
        assert len(snapshot.state.files) == 3

    def test_partial_commit_not_visible(self, temp_table_path):
        """Test that partially written commits are not visible."""
        log = DeltaLog(str(temp_table_path))

        # Create initial version
        log.commit([create_add_file(0)])

        # Create an incomplete log file manually (simulating crash)
        incomplete_file = temp_table_path / "_delta_log" / "00000000000000000001.json"
        with open(incomplete_file, "w") as f:
            # Write partial JSON that's invalid
            f.write('{"add": {"path": "incomplete.parquet"')
            # Don't close JSON - simulates crash during write

        # Reading should either skip the invalid file or handle gracefully
        # Behavior depends on implementation - here we test it doesn't crash
        try:
            actions = log._read_log_file(1)
            # If it succeeds, should return empty or valid actions
            assert actions == [] or all(isinstance(a, (AddFile, RemoveFile, Metadata, Protocol)) for a in actions)
        except json.JSONDecodeError:
            # This is also acceptable - the file is genuinely corrupt
            pass

    def test_log_file_format_integrity(self, delta_log):
        """Test that log files maintain correct format."""
        # Commit actions
        actions = [
            Protocol(min_reader_version=1, min_writer_version=2),
            Metadata(
                id="test",
                name="test",
                description="",
                schema_string="{}",
                partition_columns=[],
            ),
            create_add_file(0),
        ]
        delta_log.commit(actions)

        # Read raw file and verify format
        log_file = delta_log.log_path / "00000000000000000000.json"
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Each line should be valid JSON
        for line in lines:
            if line.strip():
                parsed = json.loads(line)
                # Each line should have exactly one action type
                assert len(parsed) == 1
                assert any(k in parsed for k in ["add", "remove", "metaData", "protocol"])


class TestTransactionIsolation:
    """Tests for transaction isolation properties."""

    def test_read_committed_isolation(self, delta_log):
        """Test that reads only see committed data."""
        # Commit version 0
        delta_log.commit([create_add_file(0)])

        # Get snapshot before next commit
        snapshot_before = delta_log.get_snapshot()

        # Commit version 1
        delta_log.commit([create_add_file(1)])

        # Previous snapshot should still show old state
        assert len(snapshot_before.state.files) == 1
        assert snapshot_before.version == 0

        # New snapshot shows new state
        snapshot_after = delta_log.get_snapshot()
        assert len(snapshot_after.state.files) == 2
        assert snapshot_after.version == 1

    def test_version_specific_reads(self, delta_log):
        """Test reading specific versions gives consistent results."""
        # Create multiple versions
        for i in range(5):
            delta_log.commit([create_add_file(i)])

        # Each version read should be consistent
        for v in range(5):
            snapshot = delta_log.get_snapshot(version=v)
            assert snapshot.version == v
            assert len(snapshot.state.files) == v + 1

            # Verify specific files
            paths = sorted([f.path for f in snapshot.state.files])
            expected_paths = sorted([f"part-{j:05d}.parquet" for j in range(v + 1)])
            assert paths == expected_paths

    def test_snapshot_immutability(self, delta_log):
        """Test that snapshots are immutable after creation."""
        delta_log.commit([create_add_file(0)])
        snapshot = delta_log.get_snapshot()

        original_file_count = len(snapshot.state.files)
        original_version = snapshot.version

        # Commit more data
        delta_log.commit([create_add_file(1)])
        delta_log.commit([create_add_file(2)])

        # Original snapshot should be unchanged
        assert len(snapshot.state.files) == original_file_count
        assert snapshot.version == original_version


class TestTransactionLogIntegrity:
    """Tests for transaction log integrity."""

    def test_version_numbers_are_sequential(self, delta_log):
        """Test that version numbers are strictly sequential."""
        versions = []
        for i in range(10):
            version = delta_log.commit([create_add_file(i)])
            versions.append(version)

        # Verify sequential
        for i, v in enumerate(versions):
            assert v == i

    def test_log_files_have_correct_names(self, delta_log):
        """Test that log files are named with 20-digit versions."""
        for i in range(3):
            delta_log.commit([create_add_file(i)])

        log_files = sorted(delta_log.log_path.glob("*.json"))
        expected_names = [
            "00000000000000000000.json",
            "00000000000000000001.json",
            "00000000000000000002.json",
        ]

        for log_file, expected_name in zip(log_files, expected_names):
            assert log_file.name == expected_name

    def test_replay_produces_consistent_state(self, delta_log):
        """Test that replaying log produces same state."""
        # Create complex transaction history
        delta_log.commit([
            Protocol(min_reader_version=1, min_writer_version=2),
            Metadata(id="t1", name="test", description="", schema_string="{}", partition_columns=[]),
        ])

        for i in range(5):
            delta_log.commit([create_add_file(i)])

        # Remove some files
        delta_log.commit([RemoveFile(path="part-00001.parquet", deletion_timestamp=999, data_change=True)])
        delta_log.commit([RemoveFile(path="part-00003.parquet", deletion_timestamp=1000, data_change=True)])

        # Add more
        delta_log.commit([create_add_file(10)])

        # Get snapshot
        snapshot = delta_log.get_snapshot()

        # Create new DeltaLog instance and replay
        new_log = DeltaLog(delta_log.table_path)
        replayed_snapshot = new_log.get_snapshot()

        # Should have same state
        assert replayed_snapshot.version == snapshot.version
        assert len(replayed_snapshot.state.files) == len(snapshot.state.files)

        original_paths = sorted([f.path for f in snapshot.state.files])
        replayed_paths = sorted([f.path for f in replayed_snapshot.state.files])
        assert original_paths == replayed_paths


class TestConcurrentTransactions:
    """Tests for concurrent transaction handling."""

    def test_concurrent_reads_safe(self, temp_table_path):
        """Test that concurrent reads are safe."""
        log = DeltaLog(str(temp_table_path))

        # Create initial data
        for i in range(10):
            log.commit([create_add_file(i)])

        results = []
        errors = []

        def reader(thread_id):
            try:
                for _ in range(10):
                    snapshot = log.get_snapshot()
                    results.append((thread_id, len(snapshot.state.files)))
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50  # 5 threads x 10 reads each
        # All reads should show 10 files
        assert all(count == 10 for _, count in results)

    def test_read_during_write(self, temp_table_path):
        """Test reading during ongoing writes."""
        log = DeltaLog(str(temp_table_path))
        log.commit([create_add_file(0)])

        write_complete = threading.Event()
        read_results = []

        def writer():
            for i in range(1, 20):
                log.commit([create_add_file(i)])
                time.sleep(0.01)
            write_complete.set()

        def reader():
            while not write_complete.is_set():
                snapshot = log.get_snapshot()
                read_results.append(len(snapshot.state.files))
                time.sleep(0.005)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # All reads should return valid counts
        assert all(1 <= count <= 20 for count in read_results)
        # Counts should generally increase (though not strictly due to timing)
        # At minimum, should see some progression
        assert max(read_results) > min(read_results)


class TestTransactionRecovery:
    """Tests for transaction recovery scenarios."""

    def test_recover_from_extraneous_files(self, temp_table_path):
        """Test recovery ignores non-log files in delta log directory."""
        log = DeltaLog(str(temp_table_path))

        # Create initial setup
        protocol = Protocol(min_reader_version=1, min_writer_version=2)
        metadata = Metadata(
            id="test",
            name="test",
            description="",
            schema_string="{}",
            partition_columns=[],
        )
        log.commit([protocol, metadata])

        # Create several versions with files
        for i in range(5):
            log.commit([create_add_file(i)])

        # Create extra files that should be ignored
        (temp_table_path / "_delta_log" / "README.md").touch()
        (temp_table_path / "_delta_log" / "_temp").mkdir(exist_ok=True)
        (temp_table_path / "_delta_log" / "checkpoint.parquet").touch()

        # New log instance should still work by reading only .json files
        new_log = DeltaLog(str(temp_table_path))
        latest = new_log._get_latest_version()

        # Should have 6 versions (0-5)
        assert latest == 5

        # Verify file count in snapshot
        snapshot = new_log.get_snapshot()
        assert len(snapshot.state.files) == 5

    def test_get_latest_version_skips_non_json_files(self, temp_table_path):
        """Test that _get_latest_version ignores non-JSON files."""
        log = DeltaLog(str(temp_table_path))
        log.commit([create_add_file(0)])
        log.commit([create_add_file(1)])

        # Create some non-JSON files
        (temp_table_path / "_delta_log" / "README.md").touch()
        (temp_table_path / "_delta_log" / "temp.txt").touch()

        assert log._get_latest_version() == 1


class TestTransactionMetrics:
    """Tests for transaction metrics and statistics."""

    def test_count_versions(self, delta_log):
        """Test counting total versions."""
        for i in range(10):
            delta_log.commit([create_add_file(i)])

        # Get all versions
        latest = delta_log._get_latest_version()
        assert latest == 9  # 0-indexed, so version 9 after 10 commits

    def test_version_timestamps(self, delta_log):
        """Test that version files have correct timestamps."""
        import os

        for i in range(3):
            delta_log.commit([create_add_file(i)])
            time.sleep(0.1)  # Small delay between commits

        # Check file modification times are in order
        log_files = sorted(delta_log.log_path.glob("*.json"))
        mtimes = [os.path.getmtime(f) for f in log_files]

        for i in range(1, len(mtimes)):
            assert mtimes[i] >= mtimes[i - 1]


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling in transactions."""

    def test_commit_to_read_only_path(self, temp_table_path):
        """Test commit behavior with read-only path."""
        log = DeltaLog(str(temp_table_path))
        log.commit([create_add_file(0)])

        # Make log directory read-only
        import os
        import stat

        original_mode = log.log_path.stat().st_mode
        try:
            os.chmod(log.log_path, stat.S_IRUSR | stat.S_IXUSR)

            # Attempt to commit should fail
            with pytest.raises((PermissionError, OSError)):
                log.commit([create_add_file(1)])
        finally:
            # Restore permissions for cleanup
            os.chmod(log.log_path, original_mode)

    def test_empty_log_path_detection(self, temp_table_path):
        """Test detection of empty delta log directory."""
        log = DeltaLog(str(temp_table_path))

        # Create empty log directory
        log.log_path.mkdir(parents=True, exist_ok=True)

        # Should still report no versions
        assert log._get_latest_version() == -1

    def test_very_large_transaction(self, delta_log):
        """Test handling very large transactions."""
        # Create transaction with many files
        large_transaction = [create_add_file(i) for i in range(1000)]

        version = delta_log.commit(large_transaction)
        assert version == 0

        snapshot = delta_log.get_snapshot()
        assert len(snapshot.state.files) == 1000

    def test_transaction_with_all_action_types(self, delta_log):
        """Test transaction containing all action types."""
        actions = [
            Protocol(min_reader_version=1, min_writer_version=2),
            Metadata(
                id="test",
                name="test_table",
                description="Full transaction test",
                schema_string='{"type": "struct", "fields": []}',
                partition_columns=["date"],
            ),
            create_add_file(0),
            create_add_file(1),
        ]

        delta_log.commit(actions)

        # Remove file in next transaction
        delta_log.commit([
            RemoveFile(path="part-00000.parquet", deletion_timestamp=999, data_change=True),
            create_add_file(2),
        ])

        snapshot = delta_log.get_snapshot()
        assert snapshot.state.protocol.min_reader_version == 1
        assert snapshot.state.metadata.name == "test_table"
        assert len(snapshot.state.files) == 2
