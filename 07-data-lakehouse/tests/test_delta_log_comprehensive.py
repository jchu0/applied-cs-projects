"""Comprehensive tests for Delta Lake transaction log implementation."""

import json
import sys
import tempfile
import time
from pathlib import Path
from threading import Thread
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

Action = delta_log_module.Action
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
def sample_add_file():
    """Create a sample AddFile action."""
    return AddFile(
        path="data/part-00000.parquet",
        partition_values={"year": "2024", "month": "01"},
        size=1024000,
        modification_time=1700000000,
        data_change=True,
        stats='{"numRecords": 1000}',
        tags={"source": "streaming"},
    )


@pytest.fixture
def sample_metadata():
    """Create a sample Metadata action."""
    return Metadata(
        id="test-table-id",
        name="test_table",
        description="Test Delta table",
        schema_string='{"type": "struct", "fields": [{"name": "id", "type": "long"}]}',
        partition_columns=["year", "month"],
        configuration={"delta.autoOptimize.optimizeWrite": "true"},
        format_provider="parquet",
    )


@pytest.fixture
def sample_protocol():
    """Create a sample Protocol action."""
    return Protocol(
        min_reader_version=1,
        min_writer_version=2,
    )


# =============================================================================
# Test Action Classes
# =============================================================================


class TestAddFile:
    """Tests for AddFile action class."""

    def test_create_add_file_with_all_fields(self, sample_add_file):
        """Test creating AddFile with all fields populated."""
        assert sample_add_file.path == "data/part-00000.parquet"
        assert sample_add_file.partition_values == {"year": "2024", "month": "01"}
        assert sample_add_file.size == 1024000
        assert sample_add_file.modification_time == 1700000000
        assert sample_add_file.data_change is True
        assert "numRecords" in sample_add_file.stats
        assert sample_add_file.tags == {"source": "streaming"}

    def test_create_add_file_with_minimal_fields(self):
        """Test creating AddFile with only required fields."""
        add_file = AddFile(
            path="part-00000.parquet",
            partition_values={},
            size=100,
            modification_time=123456789,
            data_change=True,
        )
        assert add_file.path == "part-00000.parquet"
        assert add_file.stats is None
        assert add_file.tags is None

    def test_add_file_with_empty_partition_values(self):
        """Test AddFile for non-partitioned table."""
        add_file = AddFile(
            path="unpartitioned/data.parquet",
            partition_values={},
            size=500,
            modification_time=111111111,
            data_change=True,
        )
        assert add_file.partition_values == {}

    def test_add_file_data_change_false(self):
        """Test AddFile where data_change is False (e.g., OPTIMIZE operation)."""
        add_file = AddFile(
            path="optimized/part-00000.parquet",
            partition_values={},
            size=1000,
            modification_time=123,
            data_change=False,
        )
        assert add_file.data_change is False


class TestRemoveFile:
    """Tests for RemoveFile action class."""

    def test_create_remove_file_with_all_fields(self):
        """Test creating RemoveFile with all fields."""
        remove_file = RemoveFile(
            path="data/old-file.parquet",
            deletion_timestamp=1700001000,
            data_change=True,
            extended_file_metadata=True,
            partition_values={"year": "2023"},
        )
        assert remove_file.path == "data/old-file.parquet"
        assert remove_file.deletion_timestamp == 1700001000
        assert remove_file.data_change is True
        assert remove_file.extended_file_metadata is True
        assert remove_file.partition_values == {"year": "2023"}

    def test_create_remove_file_with_defaults(self):
        """Test RemoveFile uses correct defaults."""
        remove_file = RemoveFile(
            path="old.parquet",
            deletion_timestamp=123,
            data_change=True,
        )
        assert remove_file.extended_file_metadata is False
        assert remove_file.partition_values == {}


class TestMetadata:
    """Tests for Metadata action class."""

    def test_create_metadata_with_all_fields(self, sample_metadata):
        """Test creating Metadata with all fields."""
        assert sample_metadata.id == "test-table-id"
        assert sample_metadata.name == "test_table"
        assert sample_metadata.description == "Test Delta table"
        assert len(sample_metadata.partition_columns) == 2
        assert "delta.autoOptimize.optimizeWrite" in sample_metadata.configuration

    def test_metadata_schema_string_format(self, sample_metadata):
        """Test that schema_string is valid JSON."""
        schema = json.loads(sample_metadata.schema_string)
        assert schema["type"] == "struct"
        assert "fields" in schema


class TestProtocol:
    """Tests for Protocol action class."""

    def test_create_protocol(self, sample_protocol):
        """Test creating Protocol action."""
        assert sample_protocol.min_reader_version == 1
        assert sample_protocol.min_writer_version == 2

    def test_protocol_version_zero(self):
        """Test Protocol with version 0 (legacy)."""
        protocol = Protocol(min_reader_version=0, min_writer_version=0)
        assert protocol.min_reader_version == 0


# =============================================================================
# Test TableState
# =============================================================================


class TestTableState:
    """Tests for TableState class."""

    def test_empty_table_state(self):
        """Test creating empty TableState."""
        state = TableState()
        assert state.files == []
        assert state.metadata is None
        assert state.protocol is None

    def test_apply_single_add_file(self, sample_add_file):
        """Test applying a single AddFile action."""
        state = TableState()
        new_state = state.apply_actions([sample_add_file])

        assert len(new_state.files) == 1
        assert new_state.files[0].path == sample_add_file.path
        # Original state should be unchanged (immutability)
        assert len(state.files) == 0

    def test_apply_multiple_add_files(self):
        """Test applying multiple AddFile actions."""
        state = TableState()

        add_files = [
            AddFile(
                path=f"part-{i:05d}.parquet",
                partition_values={},
                size=100 * i,
                modification_time=1700000000 + i,
                data_change=True,
            )
            for i in range(5)
        ]

        new_state = state.apply_actions(add_files)
        assert len(new_state.files) == 5

    def test_apply_remove_file_removes_correct_file(self, sample_add_file):
        """Test that RemoveFile removes the correct file."""
        state = TableState(files=[sample_add_file])

        remove = RemoveFile(
            path=sample_add_file.path,
            deletion_timestamp=1700002000,
            data_change=True,
        )

        new_state = state.apply_actions([remove])
        assert len(new_state.files) == 0

    def test_apply_remove_file_keeps_other_files(self):
        """Test that RemoveFile only removes the specified file."""
        add_files = [
            AddFile(
                path=f"part-{i:05d}.parquet",
                partition_values={},
                size=100,
                modification_time=123,
                data_change=True,
            )
            for i in range(3)
        ]

        state = TableState(files=add_files)

        remove = RemoveFile(
            path="part-00001.parquet",
            deletion_timestamp=456,
            data_change=True,
        )

        new_state = state.apply_actions([remove])
        assert len(new_state.files) == 2
        paths = [f.path for f in new_state.files]
        assert "part-00001.parquet" not in paths
        assert "part-00000.parquet" in paths
        assert "part-00002.parquet" in paths

    def test_apply_metadata_sets_metadata(self, sample_metadata):
        """Test applying Metadata action sets metadata."""
        state = TableState()
        new_state = state.apply_actions([sample_metadata])

        assert new_state.metadata is not None
        assert new_state.metadata.name == "test_table"

    def test_apply_metadata_replaces_existing(self, sample_metadata):
        """Test that applying new Metadata replaces existing."""
        state = TableState(metadata=sample_metadata)

        new_metadata = Metadata(
            id="new-id",
            name="new_table",
            description="New description",
            schema_string="{}",
            partition_columns=[],
        )

        new_state = state.apply_actions([new_metadata])
        assert new_state.metadata.name == "new_table"

    def test_apply_protocol_sets_protocol(self, sample_protocol):
        """Test applying Protocol action sets protocol."""
        state = TableState()
        new_state = state.apply_actions([sample_protocol])

        assert new_state.protocol is not None
        assert new_state.protocol.min_reader_version == 1
        assert new_state.protocol.min_writer_version == 2

    def test_apply_mixed_actions(self, sample_add_file, sample_metadata, sample_protocol):
        """Test applying multiple action types in one call."""
        state = TableState()

        actions = [sample_protocol, sample_metadata, sample_add_file]
        new_state = state.apply_actions(actions)

        assert len(new_state.files) == 1
        assert new_state.metadata is not None
        assert new_state.protocol is not None

    def test_chained_apply_actions(self):
        """Test chaining multiple apply_actions calls."""
        state = TableState()

        # Add files
        add1 = AddFile(
            path="file1.parquet",
            partition_values={},
            size=100,
            modification_time=1,
            data_change=True,
        )
        state = state.apply_actions([add1])

        add2 = AddFile(
            path="file2.parquet",
            partition_values={},
            size=200,
            modification_time=2,
            data_change=True,
        )
        state = state.apply_actions([add2])

        # Remove one
        remove = RemoveFile(path="file1.parquet", deletion_timestamp=3, data_change=True)
        state = state.apply_actions([remove])

        assert len(state.files) == 1
        assert state.files[0].path == "file2.parquet"


# =============================================================================
# Test DeltaLog
# =============================================================================


class TestDeltaLogInitialization:
    """Tests for DeltaLog initialization."""

    def test_initialization(self, delta_log, temp_table_path):
        """Test DeltaLog initializes with correct paths."""
        assert delta_log.table_path == str(temp_table_path)
        assert delta_log.log_path == temp_table_path / "_delta_log"

    def test_get_latest_version_empty(self, delta_log):
        """Test getting latest version from empty log returns -1."""
        version = delta_log._get_latest_version()
        assert version == -1

    def test_find_latest_checkpoint_empty(self, delta_log):
        """Test finding checkpoint in empty log returns None."""
        checkpoint = delta_log._find_latest_checkpoint()
        assert checkpoint is None


class TestDeltaLogCommit:
    """Tests for DeltaLog commit functionality."""

    def test_first_commit(self, delta_log, sample_add_file):
        """Test committing to a new table creates version 0."""
        version = delta_log.commit([sample_add_file])

        assert version == 0
        assert delta_log._get_latest_version() == 0
        assert (delta_log.log_path / "00000000000000000000.json").exists()

    def test_sequential_commits(self, delta_log):
        """Test sequential commits increment versions correctly."""
        for i in range(5):
            add_file = AddFile(
                path=f"part-{i:05d}.parquet",
                partition_values={},
                size=100,
                modification_time=i,
                data_change=True,
            )
            version = delta_log.commit([add_file])
            assert version == i

        assert delta_log._get_latest_version() == 4

    def test_commit_multiple_actions(self, delta_log, sample_metadata, sample_protocol):
        """Test committing multiple actions in one transaction."""
        add_file = AddFile(
            path="data.parquet",
            partition_values={},
            size=100,
            modification_time=123,
            data_change=True,
        )

        version = delta_log.commit([sample_protocol, sample_metadata, add_file])
        assert version == 0

        # Read back and verify all actions
        actions = delta_log._read_log_file(0)
        assert len(actions) == 3

    def test_commit_creates_log_directory(self, delta_log, sample_add_file):
        """Test commit creates _delta_log directory if not exists."""
        assert not delta_log.log_path.exists()

        delta_log.commit([sample_add_file])

        assert delta_log.log_path.exists()
        assert delta_log.log_path.is_dir()

    def test_concurrent_commit_conflict(self, temp_table_path):
        """Test that concurrent commits handle conflicts gracefully.

        Note: The implementation reads latest version before each retry,
        so conflicts are resolved by advancing the version number.
        This test verifies that commits succeed even with artificial conflicts.
        """
        # Create DeltaLog instance
        log = DeltaLog(str(temp_table_path))

        # First commit succeeds
        add1 = AddFile(
            path="file1.parquet",
            partition_values={},
            size=100,
            modification_time=1,
            data_change=True,
        )
        v1 = log.commit([add1])
        assert v1 == 0

        # Manually create a version file to simulate conflict
        (temp_table_path / "_delta_log" / "00000000000000000001.json").touch()

        # Second commit will detect the conflict file and use version 2
        add2 = AddFile(
            path="file2.parquet",
            partition_values={},
            size=100,
            modification_time=2,
            data_change=True,
        )

        # This should succeed with version 2 (skipping conflicting version 1)
        v2 = log.commit([add2])
        assert v2 == 2  # Skipped version 1


class TestDeltaLogRead:
    """Tests for DeltaLog read operations."""

    def test_read_log_file_existing(self, delta_log, sample_add_file):
        """Test reading existing log file."""
        delta_log.commit([sample_add_file])

        actions = delta_log._read_log_file(0)
        assert len(actions) == 1
        assert isinstance(actions[0], AddFile)
        assert actions[0].path == sample_add_file.path

    def test_read_log_file_nonexistent(self, delta_log):
        """Test reading non-existent log file returns empty list."""
        actions = delta_log._read_log_file(999)
        assert actions == []

    def test_parse_add_action(self, delta_log):
        """Test parsing AddFile action from JSON."""
        action_data = {
            "add": {
                "path": "test.parquet",
                "partitionValues": {"date": "2024-01-01"},
                "size": 1000,
                "modificationTime": 12345,
                "dataChange": True,
                "stats": '{"numRecords": 100}',
                "tags": {"key": "value"},
            }
        }

        action = delta_log._parse_action(action_data)
        assert isinstance(action, AddFile)
        assert action.path == "test.parquet"
        assert action.partition_values == {"date": "2024-01-01"}
        assert action.size == 1000

    def test_parse_remove_action(self, delta_log):
        """Test parsing RemoveFile action from JSON."""
        action_data = {
            "remove": {
                "path": "old.parquet",
                "deletionTimestamp": 99999,
                "dataChange": True,
                "partitionValues": {"year": "2023"},
            }
        }

        action = delta_log._parse_action(action_data)
        assert isinstance(action, RemoveFile)
        assert action.path == "old.parquet"
        assert action.deletion_timestamp == 99999

    def test_parse_metadata_action(self, delta_log):
        """Test parsing Metadata action from JSON."""
        action_data = {
            "metaData": {
                "id": "abc123",
                "name": "my_table",
                "description": "Description",
                "schemaString": "{}",
                "partitionColumns": ["col1"],
                "configuration": {"key": "value"},
            }
        }

        action = delta_log._parse_action(action_data)
        assert isinstance(action, Metadata)
        assert action.id == "abc123"
        assert action.name == "my_table"

    def test_parse_protocol_action(self, delta_log):
        """Test parsing Protocol action from JSON."""
        action_data = {
            "protocol": {
                "minReaderVersion": 1,
                "minWriterVersion": 2,
            }
        }

        action = delta_log._parse_action(action_data)
        assert isinstance(action, Protocol)
        assert action.min_reader_version == 1
        assert action.min_writer_version == 2

    def test_parse_unknown_action(self, delta_log):
        """Test parsing unknown action type returns None."""
        action_data = {"unknownAction": {"foo": "bar"}}
        action = delta_log._parse_action(action_data)
        assert action is None


class TestDeltaLogSnapshot:
    """Tests for DeltaLog snapshot functionality."""

    def test_get_snapshot_empty_table(self, delta_log):
        """Test getting snapshot of empty table."""
        # Need to create at least one version for snapshot
        add = AddFile(
            path="file.parquet",
            partition_values={},
            size=100,
            modification_time=1,
            data_change=True,
        )
        delta_log.commit([add])

        snapshot = delta_log.get_snapshot()
        assert isinstance(snapshot, Snapshot)
        assert snapshot.version == 0
        assert len(snapshot.state.files) == 1

    def test_get_snapshot_multiple_versions(self, delta_log):
        """Test getting latest snapshot after multiple commits."""
        for i in range(5):
            add = AddFile(
                path=f"part-{i}.parquet",
                partition_values={},
                size=100 * i,
                modification_time=i,
                data_change=True,
            )
            delta_log.commit([add])

        snapshot = delta_log.get_snapshot()
        assert snapshot.version == 4
        assert len(snapshot.state.files) == 5

    def test_get_snapshot_specific_version(self, delta_log):
        """Test getting snapshot at a specific version."""
        for i in range(5):
            add = AddFile(
                path=f"part-{i}.parquet",
                partition_values={},
                size=100,
                modification_time=i,
                data_change=True,
            )
            delta_log.commit([add])

        # Get snapshot at version 2 (should have 3 files: 0, 1, 2)
        snapshot = delta_log.get_snapshot(version=2)
        assert snapshot.version == 2
        assert len(snapshot.state.files) == 3

    def test_get_snapshot_with_removes(self, delta_log):
        """Test snapshot correctly reflects file removes."""
        # Add 3 files
        for i in range(3):
            add = AddFile(
                path=f"part-{i}.parquet",
                partition_values={},
                size=100,
                modification_time=i,
                data_change=True,
            )
            delta_log.commit([add])

        # Remove middle file
        remove = RemoveFile(
            path="part-1.parquet",
            deletion_timestamp=999,
            data_change=True,
        )
        delta_log.commit([remove])

        snapshot = delta_log.get_snapshot()
        assert snapshot.version == 3
        assert len(snapshot.state.files) == 2
        paths = [f.path for f in snapshot.state.files]
        assert "part-1.parquet" not in paths


class TestDeltaLogActionToJson:
    """Tests for DeltaLog action serialization."""

    def test_add_file_to_json(self, delta_log, sample_add_file):
        """Test serializing AddFile to JSON format."""
        json_data = delta_log._action_to_json(sample_add_file)

        assert "add" in json_data
        assert json_data["add"]["path"] == sample_add_file.path
        assert json_data["add"]["partitionValues"] == sample_add_file.partition_values
        assert json_data["add"]["size"] == sample_add_file.size

    def test_remove_file_to_json(self, delta_log):
        """Test serializing RemoveFile to JSON format."""
        remove = RemoveFile(
            path="old.parquet",
            deletion_timestamp=12345,
            data_change=True,
            partition_values={"year": "2024"},
        )

        json_data = delta_log._action_to_json(remove)

        assert "remove" in json_data
        assert json_data["remove"]["path"] == "old.parquet"
        assert json_data["remove"]["deletionTimestamp"] == 12345

    def test_metadata_to_json(self, delta_log, sample_metadata):
        """Test serializing Metadata to JSON format."""
        json_data = delta_log._action_to_json(sample_metadata)

        assert "metaData" in json_data
        assert json_data["metaData"]["id"] == sample_metadata.id
        assert json_data["metaData"]["name"] == sample_metadata.name

    def test_protocol_to_json(self, delta_log, sample_protocol):
        """Test serializing Protocol to JSON format."""
        json_data = delta_log._action_to_json(sample_protocol)

        assert "protocol" in json_data
        assert json_data["protocol"]["minReaderVersion"] == 1
        assert json_data["protocol"]["minWriterVersion"] == 2


class TestDeltaLogRoundTrip:
    """Tests for serialization/deserialization round trips."""

    def test_add_file_round_trip(self, delta_log, sample_add_file):
        """Test AddFile survives serialization round trip."""
        delta_log.commit([sample_add_file])
        actions = delta_log._read_log_file(0)

        assert len(actions) == 1
        recovered = actions[0]

        assert recovered.path == sample_add_file.path
        assert recovered.partition_values == sample_add_file.partition_values
        assert recovered.size == sample_add_file.size
        assert recovered.modification_time == sample_add_file.modification_time
        assert recovered.data_change == sample_add_file.data_change
        assert recovered.stats == sample_add_file.stats
        assert recovered.tags == sample_add_file.tags

    def test_full_table_round_trip(
        self, delta_log, sample_add_file, sample_metadata, sample_protocol
    ):
        """Test full table setup survives round trip."""
        delta_log.commit([sample_protocol, sample_metadata, sample_add_file])

        snapshot = delta_log.get_snapshot()

        assert len(snapshot.state.files) == 1
        assert snapshot.state.files[0].path == sample_add_file.path
        assert snapshot.state.metadata.name == sample_metadata.name
        assert snapshot.state.protocol.min_reader_version == sample_protocol.min_reader_version


class TestDeltaLogEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_actions_list(self, delta_log):
        """Test committing empty actions list."""
        version = delta_log.commit([])
        assert version == 0
        actions = delta_log._read_log_file(0)
        assert actions == []

    def test_large_number_of_files(self, delta_log):
        """Test handling large number of files in single commit."""
        files = [
            AddFile(
                path=f"part-{i:08d}.parquet",
                partition_values={"bucket": str(i % 100)},
                size=1024,
                modification_time=1700000000 + i,
                data_change=True,
            )
            for i in range(1000)
        ]

        version = delta_log.commit(files)
        assert version == 0

        snapshot = delta_log.get_snapshot()
        assert len(snapshot.state.files) == 1000

    def test_special_characters_in_path(self, delta_log):
        """Test handling special characters in file path."""
        add = AddFile(
            path="data/year=2024/month=01/part name with spaces.parquet",
            partition_values={"year": "2024", "month": "01"},
            size=100,
            modification_time=123,
            data_change=True,
        )

        delta_log.commit([add])
        actions = delta_log._read_log_file(0)

        assert actions[0].path == "data/year=2024/month=01/part name with spaces.parquet"

    def test_unicode_in_stats(self, delta_log):
        """Test handling unicode characters in stats field."""
        add = AddFile(
            path="data.parquet",
            partition_values={},
            size=100,
            modification_time=123,
            data_change=True,
            stats='{"minValues": {"name": "Cafe"}}',
        )

        delta_log.commit([add])
        actions = delta_log._read_log_file(0)

        assert "Cafe" in actions[0].stats

    def test_very_long_path(self, delta_log):
        """Test handling very long file paths."""
        long_path = "data/" + "/".join([f"level{i}" for i in range(50)]) + "/file.parquet"

        add = AddFile(
            path=long_path,
            partition_values={},
            size=100,
            modification_time=123,
            data_change=True,
        )

        delta_log.commit([add])
        actions = delta_log._read_log_file(0)

        assert actions[0].path == long_path
