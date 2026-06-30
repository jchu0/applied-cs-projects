"""Tests for the lakehouse processor."""

import pytest
from unittest.mock import MagicMock, patch


class TestLakehouseProcessor:
    """Test cases for LakehouseProcessor."""

    def test_import(self):
        """Test that module can be imported."""
        from lakehouse.processor import LakehouseProcessor
        assert LakehouseProcessor is not None

    def test_config_import(self):
        """Test that config types can be imported."""
        from lakehouse.config import Layer, DeltaTableConfig, LakehouseConfig

        assert Layer.BRONZE.value == "bronze"
        assert Layer.SILVER.value == "silver"
        assert Layer.GOLD.value == "gold"

    def test_delta_log_import(self):
        """Test that delta log types can be imported."""
        from lakehouse.delta_log import DeltaLog, AddFile, RemoveFile, TableState

        state = TableState()
        assert len(state.files) == 0

    def test_quality_engine_import(self):
        """Test that quality engine can be imported."""
        from lakehouse.quality import QualityEngine, ValidationResult

        engine = QualityEngine()
        assert engine is not None


class TestDeltaTableConfig:
    """Test cases for DeltaTableConfig."""

    def test_table_properties(self):
        """Test conversion to table properties."""
        from lakehouse.config import DeltaTableConfig, Layer

        config = DeltaTableConfig(
            name="test_table",
            path="/tmp/test",
            layer=Layer.SILVER,
            enable_change_data_feed=True,
            auto_compact=True,
            log_retention_days=30,
        )

        props = config.to_table_properties()
        assert props["delta.enableChangeDataFeed"] == "true"
        assert props["delta.autoOptimize.autoCompact"] == "true"
        assert "30 days" in props["delta.logRetentionDuration"]


class TestDeltaLog:
    """Test cases for DeltaLog."""

    def test_get_latest_version_empty(self, tmp_path):
        """Test getting latest version from empty log."""
        from lakehouse.delta_log import DeltaLog

        log = DeltaLog(str(tmp_path / "test_table"))
        version = log._get_latest_version()
        assert version == -1

    def test_action_to_json(self, tmp_path):
        """Test converting action to JSON."""
        from lakehouse.delta_log import DeltaLog, AddFile

        log = DeltaLog(str(tmp_path / "test_table"))

        action = AddFile(
            path="part-00000.parquet",
            partition_values={"date": "2024-01-01"},
            size=1024,
            modification_time=1234567890,
            data_change=True,
        )

        json_data = log._action_to_json(action)
        assert "add" in json_data
        assert json_data["add"]["path"] == "part-00000.parquet"
        assert json_data["add"]["size"] == 1024


class TestTableState:
    """Test cases for TableState."""

    def test_apply_add_file(self):
        """Test applying AddFile action."""
        from lakehouse.delta_log import TableState, AddFile

        state = TableState()
        action = AddFile(
            path="test.parquet",
            partition_values={},
            size=100,
            modification_time=123,
            data_change=True,
        )

        new_state = state.apply_actions([action])
        assert len(new_state.files) == 1
        assert new_state.files[0].path == "test.parquet"

    def test_apply_remove_file(self):
        """Test applying RemoveFile action."""
        from lakehouse.delta_log import TableState, AddFile, RemoveFile

        state = TableState(files=[
            AddFile(
                path="test.parquet",
                partition_values={},
                size=100,
                modification_time=123,
                data_change=True,
            )
        ])

        action = RemoveFile(
            path="test.parquet",
            deletion_timestamp=456,
            data_change=True,
        )

        new_state = state.apply_actions([action])
        assert len(new_state.files) == 0


class TestQualityEngine:
    """Test cases for QualityEngine."""

    def test_expectation_chain(self):
        """Test chaining expectations."""
        from lakehouse.quality import QualityEngine

        engine = QualityEngine()
        result = (
            engine
            .expect_column_to_exist("id")
            .expect_column_values_to_not_be_null("id")
            .expect_column_values_to_be_unique("id")
        )

        assert result is engine
        assert len(engine._expectations) == 3

    def test_clear_expectations(self):
        """Test clearing expectations."""
        from lakehouse.quality import QualityEngine

        engine = QualityEngine()
        engine.expect_column_to_exist("id")
        assert len(engine._expectations) == 1

        engine.clear_expectations()
        assert len(engine._expectations) == 0
