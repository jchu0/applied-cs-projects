"""Unit tests for the WorkflowEngine class."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Optional imports requiring yaml
try:
    from aiworkflow.engine import WorkflowEngine
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from aiworkflow.schemas import (
    FlowDefinition,
    FlowRun,
    RunStatus,
    Node as NodeDefinition,
    NodeType,
)

pytestmark = pytest.mark.skipif(not _HAS_YAML, reason="Requires yaml")


class TestWorkflowEngine:
    """Test suite for WorkflowEngine."""

    @pytest.fixture
    def engine(self):
        """Create a WorkflowEngine instance for testing."""
        return WorkflowEngine(
            max_parallel=5,
            enable_versioning=True,
            enable_optimization=True
        )

    @pytest.fixture
    def sample_flow_definition(self):
        """Create a sample flow definition for testing."""
        return FlowDefinition(
            name="test-flow",
            version="1.0.0",
            description="Test workflow",
            nodes=[
                NodeDefinition(
                    id="node1",
                    name="DataLoader",
                    type=NodeType.DATA,
                    config={"source": "test.csv"}
                ),
                NodeDefinition(
                    id="node2",
                    name="ModelTrainer",
                    type=NodeType.MODEL,
                    config={"model": "xgboost"},
                    dependencies=["node1"]
                ),
            ],
            edges=[
                {"from": "node1", "to": "node2"}
            ]
        )

    def test_engine_initialization(self, engine):
        """Test engine initialization with default parameters."""
        assert engine.scheduler is not None
        assert engine.node_registry is not None
        assert engine.retry_manager is not None
        assert engine.parser is not None
        assert engine.validator is not None
        assert engine.optimizer is not None
        assert engine.enable_versioning is True
        assert engine.enable_optimization is True
        assert engine.version_manager is not None

    def test_engine_without_versioning(self):
        """Test engine initialization without versioning."""
        engine = WorkflowEngine(enable_versioning=False)
        assert engine.enable_versioning is False
        assert not hasattr(engine, 'version_manager')

    def test_engine_without_optimization(self):
        """Test engine initialization without optimization."""
        engine = WorkflowEngine(enable_optimization=False)
        assert engine.enable_optimization is False

    @pytest.mark.asyncio
    async def test_run_flow_success(self, engine, sample_flow_definition):
        """Test successful flow execution."""
        expected_run = FlowRun(
            run_id="run-123",
            flow_id="test-flow",
            flow_version="1.0.0",
            status=RunStatus.COMPLETED,
            inputs={"data": "test"},
            outputs={"result": "success"},
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        with patch.object(engine.validator, 'validate', return_value=True):
            with patch.object(engine.optimizer, 'optimize', return_value=sample_flow_definition):
                with patch.object(engine.scheduler, 'execute', new_callable=AsyncMock) as mock_execute:
                    mock_execute.return_value = expected_run

                    result = await engine.run_flow(
                        flow_definition=sample_flow_definition,
                        inputs={"data": "test"}
                    )

                    assert result.status == RunStatus.COMPLETED
                    assert result.flow_id == "test-flow"

    @pytest.mark.asyncio
    async def test_run_flow_with_retry(self, engine, sample_flow_definition):
        """Test flow execution with retry logic."""
        expected_run = FlowRun(
            run_id="run-124",
            flow_id="test-flow",
            flow_version="1.0.0",
            status=RunStatus.COMPLETED,
            inputs={"data": "test"},
            outputs={"result": "success"},
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        with patch.object(engine.validator, 'validate', return_value=True):
            with patch.object(engine.scheduler, 'execute', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = expected_run

                result = await engine.run_flow(
                    flow_definition=sample_flow_definition,
                    inputs={"data": "test"},
                    enable_retry=True
                )

                assert mock_execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_flow_validation_failure(self, engine, sample_flow_definition):
        """Test flow execution with validation failure."""
        with patch.object(engine.validator, 'validate', return_value=False):
            with pytest.raises(ValueError, match="Flow validation failed"):
                await engine.run_flow(
                    flow_definition=sample_flow_definition,
                    inputs={"data": "test"}
                )

    @pytest.mark.asyncio
    async def test_pause_resume_flow(self, engine, sample_flow_definition):
        """Test pausing and resuming a flow."""
        run_id = "run-125"

        with patch.object(engine.scheduler, 'pause', new_callable=AsyncMock) as mock_pause:
            with patch.object(engine.scheduler, 'resume', new_callable=AsyncMock) as mock_resume:
                mock_pause.return_value = True
                mock_resume.return_value = True

                # Test pause
                result = await engine.pause_flow(run_id)
                assert result is True
                mock_pause.assert_called_once_with(run_id)

                # Test resume
                result = await engine.resume_flow(run_id)
                assert result is True
                mock_resume.assert_called_once_with(run_id)

    @pytest.mark.asyncio
    async def test_cancel_flow(self, engine):
        """Test canceling a running flow."""
        run_id = "run-126"

        with patch.object(engine.scheduler, 'cancel') as mock_cancel:
            mock_cancel.return_value = True

            result = await engine.cancel_flow(run_id)
            assert result is True
            mock_cancel.assert_called_once_with(run_id)

    def test_get_flow_status(self, engine):
        """Test getting flow execution status."""
        run_id = "run-127"
        expected_status = RunStatus.RUNNING

        with patch.object(engine.scheduler, 'get_status', return_value=expected_status):
            status = engine.get_flow_status(run_id)
            assert status == expected_status

    def test_list_active_flows(self, engine):
        """Test listing active flow executions."""
        expected_flows = ["run-128", "run-129", "run-130"]

        with patch.object(engine.scheduler, 'list_active', return_value=expected_flows):
            active_flows = engine.list_active_flows()
            assert active_flows == expected_flows
            assert len(active_flows) == 3

    @pytest.mark.asyncio
    async def test_flow_with_versioning(self, engine, sample_flow_definition):
        """Test flow execution with versioning enabled."""
        expected_run = FlowRun(
            run_id="run-131",
            flow_id="test-flow",
            flow_version="1.0.0",
            status=RunStatus.COMPLETED,
            inputs={"data": "test"},
            outputs={"result": "success"},
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        with patch.object(engine.version_manager, 'save_version') as mock_save:
            with patch.object(engine.validator, 'validate', return_value=True):
                with patch.object(engine.optimizer, 'optimize', return_value=sample_flow_definition):
                    with patch.object(engine.scheduler, 'execute', new_callable=AsyncMock) as mock_execute:
                        mock_execute.return_value = expected_run

                        await engine.run_flow(
                            flow_definition=sample_flow_definition,
                            inputs={"data": "test"}
                        )

                        mock_save.assert_called_once()

    def test_register_custom_node(self, engine):
        """Test registering a custom node executor."""
        custom_executor = Mock()
        node_type = "custom_node"

        engine.register_node_executor(node_type, custom_executor)

        assert engine.node_registry.get(node_type) == custom_executor

    @pytest.mark.asyncio
    async def test_concurrent_flow_execution(self, engine, sample_flow_definition):
        """Test concurrent execution of multiple flows."""
        num_flows = 3

        expected_run = FlowRun(
            run_id="run-concurrent",
            flow_id="test-flow",
            flow_version="1.0.0",
            status=RunStatus.COMPLETED,
            inputs={"data": "test"},
            outputs={"result": "success"},
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        with patch.object(engine.validator, 'validate', return_value=True):
            with patch.object(engine.optimizer, 'optimize', return_value=sample_flow_definition):
                with patch.object(engine.scheduler, 'execute', new_callable=AsyncMock) as mock_execute:
                    mock_execute.return_value = expected_run

                    tasks = [
                        engine.run_flow(
                            flow_definition=sample_flow_definition,
                            inputs={"data": f"test-{i}"}
                        )
                        for i in range(num_flows)
                    ]

                    results = await asyncio.gather(*tasks)

                    assert len(results) == num_flows
                    assert all(r.status == RunStatus.COMPLETED for r in results)

    def test_max_parallel_limit(self):
        """Test max parallel execution limit."""
        max_parallel = 3
        engine = WorkflowEngine(max_parallel=max_parallel)

        assert engine.scheduler.max_parallel == max_parallel
