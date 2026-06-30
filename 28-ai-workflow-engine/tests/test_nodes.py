"""Unit tests for node components."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Any, Dict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Optional imports requiring yaml
try:
    from aiworkflow.nodes import (
        NodeBase,
        NodeExecutorRegistry,
        DataNode,
        ProcessNode,
        ModelNode,
        ValidationNode,
        ConditionalNode
    )
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from aiworkflow.schemas import Node as NodeDefinition, NodeType, NodeStatus

pytestmark = pytest.mark.skipif(not _HAS_YAML, reason="Requires yaml")


class TestNodeBase:
    """Test suite for base node functionality."""

    def test_node_initialization(self):
        """Test node initialization."""
        node_def = NodeDefinition(
            id="test-node",
            name="TestNode",
            type=NodeType.PROCESS,
            config={"param": "value"}
        )

        node = NodeBase(node_def)

        assert node.id == "test-node"
        assert node.name == "TestNode"
        assert node.type == NodeType.PROCESS
        assert node.config == {"param": "value"}
        assert node.status == NodeStatus.PENDING

    @pytest.mark.asyncio
    async def test_node_execute_lifecycle(self):
        """Test node execution lifecycle."""
        node_def = NodeDefinition(
            id="test-node",
            name="TestNode",
            type=NodeType.PROCESS,
            config={}
        )

        node = NodeBase(node_def)

        # Mock the _execute method
        async def mock_execute(inputs):
            return {"result": "success"}

        node._execute = mock_execute

        # Execute node
        result = await node.execute({"input": "data"})

        assert node.status == NodeStatus.COMPLETED
        assert result == {"result": "success"}
        # >= 0: a no-op coroutine can complete within the clock's resolution
        # (0.0s on a fast run), so a strict > 0 is flaky.
        assert node.execution_time >= 0

    @pytest.mark.asyncio
    async def test_node_execute_with_error(self):
        """Test node execution with error."""
        node_def = NodeDefinition(
            id="test-node",
            name="TestNode",
            type=NodeType.PROCESS,
            config={}
        )

        node = NodeBase(node_def)

        # Mock the _execute method to raise an error
        async def mock_execute(inputs):
            raise ValueError("Execution failed")

        node._execute = mock_execute

        # Execute node
        with pytest.raises(ValueError, match="Execution failed"):
            await node.execute({"input": "data"})

        assert node.status == NodeStatus.FAILED
        assert node.error is not None

    @pytest.mark.asyncio
    async def test_node_validation(self):
        """Test node input validation."""
        node_def = NodeDefinition(
            id="test-node",
            name="TestNode",
            type=NodeType.PROCESS,
            config={},
            input_schema={
                "type": "object",
                "properties": {
                    "required_field": {"type": "string"}
                },
                "required": ["required_field"]
            }
        )

        node = NodeBase(node_def)

        # Invalid input (missing required field)
        is_valid = node.validate_inputs({})
        assert is_valid is False

        # Valid input
        is_valid = node.validate_inputs({"required_field": "value"})
        assert is_valid is True


class TestDataNode:
    """Test suite for DataNode."""

    @pytest.mark.asyncio
    async def test_csv_data_node(self):
        """Test CSV data loading node."""
        node_def = NodeDefinition(
            id="csv-loader",
            name="CSVLoader",
            type=NodeType.DATA,
            config={
                "source": "test.csv",
                "delimiter": ",",
                "has_header": True
            }
        )

        node = DataNode(node_def)

        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = Mock()
            mock_df.to_dict.return_value = {"col1": [1, 2], "col2": [3, 4]}
            mock_read_csv.return_value = mock_df

            result = await node.execute({})

            mock_read_csv.assert_called_once_with(
                "test.csv",
                delimiter=",",
                header=0
            )
            assert result["data"] == {"col1": [1, 2], "col2": [3, 4]}

    @pytest.mark.asyncio
    async def test_database_data_node(self):
        """Test database data loading node."""
        node_def = NodeDefinition(
            id="db-loader",
            name="DBLoader",
            type=NodeType.DATA,
            config={
                "source": "database",
                "connection_string": "postgresql://localhost/testdb",
                "query": "SELECT * FROM users"
            }
        )

        node = DataNode(node_def)

        with patch('sqlalchemy.create_engine') as mock_engine:
            with patch('pandas.read_sql') as mock_read_sql:
                mock_df = Mock()
                mock_df.to_dict.return_value = {"id": [1, 2], "name": ["Alice", "Bob"]}
                mock_read_sql.return_value = mock_df

                result = await node.execute({})

                assert result["data"] == {"id": [1, 2], "name": ["Alice", "Bob"]}

    @pytest.mark.asyncio
    async def test_api_data_node(self):
        """Test API data loading node."""
        node_def = NodeDefinition(
            id="api-loader",
            name="APILoader",
            type=NodeType.DATA,
            config={
                "source": "api",
                "url": "https://api.example.com/data",
                "method": "GET",
                "headers": {"Authorization": "Bearer token"}
            }
        )

        node = DataNode(node_def)

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = {"users": [{"id": 1}, {"id": 2}]}
            mock_session.request.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session

            result = await node.execute({})

            assert result["data"] == {"users": [{"id": 1}, {"id": 2}]}


class TestProcessNode:
    """Test suite for ProcessNode."""

    @pytest.mark.asyncio
    async def test_transform_process_node(self):
        """Test data transformation process node."""
        node_def = NodeDefinition(
            id="transformer",
            name="DataTransformer",
            type=NodeType.PROCESS,
            config={
                "operations": ["normalize", "scale"],
                "parameters": {
                    "scale_factor": 2.0,
                    "normalize_method": "minmax"
                }
            }
        )

        node = ProcessNode(node_def)

        input_data = {
            "data": {"values": [1, 2, 3, 4, 5]}
        }

        result = await node.execute(input_data)

        assert "processed_data" in result
        assert result["metadata"]["operations_applied"] == ["normalize", "scale"]

    @pytest.mark.asyncio
    async def test_aggregation_process_node(self):
        """Test data aggregation process node."""
        node_def = NodeDefinition(
            id="aggregator",
            name="DataAggregator",
            type=NodeType.PROCESS,
            config={
                "aggregations": {
                    "mean": True,
                    "sum": True,
                    "count": True
                },
                "group_by": ["category"]
            }
        )

        node = ProcessNode(node_def)

        input_data = {
            "data": {
                "category": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40]
            }
        }

        result = await node.execute(input_data)

        assert "aggregated_data" in result
        assert "statistics" in result


class TestModelNode:
    """Test suite for ModelNode."""

    @pytest.mark.asyncio
    async def test_train_model_node(self):
        """Test model training node."""
        node_def = NodeDefinition(
            id="trainer",
            name="ModelTrainer",
            type=NodeType.MODEL,
            config={
                "algorithm": "random_forest",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10
                },
                "training_config": {
                    "validation_split": 0.2,
                    "epochs": 10
                }
            }
        )

        node = ModelNode(node_def)

        input_data = {
            "features": [[1, 2], [3, 4], [5, 6]],
            "labels": [0, 1, 0]
        }

        with patch('sklearn.ensemble.RandomForestClassifier') as mock_rf:
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            mock_model.score.return_value = 0.95
            mock_rf.return_value = mock_model

            result = await node.execute(input_data)

            assert "model" in result
            assert "metrics" in result
            assert result["metrics"]["accuracy"] == 0.95

    @pytest.mark.asyncio
    async def test_predict_model_node(self):
        """Test model prediction node."""
        node_def = NodeDefinition(
            id="predictor",
            name="ModelPredictor",
            type=NodeType.MODEL,
            config={
                "mode": "predict",
                "model_path": "/models/trained_model.pkl"
            }
        )

        node = ModelNode(node_def)

        input_data = {
            "features": [[1, 2], [3, 4]],
            "model": Mock()  # Pre-loaded model
        }

        with patch.object(input_data["model"], 'predict', return_value=[0, 1]):
            result = await node.execute(input_data)

            assert "predictions" in result
            assert result["predictions"] == [0, 1]


class TestConditionalNode:
    """Test suite for ConditionalNode."""

    @pytest.mark.asyncio
    async def test_simple_condition(self):
        """Test simple conditional branching."""
        node_def = NodeDefinition(
            id="conditional",
            name="ConditionalBranch",
            type=NodeType.CONDITIONAL,
            config={
                "condition": {
                    "field": "score",
                    "operator": "greater_than",
                    "value": 0.8
                },
                "true_branch": "high_score_path",
                "false_branch": "low_score_path"
            }
        )

        node = ConditionalNode(node_def)

        # Test true condition
        result = await node.execute({"score": 0.9})
        assert result["next_node"] == "high_score_path"
        assert result["condition_met"] is True

        # Test false condition
        result = await node.execute({"score": 0.7})
        assert result["next_node"] == "low_score_path"
        assert result["condition_met"] is False

    @pytest.mark.asyncio
    async def test_complex_condition(self):
        """Test complex conditional logic."""
        node_def = NodeDefinition(
            id="complex_conditional",
            name="ComplexBranch",
            type=NodeType.CONDITIONAL,
            config={
                "condition": {
                    "type": "and",
                    "conditions": [
                        {
                            "field": "accuracy",
                            "operator": "greater_than",
                            "value": 0.9
                        },
                        {
                            "field": "loss",
                            "operator": "less_than",
                            "value": 0.1
                        }
                    ]
                },
                "true_branch": "deploy_model",
                "false_branch": "retrain_model"
            }
        )

        node = ConditionalNode(node_def)

        # Test all conditions met
        result = await node.execute({"accuracy": 0.95, "loss": 0.05})
        assert result["next_node"] == "deploy_model"

        # Test one condition not met
        result = await node.execute({"accuracy": 0.95, "loss": 0.15})
        assert result["next_node"] == "retrain_model"


class TestNodeExecutorRegistry:
    """Test suite for NodeExecutorRegistry."""

    def test_register_executor(self):
        """Test registering custom node executor."""
        registry = NodeExecutorRegistry()

        class CustomExecutor:
            async def execute(self, inputs):
                return {"custom": "result"}

        executor = CustomExecutor()
        registry.register("custom_type", executor)

        assert registry.get("custom_type") == executor

    def test_get_nonexistent_executor(self):
        """Test getting non-existent executor."""
        registry = NodeExecutorRegistry()

        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_executors(self):
        """Test listing registered executors."""
        registry = NodeExecutorRegistry()

        # Register default executors
        registry.register("data", DataNode)
        registry.register("process", ProcessNode)
        registry.register("model", ModelNode)

        executors = registry.list_executors()

        assert "data" in executors
        assert "process" in executors
        assert "model" in executors

    def test_unregister_executor(self):
        """Test unregistering executor."""
        registry = NodeExecutorRegistry()

        registry.register("temp", Mock())
        assert registry.has("temp")

        registry.unregister("temp")
        assert not registry.has("temp")