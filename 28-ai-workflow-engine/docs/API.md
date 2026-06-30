# AI Workflow Engine - API Documentation

## Table of Contents
1. [Python API](#python-api)
2. [REST API](#rest-api)
3. [Workflow DSL Reference](#workflow-dsl-reference)
4. [Node Types](#node-types)
5. [Configuration](#configuration)
6. [Error Handling](#error-handling)

## Python API

### WorkflowEngine

Main class for workflow execution and management.

#### Initialization

```python
from aiworkflow import WorkflowEngine

engine = WorkflowEngine(
    max_parallel=10,           # Maximum parallel node executions
    enable_versioning=True,    # Enable workflow versioning
    enable_optimization=True   # Enable automatic optimization
)
```

#### Methods

##### run_flow()
Execute a workflow asynchronously.

```python
async def run_flow(
    flow_definition: FlowDefinition,
    inputs: Dict[str, Any] = None,
    enable_retry: bool = True,
    checkpoint: bool = False
) -> FlowRun
```

**Example:**
```python
import asyncio
from aiworkflow import WorkflowEngine, FlowDefinition

engine = WorkflowEngine()

flow = FlowDefinition(
    name="example-flow",
    version="1.0.0",
    nodes=[...],
    edges=[...]
)

result = asyncio.run(engine.run_flow(flow, inputs={"data": "input"}))
print(f"Status: {result.status}")
print(f"Outputs: {result.outputs}")
```

##### pause_flow()
Pause a running workflow.

```python
async def pause_flow(run_id: str) -> bool
```

**Example:**
```python
success = await engine.pause_flow("run-123")
if success:
    print("Workflow paused successfully")
```

##### resume_flow()
Resume a paused workflow.

```python
async def resume_flow(run_id: str) -> bool
```

**Example:**
```python
success = await engine.resume_flow("run-123")
if success:
    print("Workflow resumed")
```

##### cancel_flow()
Cancel a running workflow.

```python
async def cancel_flow(run_id: str) -> bool
```

**Example:**
```python
success = await engine.cancel_flow("run-123")
if success:
    print("Workflow cancelled")
```

##### get_flow_status()
Get the current status of a workflow run.

```python
def get_flow_status(run_id: str) -> RunStatus
```

**Example:**
```python
status = engine.get_flow_status("run-123")
print(f"Current status: {status}")  # RUNNING, COMPLETED, FAILED, etc.
```

### FlowParser

Parse workflow definitions from various formats.

#### Methods

##### parse_yaml()
Parse YAML workflow definition.

```python
def parse_yaml(yaml_content: str) -> FlowDefinition
```

**Example:**
```python
from aiworkflow.compiler import FlowParser

parser = FlowParser()

yaml_content = """
name: ml-pipeline
version: 1.0.0
nodes:
  - id: loader
    name: DataLoader
    type: data
    config:
      source: data.csv
  - id: trainer
    name: ModelTrainer
    type: model
    dependencies: [loader]
edges:
  - from: loader
    to: trainer
"""

flow = parser.parse_yaml(yaml_content)
```

##### parse_json()
Parse JSON workflow definition.

```python
def parse_json(json_content: Union[str, dict]) -> FlowDefinition
```

**Example:**
```python
json_flow = {
    "name": "json-pipeline",
    "version": "1.0.0",
    "nodes": [
        {
            "id": "node1",
            "name": "Node1",
            "type": "data",
            "config": {"key": "value"}
        }
    ],
    "edges": []
}

flow = parser.parse_json(json_flow)
```

##### parse_dsl()
Parse custom DSL workflow definition.

```python
def parse_dsl(dsl_content: str) -> FlowDefinition
```

**Example:**
```python
dsl_content = """
workflow data_pipeline:
    node loader:
        type: data
        config:
            source: "database"

    node processor:
        type: process
        depends_on: loader
        config:
            operations: ["normalize", "encode"]

    flow:
        loader -> processor
"""

flow = parser.parse_dsl(dsl_content)
```

### Node Registration

Register custom node executors.

```python
from aiworkflow.nodes import NodeBase, NodeExecutorRegistry

class CustomNode(NodeBase):
    async def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Custom logic here
        result = inputs.get("data", []) * 2
        return {"processed": result}

# Register the custom node
engine.node_registry.register("custom_processor", CustomNode)
```

### Retry Configuration

Configure retry strategies for nodes.

```python
from aiworkflow.retry import ExponentialBackoffRetry, RetryConfig

# Configure retry for specific node
retry_config = RetryConfig(
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
    exponential_factor=2.0,
    jitter=True
)

node = NodeDefinition(
    id="flaky_node",
    name="FlakyOperation",
    type=NodeType.PROCESS,
    retry_config=retry_config.to_dict()
)
```

## REST API

### Endpoints

#### POST /workflows/execute
Execute a workflow.

**Request:**
```json
{
    "workflow": {
        "name": "api-workflow",
        "version": "1.0.0",
        "nodes": [...],
        "edges": [...]
    },
    "inputs": {
        "data": "input_value"
    },
    "options": {
        "enable_retry": true,
        "checkpoint": false
    }
}
```

**Response:**
```json
{
    "run_id": "run-456",
    "status": "RUNNING",
    "started_at": "2024-01-01T00:00:00Z",
    "message": "Workflow execution started"
}
```

#### GET /workflows/runs/{run_id}
Get workflow run status.

**Response:**
```json
{
    "run_id": "run-456",
    "flow_id": "api-workflow",
    "status": "COMPLETED",
    "started_at": "2024-01-01T00:00:00Z",
    "completed_at": "2024-01-01T00:05:00Z",
    "outputs": {
        "result": "processed_data"
    },
    "metrics": {
        "execution_time": 300,
        "nodes_executed": 5
    }
}
```

#### POST /workflows/runs/{run_id}/pause
Pause a running workflow.

**Response:**
```json
{
    "success": true,
    "message": "Workflow paused",
    "run_id": "run-456"
}
```

#### POST /workflows/runs/{run_id}/resume
Resume a paused workflow.

**Response:**
```json
{
    "success": true,
    "message": "Workflow resumed",
    "run_id": "run-456"
}
```

#### DELETE /workflows/runs/{run_id}
Cancel a running workflow.

**Response:**
```json
{
    "success": true,
    "message": "Workflow cancelled",
    "run_id": "run-456"
}
```

#### GET /workflows/runs
List all workflow runs.

**Query Parameters:**
- `status`: Filter by status (RUNNING, COMPLETED, FAILED)
- `limit`: Maximum number of results
- `offset`: Pagination offset

**Response:**
```json
{
    "runs": [
        {
            "run_id": "run-456",
            "flow_id": "workflow-1",
            "status": "COMPLETED",
            "started_at": "2024-01-01T00:00:00Z"
        }
    ],
    "total": 100,
    "limit": 10,
    "offset": 0
}
```

## Workflow DSL Reference

### Basic Syntax

```
workflow <workflow_name>:
    node <node_name>:
        type: <node_type>
        [depends_on: <dependency_list>]
        [config:
            <key>: <value>
            ...]
        [retry:
            max_retries: <number>
            strategy: <strategy_name>]

    flow:
        <node1> -> <node2> [-> <node3> ...]
```

### Complete Example

```
workflow ml_training_pipeline:
    # Data loading node
    node data_loader:
        type: data
        config:
            source: "s3://bucket/dataset.csv"
            format: "csv"
            delimiter: ","

    # Data validation node
    node data_validator:
        type: validation
        depends_on: data_loader
        config:
            checks:
                - null_check
                - schema_validation
                - range_check

    # Feature engineering node
    node feature_engineer:
        type: process
        depends_on: data_validator
        config:
            operations:
                - normalize
                - encode_categorical
                - create_features

    # Data splitting node
    node data_splitter:
        type: process
        depends_on: feature_engineer
        config:
            train_ratio: 0.7
            val_ratio: 0.15
            test_ratio: 0.15

    # Model training node
    node model_trainer:
        type: model
        depends_on: data_splitter
        config:
            algorithm: "xgboost"
            hyperparameters:
                n_estimators: 100
                max_depth: 10
                learning_rate: 0.1
        retry:
            max_retries: 3
            strategy: exponential_backoff

    # Model evaluation node
    node model_evaluator:
        type: validation
        depends_on: model_trainer
        config:
            metrics:
                - accuracy
                - precision
                - recall
                - f1_score

    # Conditional deployment
    node deployment_check:
        type: conditional
        depends_on: model_evaluator
        config:
            condition:
                field: "accuracy"
                operator: "greater_than"
                value: 0.9
            true_branch: "deploy_model"
            false_branch: "alert_team"

    node deploy_model:
        type: process
        depends_on: deployment_check
        config:
            target: "production"
            strategy: "blue_green"

    node alert_team:
        type: process
        depends_on: deployment_check
        config:
            notification: "email"
            recipients: ["team@example.com"]

    # Define the execution flow
    flow:
        data_loader -> data_validator -> feature_engineer
        feature_engineer -> data_splitter -> model_trainer
        model_trainer -> model_evaluator -> deployment_check
        deployment_check -> [deploy_model, alert_team]
```

## Node Types

### DataNode
Load and ingest data from various sources.

**Configuration:**
```python
{
    "type": "data",
    "config": {
        "source": "database|csv|api|s3",
        "connection_string": "...",  # For database
        "query": "SELECT ...",        # For database
        "url": "https://...",         # For API
        "method": "GET|POST",         # For API
        "headers": {...},             # For API
        "delimiter": ",",             # For CSV
        "has_header": true            # For CSV
    }
}
```

### ProcessNode
Transform and process data.

**Configuration:**
```python
{
    "type": "process",
    "config": {
        "operations": ["normalize", "scale", "encode"],
        "parameters": {
            "scale_factor": 2.0,
            "encoding": "one_hot"
        }
    }
}
```

### ModelNode
Train or run inference with ML models.

**Configuration:**
```python
{
    "type": "model",
    "config": {
        "mode": "train|predict",
        "algorithm": "xgboost|random_forest|neural_network",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        },
        "model_path": "/path/to/model"  # For loading existing model
    }
}
```

### ValidationNode
Validate data or model outputs.

**Configuration:**
```python
{
    "type": "validation",
    "config": {
        "checks": ["null_check", "schema_validation"],
        "schema": {...},              # JSON schema
        "thresholds": {
            "accuracy": 0.9,
            "loss": 0.1
        }
    }
}
```

### ConditionalNode
Conditional branching based on inputs.

**Configuration:**
```python
{
    "type": "conditional",
    "config": {
        "condition": {
            "type": "and|or",         # For multiple conditions
            "conditions": [
                {
                    "field": "score",
                    "operator": "greater_than|less_than|equals",
                    "value": 0.8
                }
            ]
        },
        "true_branch": "node_id",
        "false_branch": "node_id"
    }
}
```

## Configuration

### Engine Configuration

```python
config = {
    "engine": {
        "max_parallel": 10,
        "enable_versioning": true,
        "enable_optimization": true,
        "checkpoint_interval": 5,  # Checkpoint every 5 nodes
        "checkpoint_dir": "/tmp/checkpoints"
    },
    "retry": {
        "default_strategy": "exponential_backoff",
        "max_global_retries": 10,
        "circuit_breaker": {
            "enabled": true,
            "failure_threshold": 5,
            "recovery_timeout": 60
        }
    },
    "monitoring": {
        "enabled": true,
        "metrics_port": 9090,
        "log_level": "INFO"
    },
    "storage": {
        "backend": "s3|local|gcs",
        "bucket": "workflow-storage",
        "prefix": "workflows/"
    }
}

engine = WorkflowEngine.from_config(config)
```

### Environment Variables

```bash
# Engine configuration
AIWORKFLOW_MAX_PARALLEL=20
AIWORKFLOW_ENABLE_VERSIONING=true
AIWORKFLOW_LOG_LEVEL=DEBUG

# Storage configuration
AIWORKFLOW_STORAGE_BACKEND=s3
AIWORKFLOW_STORAGE_BUCKET=my-bucket
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx

# Monitoring
AIWORKFLOW_METRICS_ENABLED=true
AIWORKFLOW_METRICS_PORT=9090
```

## Error Handling

### Error Types

```python
from aiworkflow.exceptions import (
    WorkflowError,          # Base exception
    ParseError,             # Parsing failures
    ValidationError,        # Validation failures
    ExecutionError,         # Runtime execution errors
    RetryableError,        # Temporary failures
    NonRetryableError,     # Permanent failures
    CircularDependencyError # Circular dependencies detected
)
```

### Error Handling Examples

```python
from aiworkflow import WorkflowEngine
from aiworkflow.exceptions import WorkflowError, RetryableError

engine = WorkflowEngine()

try:
    result = await engine.run_flow(flow_definition)
except ParseError as e:
    print(f"Failed to parse workflow: {e}")
except ValidationError as e:
    print(f"Workflow validation failed: {e}")
    for error in e.errors:
        print(f"  - {error}")
except RetryableError as e:
    print(f"Temporary failure (will retry): {e}")
    print(f"Retry after: {e.retry_after} seconds")
except NonRetryableError as e:
    print(f"Permanent failure: {e}")
    print(f"Reason: {e.reason}")
except WorkflowError as e:
    print(f"Workflow error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Custom Error Handlers

```python
class CustomErrorHandler:
    async def handle_node_error(self, node_id: str, error: Exception):
        if isinstance(error, RetryableError):
            # Log and allow retry
            logger.warning(f"Node {node_id} failed (retryable): {error}")
        else:
            # Alert team for non-retryable errors
            await send_alert(f"Node {node_id} failed: {error}")

engine.set_error_handler(CustomErrorHandler())
```

### Retry Configuration with Error Types

```python
retry_config = RetryConfig(
    max_retries=5,
    retryable_exceptions=[
        ConnectionError,
        TimeoutError,
        HTTPError
    ],
    non_retryable_exceptions=[
        ValueError,
        KeyError,
        PermissionError
    ]
)
```