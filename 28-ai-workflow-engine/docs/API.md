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
    flow_definition: FlowDefinition = None,
    inputs: dict[str, Any] = None,
    enable_retry: bool = False,
    run_id: str = None
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

# Node.retry_config takes a plain dict
node = Node(
    id="flaky_node",
    name="FlakyOperation",
    type=NodeType.PROCESS,
    retry_config={
        "max_retries": 5,
        "base_delay": 1.0,
        "max_delay": 60.0,
        "exponential_factor": 2.0,
        "jitter": True,
    },
)
```

## REST API

The REST API is implemented in `aiworkflow.api.app` (FastAPI, installed via the
`api` extra). Build an app with `create_app()` or serve the module-level app:

```bash
uvicorn aiworkflow.api:app
```

### Endpoints

#### GET /health
Health check.

**Response:**
```json
{
    "status": "ok",
    "flows": 2
}
```

#### POST /flows
Register a flow definition. Returns `201` on success, `400` on parse/validation
errors.

**Request:**
```json
{
    "spec": {
        "name": "api-workflow",
        "version": "1.0.0",
        "nodes": [...],
        "edges": [...]
    },
    "description": "Optional description"
}
```

**Response:**
```json
{
    "name": "api-workflow",
    "version": "1.0.0",
    "nodes": 3
}
```

#### GET /flows
List registered flows.

**Response:**
```json
[
    {"name": "api-workflow", "version": "1.0.0", "nodes": 3}
]
```

#### GET /flows/{name}
Get a registered flow's definition (`404` if unknown).

**Response:**
```json
{
    "name": "api-workflow",
    "version": "1.0.0",
    "description": "...",
    "nodes": [
        {"id": "loader", "type": "data", "dependencies": []}
    ],
    "outputs": {}
}
```

#### GET /flows/{name}/diagram
Render a flow as a Mermaid or Graphviz DOT diagram.

**Query Parameters:**
- `fmt`: `mermaid` (default) or `dot`

**Response:**
```json
{
    "format": "mermaid",
    "diagram": "graph TD; ..."
}
```

#### POST /flows/{name}/run
Execute a registered flow and return the completed run (`404` if unknown).

**Request:**
```json
{
    "inputs": {"data": "input_value"}
}
```

**Response:**
```json
{
    "run_id": "run-456",
    "flow_id": "api-workflow",
    "flow_version": "1.0.0",
    "status": "completed",
    "inputs": {"data": "input_value"},
    "outputs": {"result": "processed_data"},
    "error": null,
    "start_time": "2024-01-01T00:00:00",
    "end_time": "2024-01-01T00:05:00",
    "node_executions": [
        {"node_id": "loader", "status": "completed", "latency_ms": 12.5, "attempts": 1, "error": null}
    ]
}
```

#### POST /runs
Execute an inline (unregistered) flow spec. Returns the same run shape as
`POST /flows/{name}/run`; `400` on parse errors.

**Request:**
```json
{
    "spec": {
        "name": "inline-flow",
        "nodes": [...],
        "edges": [...]
    },
    "inputs": {"data": "input_value"}
}
```

#### GET /runs
List run history.

**Query Parameters:**
- `flow_name`: Filter by flow name
- `limit`: Maximum number of results (default 100)

#### GET /runs/{run_id}
Get a single run by id (`404` if unknown).

#### GET /reviews
List pending human-in-the-loop reviews.

#### GET /reviews/{review_id}
Get a single review (`404` if unknown).

#### POST /reviews/{review_id}/approve
Approve a pending review (`404` unknown, `409` already resolved).

**Request:**
```json
{
    "reviewer": "alice",
    "comment": "Looks good",
    "data": {}
}
```

#### POST /reviews/{review_id}/reject
Reject a pending review. Same request/response shape as approve.

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

Configuration is passed directly to the `WorkflowEngine` constructor:

```python
from aiworkflow import WorkflowEngine

engine = WorkflowEngine(
    max_parallel=10,            # Maximum parallel node executions
    enable_versioning=True,     # Save flow versions on each run
    enable_optimization=True,   # Optimize flows before execution
    enable_checkpointing=False, # Save checkpoints after runs
    checkpoint_dir=None,        # Directory for checkpoint files
    review_store=None,          # Shared HumanReviewStore for HITL nodes
)
```

A convenience factory is also exported:

```python
from aiworkflow import create_engine

engine = create_engine(max_parallel=10, enable_versioning=True, enable_optimization=True)
```

## Error Handling

### Error Types

```python
from aiworkflow.compiler.parser import ParseError       # Parsing failures
from aiworkflow.compiler import CircularDependencyError # Circular dependencies detected
from aiworkflow.retry.strategies import (
    RetryableError,     # Temporary failures (carries retry_after, attempt)
    NonRetryableError,  # Permanent failures (carries reason)
)
```

Flow validation failures are raised as plain `ValueError` from
`WorkflowEngine.run_flow()`; node-level failures during execution do not raise —
they are recorded on the returned `FlowRun` (`status`, `error`, and per-node
`node_executions`).

### Error Handling Examples

```python
from aiworkflow import WorkflowEngine, RunStatus
from aiworkflow.compiler.parser import ParseError
from aiworkflow.compiler import CircularDependencyError

engine = WorkflowEngine()

try:
    flow_definition = engine.parser.parse_yaml(yaml_content)
    result = await engine.run_flow(flow_definition)
except ParseError as e:
    print(f"Failed to parse workflow: {e}")
except CircularDependencyError as e:
    print(f"Circular dependency detected: {e}")
except ValueError as e:
    print(f"Workflow validation failed: {e}")

if result.status == RunStatus.FAILED:
    print(f"Run failed: {result.error}")
    for exec in result.node_executions:
        if exec.error:
            print(f"  - {exec.node_id}: {exec.error}")
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