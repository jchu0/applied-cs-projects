# AI Workflow Engine

A powerful, flexible, and scalable workflow orchestration engine designed specifically for AI/ML pipelines. Build complex data processing and machine learning workflows with ease using YAML, JSON, or our custom DSL.

## Features

- **Multi-Format Support**: Define workflows using YAML, JSON, or our intuitive DSL
- **Parallel Execution**: Automatic detection and execution of parallel branches
- **Fault Tolerance**: Built-in retry mechanisms with exponential backoff
- **Extensible**: Plugin-based architecture for custom node types
- **Version Control**: Complete workflow versioning and history
- **Optimization**: Automatic workflow optimization for performance
- **Monitoring**: Comprehensive metrics and logging
- **Async Native**: Built on Python's asyncio for high performance

## Quick Start

### Installation

```bash
pip install ai-workflow-engine
```

### Basic Usage

```python
import asyncio
from aiworkflow import WorkflowEngine, FlowDefinition, NodeDefinition, NodeType

# Create engine
engine = WorkflowEngine(max_parallel=5)

# Define workflow
flow = FlowDefinition(
    name="ml-pipeline",
    version="1.0.0",
    nodes=[
        NodeDefinition(
            id="loader",
            name="DataLoader",
            type=NodeType.DATA,
            config={"source": "data.csv"}
        ),
        NodeDefinition(
            id="trainer",
            name="ModelTrainer",
            type=NodeType.MODEL,
            dependencies=["loader"],
            config={"algorithm": "random_forest"}
        )
    ],
    edges=[
        {"from": "loader", "to": "trainer"}
    ]
)

# Run workflow
async def main():
    result = await engine.run_flow(flow, inputs={"param": "value"})
    print(f"Status: {result.status}")
    print(f"Outputs: {result.outputs}")

asyncio.run(main())
```

### Using DSL

Create a file `workflow.dsl`:

```dsl
workflow ml_pipeline:
    node data_loader:
        type: data
        config:
            source: "s3://bucket/data.csv"

    node preprocessor:
        type: process
        depends_on: data_loader
        config:
            operations: ["normalize", "encode"]

    node trainer:
        type: model
        depends_on: preprocessor
        config:
            algorithm: "xgboost"
            hyperparameters:
                n_estimators: 100
                max_depth: 10

    flow:
        data_loader -> preprocessor -> trainer
```

Run the workflow:

```python
from aiworkflow import WorkflowEngine
from aiworkflow.compiler import FlowParser

# Parse DSL file
parser = FlowParser()
flow = parser.parse_file("workflow.dsl")

# Execute
engine = WorkflowEngine()
result = asyncio.run(engine.run_flow(flow))
```

## Examples

### 1. Parallel Processing

```python
flow = FlowDefinition(
    name="parallel-processing",
    version="1.0.0",
    nodes=[
        NodeDefinition(id="start", name="Start", type=NodeType.DATA),
        NodeDefinition(id="branch1", name="Process1", type=NodeType.PROCESS,
                      dependencies=["start"]),
        NodeDefinition(id="branch2", name="Process2", type=NodeType.PROCESS,
                      dependencies=["start"]),
        NodeDefinition(id="merge", name="Merge", type=NodeType.PROCESS,
                      dependencies=["branch1", "branch2"])
    ],
    edges=[
        {"from": "start", "to": "branch1"},
        {"from": "start", "to": "branch2"},
        {"from": "branch1", "to": "merge"},
        {"from": "branch2", "to": "merge"}
    ]
)
```

### 2. Conditional Execution

```python
flow = FlowDefinition(
    name="conditional-flow",
    version="1.0.0",
    nodes=[
        NodeDefinition(
            id="evaluate",
            name="Evaluate",
            type=NodeType.CONDITIONAL,
            config={
                "condition": {
                    "field": "accuracy",
                    "operator": "greater_than",
                    "value": 0.9
                },
                "true_branch": "deploy",
                "false_branch": "retrain"
            }
        ),
        NodeDefinition(id="deploy", name="Deploy", type=NodeType.PROCESS,
                      dependencies=["evaluate"]),
        NodeDefinition(id="retrain", name="Retrain", type=NodeType.MODEL,
                      dependencies=["evaluate"])
    ]
)
```

### 3. Custom Node Executor

```python
from aiworkflow.nodes import NodeBase

class CustomDataProcessor(NodeBase):
    async def _execute(self, inputs):
        # Your custom logic here
        data = inputs.get("data", [])
        processed = [x * 2 for x in data]
        return {"processed": processed}

# Register custom executor
engine.node_registry.register("custom_processor", CustomDataProcessor)

# Use in workflow
node = NodeDefinition(
    id="processor",
    name="CustomProcessor",
    type=NodeType.PROCESS,
    executor="custom_processor"
)
```

### 4. Retry Configuration

```python
from aiworkflow.retry import RetryConfig

node = NodeDefinition(
    id="flaky_operation",
    name="FlakyOperation",
    type=NodeType.PROCESS,
    retry_config={
        "max_retries": 5,
        "strategy": "exponential_backoff",
        "base_delay": 1.0,
        "max_delay": 60.0
    }
)
```

## Node Types

### Built-in Node Types

| Type | Description | Use Case |
|------|-------------|----------|
| `DATA` | Data loading and ingestion | Load from CSV, database, API |
| `PROCESS` | Data transformation | Normalize, aggregate, filter |
| `MODEL` | ML model operations | Train, predict, evaluate |
| `VALIDATION` | Data/model validation | Quality checks, thresholds |
| `CONDITIONAL` | Branching logic | Route based on conditions |

### Creating Custom Nodes

```python
from aiworkflow.nodes import NodeBase
from typing import Dict, Any

class MyCustomNode(NodeBase):
    async def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of custom node logic."""
        # Access configuration
        config = self.config

        # Process inputs
        result = await self.process(inputs)

        # Return outputs
        return {"output": result}

    async def process(self, data):
        # Your processing logic
        return processed_data
```

## Configuration

### Engine Configuration

```python
engine = WorkflowEngine(
    max_parallel=10,          # Max parallel node executions
    enable_versioning=True,   # Enable workflow versioning
    enable_optimization=True, # Automatic optimization
    checkpoint_interval=5,    # Checkpoint every 5 nodes
    checkpoint_dir="/tmp/checkpoints"
)
```

### Environment Variables

```bash
export AIWORKFLOW_MAX_PARALLEL=20
export AIWORKFLOW_LOG_LEVEL=DEBUG
export AIWORKFLOW_STORAGE_BACKEND=s3
export AIWORKFLOW_STORAGE_BUCKET=workflows
```

### Configuration File

```yaml
# config.yaml
engine:
  max_parallel: 10
  enable_versioning: true
  enable_optimization: true

storage:
  backend: s3
  bucket: workflow-storage
  region: us-west-2

monitoring:
  enabled: true
  metrics_port: 9090
  log_level: INFO
```

Load configuration:

```python
engine = WorkflowEngine.from_config("config.yaml")
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aiworkflow --cov-report=html

# Run specific test
pytest tests/test_engine.py -v

# Run integration tests
pytest tests/test_integration.py
```

## Architecture

The AI Workflow Engine consists of several key components:

- **Engine**: Main orchestrator managing workflow lifecycle
- **Compiler**: Parses and validates workflow definitions
- **Scheduler**: Manages node execution order and parallelization
- **Executor**: Runs individual nodes with retry logic
- **Registry**: Plugin system for custom node types

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## API Documentation

Comprehensive API documentation is available in [docs/API.md](docs/API.md).

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment instructions including:

- Docker deployment
- Kubernetes deployment
- AWS/GCP/Azure deployment
- Production configuration
- Monitoring setup

## Contributing

We welcome contributions! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for:

- Development setup
- Coding standards
- Testing requirements
- Pull request process

## Performance

Benchmark results on standard hardware (8 CPU cores, 16GB RAM):

| Metric | Value |
|--------|-------|
| Max parallel nodes | 100+ |
| Workflow parse time | <100ms |
| Node scheduling overhead | <10ms |
| Typical node execution | 50-500ms |
| Large workflow (100 nodes) | <5s |

## Roadmap

### Version 2.0 (Q2 2024)
- [ ] Distributed execution across multiple machines
- [ ] Real-time workflow modification
- [ ] GraphQL API
- [ ] Web UI for workflow design

### Version 2.1 (Q3 2024)
- [ ] Kubernetes operator
- [ ] Apache Airflow compatibility
- [ ] MLflow integration
- [ ] Advanced scheduling algorithms

### Version 3.0 (Q4 2024)
- [ ] GPU-aware scheduling
- [ ] Federated learning support
- [ ] Auto-ML integration
- [ ] Natural language workflow definition

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/ai-workflow-engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ai-workflow-engine/discussions)
- **Email**: support@aiworkflow.example.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python asyncio for high-performance async execution
- Inspired by Apache Airflow, Prefect, and Dagster
- Uses NetworkX for DAG operations
- Monitoring powered by Prometheus

## Citation

If you use AI Workflow Engine in your research, please cite:

```bibtex
@software{ai_workflow_engine,
  title = {AI Workflow Engine: A Flexible Orchestration System for ML Pipelines},
  author = {Your Organization},
  year = {2024},
  url = {https://github.com/your-org/ai-workflow-engine}
}
```

---

Made with ❤️ by the AI Workflow Engine Team