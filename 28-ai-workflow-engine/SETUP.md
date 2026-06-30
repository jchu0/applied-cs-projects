# Project 28: AI Workflow Engine - Setup Guide

## Overview
Workflow DSL and execution engine for complex AI pipelines with DAG-based orchestration, versioning, and distributed execution.

## Prerequisites
- Python 3.9+
- Redis or PostgreSQL (for state management)
- 8GB+ RAM
- Optional: Ray or Dask for distributed execution

## Installation

### 1. System Dependencies

**Redis (Recommended)**
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
redis-server
```

**PostgreSQL (Optional)**
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql
```

### 2. Python Environment
```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Database Setup
```bash
# Initialize database (SQLite by default)
python -m aiworkflow.db init

# Or with PostgreSQL
export DATABASE_URL="postgresql://user:pass@localhost/aiworkflow"
python -m aiworkflow.db init
```

## Configuration

### 1. Environment Variables
```bash
# .env file
# Database
DATABASE_URL=sqlite:///./workflow.db
# DATABASE_URL=postgresql://user:pass@localhost/aiworkflow

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Execution
MAX_PARALLEL_NODES=10
ENABLE_VERSIONING=true
ENABLE_OPTIMIZATION=true

# Distributed Execution (optional)
USE_RAY=false
RAY_ADDRESS=auto
USE_DASK=false
DASK_SCHEDULER=localhost:8786

# LLM Providers (for LLM nodes)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Monitoring
PROMETHEUS_PORT=9090
ENABLE_TELEMETRY=true
```

### 2. Flow Storage Directory
```bash
mkdir -p flows
mkdir -p logs
mkdir -p data
```

## Usage

### Basic Workflow

**1. Define Workflow (YAML)**
```yaml
# flows/qa_workflow.yaml
name: qa_workflow
version: "1.0.0"
description: Q&A workflow with retrieval and generation

nodes:
  - id: retriever
    type: retrieval
    inputs:
      query: "{{inputs.question}}"
    config:
      top_k: 5
      collection: "documents"

  - id: reranker
    type: rerank
    dependencies:
      - retriever
    inputs:
      query: "{{inputs.question}}"
      documents: "{{retriever.result}}"
    config:
      top_k: 3

  - id: generator
    type: llm
    dependencies:
      - reranker
    inputs:
      context: "{{reranker.result}}"
      question: "{{inputs.question}}"
    config:
      model: gpt-4
      temperature: 0.7
      max_tokens: 500
      prompt_template: |
        Context: {{context}}
        Question: {{question}}
        Answer:

outputs:
  answer: generator.result
```

**2. Execute Workflow (Python)**
```python
from aiworkflow import create_engine

# Create engine
engine = create_engine(
    max_parallel=10,
    enable_versioning=True
)

# Register flow
with open("flows/qa_workflow.yaml") as f:
    flow_spec = f.read()

flow = engine.register_flow(
    flow_spec,
    description="Q&A workflow"
)

# Execute
result = await engine.execute(
    flow_name="qa_workflow",
    inputs={"question": "What is machine learning?"}
)

print(f"Status: {result.status}")
print(f"Answer: {result.outputs['answer']}")
```

### Advanced Workflow Features

**1. Conditional Execution**
```yaml
nodes:
  - id: classifier
    type: llm
    inputs:
      text: "{{inputs.query}}"
    config:
      prompt: "Classify as: question|command|statement"

  - id: qa_handler
    type: llm
    dependencies:
      - classifier
    condition: "{{classifier.result}} == 'question'"
    inputs:
      query: "{{inputs.query}}"

  - id: command_handler
    type: custom
    dependencies:
      - classifier
    condition: "{{classifier.result}} == 'command'"
    inputs:
      command: "{{inputs.query}}"
```

**2. Parallel Execution**
```yaml
nodes:
  - id: retrieve_docs
    type: retrieval
    inputs:
      query: "{{inputs.query}}"

  - id: retrieve_web
    type: web_search
    inputs:
      query: "{{inputs.query}}"

  - id: combine
    type: merge
    dependencies:
      - retrieve_docs
      - retrieve_web
    inputs:
      sources:
        - "{{retrieve_docs.result}}"
        - "{{retrieve_web.result}}"
```

**3. Retry & Error Handling**
```yaml
nodes:
  - id: api_call
    type: llm
    inputs:
      prompt: "{{inputs.prompt}}"
    config:
      retry_strategy: exponential_backoff
      max_retries: 3
      timeout_seconds: 30
    on_error:
      action: fallback
      fallback_value: "Error occurred, using default response"
```

### Custom Node Executors

```python
from aiworkflow.nodes import BaseNodeExecutor

class CustomNodeExecutor(BaseNodeExecutor):
    """Custom node executor."""

    async def execute(self, inputs: dict, config: dict) -> dict:
        # Your custom logic
        result = await self.process(inputs, config)
        return {"result": result}

    async def process(self, inputs, config):
        # Implementation
        return "processed result"

# Register custom executor
engine.register_node_executor("custom", CustomNodeExecutor())
```

### Batch Execution
```python
# Execute workflow for multiple inputs
inputs_list = [
    {"question": "What is ML?"},
    {"question": "What is DL?"},
    {"question": "What is NLP?"}
]

results = await engine.execute_batch(
    flow_name="qa_workflow",
    inputs_list=inputs_list
)

for result in results:
    print(f"Answer: {result.outputs['answer']}")
```

### Workflow Versioning
```python
from aiworkflow import FlowVersionManager

manager = FlowVersionManager()

# Get specific version
flow_v1 = engine.get_flow("qa_workflow", version="1.0.0")
flow_v2 = engine.get_flow("qa_workflow", version="2.0.0")

# List versions
versions = manager.list_versions("qa_workflow")
```

### Monitoring & Observability

**1. Execution History**
```python
# Get run history
history = engine.get_run_history(
    flow_name="qa_workflow",
    limit=100
)

for run in history:
    print(f"Run {run.run_id}: {run.status} ({run.duration_ms}ms)")
```

**2. Flow Metrics**
```python
# Get specific run
run = engine.get_run("run-id-123")

print(f"Status: {run.status}")
print(f"Duration: {run.duration_ms}ms")
print(f"Nodes executed: {len(run.node_results)}")

if run.error:
    print(f"Error: {run.error}")
```

**3. Prometheus Integration**
```python
from prometheus_client import start_http_server

# Start metrics server
start_http_server(9090)

# Metrics available at http://localhost:9090/metrics
```

## Distributed Execution

### Using Ray
```python
import ray

# Initialize Ray
ray.init(address="auto")

# Engine will use Ray for distributed execution
engine = create_engine(
    max_parallel=100,  # Can handle more parallel nodes
    use_ray=True
)
```

### Using Dask
```python
from dask.distributed import Client

# Connect to Dask cluster
client = Client("localhost:8786")

engine = create_engine(
    max_parallel=100,
    use_dask=True
)
```

## Flow Optimization

```python
from aiworkflow import FlowOptimizer

optimizer = FlowOptimizer()

# Optimize flow
optimized_flow = optimizer.optimize(flow)

# Optimizations include:
# - Parallel execution of independent nodes
# - Caching of repeated computations
# - Dead code elimination
```

## Testing

```bash
# Run all tests
pytest

# Test specific workflow
pytest tests/test_engine.py::test_qa_workflow

# Test with fixtures
pytest tests/test_integration.py -v
```

## Common Issues

### Issue: Redis connection failed
**Solution**: Ensure Redis is running
```bash
redis-cli ping  # Should return PONG
```

### Issue: Node execution timeout
**Solution**: Increase timeout in node config
```yaml
config:
  timeout_seconds: 60  # Increase from default 30
```

### Issue: Memory leak with large workflows
**Solution**: Enable result cleanup
```python
engine = create_engine(
    cleanup_results=True,  # Clean up intermediate results
    max_result_size_mb=100
)
```

## Project Structure
```
28-ai-workflow-engine/
├── src/aiworkflow/
│   ├── engine.py           # Main engine
│   ├── compiler/          # Flow parsing & compilation
│   ├── executor/          # Execution scheduling
│   ├── nodes/             # Node executors
│   ├── retry/             # Retry strategies
│   ├── versioning/        # Flow versioning
│   └── ...
├── flows/                 # Flow definitions
├── tests/
├── requirements.txt
└── SETUP.md
```

## Next Steps
1. Define your workflows in YAML
2. Register custom node executors
3. Set up distributed execution
4. Enable monitoring
5. Deploy flows to production

## Resources
- [YAML Specification](https://yaml.org/spec/)
- [Ray Documentation](https://docs.ray.io/)
- [Dask Documentation](https://docs.dask.org/)
- [SQLAlchemy](https://docs.sqlalchemy.org/)
