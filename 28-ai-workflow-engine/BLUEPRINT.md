# Project 28: AI Workflow Engine / DSL (Dagster-for-LLMs)

## Executive Summary

A comprehensive workflow orchestration engine designed specifically for LLM-powered applications. Features a declarative DSL for defining AI workflows, intelligent retry semantics for non-deterministic LLM nodes, topological scheduling, flow versioning with diff capabilities, and complete execution lineage tracking. Think Dagster/Airflow but purpose-built for the unique challenges of AI pipelines.

> **Concepts covered:** [§04 LLM agents](../../04-ai-engineering/02-llm-applications/agents/llm-agents.md) (workflow orchestration of agent steps) · [§04 RAG systems](../../04-ai-engineering/02-llm-applications/rag/rag-systems.md) (a common workflow type) · [§02 Data pipelines / orchestration](../../02-data-engineering/01-data-pipelines/) (the Dagster/Airflow lineage). Pairs with [Project 23 (agent runtime — what runs inside a workflow node)](../23-llm-agentic-runtime/) and [Project 29 (routing — what serves the LLM calls)](../29-model-routing-layer/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Flow DSL Design](#flow-dsl-design)
3. [Node Types](#node-types)
4. [Execution Engine](#execution-engine)
5. [Retry Semantics](#retry-semantics)
6. [Versioning & Lineage](#versioning--lineage)
7. [Enterprise Features](#enterprise-features)
8. [Implementation Phases](#implementation-phases)
9. [Stretch Goals](#stretch-goals)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI Workflow Engine                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Flow Definition Layer                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │    YAML     │  │    JSON     │  │   Python    │  │   Visual    │  │  │
│  │  │    DSL      │  │    DSL      │  │    SDK      │  │   Editor    │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                       │                                      │
│                                       ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Flow Compiler                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Parser    │  │  Validator  │  │  Optimizer  │  │    DAG      │  │  │
│  │  │             │  │             │  │             │  │   Builder   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                       │                                      │
│                                       ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Execution Engine                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │  Scheduler  │  │   Retry     │  │   State     │  │   Event     │  │  │
│  │  │ (Topo Sort) │  │   Manager   │  │   Machine   │  │   Emitter   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                       │                                      │
│                                       ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Node Executors                                 │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │     LLM     │  │  Retrieval  │  │    Tool     │  │   Branch    │  │  │
│  │  │   Executor  │  │   Executor  │  │  Executor   │  │  Executor   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                       │                                      │
│                                       ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Metadata & Lineage Store                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Flow      │  │    Run      │  │   Asset     │  │   Audit     │  │  │
│  │  │  Versions   │  │   History   │  │   Catalog   │  │    Log      │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Execution Flow

```
Flow Definition (YAML/JSON/Python)
            │
            ▼
    ┌───────────────┐
    │    Parser     │ ─── Syntax validation
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   Validator   │ ─── Semantic validation
    └───────┬───────┘     Type checking
            │
            ▼
    ┌───────────────┐
    │   Optimizer   │ ─── Parallel opportunity detection
    └───────┬───────┘     Dead code elimination
            │
            ▼
    ┌───────────────┐
    │  DAG Builder  │ ─── Topological sort
    └───────┬───────┘     Dependency resolution
            │
            ▼
    ┌───────────────┐
    │   Scheduler   │ ─── Task queuing
    └───────┬───────┘     Resource allocation
            │
            ▼
    ┌───────────────┐
    │   Executors   │ ─── Node execution
    └───────┬───────┘     Retry handling
            │
            ▼
    ┌───────────────┐
    │ Lineage Store │ ─── Results & metadata
    └───────────────┘
```

---

## Flow DSL Design

### YAML DSL Specification

```yaml
# flow_definition.yaml
version: "1.0"
name: "customer-support-flow"
description: "AI-powered customer support workflow"

# Global configuration
config:
  timeout_seconds: 300
  max_retries: 3
  retry_strategy: "exponential_backoff"

# Input schema
inputs:
  customer_query:
    type: string
    required: true
  customer_id:
    type: string
    required: true
  context:
    type: object
    required: false

# Output schema
outputs:
  response:
    type: string
  confidence:
    type: float
  sources:
    type: array

# Node definitions
nodes:
  # LLM node for intent classification
  - id: classify_intent
    type: llm
    config:
      model: "gpt-4"
      temperature: 0.1
      max_tokens: 100
      prompt_template: |
        Classify the customer intent:
        Query: {{inputs.customer_query}}

        Categories: [billing, technical, general, complaint]
        Output JSON: {"intent": "...", "confidence": 0.0}
    inputs:
      query: "{{inputs.customer_query}}"
    outputs:
      intent: "$.intent"
      confidence: "$.confidence"
    retry:
      max_attempts: 3
      strategy: "exponential"
      base_delay_ms: 1000

  # Branch based on intent
  - id: route_by_intent
    type: branch
    config:
      condition: "{{classify_intent.intent}}"
      branches:
        billing: handle_billing
        technical: handle_technical
        general: handle_general
        complaint: handle_complaint
      default: handle_general
    inputs:
      intent: "{{classify_intent.intent}}"

  # Retrieval node
  - id: retrieve_docs
    type: retrieval
    config:
      index: "knowledge_base"
      top_k: 5
      filters:
        category: "{{classify_intent.intent}}"
    inputs:
      query: "{{inputs.customer_query}}"
    outputs:
      documents: "$"

  # LLM node for response generation
  - id: generate_response
    type: llm
    config:
      model: "gpt-4"
      temperature: 0.7
      max_tokens: 500
      prompt_template: |
        Generate a helpful response for this customer query.

        Query: {{inputs.customer_query}}
        Context: {{retrieve_docs.documents}}
        Customer ID: {{inputs.customer_id}}

        Be professional, accurate, and cite sources.
    inputs:
      query: "{{inputs.customer_query}}"
      context: "{{retrieve_docs.documents}}"
    outputs:
      response: "$.response"
      citations: "$.citations"
    dependencies:
      - retrieve_docs

  # Subflow for billing
  - id: handle_billing
    type: subflow
    config:
      flow_id: "billing-support-flow"
      version: "latest"
    inputs:
      query: "{{inputs.customer_query}}"
      customer_id: "{{inputs.customer_id}}"

  # Human-in-the-loop for complaints
  - id: handle_complaint
    type: human_review
    config:
      queue: "complaint_review"
      timeout_hours: 24
      escalation:
        enabled: true
        after_hours: 4
    inputs:
      query: "{{inputs.customer_query}}"
      customer_id: "{{inputs.customer_id}}"
      context: "{{retrieve_docs.documents}}"

  # Final aggregation
  - id: aggregate_response
    type: transform
    config:
      expression: |
        {
          "response": inputs.response,
          "confidence": inputs.confidence,
          "sources": inputs.citations
        }
    inputs:
      response: "{{generate_response.response}}"
      confidence: "{{classify_intent.confidence}}"
      citations: "{{generate_response.citations}}"
    outputs:
      final: "$"

# Flow edges (explicit dependencies)
edges:
  - from: classify_intent
    to: route_by_intent
  - from: route_by_intent
    to: retrieve_docs
  - from: retrieve_docs
    to: generate_response
  - from: generate_response
    to: aggregate_response
```

### JSON DSL

```json
{
  "version": "1.0",
  "name": "data-processing-flow",
  "nodes": [
    {
      "id": "extract",
      "type": "llm",
      "config": {
        "model": "gpt-4",
        "prompt_template": "Extract entities from: {{input}}"
      }
    },
    {
      "id": "transform",
      "type": "tool",
      "config": {
        "tool_name": "entity_normalizer"
      },
      "dependencies": ["extract"]
    },
    {
      "id": "load",
      "type": "tool",
      "config": {
        "tool_name": "database_insert"
      },
      "dependencies": ["transform"]
    }
  ]
}
```

### Python SDK

```python
from workflow_engine import Flow, LLMNode, RetrievalNode, BranchNode, ToolNode

# Define flow using Python SDK
flow = Flow(
    name="document-qa-flow",
    version="1.0.0"
)

# Add nodes
@flow.node(type="llm")
def classify_query(query: str) -> dict:
    return LLMNode(
        model="gpt-4",
        temperature=0.1,
        prompt=f"Classify query type: {query}"
    )

@flow.node(type="retrieval", depends_on=["classify_query"])
def retrieve_context(query: str, classification: dict) -> list:
    return RetrievalNode(
        index="documents",
        top_k=10,
        filters={"type": classification["type"]}
    )

@flow.node(type="llm", depends_on=["retrieve_context"])
def generate_answer(query: str, context: list) -> str:
    return LLMNode(
        model="gpt-4",
        temperature=0.7,
        prompt=f"Answer based on context:\nQuery: {query}\nContext: {context}"
    )

@flow.node(type="branch", depends_on=["classify_query"])
def route_complex(classification: dict):
    return BranchNode(
        condition=classification["complexity"],
        branches={
            "simple": "generate_answer",
            "complex": "expert_review"
        }
    )

# Compile and register
compiled_flow = flow.compile()
flow_registry.register(compiled_flow)
```

---

## Node Types

### LLM Node

```python
@dataclass
class LLMNodeConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 500
    prompt_template: str = ""
    system_prompt: Optional[str] = None
    response_format: Optional[str] = None  # json, text
    stop_sequences: List[str] = field(default_factory=list)
    retry_config: RetryConfig = field(default_factory=RetryConfig)

class LLMNodeExecutor(BaseNodeExecutor):
    """Executes LLM nodes with proper retry semantics."""

    def __init__(self, llm_clients: Dict[str, LLMClient]):
        self.clients = llm_clients

    async def execute(
        self,
        node: Node,
        inputs: Dict[str, Any],
        context: ExecutionContext
    ) -> NodeResult:
        config = LLMNodeConfig(**node.config)

        # Render prompt template
        prompt = self._render_template(config.prompt_template, inputs)

        # Get appropriate client
        client = self.clients[config.model]

        # Execute with retry
        response = await self._execute_with_retry(
            client,
            prompt,
            config,
            context
        )

        # Parse output
        output = self._parse_response(response, config.response_format)

        return NodeResult(
            node_id=node.id,
            output=output,
            metadata={
                "model": config.model,
                "tokens_used": response.usage.total_tokens,
                "latency_ms": response.latency_ms
            }
        )

    async def _execute_with_retry(
        self,
        client: LLMClient,
        prompt: str,
        config: LLMNodeConfig,
        context: ExecutionContext
    ) -> LLMResponse:
        retry_config = config.retry_config

        for attempt in range(retry_config.max_attempts):
            try:
                response = await client.generate(
                    prompt=prompt,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    system_prompt=config.system_prompt
                )

                # Validate response
                if config.response_format == "json":
                    self._validate_json(response.text)

                return response

            except (RateLimitError, TimeoutError) as e:
                if attempt < retry_config.max_attempts - 1:
                    delay = self._compute_delay(attempt, retry_config)
                    await asyncio.sleep(delay)
                    continue
                raise

            except ValidationError as e:
                # For validation errors, retry with guidance
                if attempt < retry_config.max_attempts - 1:
                    prompt = self._add_correction_guidance(prompt, e)
                    continue
                raise
```

### Retrieval Node

```python
@dataclass
class RetrievalNodeConfig:
    index: str
    top_k: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    rerank: bool = False
    rerank_model: Optional[str] = None

class RetrievalNodeExecutor(BaseNodeExecutor):
    """Executes retrieval nodes."""

    def __init__(self, retrieval_services: Dict[str, RetrievalService]):
        self.services = retrieval_services

    async def execute(
        self,
        node: Node,
        inputs: Dict[str, Any],
        context: ExecutionContext
    ) -> NodeResult:
        config = RetrievalNodeConfig(**node.config)
        service = self.services[config.index]

        # Execute retrieval
        results = await service.retrieve(
            query=inputs["query"],
            top_k=config.top_k,
            filters=config.filters
        )

        # Optional reranking
        if config.rerank:
            results = await self._rerank(results, inputs["query"], config)

        return NodeResult(
            node_id=node.id,
            output=results,
            metadata={
                "num_results": len(results),
                "index": config.index
            }
        )
```

### Tool Node

```python
@dataclass
class ToolNodeConfig:
    tool_name: str
    tool_config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30

class ToolNodeExecutor(BaseNodeExecutor):
    """Executes tool/function nodes."""

    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry

    async def execute(
        self,
        node: Node,
        inputs: Dict[str, Any],
        context: ExecutionContext
    ) -> NodeResult:
        config = ToolNodeConfig(**node.config)
        tool = self.registry.get(config.tool_name)

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                tool.execute(inputs, config.tool_config),
                timeout=config.timeout_seconds
            )

            return NodeResult(
                node_id=node.id,
                output=result,
                metadata={"tool": config.tool_name}
            )

        except asyncio.TimeoutError:
            raise ToolTimeoutError(f"Tool {config.tool_name} timed out")
```

### Branch Node

```python
@dataclass
class BranchNodeConfig:
    condition: str  # Expression to evaluate
    branches: Dict[str, str]  # value -> target node
    default: Optional[str] = None

class BranchNodeExecutor(BaseNodeExecutor):
    """Executes conditional branching."""

    async def execute(
        self,
        node: Node,
        inputs: Dict[str, Any],
        context: ExecutionContext
    ) -> NodeResult:
        config = BranchNodeConfig(**node.config)

        # Evaluate condition
        condition_value = self._evaluate_expression(config.condition, inputs)

        # Determine target
        target = config.branches.get(condition_value, config.default)

        if target is None:
            raise BranchError(f"No branch for value: {condition_value}")

        return NodeResult(
            node_id=node.id,
            output={"target": target, "condition_value": condition_value},
            metadata={"branched_to": target}
        )
```

### Subflow Node

```python
@dataclass
class SubflowNodeConfig:
    flow_id: str
    version: str = "latest"
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)

class SubflowNodeExecutor(BaseNodeExecutor):
    """Executes nested subflows."""

    def __init__(self, flow_registry: FlowRegistry, executor: FlowExecutor):
        self.registry = flow_registry
        self.executor = executor

    async def execute(
        self,
        node: Node,
        inputs: Dict[str, Any],
        context: ExecutionContext
    ) -> NodeResult:
        config = SubflowNodeConfig(**node.config)

        # Load subflow
        subflow = self.registry.get(config.flow_id, config.version)

        # Map inputs
        subflow_inputs = self._map_inputs(inputs, config.input_mapping)

        # Execute subflow
        result = await self.executor.execute(
            subflow,
            subflow_inputs,
            parent_context=context
        )

        # Map outputs
        output = self._map_outputs(result, config.output_mapping)

        return NodeResult(
            node_id=node.id,
            output=output,
            metadata={
                "subflow_id": config.flow_id,
                "subflow_run_id": result.run_id
            }
        )
```

---

## Execution Engine

### Topological Scheduler

```python
class TopologicalScheduler:
    """Schedules node execution in topological order."""

    def __init__(self, max_parallel: int = 10):
        self.max_parallel = max_parallel

    def schedule(self, dag: DAG) -> List[List[str]]:
        """Return execution levels (nodes in same level can run in parallel)."""
        in_degree = {node: 0 for node in dag.nodes}
        for node in dag.nodes:
            for dep in dag.get_dependencies(node):
                in_degree[node] += 1

        levels = []
        remaining = set(dag.nodes)

        while remaining:
            # Find nodes with no remaining dependencies
            level = [
                node for node in remaining
                if in_degree[node] == 0
            ]

            if not level:
                raise CycleDetectedError("DAG contains cycle")

            levels.append(level)

            # Update in-degrees
            for node in level:
                remaining.remove(node)
                for dependent in dag.get_dependents(node):
                    in_degree[dependent] -= 1

        return levels

class FlowExecutor:
    """Executes flows with scheduling and retry."""

    def __init__(
        self,
        scheduler: TopologicalScheduler,
        executors: Dict[str, BaseNodeExecutor],
        state_store: StateStore,
        event_emitter: EventEmitter
    ):
        self.scheduler = scheduler
        self.executors = executors
        self.state_store = state_store
        self.events = event_emitter

    async def execute(
        self,
        flow: CompiledFlow,
        inputs: Dict[str, Any],
        parent_context: Optional[ExecutionContext] = None
    ) -> FlowResult:
        # Create execution context
        context = ExecutionContext(
            run_id=generate_run_id(),
            flow_id=flow.id,
            flow_version=flow.version,
            parent_context=parent_context
        )

        # Initialize state
        await self.state_store.initialize(context.run_id, inputs)

        # Get execution schedule
        levels = self.scheduler.schedule(flow.dag)

        try:
            # Execute level by level
            for level_idx, level in enumerate(levels):
                await self._execute_level(flow, level, context)

            # Get final outputs
            outputs = await self.state_store.get_outputs(context.run_id)

            return FlowResult(
                run_id=context.run_id,
                status="success",
                outputs=outputs
            )

        except Exception as e:
            await self.state_store.set_status(context.run_id, "failed", str(e))
            raise

    async def _execute_level(
        self,
        flow: CompiledFlow,
        level: List[str],
        context: ExecutionContext
    ):
        """Execute all nodes in a level (possibly in parallel)."""
        tasks = []
        for node_id in level:
            node = flow.get_node(node_id)
            task = self._execute_node(node, context)
            tasks.append(task)

        # Execute in parallel with limit
        semaphore = asyncio.Semaphore(self.scheduler.max_parallel)

        async def limited_task(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(
            *[limited_task(t) for t in tasks],
            return_exceptions=True
        )

        # Check for errors
        for node_id, result in zip(level, results):
            if isinstance(result, Exception):
                raise NodeExecutionError(node_id, result)

    async def _execute_node(
        self,
        node: Node,
        context: ExecutionContext
    ) -> NodeResult:
        """Execute a single node."""
        # Emit start event
        self.events.emit(NodeStarted(node.id, context.run_id))

        # Get inputs from state
        inputs = await self.state_store.get_node_inputs(
            context.run_id,
            node.id,
            node.input_mappings
        )

        # Get executor
        executor = self.executors[node.type]

        try:
            # Execute
            result = await executor.execute(node, inputs, context)

            # Store outputs
            await self.state_store.set_node_outputs(
                context.run_id,
                node.id,
                result.output
            )

            # Emit success event
            self.events.emit(NodeCompleted(node.id, context.run_id, result))

            return result

        except Exception as e:
            self.events.emit(NodeFailed(node.id, context.run_id, e))
            raise
```

### State Machine

```python
class NodeState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class FlowStateMachine:
    """Manages state transitions for flow execution."""

    def __init__(self, state_store: StateStore):
        self.store = state_store

        # Valid transitions
        self.transitions = {
            NodeState.PENDING: [NodeState.RUNNING, NodeState.SKIPPED],
            NodeState.RUNNING: [NodeState.COMPLETED, NodeState.FAILED, NodeState.RETRYING],
            NodeState.RETRYING: [NodeState.RUNNING],
            NodeState.FAILED: [NodeState.RETRYING],  # Manual retry
            NodeState.COMPLETED: [],
            NodeState.SKIPPED: []
        }

    async def transition(
        self,
        run_id: str,
        node_id: str,
        new_state: NodeState,
        metadata: dict = None
    ):
        current_state = await self.store.get_node_state(run_id, node_id)

        if new_state not in self.transitions[current_state]:
            raise InvalidTransitionError(
                f"Cannot transition from {current_state} to {new_state}"
            )

        await self.store.set_node_state(
            run_id,
            node_id,
            new_state,
            metadata
        )
```

---

## Retry Semantics

### LLM-Specific Retry Strategies

```python
class RetryStrategy(ABC):
    @abstractmethod
    def should_retry(self, error: Exception, attempt: int) -> bool:
        pass

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        pass

class ExponentialBackoffRetry(RetryStrategy):
    """Exponential backoff for rate limits."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False

        retryable_errors = (RateLimitError, TimeoutError, ServiceUnavailableError)
        return isinstance(error, retryable_errors)

    def get_delay(self, attempt: int) -> float:
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

class LLMOutputRetry(RetryStrategy):
    """Retry for LLM output validation failures."""

    def __init__(
        self,
        max_attempts: int = 3,
        retry_delay: float = 0.5
    ):
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

    def should_retry(self, error: Exception, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False

        # Retry on validation/format errors
        return isinstance(error, (
            JSONDecodeError,
            ValidationError,
            OutputFormatError
        ))

    def get_delay(self, attempt: int) -> float:
        return self.retry_delay

class AdaptiveRetry(RetryStrategy):
    """Adapts retry strategy based on error patterns."""

    def __init__(self, error_tracker: ErrorTracker):
        self.tracker = error_tracker
        self.strategies = {
            "rate_limit": ExponentialBackoffRetry(max_attempts=5),
            "validation": LLMOutputRetry(max_attempts=3),
            "timeout": ExponentialBackoffRetry(max_attempts=3, base_delay=2.0)
        }

    def should_retry(self, error: Exception, attempt: int) -> bool:
        error_type = self._classify_error(error)
        strategy = self.strategies.get(error_type)

        if strategy:
            return strategy.should_retry(error, attempt)
        return False

    def get_delay(self, attempt: int) -> float:
        # Use recent error patterns to determine delay
        recent_errors = self.tracker.get_recent()
        if self._is_rate_limited(recent_errors):
            return min(60.0, 2 ** attempt)
        return 1.0

class RetryManager:
    """Manages retry execution."""

    def __init__(self, default_strategy: RetryStrategy):
        self.default_strategy = default_strategy

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        strategy: RetryStrategy = None,
        **kwargs
    ) -> Any:
        strategy = strategy or self.default_strategy
        last_error = None

        for attempt in range(10):  # Hard limit
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e

                if not strategy.should_retry(e, attempt):
                    raise

                delay = strategy.get_delay(attempt)
                logger.warning(
                    f"Retry attempt {attempt + 1}, "
                    f"waiting {delay}s: {e}"
                )
                await asyncio.sleep(delay)

        raise last_error
```

---

## Versioning & Lineage

### Flow Versioning

```python
class FlowVersion:
    """Represents a specific version of a flow."""

    def __init__(
        self,
        flow_id: str,
        version: str,
        definition: dict,
        created_at: datetime,
        created_by: str,
        parent_version: Optional[str] = None
    ):
        self.flow_id = flow_id
        self.version = version
        self.definition = definition
        self.created_at = created_at
        self.created_by = created_by
        self.parent_version = parent_version
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for deduplication."""
        content = json.dumps(self.definition, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

class FlowVersionManager:
    """Manages flow versions with diff capabilities."""

    def __init__(self, storage: FlowStorage):
        self.storage = storage

    async def create_version(
        self,
        flow_id: str,
        definition: dict,
        created_by: str
    ) -> FlowVersion:
        # Get latest version
        latest = await self.storage.get_latest_version(flow_id)

        # Compute new version number
        if latest:
            version = self._increment_version(latest.version)
            parent = latest.version
        else:
            version = "1.0.0"
            parent = None

        # Create version
        flow_version = FlowVersion(
            flow_id=flow_id,
            version=version,
            definition=definition,
            created_at=datetime.utcnow(),
            created_by=created_by,
            parent_version=parent
        )

        await self.storage.save_version(flow_version)
        return flow_version

    async def diff(
        self,
        flow_id: str,
        version1: str,
        version2: str
    ) -> FlowDiff:
        """Compute diff between two versions."""
        v1 = await self.storage.get_version(flow_id, version1)
        v2 = await self.storage.get_version(flow_id, version2)

        return FlowDiff(
            added_nodes=self._get_added_nodes(v1, v2),
            removed_nodes=self._get_removed_nodes(v1, v2),
            modified_nodes=self._get_modified_nodes(v1, v2),
            config_changes=self._get_config_changes(v1, v2)
        )

    async def rollback(self, flow_id: str, target_version: str) -> FlowVersion:
        """Rollback to a previous version."""
        target = await self.storage.get_version(flow_id, target_version)

        return await self.create_version(
            flow_id,
            target.definition,
            created_by="system_rollback"
        )
```

### Execution Lineage

```python
@dataclass
class ExecutionLineage:
    run_id: str
    flow_id: str
    flow_version: str
    parent_run_id: Optional[str]
    inputs: dict
    outputs: dict
    node_executions: List[NodeExecution]
    start_time: datetime
    end_time: datetime
    status: str

@dataclass
class NodeExecution:
    node_id: str
    start_time: datetime
    end_time: datetime
    inputs: dict
    outputs: dict
    status: str
    retry_count: int
    error: Optional[str]

class LineageStore:
    """Stores execution lineage for audit and debugging."""

    def __init__(self, database):
        self.db = database

    async def record_execution(
        self,
        run_id: str,
        flow_id: str,
        flow_version: str,
        inputs: dict,
        parent_run_id: Optional[str] = None
    ):
        await self.db.insert("executions", {
            "run_id": run_id,
            "flow_id": flow_id,
            "flow_version": flow_version,
            "inputs": json.dumps(inputs),
            "parent_run_id": parent_run_id,
            "start_time": datetime.utcnow(),
            "status": "running"
        })

    async def record_node_execution(
        self,
        run_id: str,
        node_execution: NodeExecution
    ):
        await self.db.insert("node_executions", {
            "run_id": run_id,
            **asdict(node_execution)
        })

    async def get_lineage(self, run_id: str) -> ExecutionLineage:
        """Get full lineage for a run."""
        execution = await self.db.get("executions", run_id)
        node_execs = await self.db.query(
            "node_executions",
            {"run_id": run_id}
        )

        return ExecutionLineage(
            run_id=run_id,
            flow_id=execution["flow_id"],
            flow_version=execution["flow_version"],
            parent_run_id=execution["parent_run_id"],
            inputs=json.loads(execution["inputs"]),
            outputs=json.loads(execution.get("outputs", "{}")),
            node_executions=[NodeExecution(**ne) for ne in node_execs],
            start_time=execution["start_time"],
            end_time=execution.get("end_time"),
            status=execution["status"]
        )

    async def get_downstream_runs(self, run_id: str) -> List[str]:
        """Get all runs that depend on this run's outputs."""
        return await self.db.query(
            "executions",
            {"parent_run_id": run_id}
        )
```

---

## Enterprise Features

### Human-in-the-Loop

```python
@dataclass
class HumanReviewConfig:
    queue: str
    timeout_hours: int = 24
    escalation: Optional[EscalationConfig] = None
    approval_required: bool = True

class HumanReviewNodeExecutor(BaseNodeExecutor):
    """Handles human-in-the-loop review steps."""

    def __init__(self, review_service: ReviewService):
        self.review_service = review_service

    async def execute(
        self,
        node: Node,
        inputs: Dict[str, Any],
        context: ExecutionContext
    ) -> NodeResult:
        config = HumanReviewConfig(**node.config)

        # Create review task
        task = await self.review_service.create_task(
            queue=config.queue,
            run_id=context.run_id,
            node_id=node.id,
            inputs=inputs,
            timeout_hours=config.timeout_hours
        )

        # Wait for completion (with escalation)
        result = await self._wait_for_review(task, config)

        return NodeResult(
            node_id=node.id,
            output=result.decision,
            metadata={
                "reviewer": result.reviewer,
                "review_time": result.duration_seconds
            }
        )

    async def _wait_for_review(
        self,
        task: ReviewTask,
        config: HumanReviewConfig
    ) -> ReviewResult:
        start_time = time.time()

        while True:
            # Check for completion
            result = await self.review_service.get_result(task.id)
            if result:
                return result

            # Check for timeout
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours > config.timeout_hours:
                raise ReviewTimeoutError(f"Review timed out after {config.timeout_hours}h")

            # Check for escalation
            if config.escalation and elapsed_hours > config.escalation.after_hours:
                await self._escalate(task, config.escalation)

            await asyncio.sleep(60)  # Poll every minute
```

### Secrets Management

```python
class SecretsManager:
    """Manages secrets for workflow execution."""

    def __init__(self, vault_client):
        self.vault = vault_client
        self.cache = TTLCache(maxsize=100, ttl=300)

    async def get_secret(self, key: str) -> str:
        """Get secret value."""
        if key in self.cache:
            return self.cache[key]

        value = await self.vault.get(key)
        self.cache[key] = value
        return value

    async def inject_secrets(
        self,
        config: dict,
        context: ExecutionContext
    ) -> dict:
        """Inject secrets into configuration."""
        result = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("$secret:"):
                secret_key = value[8:]
                result[key] = await self.get_secret(secret_key)
            elif isinstance(value, dict):
                result[key] = await self.inject_secrets(value, context)
            else:
                result[key] = value
        return result
```

### DAG Visualization UI

```python
class FlowVisualizer:
    """Generates visualization data for flows."""

    def generate_dag_layout(self, flow: CompiledFlow) -> dict:
        """Generate layout for DAG visualization."""
        # Use hierarchical layout
        levels = self._compute_levels(flow.dag)

        nodes = []
        edges = []

        for level_idx, level in enumerate(levels):
            for node_idx, node_id in enumerate(level):
                node = flow.get_node(node_id)
                nodes.append({
                    "id": node_id,
                    "type": node.type,
                    "label": node_id,
                    "x": node_idx * 200,
                    "y": level_idx * 150,
                    "config": node.config
                })

        for node_id in flow.dag.nodes:
            for dep in flow.dag.get_dependencies(node_id):
                edges.append({
                    "source": dep,
                    "target": node_id
                })

        return {"nodes": nodes, "edges": edges}

    def generate_execution_view(
        self,
        flow: CompiledFlow,
        lineage: ExecutionLineage
    ) -> dict:
        """Generate visualization with execution status."""
        layout = self.generate_dag_layout(flow)

        # Add execution status to nodes
        status_map = {
            ne.node_id: ne.status
            for ne in lineage.node_executions
        }

        for node in layout["nodes"]:
            node["status"] = status_map.get(node["id"], "pending")

        return layout
```

---

## Implementation Phases

### Phase 1: Core Engine (Weeks 1-4)

**Deliverables:**
- [ ] Flow DSL parser (YAML/JSON)
- [ ] DAG builder and validator
- [ ] Topological scheduler
- [ ] Basic node executors (LLM, Tool)
- [ ] State management

### Phase 2: Node Types (Weeks 5-7)

**Deliverables:**
- [ ] Retrieval node executor
- [ ] Branch/conditional nodes
- [ ] Subflow execution
- [ ] Transform nodes
- [ ] Node type registry

### Phase 3: Retry & Resilience (Weeks 8-10)

**Deliverables:**
- [ ] Retry strategy framework
- [ ] LLM-specific retry logic
- [ ] Circuit breaker pattern
- [ ] Error classification
- [ ] Adaptive retry

### Phase 4: Versioning & Lineage (Weeks 11-13)

**Deliverables:**
- [ ] Flow version management
- [ ] Diff computation
- [ ] Execution lineage tracking
- [ ] Metadata store
- [ ] Audit logging

### Phase 5: Enterprise Features (Weeks 14-17)

**Deliverables:**
- [ ] Human-in-the-loop nodes
- [ ] Secrets management
- [ ] DAG visualization UI
- [ ] Flow editor
- [ ] Access control

### Phase 6: Polish & Scale (Weeks 18-20)

**Deliverables:**
- [ ] Performance optimization
- [ ] Distributed execution
- [ ] Monitoring dashboard
- [ ] Documentation
- [ ] Load testing

---

## Stretch Goals

### Hot-Reloadable Flows

```python
class HotReloader:
    """Enables hot-reloading of flow definitions."""

    def __init__(self, flow_registry: FlowRegistry):
        self.registry = flow_registry
        self.watchers = {}

    async def watch(self, flow_path: str):
        """Watch flow definition for changes."""
        async for changes in awatch(flow_path):
            for change_type, path in changes:
                if change_type == Change.modified:
                    await self._reload_flow(path)

    async def _reload_flow(self, path: str):
        """Reload flow without stopping running instances."""
        # Parse new definition
        new_def = parse_flow_definition(path)

        # Validate
        validate_flow(new_def)

        # Update registry (new runs use new version)
        await self.registry.update(new_def)

        logger.info(f"Hot-reloaded flow: {new_def['name']}")
```

### Parallel Subflows

```python
class ParallelSubflowExecutor(BaseNodeExecutor):
    """Execute multiple subflows in parallel."""

    async def execute(
        self,
        node: Node,
        inputs: Dict[str, Any],
        context: ExecutionContext
    ) -> NodeResult:
        config = ParallelSubflowConfig(**node.config)

        # Create tasks for each subflow
        tasks = []
        for subflow_config in config.subflows:
            task = self._execute_subflow(subflow_config, inputs, context)
            tasks.append(task)

        # Execute in parallel
        results = await asyncio.gather(*tasks)

        # Merge results
        merged = self._merge_results(results, config.merge_strategy)

        return NodeResult(
            node_id=node.id,
            output=merged,
            metadata={"subflow_count": len(tasks)}
        )
```

### Typed Pipeline Validation

```python
class TypeValidator:
    """Validates types through the pipeline."""

    def validate_flow(self, flow: CompiledFlow) -> List[TypeError]:
        errors = []

        for node_id in flow.dag.nodes:
            node = flow.get_node(node_id)

            # Check input types
            for input_name, input_spec in node.input_schema.items():
                source = node.input_mappings.get(input_name)
                if source:
                    source_type = self._get_output_type(flow, source)
                    if not self._types_compatible(source_type, input_spec.type):
                        errors.append(TypeError(
                            f"Node {node_id}: input {input_name} expects {input_spec.type}, "
                            f"but {source} provides {source_type}"
                        ))

        return errors
```

---

## File Structure

```
28-ai-workflow-engine/
├── src/
│   ├── __init__.py
│   ├── dsl/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── validator.py
│   │   └── compiler.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── scheduler.py
│   │   ├── executor.py
│   │   ├── state.py
│   │   └── events.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── llm.py
│   │   ├── retrieval.py
│   │   ├── tool.py
│   │   ├── branch.py
│   │   └── subflow.py
│   ├── retry/
│   │   ├── __init__.py
│   │   ├── strategies.py
│   │   └── manager.py
│   ├── versioning/
│   │   ├── __init__.py
│   │   ├── versions.py
│   │   └── lineage.py
│   ├── enterprise/
│   │   ├── __init__.py
│   │   ├── human_review.py
│   │   ├── secrets.py
│   │   └── visualization.py
│   └── api/
│       ├── __init__.py
│       └── main.py
├── config/
├── tests/
├── docs/
├── ui/
│   └── dag-viewer/
├── BLUEPRINT.md
├── PROGRESS.md
└── SESSION_CONTEXT.md
```

---

## Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Flow Parse Time | < 100ms | For typical flows |
| Node Execution Overhead | < 10ms | Engine overhead |
| Retry Success Rate | > 80% | For transient errors |
| Version Diff Time | < 1s | For flow comparison |
| UI Render Time | < 500ms | DAG visualization |

---

## References

- [Dagster Architecture](https://dagster.io/blog/dagster-the-data-orchestrator)
- [Temporal Workflow Engine](https://docs.temporal.io/concepts/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [DSL Design Patterns](https://martinfowler.com/dsl.html)
