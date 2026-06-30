# LLM Agentic Runtime

## Executive Summary

Production-grade agent runtime for LLM-powered autonomous systems with tool invocation, multi-step reasoning, state persistence, and execution graph management. Supports ReAct patterns, hybrid planning, and enterprise features including sandboxed execution, guardrails, and comprehensive tracing.

> **Concepts covered:** [§04 LLM agents](../../04-ai-engineering/02-llm-applications/agents/llm-agents.md) (ReAct, tool use, multi-step reasoning) · [§04 Prompt engineering](../../04-ai-engineering/02-llm-applications/prompt-engineering/prompt-engineering.md). Pairs with [Project 28 (workflow engine — DAG orchestration)](../28-ai-workflow-engine/) and [Project 29 (model routing layer)](../29-model-routing-layer/) for the surrounding agent infrastructure. Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## System Architecture

```
+------------------------------------------------------------------+
|                      LLM Agentic Runtime                          |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +-------------------+    +-------------+ |
|  | Agent Core       |    | Planner           |    | Executor    | |
|  |------------------|    |-------------------|    |-------------| |
|  | - Memory Manager |    | - Model Planner   |    | - DAG Engine| |
|  | - Reasoning Loop |    | - Rule Planner    |    | - Sandbox   | |
|  | - State Machine  |    | - Hybrid Planner  |    | - Retry     | |
|  | - Error Handler  |    | - Plan Optimizer  |    | - Timeout   | |
|  +------------------+    +-------------------+    +-------------+ |
|           |                       |                      |        |
|           v                       v                      v        |
|  +------------------------------------------------------------------+
|  |                      Tool Registry                              |
|  |----------------------------------------------------------------|
|  | Functions | HTTP APIs | Retrieval | Database | Code Exec | ... |
|  +------------------------------------------------------------------+
|           |                                                        |
|           v                                                        |
|  +------------------------------------------------------------------+
|  |                   Enterprise Infrastructure                     |
|  |----------------------------------------------------------------|
|  | Guardrails | Tracing | Model Fallback | Rate Limiting | Audit  |
|  +------------------------------------------------------------------+
|                                                                   |
+------------------------------------------------------------------+
```

## Core Components

### 1. Agent Core Runtime

```python
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentContext:
    """Context for agent execution."""
    task: str
    max_steps: int = 20
    timeout_seconds: float = 300.0
    model_config: dict = field(default_factory=dict)
    tools_enabled: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

@dataclass
class ThoughtActionObservation:
    """Single step in agent reasoning."""
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = 0.0
    latency_ms: float = 0.0

@dataclass
class AgentResult:
    """Final result from agent execution."""
    success: bool
    answer: Optional[str]
    steps: list[ThoughtActionObservation]
    total_tokens: int
    total_latency_ms: float
    error: Optional[str] = None


class AgentRuntime:
    """Core agent runtime with ReAct-style reasoning loop."""

    def __init__(
        self,
        model_provider: "ModelProvider",
        tool_registry: "ToolRegistry",
        planner: "Planner",
        memory: "AgentMemory",
        config: "AgentRuntimeConfig",
    ):
        self.model = model_provider
        self.tools = tool_registry
        self.planner = planner
        self.memory = memory
        self.config = config

        self.state = AgentState.IDLE
        self.current_context: Optional[AgentContext] = None
        self.trace: list[ThoughtActionObservation] = []

    async def run(self, context: AgentContext) -> AgentResult:
        """Execute agent on given task."""
        self.current_context = context
        self.state = AgentState.THINKING
        self.trace = []

        start_time = time.time()
        total_tokens = 0

        try:
            for step in range(context.max_steps):
                # Check timeout
                if time.time() - start_time > context.timeout_seconds:
                    raise TimeoutError("Agent execution timed out")

                # Generate thought and action
                step_start = time.time()
                thought, action, action_input = await self._think(step)

                tao = ThoughtActionObservation(
                    step=step,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    timestamp=time.time(),
                )

                # Check for completion
                if action == "finish":
                    self.state = AgentState.COMPLETED
                    tao.observation = action_input.get("answer", "")
                    tao.latency_ms = (time.time() - step_start) * 1000
                    self.trace.append(tao)

                    return AgentResult(
                        success=True,
                        answer=tao.observation,
                        steps=self.trace,
                        total_tokens=total_tokens,
                        total_latency_ms=(time.time() - start_time) * 1000,
                    )

                # Execute action
                self.state = AgentState.ACTING
                try:
                    observation = await self._act(action, action_input)
                    tao.observation = observation
                except Exception as e:
                    tao.error = str(e)
                    tao.observation = f"Error: {e}"

                tao.latency_ms = (time.time() - step_start) * 1000
                self.trace.append(tao)

                # Update memory
                self.state = AgentState.OBSERVING
                await self.memory.add_step(tao)

                self.state = AgentState.THINKING

            # Max steps exceeded
            self.state = AgentState.FAILED
            return AgentResult(
                success=False,
                answer=None,
                steps=self.trace,
                total_tokens=total_tokens,
                total_latency_ms=(time.time() - start_time) * 1000,
                error="Maximum steps exceeded",
            )

        except Exception as e:
            self.state = AgentState.FAILED
            return AgentResult(
                success=False,
                answer=None,
                steps=self.trace,
                total_tokens=total_tokens,
                total_latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def _think(self, step: int) -> tuple[str, str, dict]:
        """Generate thought and decide on action."""
        # Build prompt with history
        messages = self._build_messages(step)

        # Call model
        response = await self.model.generate(
            messages=messages,
            tools=self.tools.get_tool_schemas(),
            **self.current_context.model_config,
        )

        # Parse response
        return self._parse_response(response)

    async def _act(self, action: str, action_input: dict) -> str:
        """Execute the chosen action."""
        tool = self.tools.get_tool(action)

        if tool is None:
            raise ValueError(f"Unknown tool: {action}")

        # Execute with sandboxing if required
        if self.config.sandbox_enabled:
            result = await self._sandboxed_execute(tool, action_input)
        else:
            result = await tool.execute(**action_input)

        return str(result)

    def _build_messages(self, step: int) -> list[dict]:
        """Build message history for model."""
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": self.current_context.task},
        ]

        # Add previous steps
        for tao in self.trace:
            # Assistant message with thought/action
            assistant_content = f"Thought: {tao.thought}"
            if tao.action:
                assistant_content += f"\nAction: {tao.action}"
                assistant_content += f"\nAction Input: {json.dumps(tao.action_input)}"

            messages.append({"role": "assistant", "content": assistant_content})

            # Observation
            if tao.observation:
                messages.append({
                    "role": "user",
                    "content": f"Observation: {tao.observation}"
                })

        return messages

    def _get_system_prompt(self) -> str:
        """Get system prompt with tool descriptions."""
        tool_descriptions = self.tools.get_formatted_descriptions()

        return f"""You are a helpful AI assistant that can use tools to accomplish tasks.

Available tools:
{tool_descriptions}

Response format:
Thought: <your reasoning about what to do next>
Action: <tool name or "finish">
Action Input: <JSON arguments for the tool>

When you have completed the task, use:
Action: finish
Action Input: {{"answer": "<your final answer>"}}

Important:
- Think step by step
- Use tools when needed
- Handle errors gracefully
- Always provide a final answer
"""

    def _parse_response(self, response: str) -> tuple[str, str, dict]:
        """Parse model response into thought, action, and input."""
        thought = ""
        action = ""
        action_input = {}

        lines = response.strip().split("\n")

        for line in lines:
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Action Input:"):
                try:
                    action_input = json.loads(line[13:].strip())
                except json.JSONDecodeError:
                    action_input = {"input": line[13:].strip()}

        return thought, action, action_input
```

### 2. Tool Registry

```python
from abc import ABC, abstractmethod
from typing import Callable, Any
import inspect
import httpx
import ast

class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the model."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for tool parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool."""
        pass


class FunctionTool(Tool):
    """Tool wrapping a Python function."""

    def __init__(self, func: Callable, name: str = None, description: str = None):
        self.func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__ or ""
        self._parameters = self._infer_parameters()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    def _infer_parameters(self) -> dict:
        """Infer JSON schema from function signature."""
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            prop = {"type": "string"}  # Default

            if param_name in hints:
                hint = hints[param_name]
                if hint == int:
                    prop = {"type": "integer"}
                elif hint == float:
                    prop = {"type": "number"}
                elif hint == bool:
                    prop = {"type": "boolean"}
                elif hint == list:
                    prop = {"type": "array"}

            properties[param_name] = prop

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    async def execute(self, **kwargs) -> Any:
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)


class HTTPTool(Tool):
    """Tool for calling HTTP APIs."""

    def __init__(
        self,
        name: str,
        description: str,
        url: str,
        method: str = "POST",
        headers: dict = None,
        parameters: dict = None,
    ):
        self._name = name
        self._description = description
        self.url = url
        self.method = method
        self.headers = headers or {}
        self._parameters = parameters or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    async def execute(self, **kwargs) -> Any:
        async with httpx.AsyncClient() as client:
            if self.method == "GET":
                response = await client.get(
                    self.url,
                    params=kwargs,
                    headers=self.headers,
                )
            else:
                response = await client.request(
                    self.method,
                    self.url,
                    json=kwargs,
                    headers=self.headers,
                )

            response.raise_for_status()
            return response.json()


class RetrievalTool(Tool):
    """Tool for RAG-style retrieval."""

    def __init__(
        self,
        name: str,
        description: str,
        retriever: "Retriever",
        top_k: int = 5,
    ):
        self._name = name
        self._description = description
        self.retriever = retriever
        self.top_k = top_k

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str) -> str:
        results = await self.retriever.search(query, top_k=self.top_k)

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result['text']}")

        return "\n\n".join(formatted)


class DatabaseTool(Tool):
    """Tool for database queries."""

    def __init__(
        self,
        name: str,
        description: str,
        connection_string: str,
        allowed_tables: list[str] = None,
        read_only: bool = True,
    ):
        self._name = name
        self._description = description
        self.connection_string = connection_string
        self.allowed_tables = allowed_tables
        self.read_only = read_only

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute"
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str) -> str:
        # Validate query
        if self.read_only and not self._is_read_only(query):
            raise ValueError("Only SELECT queries are allowed")

        if self.allowed_tables:
            self._validate_tables(query)

        # Execute query
        async with self._get_connection() as conn:
            result = await conn.fetch(query)

        return self._format_result(result)

    def _is_read_only(self, query: str) -> bool:
        """Check if query is read-only."""
        query_upper = query.strip().upper()
        return query_upper.startswith("SELECT")

    def _validate_tables(self, query: str):
        """Validate that query only accesses allowed tables."""
        # Simple validation - production would use SQL parser
        for table in self.allowed_tables:
            if table.lower() not in query.lower():
                # Would use proper SQL parsing
                pass


class CodeExecutionTool(Tool):
    """Tool for executing code (Python REPL)."""

    def __init__(
        self,
        name: str = "python_repl",
        description: str = "Execute Python code and return the result",
        timeout: float = 30.0,
        allowed_imports: list[str] = None,
    ):
        self._name = name
        self._description = description
        self.timeout = timeout
        self.allowed_imports = allowed_imports or [
            "math", "statistics", "datetime", "json", "re"
        ]
        self._globals = {"__builtins__": self._safe_builtins()}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }

    async def execute(self, code: str) -> str:
        # Validate code
        self._validate_code(code)

        # Execute in sandbox
        import asyncio

        try:
            result = await asyncio.wait_for(
                self._run_code(code),
                timeout=self.timeout,
            )
            return str(result)
        except asyncio.TimeoutError:
            return "Error: Code execution timed out"
        except Exception as e:
            return f"Error: {e}"

    def _validate_code(self, code: str):
        """Validate code for safety."""
        tree = ast.parse(code)

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.allowed_imports:
                        raise ValueError(f"Import not allowed: {alias.name}")

            if isinstance(node, ast.ImportFrom):
                if node.module not in self.allowed_imports:
                    raise ValueError(f"Import not allowed: {node.module}")

    async def _run_code(self, code: str) -> Any:
        """Execute code and return result."""
        local_vars = {}
        exec(code, self._globals, local_vars)

        # Return last expression or _result variable
        if "_result" in local_vars:
            return local_vars["_result"]
        elif local_vars:
            return list(local_vars.values())[-1]
        return "Code executed successfully"

    def _safe_builtins(self) -> dict:
        """Get safe subset of builtins."""
        safe = [
            "abs", "all", "any", "bin", "bool", "chr", "dict",
            "enumerate", "filter", "float", "format", "hex", "int",
            "len", "list", "map", "max", "min", "oct", "ord",
            "pow", "print", "range", "reversed", "round", "set",
            "slice", "sorted", "str", "sum", "tuple", "zip",
        ]
        return {k: getattr(__builtins__, k) for k in safe if hasattr(__builtins__, k)}


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_function(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
    ):
        """Register a function as a tool."""
        tool = FunctionTool(func, name, description)
        self.register(tool)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(name)

    def get_tool_schemas(self) -> list[dict]:
        """Get OpenAI-compatible tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in self._tools.values()
        ]

    def get_formatted_descriptions(self) -> str:
        """Get formatted tool descriptions for prompt."""
        descriptions = []
        for tool in self._tools.values():
            params = ", ".join(tool.parameters.get("properties", {}).keys())
            descriptions.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(descriptions)
```

### 3. Planner

```python
from abc import ABC, abstractmethod

class Plan:
    """Execution plan for agent."""

    def __init__(self, steps: list[dict]):
        self.steps = steps
        self.current_step = 0

    def next_step(self) -> Optional[dict]:
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.current_step += 1
            return step
        return None

    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)


class Planner(ABC):
    """Base class for planners."""

    @abstractmethod
    async def plan(self, task: str, tools: list[str]) -> Plan:
        """Generate execution plan for task."""
        pass


class ModelPlanner(Planner):
    """Use LLM to generate execution plan."""

    def __init__(self, model: "ModelProvider"):
        self.model = model

    async def plan(self, task: str, tools: list[str]) -> Plan:
        prompt = f"""Create a step-by-step plan to accomplish the following task.

Task: {task}

Available tools: {', '.join(tools)}

Output a JSON array of steps, where each step has:
- "action": the tool to use
- "description": what this step accomplishes
- "dependencies": list of step indices this depends on

Example:
[
  {{"action": "search", "description": "Find relevant documents", "dependencies": []}},
  {{"action": "analyze", "description": "Analyze search results", "dependencies": [0]}}
]
"""

        response = await self.model.generate([
            {"role": "user", "content": prompt}
        ])

        steps = json.loads(response)
        return Plan(steps)


class RulePlanner(Planner):
    """Rule-based planner for common patterns."""

    def __init__(self, rules: dict):
        self.rules = rules

    async def plan(self, task: str, tools: list[str]) -> Plan:
        # Match task to rules
        for pattern, plan_template in self.rules.items():
            if pattern in task.lower():
                steps = self._instantiate_template(plan_template, task)
                return Plan(steps)

        # Default: single-step plan
        return Plan([{"action": "think", "description": task}])

    def _instantiate_template(self, template: list, task: str) -> list:
        """Instantiate plan template with task."""
        return [
            {**step, "task": task}
            for step in template
        ]


class HybridPlanner(Planner):
    """Combine model and rule planning."""

    def __init__(
        self,
        model_planner: ModelPlanner,
        rule_planner: RulePlanner,
        use_model_threshold: float = 0.7,
    ):
        self.model_planner = model_planner
        self.rule_planner = rule_planner
        self.threshold = use_model_threshold

    async def plan(self, task: str, tools: list[str]) -> Plan:
        # Try rule-based first
        rule_plan = await self.rule_planner.plan(task, tools)

        # Use model if rule plan is too simple
        if len(rule_plan.steps) < 2:
            return await self.model_planner.plan(task, tools)

        return rule_plan
```

### 4. Execution Graph

```python
from typing import Any
import asyncio
from collections import defaultdict

class ExecutionNode:
    """Node in execution graph."""

    def __init__(
        self,
        id: str,
        action: str,
        inputs: dict,
        dependencies: list[str] = None,
    ):
        self.id = id
        self.action = action
        self.inputs = inputs
        self.dependencies = dependencies or []
        self.status = "pending"
        self.result = None
        self.error = None

    def is_ready(self, completed: set[str]) -> bool:
        """Check if all dependencies are completed."""
        return all(dep in completed for dep in self.dependencies)


class ExecutionGraph:
    """DAG-based execution graph for parallel tool execution."""

    def __init__(self, tool_registry: ToolRegistry):
        self.tools = tool_registry
        self.nodes: dict[str, ExecutionNode] = {}
        self.completed: set[str] = set()

    def add_node(self, node: ExecutionNode):
        """Add node to graph."""
        self.nodes[node.id] = node

    def build_from_plan(self, plan: Plan):
        """Build execution graph from plan."""
        for i, step in enumerate(plan.steps):
            node = ExecutionNode(
                id=f"step_{i}",
                action=step["action"],
                inputs=step.get("inputs", {}),
                dependencies=[
                    f"step_{d}" for d in step.get("dependencies", [])
                ],
            )
            self.add_node(node)

    async def execute(self, max_parallel: int = 5) -> dict[str, Any]:
        """Execute graph with parallel execution."""
        results = {}
        semaphore = asyncio.Semaphore(max_parallel)

        while not self._is_complete():
            # Find ready nodes
            ready = [
                node for node in self.nodes.values()
                if node.status == "pending" and node.is_ready(self.completed)
            ]

            if not ready:
                if not self._is_complete():
                    raise RuntimeError("Execution graph is stuck")
                break

            # Execute ready nodes in parallel
            tasks = [
                self._execute_node(node, semaphore, results)
                for node in ready
            ]

            await asyncio.gather(*tasks)

        return results

    async def _execute_node(
        self,
        node: ExecutionNode,
        semaphore: asyncio.Semaphore,
        results: dict,
    ):
        """Execute a single node."""
        async with semaphore:
            node.status = "running"

            try:
                # Resolve input references
                inputs = self._resolve_inputs(node.inputs, results)

                # Execute tool
                tool = self.tools.get_tool(node.action)
                result = await tool.execute(**inputs)

                node.result = result
                node.status = "completed"
                results[node.id] = result
                self.completed.add(node.id)

            except Exception as e:
                node.error = str(e)
                node.status = "failed"
                raise

    def _resolve_inputs(self, inputs: dict, results: dict) -> dict:
        """Resolve references to previous node results."""
        resolved = {}

        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to previous result
                ref_id = value[1:]
                resolved[key] = results.get(ref_id)
            else:
                resolved[key] = value

        return resolved

    def _is_complete(self) -> bool:
        """Check if all nodes are completed."""
        return len(self.completed) == len(self.nodes)
```

### 5. Memory System

```python
from typing import Optional
import numpy as np

class AgentMemory:
    """Memory system for agent context."""

    def __init__(
        self,
        short_term_limit: int = 20,
        embedding_model: Optional["EmbeddingModel"] = None,
        vector_store: Optional["VectorStore"] = None,
    ):
        self.short_term: list[ThoughtActionObservation] = []
        self.short_term_limit = short_term_limit
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    async def add_step(self, step: ThoughtActionObservation):
        """Add step to memory."""
        self.short_term.append(step)

        # Move to long-term if exceeding limit
        if len(self.short_term) > self.short_term_limit:
            overflow = self.short_term.pop(0)
            await self._store_long_term(overflow)

    async def _store_long_term(self, step: ThoughtActionObservation):
        """Store step in long-term memory (vector DB)."""
        if self.vector_store and self.embedding_model:
            text = f"Thought: {step.thought}\nAction: {step.action}\nObservation: {step.observation}"
            embedding = await self.embedding_model.encode([text])

            await self.vector_store.add(
                embeddings=embedding,
                documents=[text],
                metadata=[{
                    "step": step.step,
                    "action": step.action,
                    "timestamp": step.timestamp,
                }],
            )

    async def retrieve_relevant(self, query: str, k: int = 5) -> list[str]:
        """Retrieve relevant past experiences."""
        if not self.vector_store or not self.embedding_model:
            return []

        query_embedding = await self.embedding_model.encode([query])
        results = await self.vector_store.search(query_embedding, k=k)

        return [r["document"] for r in results]

    def get_recent_context(self) -> str:
        """Get recent steps as context string."""
        context_parts = []

        for step in self.short_term[-5:]:  # Last 5 steps
            part = f"Step {step.step}:\n"
            part += f"  Thought: {step.thought}\n"
            if step.action:
                part += f"  Action: {step.action}\n"
            if step.observation:
                part += f"  Observation: {step.observation[:200]}...\n"
            context_parts.append(part)

        return "\n".join(context_parts)

    def clear(self):
        """Clear short-term memory."""
        self.short_term = []
```

## Enterprise Features

### Sandboxed Execution

```python
import docker
import tempfile
import os

class SandboxExecutor:
    """Execute tools in isolated Docker containers."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "512m",
        cpu_quota: int = 50000,  # 50% of one CPU
        network_disabled: bool = True,
        timeout: float = 30.0,
    ):
        self.client = docker.from_env()
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.network_disabled = network_disabled
        self.timeout = timeout

    async def execute(
        self,
        code: str,
        inputs: dict,
    ) -> str:
        """Execute code in sandbox."""
        # Create temp directory for code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code and inputs
            code_path = os.path.join(tmpdir, "code.py")
            input_path = os.path.join(tmpdir, "input.json")

            with open(code_path, "w") as f:
                f.write(code)

            with open(input_path, "w") as f:
                json.dump(inputs, f)

            # Run container
            try:
                result = self.client.containers.run(
                    self.image,
                    command=f"python /code/code.py",
                    volumes={tmpdir: {"bind": "/code", "mode": "ro"}},
                    mem_limit=self.memory_limit,
                    cpu_quota=self.cpu_quota,
                    network_disabled=self.network_disabled,
                    remove=True,
                    timeout=self.timeout,
                )
                return result.decode("utf-8")

            except docker.errors.ContainerError as e:
                return f"Error: {e.stderr.decode('utf-8')}"
            except Exception as e:
                return f"Error: {e}"
```

### Guardrails

```python
from typing import Callable

class Guardrails:
    """Safety guardrails for agent execution."""

    def __init__(self):
        self.input_validators: list[Callable] = []
        self.output_validators: list[Callable] = []
        self.action_validators: list[Callable] = []
        self.blocked_patterns: list[str] = []

    def add_input_validator(self, validator: Callable[[str], bool]):
        """Add input validation function."""
        self.input_validators.append(validator)

    def add_output_validator(self, validator: Callable[[str], bool]):
        """Add output validation function."""
        self.output_validators.append(validator)

    def add_action_validator(self, validator: Callable[[str, dict], bool]):
        """Add action validation function."""
        self.action_validators.append(validator)

    def block_pattern(self, pattern: str):
        """Block specific patterns in inputs/outputs."""
        self.blocked_patterns.append(pattern)

    def validate_input(self, input_text: str) -> tuple[bool, str]:
        """Validate input text."""
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                return False, f"Blocked pattern detected: {pattern}"

        # Run validators
        for validator in self.input_validators:
            if not validator(input_text):
                return False, "Input validation failed"

        return True, ""

    def validate_output(self, output_text: str) -> tuple[bool, str]:
        """Validate output text."""
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, output_text, re.IGNORECASE):
                return False, f"Blocked pattern in output: {pattern}"

        # Run validators
        for validator in self.output_validators:
            if not validator(output_text):
                return False, "Output validation failed"

        return True, ""

    def validate_action(
        self,
        action: str,
        action_input: dict,
    ) -> tuple[bool, str]:
        """Validate action before execution."""
        for validator in self.action_validators:
            if not validator(action, action_input):
                return False, f"Action validation failed: {action}"

        return True, ""


# Common guardrail patterns
class CommonGuardrails:
    """Pre-built guardrail patterns."""

    @staticmethod
    def no_pii():
        """Block PII patterns."""
        patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",              # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        ]
        return patterns

    @staticmethod
    def no_harmful_code():
        """Block harmful code patterns."""
        return [
            r"os\.system",
            r"subprocess\.(run|call|Popen)",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__",
        ]

    @staticmethod
    def sql_injection_prevention(query: str) -> bool:
        """Validate SQL for injection attempts."""
        dangerous = [
            "DROP", "DELETE", "TRUNCATE", "ALTER",
            "CREATE", "INSERT", "UPDATE", "--", ";"
        ]
        query_upper = query.upper()
        return not any(d in query_upper for d in dangerous)
```

### Tracing and Training Data

```python
import json
from datetime import datetime

class AgentTracer:
    """Trace agent executions for debugging and training data."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.traces: list[dict] = []

    def start_trace(self, task: str, context: AgentContext) -> str:
        """Start a new trace."""
        trace_id = f"{datetime.now().isoformat()}_{hash(task) % 10000}"

        trace = {
            "trace_id": trace_id,
            "task": task,
            "context": {
                "max_steps": context.max_steps,
                "timeout": context.timeout_seconds,
                "tools": context.tools_enabled,
            },
            "steps": [],
            "result": None,
            "start_time": time.time(),
        }

        self.traces.append(trace)
        return trace_id

    def log_step(
        self,
        trace_id: str,
        step: ThoughtActionObservation,
    ):
        """Log a step in the trace."""
        trace = self._get_trace(trace_id)
        if trace:
            trace["steps"].append({
                "step": step.step,
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "error": step.error,
                "latency_ms": step.latency_ms,
            })

    def end_trace(self, trace_id: str, result: AgentResult):
        """End trace and save."""
        trace = self._get_trace(trace_id)
        if trace:
            trace["result"] = {
                "success": result.success,
                "answer": result.answer,
                "total_tokens": result.total_tokens,
                "total_latency_ms": result.total_latency_ms,
                "error": result.error,
            }
            trace["end_time"] = time.time()

            # Save to file
            self._save_trace(trace)

    def _get_trace(self, trace_id: str) -> Optional[dict]:
        for trace in self.traces:
            if trace["trace_id"] == trace_id:
                return trace
        return None

    def _save_trace(self, trace: dict):
        """Save trace to file."""
        filename = f"{trace['trace_id']}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(trace, f, indent=2)

    def export_training_data(self) -> list[dict]:
        """Export traces as training data for fine-tuning."""
        training_data = []

        for trace in self.traces:
            if not trace["result"] or not trace["result"]["success"]:
                continue

            # Convert to conversation format
            messages = [
                {"role": "user", "content": trace["task"]}
            ]

            for step in trace["steps"]:
                # Assistant turn
                assistant_msg = f"Thought: {step['thought']}\n"
                if step["action"]:
                    assistant_msg += f"Action: {step['action']}\n"
                    assistant_msg += f"Action Input: {json.dumps(step['action_input'])}"

                messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                })

                # Observation turn
                if step["observation"]:
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {step['observation']}",
                    })

            training_data.append({"messages": messages})

        return training_data
```

### Model Fallback Chains

```python
class ModelFallbackChain:
    """Chain of models with automatic fallback."""

    def __init__(self, models: list["ModelProvider"]):
        self.models = models
        self.current_index = 0

    async def generate(
        self,
        messages: list[dict],
        **kwargs
    ) -> str:
        """Generate with fallback on failure."""
        errors = []

        for i, model in enumerate(self.models):
            try:
                return await model.generate(messages, **kwargs)
            except Exception as e:
                errors.append(f"{model.name}: {e}")

                if i < len(self.models) - 1:
                    # Log and continue to next model
                    print(f"Model {model.name} failed, falling back...")
                else:
                    # All models failed
                    raise RuntimeError(
                        f"All models failed: {'; '.join(errors)}"
                    )
```

## API Design

```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LLM Agentic Runtime")

class AgentRequest(BaseModel):
    task: str
    max_steps: int = 20
    timeout_seconds: float = 300.0
    tools: list[str] | None = None
    model: str = "gpt-4"

class AgentResponse(BaseModel):
    task_id: str
    status: str
    result: AgentResult | None = None

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
):
    """Start agent execution."""
    task_id = generate_task_id()

    context = AgentContext(
        task=request.task,
        max_steps=request.max_steps,
        timeout_seconds=request.timeout_seconds,
        tools_enabled=request.tools or [],
    )

    # Run in background
    background_tasks.add_task(
        execute_agent,
        task_id=task_id,
        context=context,
    )

    return AgentResponse(
        task_id=task_id,
        status="running",
    )

@app.get("/agent/{task_id}", response_model=AgentResponse)
async def get_agent_status(task_id: str):
    """Get agent execution status."""
    result = get_task_result(task_id)

    if result is None:
        return AgentResponse(
            task_id=task_id,
            status="running",
        )

    return AgentResponse(
        task_id=task_id,
        status="completed" if result.success else "failed",
        result=result,
    )

@app.post("/agent/{task_id}/cancel")
async def cancel_agent(task_id: str):
    """Cancel running agent."""
    success = cancel_task(task_id)

    if not success:
        raise HTTPException(404, "Task not found")

    return {"status": "cancelled"}

# Tool management
@app.post("/tools/register")
async def register_tool(tool_config: dict):
    """Register a new tool."""
    # Validate and register
    pass

@app.get("/tools")
async def list_tools():
    """List available tools."""
    return tool_registry.get_tool_schemas()
```

## Implementation Phases

### Phase 1: Core Runtime (Weeks 1-2)
- [ ] Agent state machine
- [ ] Basic ReAct loop
- [ ] Thought/Action/Observation parsing
- [ ] Memory management

### Phase 2: Tool System (Weeks 3-4)
- [ ] Tool registry
- [ ] Function tools
- [ ] HTTP tools
- [ ] Retrieval tools
- [ ] Database tools
- [ ] Code execution tools

### Phase 3: Planning (Weeks 5-6)
- [ ] Model planner
- [ ] Rule planner
- [ ] Hybrid planner
- [ ] Plan optimization

### Phase 4: Execution Graph (Weeks 7-8)
- [ ] DAG representation
- [ ] Parallel execution
- [ ] Dependency resolution
- [ ] Error handling

### Phase 5: Enterprise Features (Weeks 9-10)
- [ ] Sandboxed execution
- [ ] Guardrails
- [ ] Tracing
- [ ] Model fallback

### Phase 6: Production (Weeks 11-12)
- [ ] API service
- [ ] Monitoring
- [ ] Documentation
- [ ] Examples

## Testing Strategy

```python
import pytest

class TestAgentRuntime:
    @pytest.fixture
    def runtime(self):
        model = MockModelProvider()
        tools = ToolRegistry()
        tools.register(MockTool("search"))
        tools.register(MockTool("calculate"))

        return AgentRuntime(
            model_provider=model,
            tool_registry=tools,
            planner=MockPlanner(),
            memory=AgentMemory(),
            config=AgentRuntimeConfig(),
        )

    @pytest.mark.asyncio
    async def test_simple_task(self, runtime):
        context = AgentContext(task="What is 2 + 2?")
        result = await runtime.run(context)

        assert result.success
        assert "4" in result.answer

    @pytest.mark.asyncio
    async def test_tool_use(self, runtime):
        context = AgentContext(task="Search for Python tutorials")
        result = await runtime.run(context)

        assert result.success
        assert any(s.action == "search" for s in result.steps)

    @pytest.mark.asyncio
    async def test_max_steps(self, runtime):
        context = AgentContext(task="Infinite task", max_steps=3)
        result = await runtime.run(context)

        assert not result.success
        assert "Maximum steps exceeded" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self, runtime):
        context = AgentContext(task="Slow task", timeout_seconds=0.1)
        result = await runtime.run(context)

        assert not result.success
        assert "timed out" in result.error.lower()
```

## Stretch Goals

### ReAct/Reflexion Loop

```python
class ReflexionAgent(AgentRuntime):
    """Agent with self-reflection and learning."""

    async def run(self, context: AgentContext) -> AgentResult:
        # First attempt
        result = await super().run(context)

        if not result.success:
            # Reflect on failure
            reflection = await self._reflect(result)

            # Retry with reflection
            context.metadata["reflection"] = reflection
            result = await super().run(context)

        return result

    async def _reflect(self, result: AgentResult) -> str:
        """Generate reflection on failed attempt."""
        prompt = f"""The previous attempt to complete the task failed.

Task: {self.current_context.task}
Steps taken: {len(result.steps)}
Error: {result.error}

Reflect on what went wrong and how to improve:"""

        return await self.model.generate([
            {"role": "user", "content": prompt}
        ])
```

### Debugging UI

```python
# Streamlit-based debugging interface
import streamlit as st

def agent_debugger():
    st.title("Agent Debugger")

    # Task input
    task = st.text_area("Task")

    if st.button("Run"):
        # Execute agent
        result = run_agent_sync(task)

        # Display steps
        for step in result.steps:
            with st.expander(f"Step {step.step}: {step.action}"):
                st.write("**Thought:**", step.thought)
                st.write("**Action:**", step.action)
                st.json(step.action_input)
                st.write("**Observation:**", step.observation)

        # Final result
        st.success(f"Answer: {result.answer}")
```

## References

- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [LangChain Documentation](https://python.langchain.com/)
