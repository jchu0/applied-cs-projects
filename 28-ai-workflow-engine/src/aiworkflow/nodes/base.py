"""Base node executor and node type implementations."""

from abc import ABC, abstractmethod
from typing import Any
import json
import re
import time

from ..schemas import Node, NodeType, NodeStatus


class BaseNodeExecutor(ABC):
    """Base class for node executors."""

    @abstractmethod
    async def execute(self, node: Node, inputs: dict) -> Any:
        """Execute a node.

        Args:
            node: Node to execute
            inputs: Node inputs

        Returns:
            Node output
        """
        pass


class NodeBase:
    """Base class for workflow nodes with lifecycle management."""

    def __init__(self, node_def: Node):
        self.id = node_def.id
        self.name = node_def.name
        self.type = node_def.type
        self.config = node_def.config if isinstance(node_def.config, dict) else getattr(node_def.config, 'extra', {})
        self.status = NodeStatus.PENDING
        self.execution_time = 0
        self.error = None
        self._input_schema = getattr(node_def, 'input_schema', None)

    async def execute(self, inputs: dict) -> Any:
        """Execute the node."""
        self.status = NodeStatus.RUNNING
        start = time.time()
        try:
            result = await self._execute(inputs)
            self.status = NodeStatus.COMPLETED
            self.execution_time = time.time() - start
            return result
        except Exception as e:
            self.status = NodeStatus.FAILED
            self.error = str(e)
            self.execution_time = time.time() - start
            raise

    async def _execute(self, inputs: dict) -> Any:
        """Override in subclasses."""
        raise NotImplementedError

    def validate_inputs(self, inputs: dict) -> bool:
        """Validate inputs against schema."""
        if not self._input_schema:
            return True
        required = self._input_schema.get("required", [])
        for field in required:
            if field not in inputs:
                return False
        return True


class DataNode(NodeBase):
    """Node for loading data from various sources."""

    async def _execute(self, inputs: dict) -> Any:
        source = self.config.get("source", "")

        if source.endswith(".csv") or self.config.get("delimiter"):
            import pandas
            df = pandas.read_csv(
                source,
                delimiter=self.config.get("delimiter", ","),
                header=0 if self.config.get("has_header", True) else None
            )
            return {"data": df.to_dict()}

        elif source == "database":
            import sqlalchemy
            import pandas
            engine = sqlalchemy.create_engine(self.config["connection_string"])
            df = pandas.read_sql(self.config["query"], engine)
            return {"data": df.to_dict()}

        elif source == "api":
            import aiohttp
            async with aiohttp.ClientSession() as session:
                resp_cm = await session.request(
                    self.config.get("method", "GET"),
                    self.config["url"],
                    headers=self.config.get("headers")
                )
                async with resp_cm as response:
                    data = await response.json()
                    return {"data": data}

        return {"data": None}


class ProcessNode(NodeBase):
    """Node for data processing and transformation."""

    async def _execute(self, inputs: dict) -> Any:
        operations = self.config.get("operations", [])
        data = inputs.get("data", {})

        if self.config.get("aggregations"):
            return {
                "aggregated_data": data,
                "statistics": {"operations": list(self.config.get("aggregations", {}).keys())}
            }

        return {
            "processed_data": data,
            "metadata": {"operations_applied": operations}
        }


class ModelNode(NodeBase):
    """Node for model training and prediction."""

    async def _execute(self, inputs: dict) -> Any:
        if self.config.get("mode") == "predict":
            model = inputs.get("model")
            features = inputs.get("features")
            predictions = model.predict(features)
            return {"predictions": list(predictions)}

        # Training mode
        algorithm = self.config.get("algorithm", "random_forest")
        hyperparameters = self.config.get("hyperparameters", {})

        if algorithm == "random_forest":
            import sklearn.ensemble
            model = sklearn.ensemble.RandomForestClassifier(**hyperparameters)
            features = inputs.get("features")
            labels = inputs.get("labels")
            model.fit(features, labels)
            score = model.score(features, labels)
            return {"model": model, "metrics": {"accuracy": score}}

        return {"model": None, "metrics": {}}


class ValidationNode(NodeBase):
    """Node for data validation."""

    async def _execute(self, inputs: dict) -> Any:
        rules = self.config.get("rules", [])
        data = inputs.get("data", {})
        errors = []

        for rule in rules:
            field = rule.get("field")
            check = rule.get("check")
            if field and field not in data:
                errors.append(f"Missing field: {field}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "data": data
        }


class ConditionalNode(NodeBase):
    """Node for conditional branching."""

    async def _execute(self, inputs: dict) -> Any:
        condition_config = self.config.get("condition", {})
        true_branch = self.config.get("true_branch")
        false_branch = self.config.get("false_branch")

        if isinstance(condition_config, dict):
            if "type" in condition_config:
                result = self._evaluate_compound(condition_config, inputs)
            else:
                result = self._evaluate_simple(condition_config, inputs)
        else:
            result = bool(condition_config)

        return {
            "next_node": true_branch if result else false_branch,
            "condition_met": result
        }

    def _evaluate_simple(self, condition: dict, inputs: dict) -> bool:
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        input_value = inputs.get(field)

        if operator == "greater_than":
            return input_value > value
        elif operator == "less_than":
            return input_value < value
        elif operator == "equals":
            return input_value == value
        elif operator == "not_equals":
            return input_value != value
        return False

    def _evaluate_compound(self, condition: dict, inputs: dict) -> bool:
        cond_type = condition.get("type")
        sub_conditions = condition.get("conditions", [])
        results = [self._evaluate_simple(c, inputs) for c in sub_conditions]

        if cond_type == "and":
            return all(results)
        elif cond_type == "or":
            return any(results)
        return False


class LLMNodeExecutor(BaseNodeExecutor):
    """Executor for LLM nodes."""

    def __init__(self, llm_client=None):
        """Initialize executor.

        Args:
            llm_client: LLM client for API calls
        """
        self.client = llm_client

    async def execute(self, node: Node, inputs: dict) -> Any:
        """Execute LLM node."""
        # Render prompt template
        prompt = self._render_template(node.config.prompt_template, inputs)

        # Call LLM (mock implementation)
        response = await self._call_llm(
            prompt=prompt,
            model=node.config.model or "gpt-4",
            temperature=node.config.temperature,
            max_tokens=node.config.max_tokens
        )

        return response

    def _render_template(self, template: str, values: dict) -> str:
        """Render template with values."""
        result = template
        for key, value in values.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result

    async def _call_llm(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Call LLM API."""
        # Mock implementation
        return f"[LLM Response for: {prompt[:50]}...]"


class RetrievalNodeExecutor(BaseNodeExecutor):
    """Executor for retrieval nodes."""

    def __init__(self, retriever=None):
        self.retriever = retriever

    async def execute(self, node: Node, inputs: dict) -> Any:
        """Execute retrieval node."""
        query = inputs.get("query", "")
        top_k = node.config.extra.get("top_k", 5)

        # Mock retrieval
        return [
            {"content": f"Document {i} for: {query}", "score": 1 - i * 0.1}
            for i in range(top_k)
        ]


class BranchNodeExecutor(BaseNodeExecutor):
    """Executor for conditional branch nodes."""

    async def execute(self, node: Node, inputs: dict) -> Any:
        """Execute branch node."""
        condition = node.config.extra.get("condition", "")
        branches = node.config.extra.get("branches", {})
        default = node.config.extra.get("default")

        # Evaluate condition
        value = self._evaluate_condition(condition, inputs)

        # Select branch
        branch = branches.get(value, default)

        return {"selected_branch": branch, "condition_value": value}

    def _evaluate_condition(self, condition: str, inputs: dict) -> Any:
        """Evaluate condition expression."""
        # Simple variable lookup
        if condition.startswith("{{") and condition.endswith("}}"):
            ref = condition[2:-2].strip()
            parts = ref.split(".")
            value = inputs
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            return value

        return condition


class TransformNodeExecutor(BaseNodeExecutor):
    """Executor for data transformation nodes."""

    async def execute(self, node: Node, inputs: dict) -> Any:
        """Execute transform node."""
        expression = node.config.extra.get("expression", "")

        # Simple expression evaluation
        if expression.startswith("{") and expression.endswith("}"):
            # JSON template
            result = expression
            for key, value in inputs.items():
                result = result.replace(f"inputs.{key}", json.dumps(value))
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result

        return inputs


class ToolNodeExecutor(BaseNodeExecutor):
    """Executor for tool/function nodes."""

    def __init__(self, tools: dict = None):
        self.tools = tools or {}

    async def execute(self, node: Node, inputs: dict) -> Any:
        """Execute tool node."""
        tool_name = node.config.extra.get("tool")
        tool = self.tools.get(tool_name)

        if tool:
            return await tool(**inputs)

        return {"error": f"Tool not found: {tool_name}"}


class MockNodeExecutor(BaseNodeExecutor):
    """Mock executor for testing."""

    async def execute(self, node: Node, inputs: dict) -> Any:
        return {
            "node_id": node.id,
            "type": node.type.value,
            "inputs": inputs,
            "mock": True
        }


class SubflowNodeExecutor(BaseNodeExecutor):
    """Executor that runs a nested workflow as a single node.

    A subflow node lets a workflow compose other workflows — either an inline
    flow definition or a reference to a flow registered with the engine. The
    nested run is executed by the same engine, so it gets the same scheduling,
    retry, versioning, and (recursively) subflow support.

    Configuration (read from ``node.config.extra``):
        flow: inline flow definition (dict) to run, OR
        flow_name: name of a flow registered with the engine
            (optionally with ``version``).
        input_mapping: optional ``{subflow_input: expr}`` map. ``expr`` may be a
            ``"{{key}}"`` / ``"{{a.b}}"`` reference into the parent node's inputs,
            or a literal. If omitted, the parent inputs are passed through.
        output_key: if set, the result is ``{output_key: <subflow outputs>}``;
            otherwise the subflow's output dict is returned directly.
        max_depth: optional per-node override of the recursion limit.

    Recursion is bounded: the executor threads a depth counter through the
    flow-level inputs (the scheduler propagates flow inputs to every node), so a
    subflow that (directly or transitively) calls itself fails cleanly at the
    limit instead of recursing forever.
    """

    #: Flow-level input key used to track nesting depth across subflow runs.
    DEPTH_KEY = "__subflow_depth__"

    def __init__(self, engine=None, max_depth: int = 10):
        """Initialize the executor.

        Args:
            engine: The :class:`WorkflowEngine` used to run nested flows. May be
                ``None`` in the default registry; the engine re-registers a bound
                instance on construction.
            max_depth: Default maximum subflow nesting depth.
        """
        self.engine = engine
        self.max_depth = max_depth

    async def execute(self, node: Node, inputs: dict) -> Any:
        """Run the configured nested workflow and return its outputs."""
        if self.engine is None:
            raise RuntimeError(
                f"Subflow node '{node.id}' cannot run: no engine is bound to the "
                "SubflowNodeExecutor (run subflows via WorkflowEngine, which binds one)"
            )

        extra = self._extra(node)
        max_depth = extra.get("max_depth", self.max_depth)
        depth = int(inputs.get(self.DEPTH_KEY, 0) or 0)
        if depth >= max_depth:
            raise RuntimeError(
                f"Subflow recursion limit ({max_depth}) exceeded at node '{node.id}'"
            )

        subflow = self._resolve_subflow(node, extra)
        sub_inputs = self._map_inputs(extra.get("input_mapping"), inputs)
        sub_inputs[self.DEPTH_KEY] = depth + 1

        # Import here to avoid a module-level cycle (engine imports nodes).
        from ..schemas import RunStatus

        run = await self.engine.run_flow(flow_definition=subflow, inputs=sub_inputs)
        if run.status == RunStatus.FAILED:
            raise RuntimeError(
                f"Subflow '{subflow.name}' failed in node '{node.id}': {run.error}"
            )

        output_key = extra.get("output_key")
        result = {output_key: run.outputs} if output_key else dict(run.outputs)
        if isinstance(result, dict):
            # Keep a lineage breadcrumb so the parent run can be linked to the child.
            result.setdefault("__subflow_run_id__", run.run_id)
        return result

    @staticmethod
    def _extra(node: Node) -> dict:
        """Return the node's free-form config dict regardless of config shape."""
        config = node.config
        if hasattr(config, "extra"):
            return config.extra or {}
        if isinstance(config, dict):
            return config
        return {}

    def _resolve_subflow(self, node: Node, extra: dict):
        """Build the nested FlowDefinition from inline config or a registered name."""
        if extra.get("flow"):
            return self.engine.parser.parse_dict(extra["flow"])
        flow_name = extra.get("flow_name")
        if flow_name:
            return self.engine.get_flow(flow_name, extra.get("version"))
        raise ValueError(
            f"Subflow node '{node.id}' must set either 'flow' (inline definition) "
            "or 'flow_name' (registered flow) in its config"
        )

    def _map_inputs(self, mapping: dict, inputs: dict) -> dict:
        """Build the nested flow's inputs from the optional input_mapping."""
        if not mapping:
            return {k: v for k, v in inputs.items() if k != self.DEPTH_KEY}
        return {target: self._resolve_expr(expr, inputs) for target, expr in mapping.items()}

    @staticmethod
    def _resolve_expr(expr: Any, inputs: dict) -> Any:
        """Resolve a ``{{a.b}}`` reference against the parent inputs, else literal."""
        if isinstance(expr, str) and expr.startswith("{{") and expr.endswith("}}"):
            value: Any = inputs
            for part in expr[2:-2].strip().split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            return value
        return expr


class NodeExecutorRegistry:
    """Registry for node executors."""

    def __init__(self):
        self._executors: dict[NodeType, BaseNodeExecutor] = {}
        self._named_executors: dict[str, Any] = {}
        self._default = MockNodeExecutor()
        self._setup_defaults()

    def _setup_defaults(self):
        """Set up default executors."""
        self._executors[NodeType.LLM] = LLMNodeExecutor()
        self._executors[NodeType.RETRIEVAL] = RetrievalNodeExecutor()
        self._executors[NodeType.BRANCH] = BranchNodeExecutor()
        self._executors[NodeType.TRANSFORM] = TransformNodeExecutor()
        self._executors[NodeType.TOOL] = ToolNodeExecutor()
        # Registered unbound; WorkflowEngine re-registers an engine-bound instance
        # so nested flows actually run. Unbound use raises a clear error.
        self._executors[NodeType.SUBFLOW] = SubflowNodeExecutor()

    def register(self, node_type, executor):
        """Register an executor for a node type or name."""
        if isinstance(node_type, str):
            self._named_executors[node_type] = executor
        else:
            self._executors[node_type] = executor

    def get(self, key) -> Any:
        """Get executor for node type or name.

        Raises:
            KeyError: If string key not found in named executors
        """
        if isinstance(key, str):
            if key in self._named_executors:
                return self._named_executors[key]
            raise KeyError(f"No executor registered for: {key}")
        return self._executors.get(key, self._default)

    def has(self, key) -> bool:
        """Check if an executor is registered."""
        if isinstance(key, str):
            return key in self._named_executors
        return key in self._executors

    def unregister(self, key):
        """Remove an executor."""
        if isinstance(key, str):
            self._named_executors.pop(key, None)
        else:
            self._executors.pop(key, None)

    def list_executors(self) -> list[str]:
        """List registered executor names."""
        names = list(self._named_executors.keys())
        names.extend(t.value for t in self._executors.keys())
        return names

    async def execute(self, node: Node, inputs: dict) -> Any:
        """Execute a node using appropriate executor.

        Args:
            node: Node to execute
            inputs: Node inputs

        Returns:
            Node output
        """
        # Check for named executor first
        if getattr(node, 'executor', None):
            executor = self._named_executors.get(node.executor)
            if executor:
                if isinstance(executor, BaseNodeExecutor):
                    return await executor.execute(node, inputs)
                # Plain async function
                return await executor(inputs)

        # Fall back to type-based executor
        executor = self.get(node.type)
        return await executor.execute(node, inputs)
