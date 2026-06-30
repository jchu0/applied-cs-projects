"""Topological scheduler for workflow execution."""

import asyncio
from typing import Any
from datetime import datetime

from ..schemas import (
    FlowDefinition,
    FlowRun,
    NodeExecution,
    NodeStatus,
    RunStatus,
    generate_id,
)
from ..compiler.dag import DAGBuilder


class Scheduler:
    """Schedules and executes workflow nodes."""

    def __init__(self, node_executor, max_parallel: int = 10):
        """Initialize scheduler.

        Args:
            node_executor: Executor for individual nodes
            max_parallel: Maximum parallel node executions
        """
        self.node_executor = node_executor
        self.max_parallel = max_parallel
        self.semaphore = asyncio.Semaphore(max_parallel)
        self._active_runs: dict[str, RunStatus] = {}

    async def execute_flow(
        self,
        flow: FlowDefinition,
        inputs: dict,
        run_id: str = None
    ) -> FlowRun:
        """Execute a complete workflow.

        Args:
            flow: Flow definition
            inputs: Input values
            run_id: Optional run identifier

        Returns:
            Flow run result
        """
        run_id = run_id or generate_id()
        dag = DAGBuilder(flow)

        # Initialize run
        run = FlowRun(
            run_id=run_id,
            flow_id=flow.name,
            flow_version=flow.version,
            status=RunStatus.RUNNING,
            inputs=inputs,
            start_time=datetime.utcnow()
        )

        self._active_runs[run_id] = RunStatus.RUNNING

        # State tracking
        completed: set[str] = set()
        results: dict[str, Any] = {"inputs": inputs}
        node_executions: list[NodeExecution] = []

        try:
            # Execute in topological order with parallelism
            levels = dag.get_execution_levels()

            for level in levels:
                # Execute all nodes in this level in parallel
                tasks = []
                for node_id in level:
                    node = flow.node_map[node_id]

                    # Check condition if present
                    if getattr(node, 'condition', None):
                        if not self._evaluate_condition(node.condition, results):
                            # Skip node - mark as completed with None result
                            completed.add(node_id)
                            results[node_id] = None
                            continue

                    task = self._execute_node_task(
                        node,
                        flow,
                        results,
                        completed
                    )
                    tasks.append((node_id, task))

                # Wait for all nodes in level
                level_results = await asyncio.gather(
                    *[t for _, t in tasks],
                    return_exceptions=True
                )

                # Process results
                for (node_id, _), result in zip(tasks, level_results):
                    if isinstance(result, Exception):
                        # Node failed
                        exec_record = NodeExecution(
                            node_id=node_id,
                            status=NodeStatus.FAILED,
                            start_time=datetime.utcnow(),
                            end_time=datetime.utcnow(),
                            error=str(result)
                        )
                        node_executions.append(exec_record)
                        run.status = RunStatus.FAILED
                        run.error = f"Node {node_id} failed: {result}"
                        break
                    else:
                        # Node succeeded
                        exec_record, output = result
                        node_executions.append(exec_record)
                        results[node_id] = output
                        completed.add(node_id)

                if run.status == RunStatus.FAILED:
                    break

            # Complete run
            if run.status != RunStatus.FAILED:
                run.status = RunStatus.COMPLETED
                run.outputs = self._collect_outputs(flow, results)

        except Exception as e:
            run.status = RunStatus.FAILED
            run.error = str(e)

        run.end_time = datetime.utcnow()
        run.node_executions = node_executions
        self._active_runs[run_id] = run.status

        return run

    # Alias for execute_flow
    execute = execute_flow

    async def _execute_node_task(
        self,
        node,
        flow: FlowDefinition,
        results: dict,
        completed: set[str]
    ) -> tuple:
        """Execute a single node with semaphore control.

        Args:
            node: Node to execute
            flow: Flow definition
            results: Current results
            completed: Completed node IDs

        Returns:
            Tuple of (execution record, output)
        """
        async with self.semaphore:
            start_time = datetime.utcnow()

            # Build inputs: flow inputs + dependency outputs + explicit refs
            node_inputs = dict(results.get("inputs", {}))

            # Merge outputs from upstream dependencies
            for dep_id in getattr(node, 'dependencies', []):
                dep_result = results.get(dep_id)
                if isinstance(dep_result, dict):
                    node_inputs.update(dep_result)

            # Overlay explicitly resolved node inputs
            resolved = self._resolve_inputs(node.inputs, results)
            node_inputs.update(resolved)

            # Retry logic
            retry_config = getattr(node, 'retry_config', None) or {}
            max_retries = retry_config.get('max_retries', 0)
            base_delay = retry_config.get('base_delay', 0)
            attempts = 0
            last_error = None

            while True:
                attempts += 1
                try:
                    output = await self.node_executor.execute(node, node_inputs)
                    break
                except Exception as e:
                    last_error = e
                    if attempts <= max_retries:
                        if base_delay:
                            await asyncio.sleep(base_delay)
                        continue
                    raise

            end_time = datetime.utcnow()
            latency = (end_time - start_time).total_seconds() * 1000

            exec_record = NodeExecution(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                inputs=node_inputs,
                outputs={"result": output},
                latency_ms=latency,
                attempts=attempts
            )

            return exec_record, output

    def _resolve_inputs(self, input_spec: dict, results: dict) -> dict:
        """Resolve input references to actual values.

        Args:
            input_spec: Input specification with references
            results: Available results

        Returns:
            Resolved input values
        """
        resolved = {}

        for key, value in input_spec.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # Parse reference like "{{node_id.output}}"
                ref = value[2:-2].strip()
                parts = ref.split(".")

                if parts[0] == "inputs":
                    # Reference to flow inputs
                    resolved[key] = results.get("inputs", {}).get(parts[1])
                else:
                    # Reference to node output
                    node_result = results.get(parts[0])
                    if node_result and len(parts) > 1:
                        resolved[key] = self._get_nested(node_result, parts[1:])
                    else:
                        resolved[key] = node_result
            else:
                resolved[key] = value

        return resolved

    def _get_nested(self, obj: Any, keys: list[str]) -> Any:
        """Get nested value from object."""
        for key in keys:
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                obj = getattr(obj, key, None)
        return obj

    def _collect_outputs(self, flow: FlowDefinition, results: dict) -> dict:
        """Collect final outputs from results."""
        outputs = {}
        for output_name, output_spec in flow.outputs.items():
            if isinstance(output_spec, str):
                outputs[output_name] = results.get(output_spec)
            else:
                outputs[output_name] = output_spec

        # If no explicit outputs, return last executed (non-None) node result
        if not outputs and flow.nodes:
            for node in reversed(flow.nodes):
                result = results.get(node.id)
                if result is not None:
                    if isinstance(result, dict):
                        outputs = result
                    else:
                        outputs = {"result": result}
                    break

        return outputs

    def _evaluate_condition(self, condition: str, results: dict) -> bool:
        """Evaluate a node condition against current results.

        Args:
            condition: Condition expression (e.g., "branch == 'high_score'")
            results: Current execution results

        Returns:
            True if condition is met
        """
        # Flatten all results into a single namespace
        namespace = {}
        for key, value in results.items():
            if key == "inputs":
                if isinstance(value, dict):
                    namespace.update(value)
            elif isinstance(value, dict):
                namespace.update(value)

        try:
            return bool(eval(condition, {"__builtins__": {}}, namespace))
        except Exception:
            return False

    def cancel(self, run_id: str) -> bool:
        """Cancel a running flow."""
        if run_id in self._active_runs:
            self._active_runs[run_id] = RunStatus.CANCELLED
            return True
        return False

    def get_status(self, run_id: str) -> RunStatus:
        """Get status of a run."""
        return self._active_runs.get(run_id, RunStatus.PENDING)

    def list_active(self) -> list[str]:
        """List active (running) run IDs."""
        return [
            rid for rid, status in self._active_runs.items()
            if status == RunStatus.RUNNING
        ]

    async def pause(self, run_id: str) -> bool:
        """Pause a running flow."""
        if run_id in self._active_runs:
            self._active_runs[run_id] = RunStatus.PENDING
            return True
        return False

    async def resume(self, run_id: str) -> bool:
        """Resume a paused flow."""
        if run_id in self._active_runs:
            self._active_runs[run_id] = RunStatus.RUNNING
            return True
        return False


class AsyncScheduler(Scheduler):
    """Scheduler optimized for async execution."""

    async def execute_parallel(
        self,
        flow: FlowDefinition,
        inputs_list: list[dict]
    ) -> list[FlowRun]:
        """Execute multiple runs in parallel.

        Args:
            flow: Flow definition
            inputs_list: List of inputs for each run

        Returns:
            List of flow runs
        """
        tasks = [
            self.execute_flow(flow, inputs)
            for inputs in inputs_list
        ]

        return await asyncio.gather(*tasks)
