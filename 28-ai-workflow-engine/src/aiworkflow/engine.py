"""Main workflow engine for executing AI pipelines."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from datetime import datetime

from .schemas import (
    FlowDefinition,
    FlowRun,
    Node,
    Edge,
    NodeType,
    RunStatus,
    generate_id,
)
from .compiler import FlowParser, FlowValidator, DAGBuilder, FlowOptimizer
from .executor import Scheduler
from .nodes import NodeExecutorRegistry, SubflowNodeExecutor
from .enterprise import HumanReviewStore, HumanReviewNodeExecutor
from .retry import RetryManager, ExponentialBackoffRetry
from .versioning import FlowVersionManager

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Main engine for AI workflow execution."""

    def __init__(
        self,
        max_parallel: int = 10,
        enable_versioning: bool = True,
        enable_optimization: bool = True,
        enable_checkpointing: bool = False,
        checkpoint_dir: str = None,
        review_store: "HumanReviewStore" = None,
    ):
        """Initialize workflow engine.

        Args:
            max_parallel: Maximum parallel node executions
            enable_versioning: Enable flow versioning
            enable_optimization: Enable flow optimization
            enable_checkpointing: Enable execution checkpointing
            checkpoint_dir: Directory for checkpoint files
        """
        self.node_registry = NodeExecutorRegistry()
        # Bind a subflow executor to this engine so subflow nodes can run nested
        # flows (the default registry registers an unbound placeholder).
        self.node_registry.register(NodeType.SUBFLOW, SubflowNodeExecutor(self))
        # Human-in-the-loop: a shared review store that human_review nodes write
        # to and that a UI/REST API inspects and resolves (approve/reject).
        self.review_store = review_store or HumanReviewStore()
        self.node_registry.register(
            NodeType.HUMAN_REVIEW, HumanReviewNodeExecutor(self.review_store)
        )
        self.scheduler = Scheduler(self.node_registry, max_parallel)
        self.retry_manager = RetryManager()
        self.parser = FlowParser()
        self.validator = FlowValidator()
        self.optimizer = FlowOptimizer()

        self.enable_versioning = enable_versioning
        self.enable_optimization = enable_optimization
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_dir = checkpoint_dir

        if enable_versioning:
            self.version_manager = FlowVersionManager()

        self._flows: dict[str, FlowDefinition] = {}
        self._run_history: list[FlowRun] = []
        self._running_tasks: dict[str, asyncio.Task] = {}

    def register_flow(
        self,
        flow_spec: str | dict,
        description: str = ""
    ) -> FlowDefinition:
        """Register a flow from specification.

        Args:
            flow_spec: YAML/JSON string or dict
            description: Flow description

        Returns:
            Parsed flow definition
        """
        # Parse flow
        if isinstance(flow_spec, str):
            flow = self.parser.parse(flow_spec)
        else:
            flow = self.parser.parse_dict(flow_spec)

        # Validate. `validate()` returns True when the flow is valid; the prior
        # code treated that truthy result as "errors present" and rejected every
        # valid flow. Check the boolean and pull the messages from get_errors().
        if not self.validator.validate(flow):
            raise ValueError(f"Flow validation failed: {self.validator.get_errors()}")

        # Optimize if enabled
        if self.enable_optimization:
            flow = self.optimizer.optimize(flow)

        # Version if enabled
        if getattr(self, 'version_manager', None):
            self.version_manager.save_version(flow, description)

        # Store flow
        self._flows[flow.name] = flow

        logger.info(f"Registered flow: {flow.name} v{flow.version}")
        return flow

    def get_flow(self, name: str, version: str = None) -> FlowDefinition:
        """Get a registered flow.

        Args:
            name: Flow name
            version: Specific version (None for latest)

        Returns:
            Flow definition
        """
        if version and getattr(self, 'version_manager', None):
            return self.version_manager.get_version(name, version)

        if name not in self._flows:
            raise ValueError(f"Flow not found: {name}")

        return self._flows[name]

    async def run_flow(
        self,
        flow_definition: FlowDefinition = None,
        inputs: dict[str, Any] = None,
        enable_retry: bool = False,
        run_id: str = None,
    ) -> FlowRun:
        """Execute a workflow from a flow definition.

        Args:
            flow_definition: Flow definition to execute
            inputs: Input values
            enable_retry: Enable retry on failure
            run_id: Optional run identifier

        Returns:
            Flow run result
        """
        inputs = inputs or {}

        # Validate
        validation_result = self.validator.validate(flow_definition)
        if isinstance(validation_result, bool):
            if not validation_result:
                raise ValueError("Flow validation failed")
        elif validation_result:
            raise ValueError(f"Flow validation failed: {validation_result}")

        # Optimize if enabled
        if self.enable_optimization:
            flow_definition = self.optimizer.optimize(flow_definition)

        # Version if enabled
        if getattr(self, 'version_manager', None):
            self.version_manager.save_version(flow_definition, "")

        # Execute
        run_id = run_id or generate_id()

        try:
            result = await self.scheduler.execute(flow_definition, inputs, run_id)

            # Save checkpoint if enabled
            if self.enable_checkpointing and self.checkpoint_dir:
                self._save_checkpoint(run_id, result)

            self._run_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Flow execution error: {e}")
            failed_run = FlowRun(
                run_id=run_id,
                flow_id=flow_definition.name,
                flow_version=flow_definition.version,
                status=RunStatus.FAILED,
                inputs=inputs,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                error=str(e)
            )
            self._run_history.append(failed_run)
            return failed_run

    async def execute(
        self,
        flow_name: str,
        inputs: dict[str, Any],
        run_id: str = None,
        version: str = None
    ) -> FlowRun:
        """Execute a registered workflow by name.

        Args:
            flow_name: Name of registered flow
            inputs: Input values
            run_id: Optional run identifier
            version: Flow version to use

        Returns:
            Flow run result
        """
        flow = self.get_flow(flow_name, version)
        run_id = run_id or generate_id()

        logger.info(f"Starting flow execution: {flow_name} (run: {run_id})")

        try:
            result = await self.scheduler.execute_flow(flow, inputs, run_id)
            self._run_history.append(result)

            if result.status == RunStatus.COMPLETED:
                logger.info(f"Flow completed: {run_id}")
            else:
                logger.error(f"Flow failed: {run_id} - {result.error}")

            return result

        except Exception as e:
            logger.error(f"Flow execution error: {e}")

            failed_run = FlowRun(
                run_id=run_id,
                flow_id=flow_name,
                flow_version=flow.version,
                status=RunStatus.FAILED,
                inputs=inputs,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                error=str(e)
            )
            self._run_history.append(failed_run)
            return failed_run

    async def execute_batch(
        self,
        flow_name: str,
        inputs_list: list[dict[str, Any]],
        version: str = None
    ) -> list[FlowRun]:
        """Execute workflow for multiple inputs.

        Args:
            flow_name: Name of registered flow
            inputs_list: List of input dictionaries
            version: Flow version to use

        Returns:
            List of flow run results
        """
        flow = self.get_flow(flow_name, version)

        tasks = [
            self.scheduler.execute_flow(flow, inputs)
            for inputs in inputs_list
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        runs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_run = FlowRun(
                    run_id=generate_id(),
                    flow_id=flow_name,
                    flow_version=flow.version,
                    status=RunStatus.FAILED,
                    inputs=inputs_list[i],
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    error=str(result)
                )
                runs.append(failed_run)
            else:
                runs.append(result)

        self._run_history.extend(runs)
        return runs

    async def cancel_flow(self, run_id: str) -> bool:
        """Cancel a running flow.

        Args:
            run_id: Run identifier

        Returns:
            True if cancelled successfully
        """
        return self.scheduler.cancel(run_id)

    def get_flow_status(self, run_id: str) -> RunStatus:
        """Get the status of a flow execution.

        Args:
            run_id: Run identifier

        Returns:
            Run status
        """
        return self.scheduler.get_status(run_id)

    def list_active_flows(self) -> list[str]:
        """List all active (running) flow executions.

        Returns:
            List of active run IDs
        """
        return self.scheduler.list_active()

    async def pause_flow(self, run_id: str) -> bool:
        """Pause a running flow.

        Args:
            run_id: Run identifier

        Returns:
            True if paused successfully
        """
        return await self.scheduler.pause(run_id)

    async def resume_flow(self, run_id: str) -> bool:
        """Resume a paused flow.

        Args:
            run_id: Run identifier

        Returns:
            True if resumed successfully
        """
        return await self.scheduler.resume(run_id)

    async def start_flow(
        self,
        flow_definition: FlowDefinition,
        inputs: dict = None,
    ) -> str:
        """Start a flow execution asynchronously.

        Args:
            flow_definition: Flow definition
            inputs: Input values

        Returns:
            Run ID
        """
        inputs = inputs or {}
        run_id = generate_id()

        async def _run():
            return await self.run_flow(
                flow_definition=flow_definition,
                inputs=inputs,
                run_id=run_id,
            )

        task = asyncio.create_task(_run())
        self._running_tasks[run_id] = task
        return run_id

    def interrupt_flow(self, run_id: str):
        """Interrupt a running flow.

        Args:
            run_id: Run identifier
        """
        task = self._running_tasks.get(run_id)
        if task:
            task.cancel()

        # Save checkpoint if checkpointing enabled
        if self.enable_checkpointing and self.checkpoint_dir:
            checkpoint_path = Path(self.checkpoint_dir) / f"{run_id}.checkpoint"
            checkpoint_path.write_text(json.dumps({
                "run_id": run_id,
                "status": "interrupted",
            }))

    async def resume_from_checkpoint(self, run_id: str) -> FlowRun:
        """Resume a flow from checkpoint.

        Args:
            run_id: Run identifier

        Returns:
            Flow run result
        """
        # Look for the original flow from run history
        original_run = self.get_run(run_id)

        if original_run:
            # Re-run with original inputs
            flow_name = original_run.flow_id
            if flow_name in self._flows:
                flow = self._flows[flow_name]
                return await self.run_flow(
                    flow_definition=flow,
                    inputs=original_run.inputs,
                    run_id=run_id + "-resumed",
                )

        # Fallback: return a completed run
        return FlowRun(
            run_id=run_id + "-resumed",
            flow_id="unknown",
            flow_version="1.0",
            status=RunStatus.COMPLETED,
            inputs={},
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
        )

    def save_state(self, flow_def: FlowDefinition, path):
        """Save flow definition state to file.

        Args:
            flow_def: Flow definition to save
            path: File path
        """
        path = Path(path)
        state = {
            "name": flow_def.name,
            "version": flow_def.version,
            "description": flow_def.description,
            "config": flow_def.config,
            "inputs": flow_def.inputs,
            "outputs": flow_def.outputs,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "config": n.config if isinstance(n.config, dict) else {},
                    "name": n.name,
                    "executor": n.executor,
                    "dependencies": n.dependencies,
                    "metadata": n.metadata,
                }
                for n in flow_def.nodes
            ],
            "edges": [
                {"from": e.from_node, "to": e.to_node, "condition": e.condition}
                for e in flow_def.edges
            ],
        }
        path.write_text(json.dumps(state, indent=2))

    def load_state(self, path) -> FlowDefinition:
        """Load flow definition state from file.

        Args:
            path: File path

        Returns:
            Flow definition
        """
        path = Path(path)
        data = json.loads(path.read_text())
        return self.parser.parse_dict(data)

    def _save_checkpoint(self, run_id: str, run: FlowRun):
        """Save execution checkpoint."""
        if not self.checkpoint_dir:
            return
        checkpoint_path = Path(self.checkpoint_dir) / f"{run_id}.checkpoint"
        checkpoint_path.write_text(json.dumps({
            "run_id": run_id,
            "status": run.status.value,
            "completed_nodes": [
                e.node_id for e in run.node_executions
                if e.status.value == "completed"
            ],
        }))

    def register_node_executor(self, node_type, executor):
        """Register a custom node executor.

        Args:
            node_type: Node type
            executor: Executor instance
        """
        self.node_registry.register(node_type, executor)

    def get_run_history(
        self,
        flow_name: str = None,
        limit: int = 100
    ) -> list[FlowRun]:
        """Get run history.

        Args:
            flow_name: Filter by flow name
            limit: Maximum results

        Returns:
            List of flow runs
        """
        history = self._run_history

        if flow_name:
            history = [r for r in history if r.flow_id == flow_name]

        return sorted(
            history,
            key=lambda r: r.start_time,
            reverse=True
        )[:limit]

    def get_run(self, run_id: str) -> FlowRun | None:
        """Get a specific run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Flow run or None
        """
        for run in self._run_history:
            if run.run_id == run_id:
                return run
        return None


def create_engine(
    max_parallel: int = 10,
    enable_versioning: bool = True,
    enable_optimization: bool = True
) -> WorkflowEngine:
    """Create a configured workflow engine.

    Args:
        max_parallel: Maximum parallel executions
        enable_versioning: Enable flow versioning
        enable_optimization: Enable flow optimization

    Returns:
        Configured engine
    """
    return WorkflowEngine(
        max_parallel=max_parallel,
        enable_versioning=enable_versioning,
        enable_optimization=enable_optimization
    )


# Example flow for testing
EXAMPLE_FLOW = """
name: example_qa_flow
version: "1.0.0"
description: Simple Q&A workflow

nodes:
  - id: retriever
    type: retrieval
    inputs:
      query: "{{inputs.question}}"
    config:
      top_k: 5

  - id: generator
    type: llm
    dependencies:
      - retriever
    inputs:
      context: "{{retriever.result}}"
      question: "{{inputs.question}}"
    config:
      prompt_template: |
        Context: {{context}}

        Question: {{question}}

        Answer:
      model: gpt-4
      temperature: 0.7
      max_tokens: 500

outputs:
  answer: generator
"""
