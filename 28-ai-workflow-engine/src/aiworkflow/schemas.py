"""Core data models for AI Workflow Engine."""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
from datetime import datetime
import uuid
import json
import hashlib


def generate_id() -> str:
    """Generate unique identifier."""
    return uuid.uuid4().hex[:12]


class NodeType(Enum):
    """Types of workflow nodes."""
    LLM = "llm"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    BRANCH = "branch"
    TRANSFORM = "transform"
    SUBFLOW = "subflow"
    HUMAN_REVIEW = "human_review"
    DATA = "data"
    PROCESS = "process"
    MODEL = "model"
    CONDITIONAL = "conditional"
    VALIDATION = "validation"


class NodeStatus(Enum):
    """Execution status of a node."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class RunStatus(Enum):
    """Status of a workflow run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeConfig:
    """Configuration for a workflow node."""
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    prompt_template: str = ""
    timeout_seconds: int = 60
    extra: dict = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Retry configuration for a node."""
    max_attempts: int = 3
    strategy: str = "exponential"
    base_delay_ms: int = 1000
    max_delay_ms: int = 30000


@dataclass
class Node:
    """Workflow node definition."""
    id: str
    type: NodeType
    config: Any = field(default_factory=dict)
    name: Optional[str] = None
    executor: Optional[str] = None
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    retry: Optional[RetryConfig] = None
    retry_config: Optional[dict] = None
    condition: Optional[str] = None
    checkpoint: bool = False
    metadata: dict = field(default_factory=dict)
    input_schema: Optional[dict] = None


@dataclass
class Edge:
    """Edge connecting two nodes."""
    from_node: str
    to_node: str
    condition: Optional[str] = None


@dataclass
class FlowDefinition:
    """Complete workflow definition."""
    name: str
    version: str = "1.0"
    description: str = ""
    config: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.version = str(self.version)
        self.edges = [
            Edge(from_node=e['from'], to_node=e['to'], condition=e.get('condition'))
            if isinstance(e, dict) else e
            for e in self.edges
        ]

    @property
    def node_map(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}


@dataclass
class NodeExecution:
    """Record of a node execution."""
    node_id: str
    status: NodeStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    error: Optional[str] = None
    attempts: int = 1
    latency_ms: float = 0


@dataclass
class FlowRun:
    """A single execution of a workflow."""
    run_id: str
    flow_id: str
    flow_version: str
    status: RunStatus
    inputs: dict
    outputs: dict = field(default_factory=dict)
    node_executions: list[NodeExecution] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FlowVersion:
    """Represents a specific version of a flow."""
    flow_id: str
    version: str
    definition: dict
    created_at: datetime
    created_by: str
    parent_version: Optional[str] = None

    @property
    def hash(self) -> str:
        content = json.dumps(self.definition, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class FlowDiff:
    """Difference between two flow versions."""
    added_nodes: list[str]
    removed_nodes: list[str]
    modified_nodes: list[str]
    config_changes: dict


@dataclass
class ExecutionLineage:
    """Complete lineage of an execution."""
    run_id: str
    flow_id: str
    flow_version: str
    parent_run_id: Optional[str]
    inputs: dict
    outputs: dict
    node_executions: list[NodeExecution]
    start_time: datetime
    end_time: datetime
    status: str
