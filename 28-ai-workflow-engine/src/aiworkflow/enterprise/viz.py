"""Workflow visualization — export a flow (or a run) as Mermaid or Graphviz.

These produce text diagrams from a :class:`~aiworkflow.schemas.FlowDefinition`,
so a UI or docs can render the DAG, and a completed :class:`FlowRun` can be
overlaid with per-node status for debugging.
"""

from __future__ import annotations

from ..schemas import FlowDefinition, FlowRun, NodeStatus


def _edges(flow: FlowDefinition) -> list[tuple[str, str]]:
    """Collect DAG edges from both explicit edges and node dependencies."""
    pairs: set[tuple[str, str]] = set()
    for edge in flow.edges:
        pairs.add((edge.from_node, edge.to_node))
    for node in flow.nodes:
        for dep in node.dependencies:
            pairs.add((dep, node.id))
    return sorted(pairs)


def _label(node) -> str:
    name = node.name or node.id
    return f"{name}\\n({node.type.value})"


def to_mermaid(flow: FlowDefinition) -> str:
    """Render a flow as a Mermaid ``flowchart TD`` diagram."""
    lines = ["flowchart TD"]
    for node in flow.nodes:
        lines.append(f'    {node.id}["{_label(node)}"]')
    for src, dst in _edges(flow):
        lines.append(f"    {src} --> {dst}")
    return "\n".join(lines)


def to_dot(flow: FlowDefinition) -> str:
    """Render a flow as a Graphviz DOT digraph."""
    lines = [f'digraph "{flow.name}" {{', "    rankdir=TB;", "    node [shape=box];"]
    for node in flow.nodes:
        label = f"{node.name or node.id}\\n({node.type.value})"
        lines.append(f'    "{node.id}" [label="{label}"];')
    for src, dst in _edges(flow):
        lines.append(f'    "{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)


# Mermaid CSS classes per node status, for run overlays.
_STATUS_STYLE = {
    NodeStatus.COMPLETED: ("done", "fill:#c8e6c9,stroke:#2e7d32"),
    NodeStatus.FAILED: ("failed", "fill:#ffcdd2,stroke:#c62828"),
    NodeStatus.RUNNING: ("running", "fill:#fff9c4,stroke:#f9a825"),
    NodeStatus.SKIPPED: ("skipped", "fill:#eeeeee,stroke:#9e9e9e"),
}


def run_to_mermaid(flow: FlowDefinition, run: FlowRun) -> str:
    """Render a flow as Mermaid, coloring nodes by their execution status."""
    status_by_node = {e.node_id: e.status for e in run.node_executions}

    lines = ["flowchart TD"]
    for node in flow.nodes:
        lines.append(f'    {node.id}["{_label(node)}"]')
    for src, dst in _edges(flow):
        lines.append(f"    {src} --> {dst}")

    # Class definitions and assignments for executed nodes.
    used = {}
    for node in flow.nodes:
        status = status_by_node.get(node.id)
        style = _STATUS_STYLE.get(status)
        if style:
            cls, css = style
            used[cls] = css
            lines.append(f"    class {node.id} {cls};")
    for cls, css in used.items():
        lines.append(f"    classDef {cls} {css};")
    return "\n".join(lines)
