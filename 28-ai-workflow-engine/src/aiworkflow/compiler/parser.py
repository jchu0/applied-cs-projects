"""Flow DSL parser for YAML and JSON definitions."""

import yaml
import json
from typing import Any

from ..schemas import (
    FlowDefinition,
    Node,
    NodeType,
    NodeConfig,
    RetryConfig,
    Edge,
)


class ParseError(Exception):
    """Error raised during flow parsing."""
    pass


class FlowParser:
    """Parses flow definitions from YAML/JSON/DSL."""

    def parse(self, content: str) -> FlowDefinition:
        """Parse content, auto-detecting format (YAML, JSON, or DSL).

        Args:
            content: Content string

        Returns:
            Parsed flow definition
        """
        try:
            return self.parse_yaml(content)
        except Exception:
            try:
                return self.parse_json(content)
            except Exception:
                return self.parse_dsl(content)

    def parse_dict(self, data: dict) -> FlowDefinition:
        """Parse from dictionary.

        Args:
            data: Definition dictionary

        Returns:
            Parsed flow definition
        """
        return self._parse_definition(data)

    def parse_yaml(self, content: str) -> FlowDefinition:
        """Parse YAML flow definition.

        Args:
            content: YAML content

        Returns:
            Parsed flow definition
        """
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ParseError(str(e)) from e
        if not isinstance(data, dict):
            raise ParseError("Invalid YAML: expected mapping")
        return self._parse_definition(data)

    def parse_json(self, content) -> FlowDefinition:
        """Parse JSON flow definition.

        Args:
            content: JSON string or dict

        Returns:
            Parsed flow definition
        """
        if isinstance(content, dict):
            data = content
        else:
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise ParseError(str(e)) from e
        return self._parse_definition(data)

    def parse_dsl(self, content: str) -> FlowDefinition:
        """Parse custom DSL flow definition.

        Args:
            content: DSL content

        Returns:
            Parsed flow definition
        """
        import re

        name = "unnamed"
        nodes = []
        edges = []
        current_node = None
        current_section = None
        in_config = False
        config = {}

        for line in content.strip().split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            wf_match = re.match(r'workflow\s+(\w+)\s*:', stripped)
            if wf_match:
                name = wf_match.group(1)
                continue

            node_match = re.match(r'node\s+(\w+)\s*:', stripped)
            if node_match:
                if current_node:
                    if config:
                        current_node['config'] = dict(config)
                    nodes.append(current_node)
                node_id = node_match.group(1)
                current_node = {'id': node_id, 'name': node_id}
                in_config = False
                config = {}
                current_section = 'node'
                continue

            if stripped == 'flow:':
                if current_node:
                    if config:
                        current_node['config'] = dict(config)
                    nodes.append(current_node)
                    current_node = None
                current_section = 'flow'
                in_config = False
                continue

            if stripped == 'config:':
                in_config = True
                config = {}
                continue

            if current_section == 'flow' and '->' in stripped:
                parts = [p.strip() for p in stripped.split('->')]
                for i in range(len(parts) - 1):
                    edges.append({'from': parts[i], 'to': parts[i + 1]})
                continue

            if ':' in stripped:
                key, value = stripped.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if in_config:
                    config[key] = value
                    if current_node:
                        current_node['config'] = dict(config)
                elif current_node:
                    if key == 'type':
                        current_node['type'] = value
                    elif key == 'depends_on':
                        current_node['dependencies'] = [
                            v.strip() for v in value.split(',')
                        ]

        if current_node:
            if config:
                current_node['config'] = dict(config)
            nodes.append(current_node)

        parsed_nodes = [self._parse_node(n) for n in nodes]
        parsed_edges = [
            Edge(from_node=e['from'], to_node=e['to'])
            for e in edges
        ]

        return FlowDefinition(
            name=name,
            nodes=parsed_nodes,
            edges=parsed_edges,
        )

    def parse_file(self, filepath: str) -> FlowDefinition:
        """Parse flow definition from file.

        Args:
            filepath: Path to flow file

        Returns:
            Parsed flow definition
        """
        with open(filepath) as f:
            content = f.read()

        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            return self.parse_yaml(content)
        elif filepath.endswith('.json'):
            return self.parse_json(content)
        elif filepath.endswith('.dsl'):
            return self.parse_dsl(content)
        else:
            # Try YAML first, then JSON
            try:
                return self.parse_yaml(content)
            except yaml.YAMLError:
                return self.parse_json(content)

    def _parse_definition(self, data: dict) -> FlowDefinition:
        """Parse raw definition into FlowDefinition.

        Args:
            data: Raw definition dict

        Returns:
            FlowDefinition object

        Raises:
            ParseError: If required fields are missing
        """
        if 'nodes' not in data:
            raise ParseError("Missing required field: nodes")

        # Parse nodes
        nodes = []
        for node_data in data.get('nodes', []):
            node = self._parse_node(node_data)
            nodes.append(node)

        # Parse edges
        edges = []
        for edge_data in data.get('edges', []):
            edge = Edge(
                from_node=edge_data['from'],
                to_node=edge_data['to'],
                condition=edge_data.get('condition')
            )
            edges.append(edge)

        return FlowDefinition(
            name=data.get('name', 'unnamed'),
            version=str(data.get('version', '1.0')),
            description=data.get('description', ''),
            config=data.get('config', {}),
            inputs=data.get('inputs', {}),
            outputs=data.get('outputs', {}),
            nodes=nodes,
            edges=edges
        )

    def _parse_node(self, data: dict) -> Node:
        """Parse node definition.

        Args:
            data: Node data dict

        Returns:
            Node object
        """
        # Parse node type
        node_type = NodeType(data.get('type', 'llm'))

        # Parse config
        config_data = data.get('config', {})
        config = NodeConfig(
            model=config_data.get('model'),
            temperature=config_data.get('temperature', 0.7),
            max_tokens=config_data.get('max_tokens', 1000),
            prompt_template=config_data.get('prompt_template', ''),
            timeout_seconds=config_data.get('timeout_seconds', 60),
            extra={k: v for k, v in config_data.items()
                   if k not in ['model', 'temperature', 'max_tokens',
                                'prompt_template', 'timeout_seconds']}
        )

        # Parse retry config
        retry = None
        if 'retry' in data:
            retry_data = data['retry']
            retry = RetryConfig(
                max_attempts=retry_data.get('max_attempts', 3),
                strategy=retry_data.get('strategy', 'exponential'),
                base_delay_ms=retry_data.get('base_delay_ms', 1000),
                max_delay_ms=retry_data.get('max_delay_ms', 30000)
            )

        return Node(
            id=data['id'],
            type=node_type,
            config=config,
            name=data.get('name', data.get('id')),
            executor=data.get('executor'),
            inputs=data.get('inputs', {}),
            outputs=data.get('outputs', {}),
            dependencies=data.get('dependencies', []),
            retry=retry,
            retry_config=data.get('retry_config'),
            condition=data.get('condition'),
            checkpoint=data.get('checkpoint', False),
            metadata=data.get('metadata', {})
        )


class FlowValidator:
    """Validates flow definitions."""

    def __init__(self):
        self._errors: list[str] = []

    def validate(self, flow: FlowDefinition) -> bool:
        """Validate a flow definition.

        Args:
            flow: Flow to validate

        Returns:
            True if valid, False if errors found
        """
        self._errors = []

        # Check for empty flow
        if not flow.nodes:
            self._errors.append("Empty flow: no nodes defined")

        # Check for unique node IDs
        node_ids = [node.id for node in flow.nodes]
        if len(node_ids) != len(set(node_ids)):
            self._errors.append("Duplicate node IDs found")

        # Check dependencies exist
        for node in flow.nodes:
            for dep in node.dependencies:
                if dep not in node_ids:
                    self._errors.append(f"Node {node.id} depends on non-existent node: invalid dependency {dep}")

        # Check edges reference valid nodes
        for edge in flow.edges:
            if edge.from_node not in node_ids:
                self._errors.append(f"Edge references non-existent node {edge.from_node}")
            if edge.to_node not in node_ids:
                self._errors.append(f"Edge references non-existent node {edge.to_node}")

        # Check for cycles
        if flow.nodes and self._has_cycle(flow):
            self._errors.append("Circular dependency detected")

        return len(self._errors) == 0

    def get_errors(self) -> list[str]:
        """Get validation errors from last validate() call."""
        return self._errors

    def _has_cycle(self, flow: FlowDefinition) -> bool:
        """Check if flow has cyclic dependencies."""
        # Build adjacency list
        graph = {node.id: set() for node in flow.nodes}
        for node in flow.nodes:
            for dep in node.dependencies:
                if dep in graph:
                    graph[dep].add(node.id)
        for edge in flow.edges:
            if edge.from_node in graph:
                graph[edge.from_node].add(edge.to_node)

        # DFS for cycle detection
        visited = set()
        rec_stack = set()

        def has_cycle_util(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in graph.get(node_id, []):
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in graph:
            if node_id not in visited:
                if has_cycle_util(node_id):
                    return True

        return False
