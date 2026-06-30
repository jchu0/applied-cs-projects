"""dbt integration for the semantic layer.

This module provides integration with dbt for:
- Parsing manifest.json to understand model DAG
- Loading metric and model definitions from YAML
- Syncing dbt metrics with semantic layer catalog
- Running dbt commands programmatically
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from semantic_layer.models import (
    CalculationMethod,
    MetricDefinition,
    TimeGrain,
)

logger = logging.getLogger(__name__)


class DbtResourceType(Enum):
    """Types of dbt resources."""

    MODEL = "model"
    SOURCE = "source"
    SEED = "seed"
    SNAPSHOT = "snapshot"
    METRIC = "metric"
    EXPOSURE = "exposure"
    SEMANTIC_MODEL = "semantic_model"


class DbtMaterialization(Enum):
    """dbt materialization types."""

    VIEW = "view"
    TABLE = "table"
    INCREMENTAL = "incremental"
    EPHEMERAL = "ephemeral"


@dataclass
class DbtColumn:
    """dbt column definition."""

    name: str
    description: str = ""
    data_type: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    tests: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class DbtModel:
    """dbt model definition parsed from manifest."""

    unique_id: str
    name: str
    schema: str
    database: str
    description: str = ""
    materialization: DbtMaterialization = DbtMaterialization.VIEW
    columns: Dict[str, DbtColumn] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    raw_sql: str = ""
    compiled_sql: str = ""
    path: str = ""

    @property
    def full_name(self) -> str:
        """Get fully qualified table name."""
        return f"{self.database}.{self.schema}.{self.name}"


@dataclass
class DbtSource:
    """dbt source definition."""

    unique_id: str
    name: str
    source_name: str
    schema: str
    database: str
    description: str = ""
    freshness: Optional[Dict[str, Any]] = None
    columns: Dict[str, DbtColumn] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DbtMetric:
    """dbt metric definition from manifest or YAML."""

    unique_id: str
    name: str
    label: str
    description: str = ""
    calculation_method: str = "sum"
    expression: str = ""
    timestamp: Optional[str] = None
    time_grains: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    model: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DbtManifest:
    """Parsed dbt manifest.json."""

    metadata: Dict[str, Any]
    models: Dict[str, DbtModel]
    sources: Dict[str, DbtSource]
    metrics: Dict[str, DbtMetric]
    exposures: Dict[str, Any]
    semantic_models: Dict[str, Any]
    parent_map: Dict[str, List[str]]
    child_map: Dict[str, List[str]]

    def get_upstream(self, node_id: str) -> List[str]:
        """Get upstream dependencies for a node."""
        return self.parent_map.get(node_id, [])

    def get_downstream(self, node_id: str) -> List[str]:
        """Get downstream dependencies for a node."""
        return self.child_map.get(node_id, [])

    def get_model_by_name(self, name: str) -> Optional[DbtModel]:
        """Get model by name."""
        for model in self.models.values():
            if model.name == name:
                return model
        return None

    def get_lineage_path(
        self, start_node: str, end_node: str
    ) -> Optional[List[str]]:
        """Find lineage path between two nodes."""
        # BFS to find path
        queue = [(start_node, [start_node])]
        visited = set()

        while queue:
            current, path = queue.pop(0)
            if current == end_node:
                return path
            if current in visited:
                continue
            visited.add(current)

            for child in self.get_downstream(current):
                if child not in visited:
                    queue.append((child, path + [child]))

        return None


class ManifestParser:
    """Parse dbt manifest.json file."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.manifest_path = self.project_path / "target" / "manifest.json"

    def parse(self) -> DbtManifest:
        """Parse the manifest file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {self.manifest_path}. "
                "Run 'dbt compile' or 'dbt run' first."
            )

        with open(self.manifest_path) as f:
            data = json.load(f)

        return self._parse_manifest(data)

    def _parse_manifest(self, data: Dict[str, Any]) -> DbtManifest:
        """Parse manifest data into structured objects."""
        models = {}
        sources = {}
        metrics = {}
        exposures = {}
        semantic_models = {}

        nodes = data.get("nodes", {})
        sources_data = data.get("sources", {})
        metrics_data = data.get("metrics", {})
        semantic_models_data = data.get("semantic_models", {})

        # Parse models
        for node_id, node in nodes.items():
            if node.get("resource_type") == "model":
                models[node_id] = self._parse_model(node_id, node)

        # Parse sources
        for source_id, source in sources_data.items():
            sources[source_id] = self._parse_source(source_id, source)

        # Parse metrics
        for metric_id, metric in metrics_data.items():
            metrics[metric_id] = self._parse_metric(metric_id, metric)

        # Parse semantic models (dbt 1.6+)
        for sm_id, sm in semantic_models_data.items():
            semantic_models[sm_id] = sm

        # Parse exposures
        for exp_id, exp in data.get("exposures", {}).items():
            exposures[exp_id] = exp

        return DbtManifest(
            metadata=data.get("metadata", {}),
            models=models,
            sources=sources,
            metrics=metrics,
            exposures=exposures,
            semantic_models=semantic_models,
            parent_map=data.get("parent_map", {}),
            child_map=data.get("child_map", {}),
        )

    def _parse_model(self, node_id: str, node: Dict[str, Any]) -> DbtModel:
        """Parse a model node."""
        config = node.get("config", {})
        columns = {}

        for col_name, col_data in node.get("columns", {}).items():
            columns[col_name] = DbtColumn(
                name=col_name,
                description=col_data.get("description", ""),
                data_type=col_data.get("data_type"),
                meta=col_data.get("meta", {}),
                tags=col_data.get("tags", []),
            )

        materialization_str = config.get("materialized", "view")
        try:
            materialization = DbtMaterialization(materialization_str)
        except ValueError:
            materialization = DbtMaterialization.VIEW

        return DbtModel(
            unique_id=node_id,
            name=node.get("name", ""),
            schema=node.get("schema", ""),
            database=node.get("database", ""),
            description=node.get("description", ""),
            materialization=materialization,
            columns=columns,
            depends_on=node.get("depends_on", {}).get("nodes", []),
            tags=node.get("tags", []),
            meta=node.get("meta", {}),
            raw_sql=node.get("raw_sql", ""),
            compiled_sql=node.get("compiled_sql", ""),
            path=node.get("path", ""),
        )

    def _parse_source(
        self, source_id: str, source: Dict[str, Any]
    ) -> DbtSource:
        """Parse a source definition."""
        columns = {}
        for col_name, col_data in source.get("columns", {}).items():
            columns[col_name] = DbtColumn(
                name=col_name,
                description=col_data.get("description", ""),
                data_type=col_data.get("data_type"),
                meta=col_data.get("meta", {}),
            )

        return DbtSource(
            unique_id=source_id,
            name=source.get("name", ""),
            source_name=source.get("source_name", ""),
            schema=source.get("schema", ""),
            database=source.get("database", ""),
            description=source.get("description", ""),
            freshness=source.get("freshness"),
            columns=columns,
            meta=source.get("meta", {}),
        )

    def _parse_metric(
        self, metric_id: str, metric: Dict[str, Any]
    ) -> DbtMetric:
        """Parse a metric definition."""
        return DbtMetric(
            unique_id=metric_id,
            name=metric.get("name", ""),
            label=metric.get("label", metric.get("name", "")),
            description=metric.get("description", ""),
            calculation_method=metric.get("calculation_method", "sum"),
            expression=metric.get("expression", ""),
            timestamp=metric.get("timestamp"),
            time_grains=metric.get("time_grains", []),
            dimensions=metric.get("dimensions", []),
            filters=metric.get("filters", []),
            model=metric.get("model"),
            meta=metric.get("meta", {}),
        )


class YamlParser:
    """Parse dbt YAML files for model and metric definitions."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)

    def parse_models_yaml(
        self, yaml_path: Path
    ) -> Tuple[List[DbtModel], List[DbtSource]]:
        """Parse a models YAML file."""
        if not yaml_path.exists():
            return [], []

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if not data:
            return [], []

        models = []
        sources = []

        # Parse sources
        for source in data.get("sources", []):
            source_name = source.get("name", "")
            database = source.get("database", "")
            schema = source.get("schema", "")

            for table in source.get("tables", []):
                columns = {}
                for col in table.get("columns", []):
                    columns[col["name"]] = DbtColumn(
                        name=col["name"],
                        description=col.get("description", ""),
                        data_type=col.get("data_type"),
                        meta=col.get("meta", {}),
                        tests=col.get("tests", []),
                    )

                sources.append(
                    DbtSource(
                        unique_id=f"source.{source_name}.{table['name']}",
                        name=table["name"],
                        source_name=source_name,
                        schema=schema,
                        database=database,
                        description=table.get("description", ""),
                        freshness=source.get("freshness"),
                        columns=columns,
                        meta=table.get("meta", {}),
                    )
                )

        # Parse models
        for model in data.get("models", []):
            columns = {}
            for col in model.get("columns", []):
                columns[col["name"]] = DbtColumn(
                    name=col["name"],
                    description=col.get("description", ""),
                    data_type=col.get("data_type"),
                    meta=col.get("meta", {}),
                    tests=col.get("tests", []),
                )

            config = model.get("config", {})
            materialization_str = config.get("materialized", "view")
            try:
                materialization = DbtMaterialization(materialization_str)
            except ValueError:
                materialization = DbtMaterialization.VIEW

            models.append(
                DbtModel(
                    unique_id=f"model.{model['name']}",
                    name=model["name"],
                    schema="",
                    database="",
                    description=model.get("description", ""),
                    materialization=materialization,
                    columns=columns,
                    tags=model.get("tags", []),
                    meta=model.get("meta", {}),
                )
            )

        return models, sources

    def parse_metrics_yaml(self, yaml_path: Path) -> List[DbtMetric]:
        """Parse a metrics YAML file."""
        if not yaml_path.exists():
            return []

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if not data:
            return []

        metrics = []
        for metric in data.get("metrics", []):
            metrics.append(
                DbtMetric(
                    unique_id=f"metric.{metric['name']}",
                    name=metric["name"],
                    label=metric.get("label", metric["name"]),
                    description=metric.get("description", ""),
                    calculation_method=metric.get("calculation_method", "sum"),
                    expression=metric.get("expression", ""),
                    timestamp=metric.get("timestamp"),
                    time_grains=metric.get("time_grains", []),
                    dimensions=metric.get("dimensions", []),
                    filters=metric.get("filters", []),
                    model=metric.get("model"),
                    meta=metric.get("meta", {}),
                )
            )

        return metrics

    def find_all_yaml_files(self) -> List[Path]:
        """Find all YAML files in the project."""
        yaml_files = []
        for pattern in ["**/*.yml", "**/*.yaml"]:
            yaml_files.extend(self.project_path.glob(pattern))
        return yaml_files


class MetricSyncer:
    """Sync dbt metrics with semantic layer catalog."""

    CALC_METHOD_MAP = {
        "sum": CalculationMethod.SUM,
        "count": CalculationMethod.COUNT,
        "count_distinct": CalculationMethod.COUNT_DISTINCT,
        "average": CalculationMethod.AVERAGE,
        "avg": CalculationMethod.AVERAGE,
        "min": CalculationMethod.MIN,
        "max": CalculationMethod.MAX,
        "derived": CalculationMethod.DERIVED,
    }

    TIME_GRAIN_MAP = {
        "day": TimeGrain.DAY,
        "week": TimeGrain.WEEK,
        "month": TimeGrain.MONTH,
        "quarter": TimeGrain.QUARTER,
        "year": TimeGrain.YEAR,
    }

    def convert_dbt_metric(self, dbt_metric: DbtMetric) -> MetricDefinition:
        """Convert a dbt metric to semantic layer MetricDefinition."""
        calc_method = self.CALC_METHOD_MAP.get(
            dbt_metric.calculation_method.lower(),
            CalculationMethod.SUM,
        )

        time_grains = []
        for grain in dbt_metric.time_grains:
            if grain.lower() in self.TIME_GRAIN_MAP:
                time_grains.append(self.TIME_GRAIN_MAP[grain.lower()])

        # Default time grains if none specified
        if not time_grains:
            time_grains = [TimeGrain.DAY, TimeGrain.WEEK, TimeGrain.MONTH]

        return MetricDefinition(
            name=dbt_metric.name,
            label=dbt_metric.label,
            description=dbt_metric.description,
            model=dbt_metric.model or "",
            calculation_method=calc_method,
            expression=dbt_metric.expression,
            timestamp=dbt_metric.timestamp or "created_at",
            time_grains=time_grains,
            dimensions=dbt_metric.dimensions,
            filters=self._convert_filters(dbt_metric.filters),
            meta=dbt_metric.meta,
        )

    def _convert_filters(
        self, dbt_filters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert dbt filter format to semantic layer format."""
        filters = []
        for f in dbt_filters:
            filters.append({
                "field": f.get("field", ""),
                "operator": f.get("operator", "="),
                "value": f.get("value", ""),
            })
        return filters

    def sync_manifest(
        self, manifest: DbtManifest
    ) -> List[MetricDefinition]:
        """Sync all metrics from manifest."""
        synced = []
        for dbt_metric in manifest.metrics.values():
            try:
                metric = self.convert_dbt_metric(dbt_metric)
                synced.append(metric)
                logger.info(f"Synced metric: {metric.name}")
            except Exception as e:
                logger.error(
                    f"Failed to sync metric {dbt_metric.name}: {e}"
                )
        return synced


class DbtRunner:
    """Run dbt commands programmatically."""

    def __init__(
        self,
        project_path: str,
        profiles_dir: Optional[str] = None,
        target: Optional[str] = None,
    ):
        self.project_path = Path(project_path)
        self.profiles_dir = profiles_dir
        self.target = target

    def _build_command(
        self, command: str, *args: str, **kwargs: Any
    ) -> List[str]:
        """Build dbt command with arguments."""
        cmd = ["dbt", command]

        if self.profiles_dir:
            cmd.extend(["--profiles-dir", self.profiles_dir])
        if self.target:
            cmd.extend(["--target", self.target])

        cmd.extend(args)

        for key, value in kwargs.items():
            if value is True:
                cmd.append(f"--{key.replace('_', '-')}")
            elif value is not False and value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        return cmd

    def run(
        self,
        models: Optional[List[str]] = None,
        select: Optional[str] = None,
        exclude: Optional[str] = None,
        full_refresh: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run dbt run command."""
        cmd = self._build_command("run")

        if models:
            cmd.extend(["--models", " ".join(models)])
        if select:
            cmd.extend(["--select", select])
        if exclude:
            cmd.extend(["--exclude", exclude])
        if full_refresh:
            cmd.append("--full-refresh")

        return self._execute(cmd)

    def test(
        self,
        models: Optional[List[str]] = None,
        select: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Run dbt test command."""
        cmd = self._build_command("test")

        if models:
            cmd.extend(["--models", " ".join(models)])
        if select:
            cmd.extend(["--select", select])

        return self._execute(cmd)

    def compile(self) -> subprocess.CompletedProcess:
        """Run dbt compile command."""
        cmd = self._build_command("compile")
        return self._execute(cmd)

    def deps(self) -> subprocess.CompletedProcess:
        """Run dbt deps command."""
        cmd = self._build_command("deps")
        return self._execute(cmd)

    def seed(
        self,
        select: Optional[str] = None,
        full_refresh: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run dbt seed command."""
        cmd = self._build_command("seed")

        if select:
            cmd.extend(["--select", select])
        if full_refresh:
            cmd.append("--full-refresh")

        return self._execute(cmd)

    def snapshot(
        self,
        select: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Run dbt snapshot command."""
        cmd = self._build_command("snapshot")

        if select:
            cmd.extend(["--select", select])

        return self._execute(cmd)

    def source_freshness(
        self,
        select: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Run dbt source freshness command."""
        cmd = self._build_command("source", "freshness")

        if select:
            cmd.extend(["--select", select])

        return self._execute(cmd)

    def parse(self) -> subprocess.CompletedProcess:
        """Run dbt parse command."""
        cmd = self._build_command("parse")
        return self._execute(cmd)

    def ls(
        self,
        resource_type: Optional[str] = None,
        select: Optional[str] = None,
        output: str = "name",
    ) -> subprocess.CompletedProcess:
        """Run dbt ls command."""
        cmd = self._build_command("ls", output=output)

        if resource_type:
            cmd.extend(["--resource-type", resource_type])
        if select:
            cmd.extend(["--select", select])

        return self._execute(cmd)

    def _execute(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Execute a dbt command."""
        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=self.project_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"dbt command failed: {result.stderr}")
        else:
            logger.debug(f"dbt output: {result.stdout}")

        return result


@dataclass
class DbtProjectConfig:
    """Configuration for dbt project integration."""

    project_path: str
    profiles_dir: Optional[str] = None
    target: Optional[str] = None
    auto_compile: bool = True
    sync_on_change: bool = True


class DbtIntegration:
    """Main integration class for dbt with semantic layer."""

    def __init__(self, config: DbtProjectConfig):
        self.config = config
        self.manifest_parser = ManifestParser(config.project_path)
        self.yaml_parser = YamlParser(config.project_path)
        self.metric_syncer = MetricSyncer()
        self.runner = DbtRunner(
            config.project_path,
            config.profiles_dir,
            config.target,
        )
        self._manifest: Optional[DbtManifest] = None

    @property
    def manifest(self) -> DbtManifest:
        """Get cached or load manifest."""
        if self._manifest is None:
            self.load_manifest()
        return self._manifest

    def load_manifest(self, force_compile: bool = False) -> DbtManifest:
        """Load dbt manifest, optionally compiling first."""
        manifest_path = Path(self.config.project_path) / "target" / "manifest.json"

        if force_compile or (
            self.config.auto_compile and not manifest_path.exists()
        ):
            logger.info("Compiling dbt project...")
            result = self.runner.compile()
            if result.returncode != 0:
                raise RuntimeError(f"dbt compile failed: {result.stderr}")

        self._manifest = self.manifest_parser.parse()
        logger.info(
            f"Loaded manifest with {len(self._manifest.models)} models, "
            f"{len(self._manifest.metrics)} metrics"
        )
        return self._manifest

    def sync_metrics(self) -> List[MetricDefinition]:
        """Sync all metrics from dbt to semantic layer."""
        return self.metric_syncer.sync_manifest(self.manifest)

    def get_model_lineage(
        self, model_name: str
    ) -> Dict[str, List[str]]:
        """Get upstream and downstream lineage for a model."""
        model = self.manifest.get_model_by_name(model_name)
        if not model:
            raise ValueError(f"Model not found: {model_name}")

        return {
            "upstream": self.manifest.get_upstream(model.unique_id),
            "downstream": self.manifest.get_downstream(model.unique_id),
        }

    def get_metric_lineage(
        self, metric_name: str
    ) -> Dict[str, Any]:
        """Get lineage for a metric including source models."""
        for metric in self.manifest.metrics.values():
            if metric.name == metric_name:
                return {
                    "metric": metric_name,
                    "model": metric.model,
                    "dimensions": metric.dimensions,
                    "upstream": (
                        self.manifest.get_upstream(metric.model)
                        if metric.model else []
                    ),
                }
        raise ValueError(f"Metric not found: {metric_name}")

    def validate_project(self) -> Dict[str, Any]:
        """Validate dbt project structure and metrics."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics_count": 0,
            "models_count": 0,
        }

        try:
            manifest = self.load_manifest()
            result["models_count"] = len(manifest.models)
            result["metrics_count"] = len(manifest.metrics)

            # Validate metrics
            for metric in manifest.metrics.values():
                if not metric.description:
                    result["warnings"].append(
                        f"Metric '{metric.name}' has no description"
                    )
                if not metric.model:
                    result["warnings"].append(
                        f"Metric '{metric.name}' has no model reference"
                    )

            # Validate models
            for model in manifest.models.values():
                if not model.description:
                    result["warnings"].append(
                        f"Model '{model.name}' has no description"
                    )
                if not model.columns:
                    result["warnings"].append(
                        f"Model '{model.name}' has no column definitions"
                    )

        except Exception as e:
            result["valid"] = False
            result["errors"].append(str(e))

        return result

    def run_model(
        self,
        model_name: str,
        full_refresh: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run a specific dbt model."""
        return self.runner.run(models=[model_name], full_refresh=full_refresh)

    def test_model(self, model_name: str) -> subprocess.CompletedProcess:
        """Test a specific dbt model."""
        return self.runner.test(models=[model_name])

    def refresh_sources(self) -> subprocess.CompletedProcess:
        """Check source freshness."""
        return self.runner.source_freshness()


def create_dbt_integration(
    project_path: str,
    profiles_dir: Optional[str] = None,
    target: Optional[str] = None,
) -> DbtIntegration:
    """Factory function to create dbt integration."""
    config = DbtProjectConfig(
        project_path=project_path,
        profiles_dir=profiles_dir,
        target=target,
    )
    return DbtIntegration(config)
