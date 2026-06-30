"""Pipeline executor for feature transformations."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import traceback

from feature_platform.transformers.base import BaseTransformer
from feature_platform.pipeline.dag import DAG, DAGNode, DAGExecutor


class ExecutionStatus(Enum):
    """Status of pipeline execution."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of a pipeline execution."""

    status: ExecutionStatus
    data: Any = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "metrics": self.metrics,
        }


@dataclass
class PipelineStep:
    """A step in the feature pipeline."""

    name: str
    transformer: Union[BaseTransformer, Callable]
    input_columns: Optional[List[str]] = None
    output_columns: Optional[List[str]] = None
    config: Dict[str, Any] = field(default_factory=dict)


class PipelineExecutor:
    """
    Executor for feature transformation pipelines.

    Provides:
    - Sequential step execution
    - Fit and transform operations
    - Checkpointing
    - Progress tracking
    """

    def __init__(
        self,
        steps: Optional[List[PipelineStep]] = None,
        name: str = "feature_pipeline",
        checkpoint_path: Optional[str] = None,
    ):
        self.steps = steps or []
        self.name = name
        self.checkpoint_path = checkpoint_path
        self._fitted = False
        self._current_step = 0
        self._execution_history: List[ExecutionResult] = []

    def add_step(
        self,
        name: str,
        transformer: Union[BaseTransformer, Callable],
        input_columns: Optional[List[str]] = None,
        output_columns: Optional[List[str]] = None,
        **config,
    ) -> "PipelineExecutor":
        """Add a step to the pipeline."""
        step = PipelineStep(
            name=name,
            transformer=transformer,
            input_columns=input_columns,
            output_columns=output_columns,
            config=config,
        )
        self.steps.append(step)
        return self

    def fit(
        self,
        data: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "PipelineExecutor":
        """
        Fit all transformers in the pipeline.

        Parameters:
            data: Input data array
            y: Target array (optional)
            columns: Column names

        Returns:
            self
        """
        result = ExecutionResult(
            status=ExecutionStatus.RUNNING,
            start_time=datetime.utcnow(),
        )

        try:
            current_data = data
            current_columns = columns

            for i, step in enumerate(self.steps):
                self._current_step = i

                if isinstance(step.transformer, BaseTransformer):
                    # Select columns if specified
                    if step.input_columns and current_columns:
                        col_indices = [
                            current_columns.index(c) for c in step.input_columns
                            if c in current_columns
                        ]
                        step_data = current_data[:, col_indices]
                        step_columns = step.input_columns
                    else:
                        step_data = current_data
                        step_columns = current_columns

                    # Fit transformer
                    step.transformer.fit(step_data, y, step_columns)

                    # Transform for next step
                    transformed = step.transformer.transform(step_data)
                    output_columns = step.transformer.get_feature_names_out(step_columns)

                    # Update data for next step
                    if step.input_columns and current_columns:
                        # Replace original columns with transformed
                        current_data = self._replace_columns(
                            current_data, current_columns,
                            step.input_columns, transformed, output_columns
                        )
                        current_columns = self._update_column_names(
                            current_columns, step.input_columns, output_columns
                        )
                    else:
                        current_data = transformed
                        current_columns = output_columns

                result.step_results[step.name] = {
                    "status": "completed",
                    "output_shape": current_data.shape,
                }

            self._fitted = True
            result.status = ExecutionStatus.COMPLETED
            result.end_time = datetime.utcnow()
            result.data = current_data

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.error_traceback = traceback.format_exc()
            result.end_time = datetime.utcnow()

        self._execution_history.append(result)
        return self

    def transform(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """
        Transform data through the pipeline.

        Parameters:
            data: Input data array
            columns: Column names

        Returns:
            ExecutionResult with transformed data
        """
        if not self._fitted:
            raise ValueError("Pipeline has not been fitted")

        result = ExecutionResult(
            status=ExecutionStatus.RUNNING,
            start_time=datetime.utcnow(),
        )

        try:
            current_data = data
            current_columns = columns

            for step in self.steps:
                if isinstance(step.transformer, BaseTransformer):
                    # Select columns if specified
                    if step.input_columns and current_columns:
                        col_indices = [
                            current_columns.index(c) for c in step.input_columns
                            if c in current_columns
                        ]
                        step_data = current_data[:, col_indices]
                    else:
                        step_data = current_data

                    # Transform
                    transformed = step.transformer.transform(step_data)
                    output_columns = step.transformer.get_feature_names_out(
                        step.input_columns or current_columns
                    )

                    # Update data
                    if step.input_columns and current_columns:
                        current_data = self._replace_columns(
                            current_data, current_columns,
                            step.input_columns, transformed, output_columns
                        )
                        current_columns = self._update_column_names(
                            current_columns, step.input_columns, output_columns
                        )
                    else:
                        current_data = transformed
                        current_columns = output_columns

                elif callable(step.transformer):
                    # Custom function
                    current_data = step.transformer(current_data, **step.config)

                result.step_results[step.name] = {
                    "status": "completed",
                    "output_shape": current_data.shape,
                }

            result.status = ExecutionStatus.COMPLETED
            result.data = current_data
            result.end_time = datetime.utcnow()
            result.metrics["output_columns"] = current_columns

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.error_traceback = traceback.format_exc()
            result.end_time = datetime.utcnow()

        self._execution_history.append(result)
        return result

    def fit_transform(
        self,
        data: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """Fit and transform in one step."""
        self.fit(data, y, columns)
        return self.transform(data, columns)

    def _replace_columns(
        self,
        data: np.ndarray,
        all_columns: List[str],
        input_columns: List[str],
        new_data: np.ndarray,
        new_columns: List[str],
    ) -> np.ndarray:
        """Replace specific columns in the data array."""
        # Get indices of columns to keep
        keep_indices = [
            i for i, c in enumerate(all_columns)
            if c not in input_columns
        ]

        if not keep_indices:
            return new_data

        kept_data = data[:, keep_indices]
        return np.hstack([kept_data, new_data])

    def _update_column_names(
        self,
        all_columns: List[str],
        input_columns: List[str],
        new_columns: List[str],
    ) -> List[str]:
        """Update column names after transformation."""
        kept_columns = [c for c in all_columns if c not in input_columns]
        return kept_columns + list(new_columns)

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_transformer(self, name: str) -> Optional[BaseTransformer]:
        """Get a transformer by step name."""
        step = self.get_step(name)
        if step and isinstance(step.transformer, BaseTransformer):
            return step.transformer
        return None

    def get_execution_history(self) -> List[ExecutionResult]:
        """Get execution history."""
        return self._execution_history.copy()

    def get_last_result(self) -> Optional[ExecutionResult]:
        """Get the last execution result."""
        if self._execution_history:
            return self._execution_history[-1]
        return None

    def to_dag(self) -> DAG:
        """Convert pipeline to a DAG for parallel execution."""
        dag = DAG(name=self.name)

        previous_name = None
        for step in self.steps:
            def make_step_func(s):
                def step_func(**kwargs):
                    data = kwargs.get(previous_name if previous_name else "input", None)
                    if isinstance(s.transformer, BaseTransformer):
                        return s.transformer.transform(data)
                    elif callable(s.transformer):
                        return s.transformer(data, **s.config)
                    return data
                return step_func

            node = DAGNode(
                name=step.name,
                func=make_step_func(step),
                inputs=[previous_name] if previous_name else [],
                config=step.config,
            )
            dag.add_node(node)

            if previous_name:
                dag.add_edge(previous_name, step.name)

            previous_name = step.name

        return dag

    def save(self, path: str) -> None:
        """Save the fitted pipeline to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "PipelineExecutor":
        """Load a pipeline from disk."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


class FeaturePipelineBuilder:
    """
    Builder for creating feature pipelines.

    Provides a fluent interface for constructing pipelines.
    """

    def __init__(self, name: str = "feature_pipeline"):
        self.name = name
        self._steps: List[PipelineStep] = []

    def add_numeric_scaler(
        self,
        columns: List[str],
        method: str = "standard",
        name: Optional[str] = None,
    ) -> "FeaturePipelineBuilder":
        """Add a numeric scaling step."""
        from feature_platform.transformers.numeric import (
            StandardScaler, MinMaxScaler, RobustScaler
        )

        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }

        if method not in scalers:
            raise ValueError(f"Unknown scaler method: {method}")

        step = PipelineStep(
            name=name or f"{method}_scaler",
            transformer=scalers[method](),
            input_columns=columns,
        )
        self._steps.append(step)
        return self

    def add_categorical_encoder(
        self,
        columns: List[str],
        method: str = "onehot",
        name: Optional[str] = None,
    ) -> "FeaturePipelineBuilder":
        """Add a categorical encoding step."""
        from feature_platform.transformers.categorical import (
            OneHotEncoder, LabelEncoder, OrdinalEncoder
        )

        encoders = {
            "onehot": OneHotEncoder,
            "label": LabelEncoder,
            "ordinal": OrdinalEncoder,
        }

        if method not in encoders:
            raise ValueError(f"Unknown encoder method: {method}")

        step = PipelineStep(
            name=name or f"{method}_encoder",
            transformer=encoders[method](),
            input_columns=columns,
        )
        self._steps.append(step)
        return self

    def add_date_features(
        self,
        columns: List[str],
        parts: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> "FeaturePipelineBuilder":
        """Add date feature extraction step."""
        from feature_platform.transformers.temporal import DatePartsExtractor

        step = PipelineStep(
            name=name or "date_features",
            transformer=DatePartsExtractor(parts=parts),
            input_columns=columns,
        )
        self._steps.append(step)
        return self

    def add_text_vectorizer(
        self,
        columns: List[str],
        method: str = "tfidf",
        max_features: int = 100,
        name: Optional[str] = None,
    ) -> "FeaturePipelineBuilder":
        """Add text vectorization step."""
        from feature_platform.transformers.text import TfidfVectorizer, CountVectorizer

        vectorizers = {
            "tfidf": lambda: TfidfVectorizer(max_features=max_features),
            "count": lambda: CountVectorizer(max_features=max_features),
        }

        if method not in vectorizers:
            raise ValueError(f"Unknown vectorizer method: {method}")

        step = PipelineStep(
            name=name or f"{method}_vectorizer",
            transformer=vectorizers[method](),
            input_columns=columns,
        )
        self._steps.append(step)
        return self

    def add_custom_step(
        self,
        name: str,
        transformer: Union[BaseTransformer, Callable],
        input_columns: Optional[List[str]] = None,
        **config,
    ) -> "FeaturePipelineBuilder":
        """Add a custom transformation step."""
        step = PipelineStep(
            name=name,
            transformer=transformer,
            input_columns=input_columns,
            config=config,
        )
        self._steps.append(step)
        return self

    def build(self) -> PipelineExecutor:
        """Build the pipeline."""
        return PipelineExecutor(steps=self._steps, name=self.name)
