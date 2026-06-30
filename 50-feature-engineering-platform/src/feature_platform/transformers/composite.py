"""Composite feature transformers for chaining and combining transformations."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from feature_platform.transformers.base import BaseTransformer, TransformerState


class Pipeline(BaseTransformer):
    """
    Chain multiple transformers in sequence.

    Each transformer's output becomes the next transformer's input.

    Parameters:
        steps: List of (name, transformer) tuples
    """

    def __init__(
        self,
        steps: List[Tuple[str, BaseTransformer]],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self) -> None:
        """Validate that steps are properly formatted."""
        names = set()
        for step in self.steps:
            if len(step) != 2:
                raise ValueError(
                    "Each step must be a (name, transformer) tuple"
                )
            name, transformer = step
            if not isinstance(name, str):
                raise ValueError("Step name must be a string")
            if not isinstance(transformer, BaseTransformer):
                raise ValueError("Step transformer must be a BaseTransformer")
            if name in names:
                raise ValueError(f"Duplicate step name: {name}")
            names.add(name)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "Pipeline":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self._input_columns = columns or [f"x{i}" for i in range(X.shape[1])]

        current_X = X
        current_columns = self._input_columns

        for name, transformer in self.steps[:-1]:
            transformer.fit(current_X, y, current_columns)
            current_X = transformer.transform(current_X)
            current_columns = transformer.get_feature_names_out(current_columns)

        # Fit last transformer without transforming
        if self.steps:
            name, transformer = self.steps[-1]
            transformer.fit(current_X, y, current_columns)
            self._output_columns = transformer.get_feature_names_out(current_columns)
        else:
            self._output_columns = self._input_columns.copy()

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        current_X = X
        for name, transformer in self.steps:
            current_X = transformer.transform(current_X)

        return current_X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        current_X = X
        for name, transformer in reversed(self.steps):
            current_X = transformer.inverse_transform(current_X)

        return current_X

    def get_step(self, name: str) -> BaseTransformer:
        """Get a transformer by name."""
        for step_name, transformer in self.steps:
            if step_name == name:
                return transformer
        raise KeyError(f"Step not found: {name}")

    def _get_parameters(self) -> Dict[str, Any]:
        return {"steps": [(name, type(t).__name__) for name, t in self.steps]}


class FeatureUnion(BaseTransformer):
    """
    Concatenate results of multiple transformers.

    Parameters:
        transformers: List of (name, transformer) tuples
        n_jobs: Number of parallel jobs (not implemented)
    """

    def __init__(
        self,
        transformers: List[Tuple[str, BaseTransformer]],
        n_jobs: int = 1,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.transformers = transformers
        self.n_jobs = n_jobs
        self._validate_transformers()

    def _validate_transformers(self) -> None:
        """Validate that transformers are properly formatted."""
        names = set()
        for item in self.transformers:
            if len(item) != 2:
                raise ValueError(
                    "Each item must be a (name, transformer) tuple"
                )
            name, transformer = item
            if not isinstance(name, str):
                raise ValueError("Transformer name must be a string")
            if not isinstance(transformer, BaseTransformer):
                raise ValueError("Must be a BaseTransformer")
            if name in names:
                raise ValueError(f"Duplicate transformer name: {name}")
            names.add(name)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "FeatureUnion":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self._input_columns = columns or [f"x{i}" for i in range(X.shape[1])]

        self._output_columns = []
        for name, transformer in self.transformers:
            transformer.fit(X, y, self._input_columns)
            out_cols = transformer.get_feature_names_out(self._input_columns)
            self._output_columns.extend([f"{name}__{col}" for col in out_cols])

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        results = []
        for name, transformer in self.transformers:
            result = transformer.transform(X)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            results.append(result)

        return np.hstack(results)

    def get_transformer(self, name: str) -> BaseTransformer:
        """Get a transformer by name."""
        for t_name, transformer in self.transformers:
            if t_name == name:
                return transformer
        raise KeyError(f"Transformer not found: {name}")

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "transformers": [(name, type(t).__name__) for name, t in self.transformers],
            "n_jobs": self.n_jobs,
        }


class ColumnTransformer(BaseTransformer):
    """
    Apply different transformers to different columns.

    Parameters:
        transformers: List of (name, transformer, columns) tuples
            columns can be:
            - List of column indices (int)
            - List of column names (str)
            - A callable that takes column names and returns indices
        remainder: What to do with remaining columns ('drop', 'passthrough')
    """

    def __init__(
        self,
        transformers: List[Tuple[str, BaseTransformer, Union[List[int], List[str]]]],
        remainder: str = "drop",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.transformers = transformers
        self.remainder = remainder
        self._column_indices: Dict[str, List[int]] = {}
        self._remainder_indices: List[int] = []

    def _get_column_indices(
        self,
        columns: Union[List[int], List[str]],
        all_columns: List[str],
    ) -> List[int]:
        """Convert column specification to indices."""
        if not columns:
            return []

        if isinstance(columns[0], int):
            return list(columns)
        else:
            # Column names
            col_map = {name: idx for idx, name in enumerate(all_columns)}
            return [col_map[name] for name in columns if name in col_map]

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "ColumnTransformer":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        n_cols = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(n_cols)]

        # Track which columns are used
        used_columns = set()
        self._output_columns = []

        for name, transformer, cols in self.transformers:
            indices = self._get_column_indices(cols, self._input_columns)
            self._column_indices[name] = indices
            used_columns.update(indices)

            # Subset the data
            X_subset = X[:, indices]
            col_names = [self._input_columns[i] for i in indices]

            # Fit transformer
            transformer.fit(X_subset, y, col_names)
            out_cols = transformer.get_feature_names_out(col_names)
            self._output_columns.extend([f"{name}__{col}" for col in out_cols])

        # Handle remainder
        self._remainder_indices = [i for i in range(n_cols) if i not in used_columns]
        if self.remainder == "passthrough" and self._remainder_indices:
            for i in self._remainder_indices:
                self._output_columns.append(f"remainder__{self._input_columns[i]}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        results = []

        for name, transformer, _ in self.transformers:
            indices = self._column_indices[name]
            X_subset = X[:, indices]
            result = transformer.transform(X_subset)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            results.append(result)

        # Handle remainder
        if self.remainder == "passthrough" and self._remainder_indices:
            results.append(X[:, self._remainder_indices])

        if results:
            return np.hstack(results)
        else:
            return np.zeros((X.shape[0], 0))

    def get_transformer(self, name: str) -> BaseTransformer:
        """Get a transformer by name."""
        for t_name, transformer, _ in self.transformers:
            if t_name == name:
                return transformer
        raise KeyError(f"Transformer not found: {name}")

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "transformers": [
                (name, type(t).__name__, cols)
                for name, t, cols in self.transformers
            ],
            "remainder": self.remainder,
        }


class SequentialTransformer(BaseTransformer):
    """
    Apply transformers sequentially, passing data through each.

    Similar to Pipeline but with simpler interface.

    Parameters:
        transformers: List of transformers to apply in sequence
    """

    def __init__(
        self,
        transformers: List[BaseTransformer],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.transformers = transformers

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "SequentialTransformer":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self._input_columns = columns or [f"x{i}" for i in range(X.shape[1])]

        current_X = X
        current_columns = self._input_columns

        for transformer in self.transformers:
            transformer.fit(current_X, y, current_columns)
            current_X = transformer.transform(current_X)
            current_columns = transformer.get_feature_names_out(current_columns)

        self._output_columns = current_columns

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        current_X = X
        for transformer in self.transformers:
            current_X = transformer.transform(current_X)

        return current_X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        current_X = X
        for transformer in reversed(self.transformers):
            current_X = transformer.inverse_transform(current_X)

        return current_X

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "transformers": [type(t).__name__ for t in self.transformers]
        }


class SelectKBest(BaseTransformer):
    """
    Select top K features based on a scoring function.

    Parameters:
        k: Number of features to select
        score_func: Scoring function ('f_classif', 'f_regression', 'mutual_info')
    """

    def __init__(
        self,
        k: int = 10,
        score_func: str = "f_classif",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.k = k
        self.score_func = score_func
        self.scores_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def _f_classif(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """F-statistic for classification."""
        classes = np.unique(y)
        n_features = X.shape[1]
        scores = np.zeros(n_features)

        for i in range(n_features):
            class_means = [X[y == c, i].mean() for c in classes]
            overall_mean = X[:, i].mean()

            # Between-class variance
            ssb = sum(
                np.sum(y == c) * (cm - overall_mean) ** 2
                for c, cm in zip(classes, class_means)
            )

            # Within-class variance
            ssw = sum(
                np.sum((X[y == c, i] - cm) ** 2)
                for c, cm in zip(classes, class_means)
            )

            if ssw > 0:
                scores[i] = ssb / ssw * (len(y) - len(classes)) / (len(classes) - 1)
            else:
                scores[i] = 0

        return scores

    def _f_regression(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """F-statistic for regression."""
        n_features = X.shape[1]
        scores = np.zeros(n_features)

        y_mean = y.mean()
        ss_tot = np.sum((y - y_mean) ** 2)

        for i in range(n_features):
            # Simple linear regression
            x = X[:, i]
            x_mean = x.mean()

            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            intercept = y_mean - slope * x_mean

            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)

            if ss_res > 0:
                scores[i] = (ss_tot - ss_res) / ss_res * (len(y) - 2)
            else:
                scores[i] = float('inf')

        return scores

    def _mutual_info(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mutual information estimation."""
        n_features = X.shape[1]
        scores = np.zeros(n_features)

        # Discretize continuous features
        n_bins = min(10, len(np.unique(y)))

        for i in range(n_features):
            x = X[:, i]
            x_binned = np.digitize(x, np.percentile(x, np.linspace(0, 100, n_bins + 1)[1:-1]))

            # Compute joint and marginal probabilities
            joint = np.zeros((n_bins, n_bins))
            for xi, yi in zip(x_binned, y):
                yi_idx = int(yi) % n_bins
                joint[xi % n_bins, yi_idx] += 1
            joint /= len(y)

            px = joint.sum(axis=1)
            py = joint.sum(axis=0)

            # Compute MI
            mi = 0
            for xi in range(n_bins):
                for yi in range(n_bins):
                    if joint[xi, yi] > 0 and px[xi] > 0 and py[yi] > 0:
                        mi += joint[xi, yi] * np.log(joint[xi, yi] / (px[xi] * py[yi]))

            scores[i] = mi

        return scores

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "SelectKBest":
        if y is None:
            raise ValueError("SelectKBest requires target variable y")

        X = self._validate_input(X)
        X = self._ensure_2d(X)
        y = np.asarray(y).ravel()

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Compute scores
        if self.score_func == "f_classif":
            self.scores_ = self._f_classif(X, y)
        elif self.score_func == "f_regression":
            self.scores_ = self._f_regression(X, y)
        elif self.score_func == "mutual_info":
            self.scores_ = self._mutual_info(X, y)
        else:
            raise ValueError(f"Unknown score function: {self.score_func}")

        # Select top K
        k = min(self.k, self.n_features_)
        self.selected_indices_ = np.argsort(self.scores_)[-k:][::-1]

        self._output_columns = [self._input_columns[i] for i in self.selected_indices_]

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)
        return X[:, self.selected_indices_]

    def _get_parameters(self) -> Dict[str, Any]:
        return {"k": self.k, "score_func": self.score_func}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "scores": self.scores_.tolist() if self.scores_ is not None else None,
            "selected_indices": self.selected_indices_.tolist() if self.selected_indices_ is not None else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.scores_ = np.array(stats["scores"]) if stats.get("scores") else None
        self.selected_indices_ = np.array(stats["selected_indices"]) if stats.get("selected_indices") else None
        self.n_features_ = stats.get("n_features", 0)
