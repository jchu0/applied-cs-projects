"""Base transformer classes for Feature Engineering Platform."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import json
import pickle
import numpy as np


@dataclass
class TransformerState:
    """State of a fitted transformer."""

    name: str
    transformer_type: str
    parameters: Dict[str, Any]
    fitted_at: datetime
    input_columns: List[str]
    output_columns: List[str]
    statistics: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "name": self.name,
            "transformer_type": self.transformer_type,
            "parameters": self.parameters,
            "fitted_at": self.fitted_at.isoformat(),
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "statistics": self.statistics,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformerState":
        """Create state from dictionary."""
        data = data.copy()
        data["fitted_at"] = datetime.fromisoformat(data["fitted_at"])
        return cls(**data)

    def to_json(self) -> str:
        """Serialize state to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "TransformerState":
        """Deserialize state from JSON."""
        return cls.from_dict(json.loads(json_str))


class TransformerMixin:
    """Mixin class providing common transformer utilities."""

    def _validate_input(
        self,
        X: np.ndarray,
        require_fitted: bool = False,
    ) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if hasattr(X, "values"):  # pandas DataFrame/Series
            X = X.values
        X = np.asarray(X)

        if require_fitted and not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} has not been fitted")

        return X

    def _get_output_columns(
        self,
        input_columns: List[str],
        suffix: str = "",
    ) -> List[str]:
        """Generate output column names."""
        if suffix:
            return [f"{col}_{suffix}" for col in input_columns]
        return input_columns.copy()

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        """Ensure array is 2D."""
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    def _ensure_1d(self, X: np.ndarray) -> np.ndarray:
        """Ensure array is 1D."""
        if X.ndim == 2 and X.shape[1] == 1:
            return X.ravel()
        return X


class BaseTransformer(ABC, TransformerMixin):
    """
    Base class for all feature transformers.

    All transformers must implement:
    - fit(X, y=None): Learn parameters from training data
    - transform(X): Apply the transformation
    - fit_transform(X, y=None): Fit and transform in one step

    Optional methods:
    - inverse_transform(X): Reverse the transformation
    - get_feature_names_out(): Get output feature names
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        self._input_columns: List[str] = []
        self._output_columns: List[str] = []
        self._fitted_at: Optional[datetime] = None

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "BaseTransformer":
        """
        Fit the transformer to training data.

        Parameters:
            X: Input data array
            y: Target array (optional, for supervised transformers)
            columns: Column names for the input data

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data.

        Parameters:
            X: Input data array

        Returns:
            Transformed array
        """
        pass

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters:
            X: Input data array
            y: Target array (optional)
            columns: Column names

        Returns:
            Transformed array
        """
        return self.fit(X, y, columns).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the transformation.

        Parameters:
            X: Transformed data array

        Returns:
            Original data array

        Raises:
            NotImplementedError: If inverse transform is not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )

    def get_feature_names_out(
        self,
        input_features: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get output feature names.

        Parameters:
            input_features: Input feature names

        Returns:
            Output feature names
        """
        if self._output_columns:
            return self._output_columns
        if input_features:
            return input_features
        return self._input_columns

    def get_state(self) -> TransformerState:
        """Get the current state of the transformer."""
        if not self.is_fitted:
            raise ValueError("Transformer has not been fitted")

        return TransformerState(
            name=self.name,
            transformer_type=self.__class__.__name__,
            parameters=self._get_parameters(),
            fitted_at=self._fitted_at,
            input_columns=self._input_columns,
            output_columns=self._output_columns,
            statistics=self._get_statistics(),
        )

    def set_state(self, state: TransformerState) -> "BaseTransformer":
        """
        Restore transformer from saved state.

        Parameters:
            state: Saved transformer state

        Returns:
            self
        """
        self._set_parameters(state.parameters)
        self._set_statistics(state.statistics)
        self._input_columns = state.input_columns
        self._output_columns = state.output_columns
        self._fitted_at = state.fitted_at
        self.is_fitted = True
        return self

    def _get_parameters(self) -> Dict[str, Any]:
        """Get transformer parameters for serialization."""
        return {}

    def _set_parameters(self, params: Dict[str, Any]) -> None:
        """Set transformer parameters from deserialization."""
        pass

    def _get_statistics(self) -> Dict[str, Any]:
        """Get learned statistics for serialization."""
        return {}

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        """Set learned statistics from deserialization."""
        pass

    def save(self, path: str) -> None:
        """
        Save transformer to file.

        Parameters:
            path: File path to save to
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseTransformer":
        """
        Load transformer from file.

        Parameters:
            path: File path to load from

        Returns:
            Loaded transformer
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        params = self._get_parameters()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        fitted_str = " (fitted)" if self.is_fitted else ""
        return f"{self.__class__.__name__}({param_str}){fitted_str}"


class IdentityTransformer(BaseTransformer):
    """Transformer that returns input unchanged."""

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "IdentityTransformer":
        X = self._validate_input(X)
        self._input_columns = columns or [f"x{i}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
        self._output_columns = self._input_columns.copy()
        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._validate_input(X, require_fitted=True)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self._validate_input(X, require_fitted=True)


class FunctionTransformer(BaseTransformer):
    """Transformer that applies a custom function."""

    def __init__(
        self,
        func: callable,
        inverse_func: Optional[callable] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.func = func
        self.inverse_func = inverse_func

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "FunctionTransformer":
        X = self._validate_input(X)
        self._input_columns = columns or [f"x{i}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
        self._output_columns = self._input_columns.copy()
        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        return self.func(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.inverse_func is None:
            raise NotImplementedError("No inverse function provided")
        X = self._validate_input(X, require_fitted=True)
        return self.inverse_func(X)
