"""Categorical feature transformers."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

from feature_platform.transformers.base import BaseTransformer


class LabelEncoder(BaseTransformer):
    """
    Encode categorical labels as integers.

    Parameters:
        handle_unknown: How to handle unknown categories ('error', 'use_default')
        default_value: Default value for unknown categories when handle_unknown='use_default'
    """

    def __init__(
        self,
        handle_unknown: str = "error",
        default_value: int = -1,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.handle_unknown = handle_unknown
        self.default_value = default_value
        self.classes_: Optional[Dict[int, np.ndarray]] = None
        self.class_to_idx_: Optional[Dict[int, Dict[Any, int]]] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "LabelEncoder":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"{col}_encoded" for col in self._input_columns]

        self.classes_ = {}
        self.class_to_idx_ = {}

        for i in range(self.n_features_):
            unique_values = np.unique(X[:, i].astype(str))
            self.classes_[i] = unique_values
            self.class_to_idx_[i] = {val: idx for idx, val in enumerate(unique_values)}

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.zeros((X.shape[0], self.n_features_), dtype=np.int64)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                val = str(X[j, i])
                if val in self.class_to_idx_[i]:
                    result[j, i] = self.class_to_idx_[i][val]
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {val}")
                else:
                    result[j, i] = self.default_value

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.empty((X.shape[0], self.n_features_), dtype=object)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                idx = int(X[j, i])
                if 0 <= idx < len(self.classes_[i]):
                    result[j, i] = self.classes_[i][idx]
                else:
                    result[j, i] = None

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "handle_unknown": self.handle_unknown,
            "default_value": self.default_value,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "classes": {str(k): v.tolist() for k, v in self.classes_.items()} if self.classes_ else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        if stats.get("classes"):
            self.classes_ = {int(k): np.array(v) for k, v in stats["classes"].items()}
            self.class_to_idx_ = {
                int(k): {val: idx for idx, val in enumerate(v)}
                for k, v in stats["classes"].items()
            }
        self.n_features_ = stats.get("n_features", 0)


class OneHotEncoder(BaseTransformer):
    """
    Encode categorical features as one-hot vectors.

    Parameters:
        handle_unknown: How to handle unknown categories ('error', 'ignore')
        drop: Strategy for dropping one category ('first', 'none', or specific category)
        sparse: If True, return sparse matrix (not implemented, always returns dense)
    """

    def __init__(
        self,
        handle_unknown: str = "error",
        drop: Optional[str] = None,
        sparse: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.sparse = sparse
        self.categories_: Optional[Dict[int, np.ndarray]] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "OneHotEncoder":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        self.categories_ = {}
        self._output_columns = []

        for i in range(self.n_features_):
            unique_values = np.unique(X[:, i].astype(str))
            self.categories_[i] = unique_values

            # Generate column names
            col_name = self._input_columns[i]
            start_idx = 1 if self.drop == "first" else 0
            for val in unique_values[start_idx:]:
                self._output_columns.append(f"{col_name}_{val}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        n_output_cols = len(self._output_columns)
        result = np.zeros((X.shape[0], n_output_cols), dtype=np.float64)

        col_offset = 0
        for i in range(self.n_features_):
            categories = self.categories_[i]
            start_idx = 1 if self.drop == "first" else 0
            n_cats = len(categories) - start_idx

            for j in range(X.shape[0]):
                val = str(X[j, i])
                if val in categories:
                    cat_idx = np.where(categories == val)[0][0]
                    if cat_idx >= start_idx:
                        result[j, col_offset + cat_idx - start_idx] = 1.0
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {val}")
                # For 'ignore', all zeros (which is already the case)

            col_offset += n_cats

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.empty((X.shape[0], self.n_features_), dtype=object)

        col_offset = 0
        for i in range(self.n_features_):
            categories = self.categories_[i]
            start_idx = 1 if self.drop == "first" else 0
            n_cats = len(categories) - start_idx

            for j in range(X.shape[0]):
                cat_slice = X[j, col_offset:col_offset + n_cats]
                if np.any(cat_slice == 1):
                    cat_idx = np.argmax(cat_slice) + start_idx
                    result[j, i] = categories[cat_idx]
                elif self.drop == "first":
                    result[j, i] = categories[0]
                else:
                    result[j, i] = None

            col_offset += n_cats

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "handle_unknown": self.handle_unknown,
            "drop": self.drop,
            "sparse": self.sparse,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "categories": {str(k): v.tolist() for k, v in self.categories_.items()} if self.categories_ else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        if stats.get("categories"):
            self.categories_ = {int(k): np.array(v) for k, v in stats["categories"].items()}
        self.n_features_ = stats.get("n_features", 0)


class OrdinalEncoder(BaseTransformer):
    """
    Encode categorical features as ordinal integers based on specified order.

    Parameters:
        categories: Dict mapping column index to ordered list of categories
        handle_unknown: How to handle unknown categories ('error', 'use_default')
        default_value: Default value for unknown categories
    """

    def __init__(
        self,
        categories: Optional[Dict[int, List[str]]] = None,
        handle_unknown: str = "error",
        default_value: int = -1,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.default_value = default_value
        self.categories_: Optional[Dict[int, np.ndarray]] = None
        self.category_to_idx_: Optional[Dict[int, Dict[str, int]]] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "OrdinalEncoder":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"{col}_ordinal" for col in self._input_columns]

        self.categories_ = {}
        self.category_to_idx_ = {}

        for i in range(self.n_features_):
            if self.categories and i in self.categories:
                # Use provided order
                cats = np.array(self.categories[i])
            else:
                # Default to sorted order
                cats = np.sort(np.unique(X[:, i].astype(str)))

            self.categories_[i] = cats
            self.category_to_idx_[i] = {cat: idx for idx, cat in enumerate(cats)}

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.zeros((X.shape[0], self.n_features_), dtype=np.int64)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                val = str(X[j, i])
                if val in self.category_to_idx_[i]:
                    result[j, i] = self.category_to_idx_[i][val]
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {val}")
                else:
                    result[j, i] = self.default_value

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.empty((X.shape[0], self.n_features_), dtype=object)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                idx = int(X[j, i])
                if 0 <= idx < len(self.categories_[i]):
                    result[j, i] = self.categories_[i][idx]
                else:
                    result[j, i] = None

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "handle_unknown": self.handle_unknown,
            "default_value": self.default_value,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "categories": {str(k): v.tolist() for k, v in self.categories_.items()} if self.categories_ else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        if stats.get("categories"):
            self.categories_ = {int(k): np.array(v) for k, v in stats["categories"].items()}
            self.category_to_idx_ = {
                int(k): {cat: idx for idx, cat in enumerate(v)}
                for k, v in stats["categories"].items()
            }
        self.n_features_ = stats.get("n_features", 0)


class TargetEncoder(BaseTransformer):
    """
    Encode categorical features using target variable statistics.

    Parameters:
        smoothing: Smoothing factor for regularization
        min_samples: Minimum samples for a category to use its mean
    """

    def __init__(
        self,
        smoothing: float = 1.0,
        min_samples: int = 1,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.encodings_: Optional[Dict[int, Dict[str, float]]] = None
        self.global_mean_: Optional[float] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "TargetEncoder":
        if y is None:
            raise ValueError("TargetEncoder requires target variable y")

        X = self._validate_input(X)
        X = self._ensure_2d(X)
        y = np.asarray(y).ravel()

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"{col}_target" for col in self._input_columns]

        self.global_mean_ = np.nanmean(y)
        self.encodings_ = {}

        for i in range(self.n_features_):
            self.encodings_[i] = {}
            categories = np.unique(X[:, i].astype(str))

            for cat in categories:
                mask = X[:, i].astype(str) == cat
                n_samples = np.sum(mask)
                cat_mean = np.nanmean(y[mask])

                if n_samples >= self.min_samples:
                    # Apply smoothing
                    weight = n_samples / (n_samples + self.smoothing)
                    encoding = weight * cat_mean + (1 - weight) * self.global_mean_
                else:
                    encoding = self.global_mean_

                self.encodings_[i][cat] = encoding

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.zeros((X.shape[0], self.n_features_), dtype=np.float64)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                val = str(X[j, i])
                result[j, i] = self.encodings_[i].get(val, self.global_mean_)

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "smoothing": self.smoothing,
            "min_samples": self.min_samples,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "encodings": {str(k): v for k, v in self.encodings_.items()} if self.encodings_ else None,
            "global_mean": self.global_mean_,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        if stats.get("encodings"):
            self.encodings_ = {int(k): v for k, v in stats["encodings"].items()}
        self.global_mean_ = stats.get("global_mean")
        self.n_features_ = stats.get("n_features", 0)


class FrequencyEncoder(BaseTransformer):
    """
    Encode categorical features using their frequency in the training data.

    Parameters:
        normalize: If True, return normalized frequencies (0-1)
    """

    def __init__(
        self,
        normalize: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.normalize = normalize
        self.frequencies_: Optional[Dict[int, Dict[str, float]]] = None
        self.n_samples_: int = 0
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "FrequencyEncoder":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"{col}_freq" for col in self._input_columns]

        self.frequencies_ = {}

        for i in range(self.n_features_):
            self.frequencies_[i] = {}
            categories, counts = np.unique(X[:, i].astype(str), return_counts=True)

            for cat, count in zip(categories, counts):
                if self.normalize:
                    self.frequencies_[i][cat] = count / self.n_samples_
                else:
                    self.frequencies_[i][cat] = count

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.zeros((X.shape[0], self.n_features_), dtype=np.float64)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                val = str(X[j, i])
                result[j, i] = self.frequencies_[i].get(val, 0.0)

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"normalize": self.normalize}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "frequencies": {str(k): v for k, v in self.frequencies_.items()} if self.frequencies_ else None,
            "n_samples": self.n_samples_,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        if stats.get("frequencies"):
            self.frequencies_ = {int(k): v for k, v in stats["frequencies"].items()}
        self.n_samples_ = stats.get("n_samples", 0)
        self.n_features_ = stats.get("n_features", 0)


class BinaryEncoder(BaseTransformer):
    """
    Encode categorical features using binary representation.

    Each category is assigned an integer, which is then encoded as binary digits.

    Parameters:
        handle_unknown: How to handle unknown categories ('error', 'use_default')
    """

    def __init__(
        self,
        handle_unknown: str = "error",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.handle_unknown = handle_unknown
        self.categories_: Optional[Dict[int, np.ndarray]] = None
        self.n_bits_: Optional[Dict[int, int]] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "BinaryEncoder":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        self.categories_ = {}
        self.n_bits_ = {}
        self._output_columns = []

        for i in range(self.n_features_):
            categories = np.unique(X[:, i].astype(str))
            self.categories_[i] = categories

            # Calculate number of bits needed
            n_cats = len(categories)
            n_bits = max(1, int(np.ceil(np.log2(n_cats + 1))))  # +1 for unknown
            self.n_bits_[i] = n_bits

            # Generate column names
            col_name = self._input_columns[i]
            for b in range(n_bits):
                self._output_columns.append(f"{col_name}_bit_{b}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        n_output_cols = sum(self.n_bits_.values())
        result = np.zeros((X.shape[0], n_output_cols), dtype=np.float64)

        col_offset = 0
        for i in range(self.n_features_):
            n_bits = self.n_bits_[i]
            categories = self.categories_[i]

            for j in range(X.shape[0]):
                val = str(X[j, i])
                if val in categories:
                    cat_idx = np.where(categories == val)[0][0] + 1  # 1-indexed
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {val}")
                else:
                    cat_idx = 0  # All zeros for unknown

                # Convert to binary
                for b in range(n_bits):
                    result[j, col_offset + b] = (cat_idx >> b) & 1

            col_offset += n_bits

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"handle_unknown": self.handle_unknown}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "categories": {str(k): v.tolist() for k, v in self.categories_.items()} if self.categories_ else None,
            "n_bits": self.n_bits_,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        if stats.get("categories"):
            self.categories_ = {int(k): np.array(v) for k, v in stats["categories"].items()}
        if stats.get("n_bits"):
            self.n_bits_ = {int(k): v for k, v in stats["n_bits"].items()}
        self.n_features_ = stats.get("n_features", 0)


class HashingEncoder(BaseTransformer):
    """
    Encode categorical features using hashing trick.

    Maps categories to a fixed-size feature space using hashing.

    Parameters:
        n_features: Number of output features (hash buckets)
        hash_function: Hash function to use ('murmurhash', 'md5')
    """

    def __init__(
        self,
        n_features: int = 8,
        hash_function: str = "murmurhash",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.n_output_features = n_features
        self.hash_function = hash_function
        self.n_features_: int = 0

    def _hash(self, value: str) -> int:
        """Hash a string value."""
        if self.hash_function == "murmurhash":
            # Simple multiplicative hash
            h = 0
            for char in value:
                h = (h * 31 + ord(char)) & 0xFFFFFFFF
            return h
        elif self.hash_function == "md5":
            import hashlib
            return int(hashlib.md5(value.encode()).hexdigest(), 16)
        else:
            return hash(value)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "HashingEncoder":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Generate output column names
        self._output_columns = []
        for col in self._input_columns:
            for h in range(self.n_output_features):
                self._output_columns.append(f"{col}_hash_{h}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        n_output = self.n_features_ * self.n_output_features
        result = np.zeros((X.shape[0], n_output), dtype=np.float64)

        for i in range(self.n_features_):
            offset = i * self.n_output_features
            for j in range(X.shape[0]):
                val = str(X[j, i])
                h = self._hash(val) % self.n_output_features
                # Use signed hash for alternating +1/-1
                sign = 1 if (self._hash(val + "_sign") % 2) == 0 else -1
                result[j, offset + h] += sign

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_output_features,
            "hash_function": self.hash_function,
        }


class ImputerCategorical(BaseTransformer):
    """
    Impute missing values in categorical features.

    Parameters:
        strategy: Imputation strategy ('most_frequent', 'constant')
        fill_value: Value to use for 'constant' strategy
    """

    def __init__(
        self,
        strategy: str = "most_frequent",
        fill_value: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.strategy = strategy
        self.fill_value = fill_value
        self.fill_values_: Optional[Dict[int, str]] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "ImputerCategorical":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = self._input_columns.copy()

        self.fill_values_ = {}

        for i in range(self.n_features_):
            if self.strategy == "most_frequent":
                # Filter out None/nan values
                col_data = X[:, i].astype(str)
                mask = (col_data != "None") & (col_data != "nan") & (col_data != "")
                valid_data = col_data[mask]

                if len(valid_data) > 0:
                    values, counts = np.unique(valid_data, return_counts=True)
                    self.fill_values_[i] = values[np.argmax(counts)]
                else:
                    self.fill_values_[i] = self.fill_value or "UNKNOWN"
            else:
                self.fill_values_[i] = self.fill_value or "UNKNOWN"

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)
        X = X.astype(object).copy()

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                val = str(X[j, i])
                if val in ("None", "nan", ""):
                    X[j, i] = self.fill_values_[i]

        return X

    def _get_parameters(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "fill_value": self.fill_value}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "fill_values": self.fill_values_,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        if stats.get("fill_values"):
            self.fill_values_ = {int(k): v for k, v in stats["fill_values"].items()}
        self.n_features_ = stats.get("n_features", 0)
