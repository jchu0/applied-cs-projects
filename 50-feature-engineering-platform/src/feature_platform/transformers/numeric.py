"""Numeric feature transformers."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np

from feature_platform.transformers.base import BaseTransformer


class StandardScaler(BaseTransformer):
    """
    Standardize features by removing the mean and scaling to unit variance.

    z = (x - mean) / std

    Parameters:
        with_mean: If True, center the data before scaling
        with_std: If True, scale to unit variance
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "StandardScaler":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = self._input_columns.copy()

        if self.with_mean:
            self.mean_ = np.nanmean(X, axis=0)
        else:
            self.mean_ = np.zeros(self.n_features_)

        if self.with_std:
            self.std_ = np.nanstd(X, axis=0)
            # Avoid division by zero
            self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        else:
            self.std_ = np.ones(self.n_features_)

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)
        return X * self.std_ + self.mean_

    def _get_parameters(self) -> Dict[str, Any]:
        return {"with_mean": self.with_mean, "with_std": self.with_std}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.mean_ = np.array(stats["mean"]) if stats.get("mean") else None
        self.std_ = np.array(stats["std"]) if stats.get("std") else None
        self.n_features_ = stats.get("n_features", 0)


class MinMaxScaler(BaseTransformer):
    """
    Scale features to a given range [min_val, max_val].

    X_scaled = (X - X.min) / (X.max - X.min) * (max_val - min_val) + min_val

    Parameters:
        feature_range: Desired range of transformed data
    """

    def __init__(
        self,
        feature_range: tuple = (0, 1),
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "MinMaxScaler":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = self._input_columns.copy()

        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)

        data_range = self.max_ - self.min_
        # Avoid division by zero
        data_range = np.where(data_range == 0, 1.0, data_range)

        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / data_range

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        feature_min, _ = self.feature_range
        return (X - self.min_) * self.scale_ + feature_min

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        feature_min, _ = self.feature_range
        return (X - feature_min) / self.scale_ + self.min_

    def _get_parameters(self) -> Dict[str, Any]:
        return {"feature_range": self.feature_range}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "min": self.min_.tolist() if self.min_ is not None else None,
            "max": self.max_.tolist() if self.max_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.min_ = np.array(stats["min"]) if stats.get("min") else None
        self.max_ = np.array(stats["max"]) if stats.get("max") else None
        self.scale_ = np.array(stats["scale"]) if stats.get("scale") else None
        self.n_features_ = stats.get("n_features", 0)


class RobustScaler(BaseTransformer):
    """
    Scale features using statistics that are robust to outliers.

    Uses the median and interquartile range (IQR).
    X_scaled = (X - median) / IQR

    Parameters:
        with_centering: If True, center the data before scaling
        with_scaling: If True, scale to interquartile range
        quantile_range: Quantile range used to calculate scale
    """

    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple = (25.0, 75.0),
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "RobustScaler":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = self._input_columns.copy()

        if self.with_centering:
            self.center_ = np.nanmedian(X, axis=0)
        else:
            self.center_ = np.zeros(self.n_features_)

        if self.with_scaling:
            q_min, q_max = self.quantile_range
            q_low = np.nanpercentile(X, q_min, axis=0)
            q_high = np.nanpercentile(X, q_max, axis=0)
            self.scale_ = q_high - q_low
            # Avoid division by zero
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.scale_ = np.ones(self.n_features_)

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)
        return (X - self.center_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)
        return X * self.scale_ + self.center_

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "with_centering": self.with_centering,
            "with_scaling": self.with_scaling,
            "quantile_range": self.quantile_range,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "center": self.center_.tolist() if self.center_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.center_ = np.array(stats["center"]) if stats.get("center") else None
        self.scale_ = np.array(stats["scale"]) if stats.get("scale") else None
        self.n_features_ = stats.get("n_features", 0)


class LogTransformer(BaseTransformer):
    """
    Apply logarithmic transformation to features.

    X_scaled = log(X + offset)

    Parameters:
        base: Logarithm base (e, 2, or 10)
        offset: Constant added before log to handle zeros
    """

    def __init__(
        self,
        base: str = "e",
        offset: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.base = base
        self.offset = offset
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "LogTransformer":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"{col}_log" for col in self._input_columns]

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        X_shifted = X + self.offset

        if self.base == "e":
            return np.log(X_shifted)
        elif self.base == "2":
            return np.log2(X_shifted)
        elif self.base == "10":
            return np.log10(X_shifted)
        else:
            return np.log(X_shifted) / np.log(float(self.base))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        if self.base == "e":
            return np.exp(X) - self.offset
        elif self.base == "2":
            return np.power(2, X) - self.offset
        elif self.base == "10":
            return np.power(10, X) - self.offset
        else:
            return np.power(float(self.base), X) - self.offset

    def _get_parameters(self) -> Dict[str, Any]:
        return {"base": self.base, "offset": self.offset}


class PowerTransformer(BaseTransformer):
    """
    Apply power transformation to make data more Gaussian-like.

    Supports Box-Cox and Yeo-Johnson transformations.

    Parameters:
        method: Transformation method ('box-cox' or 'yeo-johnson')
        standardize: If True, apply zero-mean, unit-variance normalization
    """

    def __init__(
        self,
        method: str = "yeo-johnson",
        standardize: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.method = method
        self.standardize = standardize
        self.lambdas_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """Apply Yeo-Johnson transformation."""
        result = np.zeros_like(x)
        pos = x >= 0
        neg = ~pos

        if np.abs(lmbda) < 1e-10:
            result[pos] = np.log1p(x[pos])
        else:
            result[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda

        if np.abs(lmbda - 2) < 1e-10:
            result[neg] = -np.log1p(-x[neg])
        else:
            result[neg] = -(np.power(-x[neg] + 1, 2 - lmbda) - 1) / (2 - lmbda)

        return result

    def _box_cox_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """Apply Box-Cox transformation."""
        if np.abs(lmbda) < 1e-10:
            return np.log(x)
        else:
            return (np.power(x, lmbda) - 1) / lmbda

    def _estimate_lambda(self, x: np.ndarray) -> float:
        """Estimate optimal lambda using maximum likelihood."""
        from scipy import stats

        # Simple grid search for lambda
        best_lambda = 0.0
        best_score = -np.inf

        for lmbda in np.linspace(-2, 2, 41):
            try:
                if self.method == "box-cox":
                    if np.any(x <= 0):
                        continue
                    transformed = self._box_cox_transform(x, lmbda)
                else:
                    transformed = self._yeo_johnson_transform(x, lmbda)

                # Use negative log-likelihood as score
                _, p_value = stats.normaltest(transformed)
                if p_value > best_score:
                    best_score = p_value
                    best_lambda = lmbda
            except Exception:
                continue

        return best_lambda

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "PowerTransformer":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"{col}_power" for col in self._input_columns]

        # Estimate lambda for each feature
        self.lambdas_ = np.zeros(self.n_features_)
        for i in range(self.n_features_):
            col_data = X[:, i][~np.isnan(X[:, i])]
            self.lambdas_[i] = self._estimate_lambda(col_data)

        # Compute mean and std for standardization
        if self.standardize:
            transformed = self._apply_transform(X)
            self.mean_ = np.nanmean(transformed, axis=0)
            self.std_ = np.nanstd(transformed, axis=0)
            self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def _apply_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply power transformation without standardization."""
        result = np.zeros_like(X)
        for i in range(self.n_features_):
            if self.method == "box-cox":
                result[:, i] = self._box_cox_transform(X[:, i], self.lambdas_[i])
            else:
                result[:, i] = self._yeo_johnson_transform(X[:, i], self.lambdas_[i])
        return result

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = self._apply_transform(X)

        if self.standardize:
            result = (result - self.mean_) / self.std_

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"method": self.method, "standardize": self.standardize}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "lambdas": self.lambdas_.tolist() if self.lambdas_ is not None else None,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.lambdas_ = np.array(stats["lambdas"]) if stats.get("lambdas") else None
        self.mean_ = np.array(stats["mean"]) if stats.get("mean") else None
        self.std_ = np.array(stats["std"]) if stats.get("std") else None
        self.n_features_ = stats.get("n_features", 0)


class Binner(BaseTransformer):
    """
    Bin continuous features into discrete intervals.

    Parameters:
        n_bins: Number of bins
        strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
        encode: Output encoding ('ordinal', 'onehot')
    """

    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = "quantile",
        encode: str = "ordinal",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.bin_edges_: Optional[List[np.ndarray]] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "Binner":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        self.bin_edges_ = []

        for i in range(self.n_features_):
            col_data = X[:, i][~np.isnan(X[:, i])]

            if self.strategy == "uniform":
                edges = np.linspace(col_data.min(), col_data.max(), self.n_bins + 1)
            elif self.strategy == "quantile":
                percentiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.percentile(col_data, percentiles)
                # Remove duplicate edges
                edges = np.unique(edges)
            elif self.strategy == "kmeans":
                from scipy.cluster.vq import kmeans
                centers, _ = kmeans(col_data.astype(float), self.n_bins)
                centers = np.sort(centers)
                edges = np.concatenate([
                    [col_data.min()],
                    (centers[:-1] + centers[1:]) / 2,
                    [col_data.max()]
                ])
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            self.bin_edges_.append(edges)

        # Generate output column names
        if self.encode == "ordinal":
            self._output_columns = [f"{col}_binned" for col in self._input_columns]
        else:  # onehot
            self._output_columns = []
            for i, col in enumerate(self._input_columns):
                n_actual_bins = len(self.bin_edges_[i]) - 1
                for j in range(n_actual_bins):
                    self._output_columns.append(f"{col}_bin_{j}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        if self.encode == "ordinal":
            result = np.zeros((X.shape[0], self.n_features_))
            for i in range(self.n_features_):
                result[:, i] = np.digitize(X[:, i], self.bin_edges_[i][1:-1])
            return result
        else:  # onehot
            all_bins = []
            for i in range(self.n_features_):
                n_actual_bins = len(self.bin_edges_[i]) - 1
                bin_indices = np.digitize(X[:, i], self.bin_edges_[i][1:-1])
                onehot = np.zeros((X.shape[0], n_actual_bins))
                for j in range(X.shape[0]):
                    if bin_indices[j] < n_actual_bins:
                        onehot[j, bin_indices[j]] = 1
                all_bins.append(onehot)
            return np.hstack(all_bins)

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "strategy": self.strategy,
            "encode": self.encode,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "bin_edges": [e.tolist() for e in self.bin_edges_] if self.bin_edges_ else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        if stats.get("bin_edges"):
            self.bin_edges_ = [np.array(e) for e in stats["bin_edges"]]
        self.n_features_ = stats.get("n_features", 0)


class QuantileTransformer(BaseTransformer):
    """
    Transform features using quantile information.

    Parameters:
        n_quantiles: Number of quantiles to compute
        output_distribution: Output distribution ('uniform' or 'normal')
    """

    def __init__(
        self,
        n_quantiles: int = 1000,
        output_distribution: str = "uniform",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.quantiles_: Optional[np.ndarray] = None
        self.references_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "QuantileTransformer":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"{col}_quantile" for col in self._input_columns]

        # Compute quantiles for each feature
        self.references_ = np.linspace(0, 1, self.n_quantiles)
        self.quantiles_ = np.zeros((self.n_quantiles, self.n_features_))

        for i in range(self.n_features_):
            col_data = X[:, i][~np.isnan(X[:, i])]
            self.quantiles_[:, i] = np.percentile(
                col_data, self.references_ * 100
            )

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.zeros_like(X)

        for i in range(self.n_features_):
            # Find quantile positions using interpolation
            result[:, i] = np.interp(
                X[:, i],
                self.quantiles_[:, i],
                self.references_
            )

        if self.output_distribution == "normal":
            from scipy import stats
            # Clip to avoid infinite values
            result = np.clip(result, 1e-7, 1 - 1e-7)
            result = stats.norm.ppf(result)

        return result

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        if self.output_distribution == "normal":
            from scipy import stats
            X = stats.norm.cdf(X)

        result = np.zeros_like(X)

        for i in range(self.n_features_):
            result[:, i] = np.interp(
                X[:, i],
                self.references_,
                self.quantiles_[:, i]
            )

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "n_quantiles": self.n_quantiles,
            "output_distribution": self.output_distribution,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "quantiles": self.quantiles_.tolist() if self.quantiles_ is not None else None,
            "references": self.references_.tolist() if self.references_ is not None else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.quantiles_ = np.array(stats["quantiles"]) if stats.get("quantiles") else None
        self.references_ = np.array(stats["references"]) if stats.get("references") else None
        self.n_features_ = stats.get("n_features", 0)


class Normalizer(BaseTransformer):
    """
    Normalize samples individually to unit norm.

    Parameters:
        norm: Norm type ('l1', 'l2', 'max')
    """

    def __init__(
        self,
        norm: str = "l2",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.norm = norm
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "Normalizer":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = self._input_columns.copy()

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        if self.norm == "l1":
            norms = np.abs(X).sum(axis=1, keepdims=True)
        elif self.norm == "l2":
            norms = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
        elif self.norm == "max":
            norms = np.abs(X).max(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return X / norms

    def _get_parameters(self) -> Dict[str, Any]:
        return {"norm": self.norm}


class ClipTransformer(BaseTransformer):
    """
    Clip values to a specified range.

    Parameters:
        lower: Lower bound (None for no lower bound)
        upper: Upper bound (None for no upper bound)
        use_percentiles: If True, interpret bounds as percentiles
    """

    def __init__(
        self,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        use_percentiles: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.lower = lower
        self.upper = upper
        self.use_percentiles = use_percentiles
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "ClipTransformer":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = self._input_columns.copy()

        if self.use_percentiles:
            if self.lower is not None:
                self.lower_bounds_ = np.percentile(X, self.lower, axis=0)
            else:
                self.lower_bounds_ = np.full(self.n_features_, -np.inf)

            if self.upper is not None:
                self.upper_bounds_ = np.percentile(X, self.upper, axis=0)
            else:
                self.upper_bounds_ = np.full(self.n_features_, np.inf)
        else:
            self.lower_bounds_ = np.full(
                self.n_features_,
                self.lower if self.lower is not None else -np.inf
            )
            self.upper_bounds_ = np.full(
                self.n_features_,
                self.upper if self.upper is not None else np.inf
            )

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "use_percentiles": self.use_percentiles,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "lower_bounds": self.lower_bounds_.tolist() if self.lower_bounds_ is not None else None,
            "upper_bounds": self.upper_bounds_.tolist() if self.upper_bounds_ is not None else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.lower_bounds_ = np.array(stats["lower_bounds"]) if stats.get("lower_bounds") else None
        self.upper_bounds_ = np.array(stats["upper_bounds"]) if stats.get("upper_bounds") else None
        self.n_features_ = stats.get("n_features", 0)


class ImputerNumeric(BaseTransformer):
    """
    Impute missing values in numeric features.

    Parameters:
        strategy: Imputation strategy ('mean', 'median', 'constant', 'most_frequent')
        fill_value: Value to use for 'constant' strategy
    """

    def __init__(
        self,
        strategy: str = "mean",
        fill_value: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "ImputerNumeric":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = self._input_columns.copy()

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "constant":
            self.statistics_ = np.full(self.n_features_, self.fill_value or 0.0)
        elif self.strategy == "most_frequent":
            self.statistics_ = np.zeros(self.n_features_)
            for i in range(self.n_features_):
                col_data = X[:, i][~np.isnan(X[:, i])]
                if len(col_data) > 0:
                    values, counts = np.unique(col_data, return_counts=True)
                    self.statistics_[i] = values[np.argmax(counts)]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)
        X = X.copy()

        for i in range(self.n_features_):
            mask = np.isnan(X[:, i])
            X[mask, i] = self.statistics_[i]

        return X

    def _get_parameters(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "fill_value": self.fill_value}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "statistics": self.statistics_.tolist() if self.statistics_ is not None else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.statistics_ = np.array(stats["statistics"]) if stats.get("statistics") else None
        self.n_features_ = stats.get("n_features", 0)
