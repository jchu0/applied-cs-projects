"""Tests for numeric transformers."""

import numpy as np
import pytest
import importlib.util

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None

from feature_platform.transformers.numeric import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LogTransformer,
    PowerTransformer,
    Binner,
    QuantileTransformer,
    Normalizer,
    ClipTransformer,
    ImputerNumeric,
)


class TestStandardScaler:
    """Tests for StandardScaler."""

    def test_fit_transform(self):
        """Test basic fit and transform."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        result = scaler.fit_transform(X)

        assert result.shape == X.shape
        assert scaler.is_fitted
        # Check zero mean
        np.testing.assert_array_almost_equal(result.mean(axis=0), [0, 0], decimal=5)
        # Check unit variance
        np.testing.assert_array_almost_equal(result.std(axis=0), [1, 1], decimal=5)

    def test_inverse_transform(self):
        """Test inverse transform."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        transformed = scaler.fit_transform(X)
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(X, recovered, decimal=5)

    def test_without_mean(self):
        """Test without centering."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler(with_mean=False)
        result = scaler.fit_transform(X)

        assert result.shape == X.shape
        # Should still have unit variance
        np.testing.assert_array_almost_equal(result.std(axis=0), [1, 1], decimal=5)

    def test_without_std(self):
        """Test without scaling."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler(with_std=False)
        result = scaler.fit_transform(X)

        # Should have zero mean but not unit variance
        np.testing.assert_array_almost_equal(result.mean(axis=0), [0, 0], decimal=5)

    def test_handles_nan(self):
        """Test handling of NaN values."""
        X = np.array([[1, np.nan], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaler.fit(X)

        assert scaler.is_fitted
        assert not np.isnan(scaler.mean_[0])

    def test_get_state(self):
        """Test state serialization."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaler.fit(X)

        state = scaler.get_state()
        assert state.transformer_type == "StandardScaler"
        assert "mean" in state.statistics
        assert "std" in state.statistics


class TestMinMaxScaler:
    """Tests for MinMaxScaler."""

    def test_default_range(self):
        """Test default 0-1 range."""
        X = np.array([[1], [2], [3], [4], [5]])
        scaler = MinMaxScaler()
        result = scaler.fit_transform(X)

        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_custom_range(self):
        """Test custom range."""
        X = np.array([[1], [2], [3], [4], [5]])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        result = scaler.fit_transform(X)

        assert result.min() == pytest.approx(-1.0)
        assert result.max() == pytest.approx(1.0)

    def test_inverse_transform(self):
        """Test inverse transform."""
        X = np.array([[1, 10], [2, 20], [3, 30]])
        scaler = MinMaxScaler()
        transformed = scaler.fit_transform(X)
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(X, recovered, decimal=5)

    def test_constant_column(self):
        """Test with constant column."""
        X = np.array([[1, 5], [1, 10], [1, 15]])
        scaler = MinMaxScaler()
        result = scaler.fit_transform(X)

        # Constant column should not cause division by zero
        assert not np.any(np.isnan(result))


class TestRobustScaler:
    """Tests for RobustScaler."""

    def test_fit_transform(self):
        """Test basic fit and transform."""
        X = np.array([[1], [2], [3], [4], [100]])  # With outlier
        scaler = RobustScaler()
        result = scaler.fit_transform(X)

        # Median should be 0
        assert np.median(result) == pytest.approx(0.0, abs=0.1)

    def test_custom_quantile_range(self):
        """Test custom quantile range."""
        X = np.array([[1], [2], [3], [4], [5]])
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        scaler.fit(X)

        assert scaler.is_fitted

    def test_without_centering(self):
        """Test without centering."""
        X = np.array([[1], [2], [3], [4], [5]])
        scaler = RobustScaler(with_centering=False)
        result = scaler.fit_transform(X)

        # Values should still be positive
        assert result.min() > 0


class TestLogTransformer:
    """Tests for LogTransformer."""

    def test_natural_log(self):
        """Test natural logarithm."""
        X = np.array([[1], [np.e], [np.e**2]])
        transformer = LogTransformer(base="e", offset=0)
        result = transformer.fit_transform(X)

        np.testing.assert_array_almost_equal(result.flatten(), [0, 1, 2], decimal=5)

    def test_log_base_10(self):
        """Test log base 10."""
        X = np.array([[1], [10], [100]])
        transformer = LogTransformer(base="10", offset=0)
        result = transformer.fit_transform(X)

        np.testing.assert_array_almost_equal(result.flatten(), [0, 1, 2], decimal=5)

    def test_log_base_2(self):
        """Test log base 2."""
        X = np.array([[1], [2], [4]])
        transformer = LogTransformer(base="2", offset=0)
        result = transformer.fit_transform(X)

        np.testing.assert_array_almost_equal(result.flatten(), [0, 1, 2], decimal=5)

    def test_offset(self):
        """Test with offset for zeros."""
        X = np.array([[0], [1], [2]])
        transformer = LogTransformer(base="e", offset=1)
        result = transformer.fit_transform(X)

        assert not np.any(np.isinf(result))

    def test_inverse_transform(self):
        """Test inverse transform."""
        X = np.array([[1], [2], [3]])
        transformer = LogTransformer()
        transformed = transformer.fit_transform(X)
        recovered = transformer.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(X, recovered, decimal=5)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestPowerTransformer:
    """Tests for PowerTransformer."""

    def test_yeo_johnson(self):
        """Test Yeo-Johnson transformation."""
        np.random.seed(42)
        X = np.random.exponential(2, (100, 1))
        transformer = PowerTransformer(method="yeo-johnson")
        result = transformer.fit_transform(X)

        assert result.shape == X.shape
        assert transformer.is_fitted

    def test_handles_negative(self):
        """Test Yeo-Johnson handles negative values."""
        X = np.array([[-2], [-1], [0], [1], [2]])
        transformer = PowerTransformer(method="yeo-johnson")
        result = transformer.fit_transform(X)

        assert not np.any(np.isnan(result))

    def test_standardization(self):
        """Test output standardization."""
        np.random.seed(42)
        X = np.random.exponential(2, (100, 1))
        transformer = PowerTransformer(standardize=True)
        result = transformer.fit_transform(X)

        # Should be approximately zero mean, unit variance
        assert np.abs(result.mean()) < 0.1
        assert np.abs(result.std() - 1) < 0.1


class TestBinner:
    """Tests for Binner."""

    def test_uniform_binning(self):
        """Test uniform binning."""
        X = np.array([[1], [2], [3], [4], [5]])
        binner = Binner(n_bins=5, strategy="uniform")
        result = binner.fit_transform(X)

        assert result.shape == X.shape
        assert len(np.unique(result)) <= 5

    def test_quantile_binning(self):
        """Test quantile binning."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        binner = Binner(n_bins=4, strategy="quantile")
        result = binner.fit_transform(X)

        # Each bin should have roughly equal count
        counts = np.bincount(result.flatten().astype(int))
        assert max(counts) - min(counts[counts > 0]) <= 2

    def test_onehot_encoding(self):
        """Test one-hot encoding of bins."""
        X = np.array([[1], [2], [3], [4], [5]])
        binner = Binner(n_bins=3, encode="onehot")
        result = binner.fit_transform(X)

        # Each row should sum to 1 (one-hot)
        np.testing.assert_array_equal(result.sum(axis=1), np.ones(5))


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestQuantileTransformer:
    """Tests for QuantileTransformer."""

    def test_uniform_output(self):
        """Test uniform output distribution."""
        np.random.seed(42)
        X = np.random.exponential(2, (100, 1))
        transformer = QuantileTransformer(output_distribution="uniform")
        result = transformer.fit_transform(X)

        # Output should be between 0 and 1
        assert result.min() >= 0
        assert result.max() <= 1

    def test_normal_output(self):
        """Test normal output distribution."""
        np.random.seed(42)
        X = np.random.exponential(2, (100, 1))
        transformer = QuantileTransformer(output_distribution="normal")
        result = transformer.fit_transform(X)

        # Should be approximately standard normal
        assert np.abs(result.mean()) < 0.2
        assert np.abs(result.std() - 1) < 0.3

    def test_inverse_transform(self):
        """Test inverse transform."""
        np.random.seed(42)
        X = np.random.uniform(0, 10, (50, 1))
        transformer = QuantileTransformer(output_distribution="uniform")
        transformed = transformer.fit_transform(X)
        recovered = transformer.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(X, recovered, decimal=2)


class TestNormalizer:
    """Tests for Normalizer."""

    def test_l2_norm(self):
        """Test L2 normalization."""
        X = np.array([[3, 4], [1, 0], [0, 1]])
        normalizer = Normalizer(norm="l2")
        result = normalizer.fit_transform(X)

        # Each row should have L2 norm of 1
        norms = np.sqrt((result ** 2).sum(axis=1))
        np.testing.assert_array_almost_equal(norms, [1, 1, 1], decimal=5)

    def test_l1_norm(self):
        """Test L1 normalization."""
        X = np.array([[3, 4], [1, 2], [2, 1]])
        normalizer = Normalizer(norm="l1")
        result = normalizer.fit_transform(X)

        # Each row should have L1 norm of 1
        norms = np.abs(result).sum(axis=1)
        np.testing.assert_array_almost_equal(norms, [1, 1, 1], decimal=5)

    def test_max_norm(self):
        """Test max normalization."""
        X = np.array([[3, 4], [1, 2], [5, 1]])
        normalizer = Normalizer(norm="max")
        result = normalizer.fit_transform(X)

        # Each row's max should be 1
        max_vals = np.abs(result).max(axis=1)
        np.testing.assert_array_almost_equal(max_vals, [1, 1, 1], decimal=5)


class TestClipTransformer:
    """Tests for ClipTransformer."""

    def test_fixed_bounds(self):
        """Test fixed bounds clipping."""
        X = np.array([[1], [5], [10], [15], [20]])
        clipper = ClipTransformer(lower=5, upper=15)
        result = clipper.fit_transform(X)

        assert result.min() >= 5
        assert result.max() <= 15

    def test_percentile_bounds(self):
        """Test percentile-based clipping."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [100]])
        clipper = ClipTransformer(lower=10, upper=90, use_percentiles=True)
        result = clipper.fit_transform(X)

        # The outlier (100) should be clipped
        assert result.max() < 100


class TestImputerNumeric:
    """Tests for ImputerNumeric."""

    def test_mean_imputation(self):
        """Test mean imputation."""
        X = np.array([[1], [2], [np.nan], [4], [5]])
        imputer = ImputerNumeric(strategy="mean")
        result = imputer.fit_transform(X)

        assert not np.any(np.isnan(result))
        assert result[2, 0] == pytest.approx(3.0)  # Mean of 1, 2, 4, 5

    def test_median_imputation(self):
        """Test median imputation."""
        X = np.array([[1], [2], [np.nan], [4], [100]])  # With outlier
        imputer = ImputerNumeric(strategy="median")
        result = imputer.fit_transform(X)

        assert result[2, 0] == pytest.approx(3.0)  # Median of 1, 2, 4, 100

    def test_constant_imputation(self):
        """Test constant imputation."""
        X = np.array([[1], [np.nan], [3]])
        imputer = ImputerNumeric(strategy="constant", fill_value=-1)
        result = imputer.fit_transform(X)

        assert result[1, 0] == -1

    def test_most_frequent_imputation(self):
        """Test most frequent imputation."""
        X = np.array([[1], [1], [np.nan], [2], [1]])
        imputer = ImputerNumeric(strategy="most_frequent")
        result = imputer.fit_transform(X)

        assert result[2, 0] == 1  # Most frequent value


class TestTransformerPersistence:
    """Test transformer state persistence."""

    def test_save_load_scaler(self, tmp_path):
        """Test saving and loading a scaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaler.fit(X)

        # Save
        path = tmp_path / "scaler.pkl"
        scaler.save(str(path))

        # Load
        loaded = StandardScaler.load(str(path))

        # Compare results
        result1 = scaler.transform(X)
        result2 = loaded.transform(X)

        np.testing.assert_array_equal(result1, result2)

    def test_state_restoration(self):
        """Test state restoration."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler1 = StandardScaler()
        scaler1.fit(X)

        state = scaler1.get_state()

        scaler2 = StandardScaler()
        scaler2.set_state(state)

        result1 = scaler1.transform(X)
        result2 = scaler2.transform(X)

        np.testing.assert_array_equal(result1, result2)
