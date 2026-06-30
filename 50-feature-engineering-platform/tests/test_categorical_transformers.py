"""Tests for categorical transformers."""

import numpy as np
import pytest

from feature_platform.transformers.categorical import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    FrequencyEncoder,
    BinaryEncoder,
    HashingEncoder,
    ImputerCategorical,
)


class TestLabelEncoder:
    """Tests for LabelEncoder."""

    def test_basic_encoding(self):
        """Test basic label encoding."""
        X = np.array([["a"], ["b"], ["c"], ["a"], ["b"]])
        encoder = LabelEncoder()
        result = encoder.fit_transform(X)

        assert result.shape == X.shape
        assert len(np.unique(result)) == 3

    def test_inverse_transform(self):
        """Test inverse transform."""
        X = np.array([["cat"], ["dog"], ["bird"], ["cat"]])
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(X)
        decoded = encoder.inverse_transform(encoded)

        np.testing.assert_array_equal(X, decoded)

    def test_multiple_columns(self):
        """Test with multiple columns."""
        X = np.array([["a", "x"], ["b", "y"], ["a", "z"]])
        encoder = LabelEncoder()
        result = encoder.fit_transform(X)

        assert result.shape == X.shape
        assert result.dtype == np.int64

    def test_unknown_handling_error(self):
        """Test unknown category handling with error."""
        X_train = np.array([["a"], ["b"]])
        X_test = np.array([["a"], ["c"]])

        encoder = LabelEncoder(handle_unknown="error")
        encoder.fit(X_train)

        with pytest.raises(ValueError):
            encoder.transform(X_test)

    def test_unknown_handling_default(self):
        """Test unknown category handling with default."""
        X_train = np.array([["a"], ["b"]])
        X_test = np.array([["a"], ["c"]])

        encoder = LabelEncoder(handle_unknown="use_default", default_value=-1)
        encoder.fit(X_train)
        result = encoder.transform(X_test)

        assert result[1, 0] == -1


class TestOneHotEncoder:
    """Tests for OneHotEncoder."""

    def test_basic_encoding(self):
        """Test basic one-hot encoding."""
        X = np.array([["a"], ["b"], ["c"]])
        encoder = OneHotEncoder()
        result = encoder.fit_transform(X)

        # Should have 3 columns (one per category)
        assert result.shape == (3, 3)
        # Each row sums to 1
        np.testing.assert_array_equal(result.sum(axis=1), [1, 1, 1])

    def test_drop_first(self):
        """Test dropping first category."""
        X = np.array([["a"], ["b"], ["c"]])
        encoder = OneHotEncoder(drop="first")
        result = encoder.fit_transform(X)

        # Should have 2 columns (n-1)
        assert result.shape == (3, 2)

    def test_multiple_columns(self):
        """Test with multiple columns."""
        X = np.array([["a", "x"], ["b", "y"], ["a", "y"]])
        encoder = OneHotEncoder()
        result = encoder.fit_transform(X)

        # 2 categories from first col + 2 from second = 4
        assert result.shape[1] == 4

    def test_inverse_transform(self):
        """Test inverse transform."""
        X = np.array([["cat"], ["dog"], ["bird"]])
        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(X)
        decoded = encoder.inverse_transform(encoded)

        np.testing.assert_array_equal(X, decoded)

    def test_unknown_handling_ignore(self):
        """Test unknown category handling with ignore."""
        X_train = np.array([["a"], ["b"]])
        X_test = np.array([["a"], ["c"]])

        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(X_train)
        result = encoder.transform(X_test)

        # Unknown category should result in all zeros
        np.testing.assert_array_equal(result[1], [0, 0])


class TestOrdinalEncoder:
    """Tests for OrdinalEncoder."""

    def test_default_ordering(self):
        """Test default alphabetical ordering."""
        X = np.array([["c"], ["a"], ["b"]])
        encoder = OrdinalEncoder()
        result = encoder.fit_transform(X)

        # Should be sorted alphabetically: a=0, b=1, c=2
        np.testing.assert_array_equal(result.flatten(), [2, 0, 1])

    def test_custom_ordering(self):
        """Test custom ordering."""
        X = np.array([["low"], ["medium"], ["high"]])
        encoder = OrdinalEncoder(categories={0: ["low", "medium", "high"]})
        result = encoder.fit_transform(X)

        np.testing.assert_array_equal(result.flatten(), [0, 1, 2])

    def test_inverse_transform(self):
        """Test inverse transform."""
        X = np.array([["small"], ["medium"], ["large"]])
        encoder = OrdinalEncoder(categories={0: ["small", "medium", "large"]})
        encoded = encoder.fit_transform(X)
        decoded = encoder.inverse_transform(encoded)

        np.testing.assert_array_equal(X, decoded)


class TestTargetEncoder:
    """Tests for TargetEncoder."""

    def test_basic_encoding(self):
        """Test basic target encoding."""
        X = np.array([["a"], ["a"], ["b"], ["b"]])
        y = np.array([0, 1, 1, 1])

        encoder = TargetEncoder(smoothing=0)
        result = encoder.fit_transform(X, y)

        # 'a' mean = 0.5, 'b' mean = 1.0
        assert result[0, 0] == pytest.approx(0.5)
        assert result[2, 0] == pytest.approx(1.0)

    def test_smoothing(self):
        """Test smoothing effect."""
        X = np.array([["a"], ["b"], ["b"], ["b"], ["b"]])
        y = np.array([0, 1, 1, 1, 1])

        # With high smoothing, rare category should be pulled toward global mean
        encoder = TargetEncoder(smoothing=10)
        result = encoder.fit_transform(X, y)

        global_mean = y.mean()
        # 'a' with only 1 sample should be closer to global mean
        assert result[0, 0] > 0  # Pulled up from 0
        assert result[0, 0] < global_mean  # But not all the way to global mean

    def test_requires_target(self):
        """Test that target is required."""
        X = np.array([["a"], ["b"]])
        encoder = TargetEncoder()

        with pytest.raises(ValueError):
            encoder.fit(X)


class TestFrequencyEncoder:
    """Tests for FrequencyEncoder."""

    def test_normalized_frequencies(self):
        """Test normalized frequency encoding."""
        X = np.array([["a"], ["a"], ["b"], ["b"], ["b"]])
        encoder = FrequencyEncoder(normalize=True)
        result = encoder.fit_transform(X)

        # 'a' appears 2/5 = 0.4, 'b' appears 3/5 = 0.6
        assert result[0, 0] == pytest.approx(0.4)
        assert result[2, 0] == pytest.approx(0.6)

    def test_raw_frequencies(self):
        """Test raw frequency encoding."""
        X = np.array([["a"], ["a"], ["b"]])
        encoder = FrequencyEncoder(normalize=False)
        result = encoder.fit_transform(X)

        assert result[0, 0] == 2  # 'a' count
        assert result[2, 0] == 1  # 'b' count

    def test_unknown_category(self):
        """Test handling of unknown categories."""
        X_train = np.array([["a"], ["b"]])
        X_test = np.array([["c"]])

        encoder = FrequencyEncoder()
        encoder.fit(X_train)
        result = encoder.transform(X_test)

        # Unknown category should have 0 frequency
        assert result[0, 0] == 0


class TestBinaryEncoder:
    """Tests for BinaryEncoder."""

    def test_basic_encoding(self):
        """Test basic binary encoding."""
        X = np.array([["a"], ["b"], ["c"], ["d"]])
        encoder = BinaryEncoder()
        result = encoder.fit_transform(X)

        # 4 categories need ceil(log2(5)) = 3 bits
        assert result.shape[1] == 3

    def test_unique_encodings(self):
        """Test that each category gets unique encoding."""
        X = np.array([["a"], ["b"], ["c"], ["d"]])
        encoder = BinaryEncoder()
        result = encoder.fit_transform(X)

        # All rows should be unique
        unique_rows = np.unique(result, axis=0)
        assert len(unique_rows) == 4

    def test_unknown_handling(self):
        """Test unknown category handling."""
        X_train = np.array([["a"], ["b"]])
        X_test = np.array([["c"]])

        encoder = BinaryEncoder(handle_unknown="use_default")
        encoder.fit(X_train)
        result = encoder.transform(X_test)

        # Unknown should be all zeros
        np.testing.assert_array_equal(result[0], [0, 0])


class TestHashingEncoder:
    """Tests for HashingEncoder."""

    def test_fixed_dimensions(self):
        """Test fixed output dimensions."""
        X = np.array([["a"], ["b"], ["c"], ["d"], ["e"]])
        encoder = HashingEncoder(n_features=4)
        result = encoder.fit_transform(X)

        assert result.shape == (5, 4)

    def test_handles_unknown(self):
        """Test handling of unknown categories."""
        X_train = np.array([["a"], ["b"]])
        X_test = np.array([["c"], ["d"]])

        encoder = HashingEncoder(n_features=4)
        encoder.fit(X_train)
        result = encoder.transform(X_test)

        # Should work without errors
        assert result.shape == (2, 4)

    def test_deterministic(self):
        """Test that hashing is deterministic."""
        X = np.array([["a"], ["b"]])
        encoder = HashingEncoder(n_features=8)

        result1 = encoder.fit_transform(X)
        result2 = encoder.transform(X)

        np.testing.assert_array_equal(result1, result2)


class TestImputerCategorical:
    """Tests for ImputerCategorical."""

    def test_most_frequent_imputation(self):
        """Test most frequent imputation."""
        X = np.array([["a"], ["a"], [None], ["b"]])
        imputer = ImputerCategorical(strategy="most_frequent")
        result = imputer.fit_transform(X)

        assert result[2, 0] == "a"

    def test_constant_imputation(self):
        """Test constant imputation."""
        X = np.array([["a"], [None], ["b"]])
        imputer = ImputerCategorical(strategy="constant", fill_value="MISSING")
        result = imputer.fit_transform(X)

        assert result[1, 0] == "MISSING"

    def test_handles_various_null_representations(self):
        """Test handling of various null representations."""
        X = np.array([["a"], [None], ["nan"], [""]])
        imputer = ImputerCategorical(strategy="most_frequent")
        result = imputer.fit_transform(X)

        # All nulls should be filled
        assert "a" in result[1, 0] or result[1, 0] == "a"


class TestEncoderPersistence:
    """Tests for encoder state persistence."""

    def test_label_encoder_state(self):
        """Test LabelEncoder state restoration."""
        X = np.array([["a"], ["b"], ["c"]])

        encoder1 = LabelEncoder()
        encoder1.fit(X)

        state = encoder1.get_state()

        encoder2 = LabelEncoder()
        encoder2.set_state(state)

        result1 = encoder1.transform(X)
        result2 = encoder2.transform(X)

        np.testing.assert_array_equal(result1, result2)

    def test_onehot_encoder_state(self):
        """Test OneHotEncoder state restoration."""
        X = np.array([["cat"], ["dog"]])

        encoder1 = OneHotEncoder()
        encoder1.fit(X)

        state = encoder1.get_state()

        encoder2 = OneHotEncoder()
        encoder2.set_state(state)

        result1 = encoder1.transform(X)
        result2 = encoder2.transform(X)

        np.testing.assert_array_equal(result1, result2)


class TestEncoderEdgeCases:
    """Tests for edge cases in categorical encoders."""

    def test_single_category(self):
        """Test encoding with single category."""
        X = np.array([["a"], ["a"], ["a"]])

        encoder = OneHotEncoder()
        result = encoder.fit_transform(X)

        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.flatten(), [1, 1, 1])

    def test_empty_strings(self):
        """Test handling of empty strings."""
        X = np.array([["a"], [""], ["b"]])
        encoder = LabelEncoder()
        result = encoder.fit_transform(X)

        # Should encode empty string as a category
        assert len(np.unique(result)) == 3

    def test_unicode_categories(self):
        """Test handling of unicode categories."""
        X = np.array([["cafe"], ["cafe"], ["nihao"]])
        encoder = LabelEncoder()
        result = encoder.fit_transform(X)

        assert encoder.is_fitted
        assert len(np.unique(result)) == 2

    def test_numeric_strings(self):
        """Test handling of numeric strings."""
        X = np.array([["1"], ["2"], ["1"], ["3"]])
        encoder = LabelEncoder()
        result = encoder.fit_transform(X)

        assert len(np.unique(result)) == 3
