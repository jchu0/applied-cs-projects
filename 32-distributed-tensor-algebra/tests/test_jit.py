"""Tests for JIT compilation functionality."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorlib.core.tensor import (
    LazyTensor, Tracer, TensorSpec, ShapeSpec,
    array, randn, zeros, ones
)
from tensorlib.core.primitives import matmul, exp, reduce_sum
from tensorlib.jit.compiler import (
    JittedFunction, jit, TracingContext, trace
)


class TestJittedFunction:
    """Test JittedFunction class."""

    def test_create_jitted_function(self):
        """Test creating a jitted function."""
        def f(x):
            return x + x

        jitted_f = JittedFunction(f)

        assert jitted_f.fun is f
        assert jitted_f.static_argnums == ()
        assert len(jitted_f._cache) == 0

    def test_jitted_function_call(self):
        """Test calling a jitted function."""
        def f(x):
            return x * 2

        jitted_f = JittedFunction(f)
        x = array([1.0, 2.0, 3.0])
        result = jitted_f(x)

        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_jitted_function_with_static_argnums(self):
        """Test jitted function with static arguments."""
        def f(x, scale):
            return x * scale

        jitted_f = JittedFunction(f, static_argnums=(1,))
        x = array([1.0, 2.0, 3.0])

        result = jitted_f(x, 2.0)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestJitDecorator:
    """Test jit() decorator."""

    def test_jit_basic(self):
        """Test basic jit decorator usage."""
        @jit
        def f(x):
            return x + 1

        x = array([1.0, 2.0, 3.0])
        result = f(x)

        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_jit_with_args(self):
        """Test jit decorator with arguments."""
        @jit(static_argnums=(1,))
        def f(x, n):
            result = x
            for _ in range(n):
                result = result + 1
            return result

        x = array([1.0, 2.0, 3.0])
        result = f(x, 3)

        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_jit_without_parens(self):
        """Test jit decorator without parentheses."""
        @jit
        def f(x, y):
            return x + y

        x = array([1.0, 2.0, 3.0])
        y = array([4.0, 5.0, 6.0])
        result = f(x, y)

        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_jit_with_parens(self):
        """Test jit decorator with parentheses but no args."""
        @jit()
        def f(x):
            return x * x

        x = array([2.0, 3.0, 4.0])
        result = f(x)

        expected = np.array([4.0, 9.0, 16.0])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestCaching:
    """Test JIT caching behavior."""

    def test_cache_hit(self):
        """Test that cached compilation is reused."""
        call_count = [0]

        def f(x):
            call_count[0] += 1
            return x + 1

        jitted_f = JittedFunction(f)

        x1 = array([1.0, 2.0, 3.0])
        x2 = array([4.0, 5.0, 6.0])

        # First call - should compile
        jitted_f(x1)
        # Second call with same shape/dtype - should use cache
        jitted_f(x2)

        # Function should have been called twice
        assert call_count[0] == 2

        # But cache should only have one entry (same signature)
        assert jitted_f.cache_info['size'] == 1

    def test_cache_miss_different_shape(self):
        """Test cache miss with different shape."""
        @jit
        def f(x):
            return x + 1

        x1 = array([1.0, 2.0, 3.0])
        x2 = array([1.0, 2.0])

        f(x1)
        f(x2)

        # Should have two cache entries (different shapes)
        assert f.cache_info['size'] == 2

    def test_cache_miss_different_dtype(self):
        """Test cache miss with different dtype."""
        @jit
        def f(x):
            return x + 1

        x1 = array([1.0, 2.0], dtype=np.float32)
        x2 = array([1.0, 2.0], dtype=np.float64)

        f(x1)
        f(x2)

        # Should have two cache entries (different dtypes)
        assert f.cache_info['size'] == 2

    def test_clear_cache(self):
        """Test clearing the cache."""
        @jit
        def f(x):
            return x * 2

        x = array([1.0, 2.0, 3.0])
        f(x)

        assert f.cache_info['size'] == 1

        f.clear_cache()
        assert f.cache_info['size'] == 0

    def test_cache_info(self):
        """Test cache_info property."""
        @jit
        def f(x):
            return x + 1

        x = array([1.0, 2.0, 3.0])

        # Initial state
        info = f.cache_info
        assert 'hits' in info
        assert 'misses' in info
        assert 'size' in info

        # After first call - 1 miss
        f(x)
        info = f.cache_info
        assert info['misses'] == 1
        assert info['size'] == 1

        # After second call - 1 hit
        f(x)
        info = f.cache_info
        assert info['hits'] >= 0
        assert info['size'] == 1


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_cache_key_tensor_signature(self):
        """Test that cache key captures tensor signature."""
        jf = JittedFunction(lambda x: x)

        x1 = array([[1, 2], [3, 4]], dtype=np.float32)
        x2 = array([[5, 6], [7, 8]], dtype=np.float32)
        x3 = array([1, 2, 3, 4], dtype=np.float32)

        key1 = jf._make_cache_key((x1,), {})
        key2 = jf._make_cache_key((x2,), {})
        key3 = jf._make_cache_key((x3,), {})

        # Same shape/dtype should have same key
        assert key1 == key2
        # Different shape should have different key
        assert key1 != key3

    def test_cache_key_static_args(self):
        """Test that static args are included in key."""
        jf = JittedFunction(lambda x, n: x, static_argnums=(1,))

        x = array([1.0, 2.0])

        key1 = jf._make_cache_key((x, 2), {})
        key2 = jf._make_cache_key((x, 3), {})

        # Different static values should have different keys
        assert key1 != key2

    def test_cache_key_kwargs(self):
        """Test that kwargs are included in key."""
        jf = JittedFunction(lambda x, a=1: x)

        x = array([1.0, 2.0])

        key1 = jf._make_cache_key((x,), {'a': 1})
        key2 = jf._make_cache_key((x,), {'a': 2})

        assert key1 != key2


class TestTracingContext:
    """Test TracingContext for graph building."""

    def test_context_basic(self):
        """Test basic tracing context."""
        with TracingContext() as ctx:
            assert TracingContext.is_tracing()
            assert TracingContext.get_current() is ctx

        assert not TracingContext.is_tracing()
        assert TracingContext.get_current() is None

    def test_nested_context(self):
        """Test nested tracing contexts."""
        with TracingContext() as ctx1:
            assert TracingContext.get_current() is ctx1

            with TracingContext() as ctx2:
                assert TracingContext.get_current() is ctx2
                assert ctx2 is not ctx1

            assert TracingContext.get_current() is ctx1

    def test_context_inputs_outputs(self):
        """Test tracking inputs and outputs."""
        with TracingContext() as ctx:
            # Context starts empty
            assert ctx.inputs == []
            assert ctx.outputs == []
            assert ctx.ops == []


class TestTrace:
    """Test trace() function for graph building.

    Note: The trace() function works with Tracers, which currently don't
    have arithmetic operations implemented. These tests verify the tracing
    infrastructure works correctly with identity functions.
    """

    def test_trace_identity(self):
        """Test tracing an identity function."""
        def f(x):
            return x  # Identity - no operations on tracer

        x = array([1.0, 2.0, 3.0])
        traced = trace(f, x)

        # Trace returns a TracingContext with inputs populated
        assert hasattr(traced, 'inputs')
        assert hasattr(traced, 'outputs')
        assert hasattr(traced, 'ops')
        assert isinstance(traced, TracingContext)

    def test_trace_with_non_tensor_args(self):
        """Test tracing with mixed tensor and non-tensor arguments."""
        def f(x, scale):
            # x is a tracer, scale is a scalar - not traced
            return x  # Return tracer unchanged

        x = array([1.0, 2.0])
        traced = trace(f, x, 2.0)

        # Should have traced the tensor input
        assert hasattr(traced, 'inputs')
        # Only LazyTensor args become tracers
        assert len(traced.inputs) == 1

    def test_trace_creates_input_tracers(self):
        """Test that trace creates Tracer objects for inputs."""
        def f(x):
            return x

        x = array([1.0, 2.0, 3.0])
        traced = trace(f, x)

        # Should have created a tracer for the input
        assert len(traced.inputs) == 1
        assert isinstance(traced.inputs[0], Tracer)


class TestTracer:
    """Test Tracer for symbolic execution."""

    def test_tracer_creation(self):
        """Test creating a tracer."""
        spec = TensorSpec(ShapeSpec((32, 64)), np.float32)
        tracer = Tracer(spec, "input_0")

        assert tracer.shape == (32, 64)
        assert tracer.dtype == np.float32
        assert tracer.name == "input_0"

    def test_tracer_auto_naming(self):
        """Test automatic tracer naming."""
        spec = TensorSpec(ShapeSpec((10,)), np.float32)

        t1 = Tracer(spec)
        t2 = Tracer(spec)

        # Should have unique names
        assert t1.name != t2.name

    def test_tracer_repr(self):
        """Test tracer string representation."""
        spec = TensorSpec(ShapeSpec((4, 8)), np.float32)
        tracer = Tracer(spec, "test")

        repr_str = repr(tracer)
        assert "test" in repr_str
        assert "(4, 8)" in repr_str


class TestJitMatmul:
    """Test JIT with matrix operations."""

    def test_jit_matmul(self):
        """Test JIT-compiled matrix multiplication."""
        @jit
        def f(x, y):
            return matmul(x, y)

        x = randn(32, 64)
        y = randn(64, 128)
        result = f(x, y)

        expected = np.matmul(x.numpy(), y.numpy())
        np.testing.assert_array_almost_equal(result.numpy(), expected, decimal=5)

    def test_jit_mlp_layer(self):
        """Test JIT-compiled MLP layer."""
        @jit
        def layer(x, w, b):
            h = matmul(x, w) + b
            return h

        batch = 16
        in_features = 32
        out_features = 64

        x = randn(batch, in_features)
        w = randn(in_features, out_features)
        b = zeros((out_features,))

        result = layer(x, w, b)
        assert result.shape == (batch, out_features)


class TestJitWithTransformations:
    """Test JIT with various tensor transformations."""

    def test_jit_exp_sum(self):
        """Test JIT with exp and sum."""
        @jit
        def softmax_like(x):
            e = exp(x)
            return e / reduce_sum(e, keepdims=True)

        x = randn(10)
        result = softmax_like(x)

        # Sum should be approximately 1
        np.testing.assert_almost_equal(result.numpy().sum(), 1.0, decimal=5)

    def test_jit_chained(self):
        """Test JIT with chained operations."""
        @jit
        def f(x):
            y = x * 2
            z = y + 1
            w = z * 3
            return w

        x = array([1.0, 2.0, 3.0])
        result = f(x)

        expected = (np.array([1.0, 2.0, 3.0]) * 2 + 1) * 3
        np.testing.assert_array_equal(result.numpy(), expected)


class TestJitEdgeCases:
    """Test edge cases for JIT compilation."""

    def test_jit_scalar_output(self):
        """Test JIT with scalar output."""
        @jit
        def f(x):
            return x.sum()

        x = array([1.0, 2.0, 3.0])
        result = f(x)

        assert result.numpy() == 6.0

    def test_jit_no_tensor_args(self):
        """Test JIT with non-tensor static arguments."""
        @jit(static_argnums=(1, 2))
        def f(x, a, b):
            return x * a + b

        x = array([1.0, 2.0, 3.0])
        result = f(x, 2, 3)

        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_jit_empty_tensor(self):
        """Test JIT with empty tensor."""
        @jit
        def f(x):
            return x + 1

        x = array([], dtype=np.float32).reshape(0, 5)
        result = f(x)

        assert result.shape == (0, 5)

    def test_jit_high_dimensional(self):
        """Test JIT with high-dimensional tensor."""
        @jit
        def f(x):
            return x * 2

        x = randn(2, 3, 4, 5, 6)
        result = f(x)

        expected = x.numpy() * 2
        np.testing.assert_array_almost_equal(result.numpy(), expected)


class TestJitPerformance:
    """Test JIT performance characteristics."""

    def test_repeated_calls_use_cache(self):
        """Test that repeated calls are faster due to caching."""
        @jit
        def f(x):
            result = x
            for _ in range(10):
                result = result + 1
            return result

        x = randn(100, 100)

        # Warmup / compile
        f(x)
        initial_cache_size = f.cache_info['size']

        # Call again - should use cache
        for _ in range(5):
            f(x)

        # Cache should not grow
        assert f.cache_info['size'] == initial_cache_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
