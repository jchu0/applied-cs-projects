"""Tests for JIT tracing functionality."""

import pytest
import numpy as np

from dynagraph import Tensor, jit_trace, trace_graph
from dynagraph.autograd import is_tracing, TracedFunction, LazyTracedFunction


class TestJITTrace:
    """Tests for jit_trace decorator."""

    def test_basic_trace(self):
        """Test basic function tracing."""

        def simple_func(x):
            return x * 2

        x = Tensor([1.0, 2.0, 3.0])
        traced = jit_trace(simple_func)

        # Should return a LazyTracedFunction
        assert isinstance(traced, LazyTracedFunction)

        # Should execute correctly
        result = traced(x)
        expected = x.data * 2
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_trace_with_sample_inputs(self):
        """Test tracing with sample inputs."""

        @jit_trace(sample_inputs=(Tensor([1.0, 2.0]),))
        def mul_func(x):
            return x * 3

        # Should be immediately traced
        assert isinstance(mul_func, TracedFunction)

        # Should execute correctly
        x = Tensor([4.0, 5.0])
        result = mul_func(x)
        expected = x.data * 3
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_traced_function_caching(self):
        """Test that traced functions cache by input shape."""

        @jit_trace
        def add_one(x):
            return x + 1

        # First call traces
        x1 = Tensor([1.0, 2.0])
        result1 = add_one(x1)

        # Second call with same shape should use cache
        x2 = Tensor([3.0, 4.0])
        result2 = add_one(x2)

        np.testing.assert_array_almost_equal(result1.data, [2.0, 3.0])
        np.testing.assert_array_almost_equal(result2.data, [4.0, 5.0])

    def test_trace_graph(self):
        """Test trace_graph function."""

        def my_func(x):
            return x * x + 1

        x = Tensor([1.0, 2.0, 3.0])
        graph = trace_graph(my_func, x)

        # Should return a graph dictionary
        assert isinstance(graph, dict)
        assert 'inputs' in graph
        assert 'outputs' in graph

    def test_is_tracing(self):
        """Test is_tracing function."""
        # Should not be tracing initially
        assert not is_tracing()

    def test_trace_with_multiple_ops(self):
        """Test tracing function with multiple operations."""

        @jit_trace
        def multi_op(x, y):
            z = x + y
            w = z * 2
            return w - 1

        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0])

        result = multi_op(x, y)
        expected = (x.data + y.data) * 2 - 1
        np.testing.assert_array_almost_equal(result.data, expected)


class TestTracedFunction:
    """Tests for TracedFunction class."""

    def test_traced_function_creation(self):
        """Test creating a TracedFunction."""

        def func(x):
            return x * 2

        traced_graph = {'inputs': 1, 'outputs': 1, 'ops': []}
        traced = TracedFunction(func, traced_graph)

        assert traced.func is func
        assert traced.traced_graph == traced_graph

    def test_cache_key_generation(self):
        """Test cache key generation for different inputs."""

        def func(x):
            return x

        traced = TracedFunction(func, {'inputs': 1, 'outputs': 1, 'ops': []})

        # Different shapes should produce different cache keys
        key1 = traced._make_cache_key([Tensor(np.zeros((2, 3)))])
        key2 = traced._make_cache_key([Tensor(np.zeros((3, 2)))])
        key3 = traced._make_cache_key([Tensor(np.zeros((2, 3)))])

        assert key1 != key2
        assert key1 == key3


class TestLazyTracedFunction:
    """Tests for LazyTracedFunction class."""

    def test_lazy_tracing(self):
        """Test that tracing is deferred until first call."""

        trace_count = [0]

        def func(x):
            trace_count[0] += 1
            return x * 2

        lazy = LazyTracedFunction(func)

        # Should not have traced yet
        assert lazy._traced is None

        # First call should trace
        x = Tensor([1.0, 2.0])
        result = lazy(x)

        # Should have traced now
        assert lazy._traced is not None
        np.testing.assert_array_almost_equal(result.data, [2.0, 4.0])

    def test_lazy_function_reuse(self):
        """Test that lazy traced function reuses trace."""

        def func(x):
            return x + 1

        lazy = LazyTracedFunction(func)

        x1 = Tensor([1.0])
        result1 = lazy(x1)

        # Get the traced function
        traced = lazy._traced

        x2 = Tensor([2.0])
        result2 = lazy(x2)

        # Should reuse the same traced function
        assert lazy._traced is traced
