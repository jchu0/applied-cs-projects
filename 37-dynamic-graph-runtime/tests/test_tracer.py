"""Tests for the bytecode tracer's real op-capture behavior."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from dynamicgraph.tracer import BytecodeTracer
from dynamicgraph.core.graph import OpType


def _vec(n=3):
    return np.zeros(n, dtype=np.float32)


def _ops(graph, op_type):
    return [n for n in graph.nodes.values() if n.op_type == op_type]


def test_traces_binary_add():
    def f(x, y):
        return x + y

    tracer = BytecodeTracer()
    g = tracer.trace_function(f, _vec(), _vec())

    assert tracer.graph_break_reasons == []
    adds = _ops(g, OpType.ADD)
    assert len(adds) == 1
    assert len(adds[0].inputs) == 2          # wired to both input tensors
    assert len(_ops(g, OpType.INPUT)) == 2
    assert len(_ops(g, OpType.OUTPUT)) == 1


def test_traces_chained_ops():
    def f(x, y):
        z = x * y
        return z + x

    tracer = BytecodeTracer()
    g = tracer.trace_function(f, _vec(), _vec())

    assert tracer.graph_break_reasons == []
    assert len(_ops(g, OpType.MUL)) == 1
    adds = _ops(g, OpType.ADD)
    assert len(adds) == 1
    # the add consumes the mul's result and the x input
    mul_id = _ops(g, OpType.MUL)[0].id
    assert mul_id in adds[0].inputs


def test_traces_scalar_constant():
    def f(x):
        return x * 2.0

    tracer = BytecodeTracer()
    g = tracer.trace_function(f, _vec())

    muls = _ops(g, OpType.MUL)
    assert len(muls) == 1
    consts = _ops(g, OpType.CONSTANT)
    assert len(consts) == 1
    assert consts[0].attributes.get("value") == 2.0
    assert consts[0].id in muls[0].inputs


def test_traces_matmul():
    def f(x, y):
        return x @ y

    tracer = BytecodeTracer()
    g = tracer.trace_function(f, np.zeros((2, 3), np.float32), np.zeros((3, 4), np.float32))
    assert tracer.graph_break_reasons == []
    assert len(_ops(g, OpType.MATMUL)) == 1


def test_shape_inference_on_elementwise():
    def f(x, y):
        return x + y

    tracer = BytecodeTracer()
    g = tracer.trace_function(f, np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32))
    add = _ops(g, OpType.ADD)[0]
    assert add.metadata.shape == (2, 3)


def test_graph_break_on_function_call():
    def f(x):
        return abs(x)  # LOAD_GLOBAL + CALL -> unsupported

    tracer = BytecodeTracer()
    g = tracer.trace_function(f, _vec())
    assert len(tracer.graph_break_reasons) >= 1
    # the input was still captured before the break
    assert len(_ops(g, OpType.INPUT)) == 1


def test_graph_break_on_branch():
    def f(x, y):
        if x:
            return y
        return x

    tracer = BytecodeTracer()
    tracer.trace_function(f, _vec(), _vec())
    assert len(tracer.graph_break_reasons) >= 1


def test_reset_clears_state():
    def f(x, y):
        return x + y

    tracer = BytecodeTracer()
    tracer.trace_function(f, _vec(), _vec())
    assert len(tracer.graph.nodes) > 0
    tracer.reset()
    assert len(tracer.graph.nodes) == 0
    assert tracer.symbolic_values == {}
    assert tracer.graph_break_reasons == []
