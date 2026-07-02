# Dynamic Graph Execution Runtime - API Documentation

This document covers the public API exported from the top-level `dynamicgraph`
package (everything in `dynamicgraph.__all__`). Every symbol below is importable
directly, e.g. `from dynamicgraph import Graph, compile`.

## Compiler API

The high-level entry point. `compile` is a `torch.compile()`-style decorator
that traces a function, optimizes its graph, and executes it on a backend, with
automatic caching and guard generation.

```python
def compile(
    func: Optional[Callable] = None,
    *,
    backend: str = "eager_numpy",
    optimization_level: int = 1,
    **kwargs,
) -> Callable

def optimize(func: Callable) -> Callable   # alias for compile() with defaults
def trace(func: Callable, *args, **kwargs) -> Graph
def explain(func: Callable, *args, **kwargs) -> str
```

**Example:**
```python
import numpy as np
from dynamicgraph import compile, trace, explain

@compile(backend="eager_numpy", optimization_level=2)
def forward(x, y):
    return x + y * 2

x = np.ones((4, 4), dtype=np.float32)
y = np.ones((4, 4), dtype=np.float32)
result = forward(x, y)

# Inspect the captured graph without executing it
graph = trace(forward, x, y)
print(explain(forward, x, y))   # human-readable graph + optimization summary
```

### DynamicCompiler

The class behind `compile`. Instances are callable decorators.

```python
class DynamicCompiler:
    def __init__(
        self,
        backend: str = "eager_numpy",
        optimization_level: int = 1,
        cache_enabled: bool = True,
        fallback_to_eager: bool = True,
        verbose: bool = False,
    )

    def __call__(self, func: Callable) -> Callable   # use as @compiler
    def get_stats(self) -> Dict[str, Any]
    def reset_stats(self) -> None
```

```python
from dynamicgraph import DynamicCompiler

compiler = DynamicCompiler(optimization_level=2, verbose=True)

@compiler
def model(x, y):
    return x @ y + x

out = model(x, y)
print(compiler.get_stats())
```

### Compilation Cache and Guards

`DynamicCompiler` caches compiled functions keyed by source location, guarded by
runtime checks on argument shape/dtype.

```python
class CompilationCache:
    def __init__(self, max_entries: int = 1000)
    def lookup(self, func, *args, **kwargs) -> Optional[CompiledFunction]
    def insert(self, func, compiled_fn, graph, guards, compile_time_ms) -> None
    def get_stats(self) -> Dict[str, Any]
    def clear(self) -> None

class Guard:
    def __init__(self, name: str, check_fn: Callable[..., bool], description: str = "")
    def check(self, *args, **kwargs) -> bool

class ShapeGuard(Guard):
    def __init__(self, arg_index: int, expected_shape: Tuple[int, ...])

class DtypeGuard(Guard):
    def __init__(self, arg_index: int, expected_dtype: str)

@dataclass
class CacheEntry:
    compiled_fn: CompiledFunction
    guards: List[Guard]
    graph: Graph
    hit_count: int = 0
    compile_time_ms: float = 0.0
    created_at: float = ...
    def check_guards(self, *args, **kwargs) -> bool
```

## Core Graph API

### Graph

```python
class Graph:
    def __init__(self, name: Optional[str] = None)

    def add_node(self, node: Node) -> str
    def remove_node(self, node_id: str) -> None
    def add_edge(self, source: str, target: str, index: int = 0,
                 attributes: Optional[Dict] = None) -> Edge
    def get_predecessors(self, node_id: str) -> List[str]
    def get_successors(self, node_id: str) -> List[str]
    def topological_sort(self) -> List[str]
    def has_cycle(self) -> bool
    def subgraph(self, node_ids: Set[str]) -> Graph
    def clone(self) -> Graph
    def validate(self) -> List[str]
    def to_dict(self) -> Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Graph
```

**Example:**
```python
from dynamicgraph import Graph, Node, OpType

graph = Graph(name="my_model")

input_node = Node(op_type=OpType.INPUT, name="input")
add_node = Node(op_type=OpType.ADD, name="add")
output_node = Node(op_type=OpType.OUTPUT, name="output")

input_id = graph.add_node(input_node)
add_id = graph.add_node(add_node)
output_id = graph.add_node(output_node)

graph.add_edge(input_id, add_id)
graph.add_edge(add_id, output_id)

issues = graph.validate()      # [] when the graph is well-formed
if not issues:
    order = graph.topological_sort()
```

### Node, Edge, NodeMetadata, OpType

```python
@dataclass
class Node:
    id: str                       # auto-generated "node_<hex>" if omitted
    op_type: OpType = OpType.CUSTOM
    name: Optional[str] = None
    inputs: List[str] = []        # source node ids
    outputs: List[str] = []       # target node ids
    attributes: Dict[str, Any] = {}
    metadata: NodeMetadata = NodeMetadata()

    def add_input(self, node_id: str) -> None
    def add_output(self, node_id: str) -> None
    def remove_input(self, node_id: str) -> None
    def remove_output(self, node_id: str) -> None

@dataclass
class Edge:
    source: str
    target: str
    index: int = 0                # input slot at the target node
    attributes: Dict[str, Any] = {}

@dataclass
class NodeMetadata:
    dtype: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None
    device: Optional[str] = None
    requires_grad: bool = False
    is_parameter: bool = False
    is_buffer: bool = False
    source_location: Optional[str] = None
    original_name: Optional[str] = None
    compile_hints: Dict[str, Any] = {}
```

`OpType` is an `Enum` of supported operations, including: `ADD`, `SUB`, `MUL`,
`DIV`, `MATMUL`, `LINEAR`, `CONV2D`, `RELU`, `SIGMOID`, `SOFTMAX`, `BATCHNORM`,
`DROPOUT`, shape ops (`RESHAPE`, `TRANSPOSE`, `PERMUTE`, `SQUEEZE`, `UNSQUEEZE`),
reductions (`SUM`, `MEAN`, `MAX`, `MIN`), control flow (`IF_THEN_ELSE`,
`WHILE_LOOP`, `FOR_LOOP`), memory ops (`COPY`, `CLONE`, `DETACH`), placeholders
(`INPUT`, `OUTPUT`, `CONSTANT`, `PARAMETER`, `BUFFER`), and `CUSTOM`.

## Tensor API

### SymbolicTensor

Represents a tensor symbolically during graph construction. Supports operator
overloads (`+`, `-`, `*`, `/`, `@`, unary `-`) that construct result tensors
with inferred shapes/dtypes.

```python
class SymbolicTensor:
    def __init__(
        self,
        node_id: Optional[str] = None,
        metadata: Optional[TensorMetadata] = None,
        concrete_value: Optional[np.ndarray] = None,
        source: Optional[str] = None,
    )

    # properties
    shape: Optional[Tuple[int, ...]]
    dtype: Optional[np.dtype]
    device: str
    requires_grad: bool          # settable
    grad: Optional[SymbolicTensor]  # settable; requires requires_grad

    def detach(self) -> SymbolicTensor
    def clone(self) -> SymbolicTensor
    def to(self, device: str) -> SymbolicTensor
    def is_concrete(self) -> bool
    def is_symbolic(self) -> bool
    def numel(self) -> Optional[int]
    def ndim(self) -> Optional[int]
    def size(self, dim: Optional[int] = None) -> Union[Tuple[int, ...], int, None]

@dataclass
class TensorMetadata:
    dtype: Optional[np.dtype] = None
    shape: Optional[Tuple[int, ...]] = None
    device: str = "cpu"
    requires_grad: bool = False
    is_parameter: bool = False
    is_buffer: bool = False
    name: Optional[str] = None
```

### TensorFactory

```python
class TensorFactory:
    @staticmethod
    def zeros(shape, dtype=np.float32, device="cpu", requires_grad=False) -> SymbolicTensor
    @staticmethod
    def ones(shape, dtype=np.float32, device="cpu", requires_grad=False) -> SymbolicTensor
    @staticmethod
    def randn(shape, dtype=np.float32, device="cpu", requires_grad=False) -> SymbolicTensor
    @staticmethod
    def from_numpy(array, device="cpu", requires_grad=False) -> SymbolicTensor
```

**Example:**
```python
import numpy as np
from dynamicgraph import SymbolicTensor, TensorMetadata, TensorFactory

a = SymbolicTensor(metadata=TensorMetadata(shape=(10, 20), dtype=np.float32))
b = TensorFactory.randn((20, 5), requires_grad=True)

c = a @ b          # SymbolicTensor with shape (10, 5)
d = TensorFactory.ones((10, 5)) + c
```

## Tracing API

### BytecodeTracer

Walks a function's CPython bytecode and symbolically executes the straight-line
tensor-math subset, recording each op as a graph node. Unsupported constructs
(calls, branches, loops) trigger a *graph break* and end tracing, returning the
partial graph.

```python
class BytecodeTracer:
    def __init__(self, graph: Optional[Graph] = None)
    def trace_function(self, func: Callable, *args, **kwargs) -> Graph
    def should_trace(self, frame: types.FrameType) -> bool
    def record_graph_break(self, reason: str) -> None
    def reset(self) -> None

    # attributes
    graph: Graph
    graph_break_reasons: List[str]

class TracingMode(Enum):    # NONE | SYMBOLIC | CONCRETE

@dataclass
class TraceFrame:
    code: types.CodeType
    locals: Dict[str, Any]
    globals: Dict[str, Any]
    instructions: List[dis.Instruction]
    ip: int = 0
    stack: List[Any] = []
    block_stack: List[tuple] = []
```

**Example:**
```python
from dynamicgraph import BytecodeTracer

def my_function(x, y):
    z = x + y
    return z * 2

tracer = BytecodeTracer()
graph = tracer.trace_function(my_function, x, y)

print(f"Captured {len(graph.nodes)} operations")
print(f"Graph breaks: {tracer.graph_break_reasons}")
```

### FrameGuard

```python
class GuardCondition(Enum):   # TYPE_MATCH | SHAPE_MATCH | VALUE_MATCH | ATTRIBUTE_MATCH

@dataclass
class GuardFailure:
    condition: GuardCondition
    expected: Any
    actual: Any
    message: str = ""

class FrameGuard:
    def __init__(self)
    def add_condition(self, condition, check, expected=None, actual_fn=None) -> None
    def check(self) -> bool
    def get_failures(self) -> list[GuardFailure]
    def reset(self) -> None
```

## Optimization API

### GraphOptimizer

Runs optimization passes to convergence. **Note:** `optimize()` returns a
`(Graph, OptimizationStats)` tuple and does not mutate the input graph.

```python
class GraphOptimizer:
    def __init__(self, optimization_level: int = 1,
                 max_iterations: int = 10, verbose: bool = False)

    def add_pass(self, pass_: OptimizationPass, index: Optional[int] = None) -> None
    def remove_pass(self, name: str) -> None
    def enable_pass(self, name: str) -> None
    def disable_pass(self, name: str) -> None
    def optimize(self, graph: Graph) -> Tuple[Graph, OptimizationStats]
    def optimize_for_inference(self, graph: Graph) -> Tuple[Graph, OptimizationStats]
    def optimize_for_training(self, graph: Graph) -> Tuple[Graph, OptimizationStats]
    def get_pass_stats(self) -> Dict[str, dict]
    def reset_stats(self) -> None

def optimize_graph(graph: Graph, level: int = 1, mode: str = "default") -> Graph
```

`optimization_level`: `0` = none, `1` = basic (shape inference, constant
folding, algebraic simplification, dead-code elimination), `2` = aggressive
(adds CSE, operator fusion, layout optimization). `mode` for `optimize_graph` is
`"default"`, `"inference"`, or `"training"`.

**Example:**
```python
from dynamicgraph import (
    GraphOptimizer, optimize_graph,
    ConstantFolding, OperatorFusion,
)

optimizer = GraphOptimizer(optimization_level=2)
optimizer.add_pass(OperatorFusion())        # pass an instance

optimized_graph, stats = optimizer.optimize(original_graph)
print(stats.total_passes, stats.nodes_removed)

# Or the one-shot convenience wrapper:
optimized_graph = optimize_graph(original_graph, level=2, mode="inference")
```

### Optimization Passes

All passes subclass `OptimizationPass` and implement
`run(graph) -> PassResult`. Instantiate with no arguments.

```python
class OptimizationPass(ABC):
    def __init__(self, name: str)
    @abstractmethod
    def run(self, graph: Graph) -> PassResult
    # attributes: name, enabled, stats

# Concrete passes (exported):
ConstantFolding()
DeadCodeElimination()
CommonSubexpressionElimination()
AlgebraicSimplification()
OperatorFusion()
```

`PassResult` (returned by `run`, not exported at top level) carries `changed`,
`nodes_removed`, `nodes_added`, `nodes_modified`, and `message`.

## Backend API

### Backend

Abstract execution backend. Subclasses implement `name`, `compile`, and
`is_available`.

```python
class Backend(ABC):
    @abstractmethod
    def name(self) -> str
    @abstractmethod
    def compile(self, graph: Graph) -> CompiledFunction
    @abstractmethod
    def is_available(self) -> bool
    def supports_op(self, op_type: OpType) -> bool   # default: True

class EagerBackend(Backend):
    """NumPy eager-mode interpreter. name() == 'eager_numpy'."""

@dataclass
class CompiledFunction:
    execute_fn: Callable
    graph: Graph
    backend_name: str
    input_names: List[str]
    output_names: List[str]
    metadata: Dict[str, Any] = {}
    def __call__(self, *args, **kwargs) -> Any
```

### BackendRegistry

A registry of backend instances. `EagerBackend` (name `"eager_numpy"`) is
registered as the default at import time.

```python
class BackendRegistry:
    @classmethod
    def register(cls, backend: Backend, set_default: bool = False) -> None
    @classmethod
    def get(cls, name: str) -> Optional[Backend]
    @classmethod
    def get_default(cls) -> Optional[Backend]
    @classmethod
    def set_default(cls, name: str) -> None
    @classmethod
    def list_backends(cls) -> List[str]
    @classmethod
    def list_available(cls) -> List[str]
```

**Example:**
```python
from dynamicgraph import BackendRegistry, EagerBackend

backend = BackendRegistry.get("eager_numpy")
compiled = backend.compile(graph)     # CompiledFunction
result = compiled(x, y)

print(BackendRegistry.list_available())
```

## Context Management

### ExecutionContext and CompilationContext

```python
@dataclass
class CompilationContext:
    graph: Optional[Graph] = None
    optimization_level: int = 1
    backend: str = "eager"
    device: str = "cpu"
    enable_profiling: bool = False
    enable_debugging: bool = False
    cache_compiled: bool = True
    max_graph_size: int = 10000
    min_graph_size: int = 10
    # plus compilation statistics counters
    def should_compile(self, graph: Graph) -> bool

@dataclass
class ExecutionContext:
    compilation_context: CompilationContext = ...
    is_training: bool = True
    is_grad_enabled: bool = True
    def push_graph(self, graph: Graph) -> None
    def pop_graph(self) -> Optional[Graph]
    def get_tensor(self, node_id: str) -> Optional[SymbolicTensor]
    def set_tensor(self, node_id: str, tensor: SymbolicTensor) -> None
    def clear_cache(self) -> None
    def set_parameter(self, name: str, value: Any) -> None
    def get_parameter(self, name: str) -> Optional[Any]
    def set_buffer(self, name: str, value: Any) -> None
    def get_buffer(self, name: str) -> Optional[Any]
    def compute_gradients(self, loss: SymbolicTensor) -> Dict[str, SymbolicTensor]
    def zero_gradients(self) -> None
```

### GlobalContext and helpers

```python
class GlobalContext:
    @classmethod
    def get_instance(cls) -> GlobalContext
    def get_context(self, thread_id: Optional[int] = None) -> ExecutionContext
    def set_context(self, context: ExecutionContext, thread_id: Optional[int] = None) -> None
    def clear_context(self, thread_id: Optional[int] = None) -> None

def get_current_context() -> ExecutionContext
def set_current_context(context: ExecutionContext) -> None
```

### Gradient-mode context managers

`no_grad` and `enable_grad` toggle `is_grad_enabled` on the current context.

```python
from dynamicgraph import no_grad, enable_grad

with no_grad():
    ...   # graph built without gradient tracking

with enable_grad():
    ...   # gradient tracking re-enabled
```

## Notes

- The exported backend is `EagerBackend` (`"eager_numpy"`), a NumPy interpreter.
  A PyTorch backend exists internally but is only registered when `torch` is
  importable; it is not part of the top-level public API.
- `optimization_level` semantics are shared across the compiler, optimizer, and
  compilation context: `0` = none, `1` = basic, `2` = aggressive.
- For a broader architectural overview see [ARCHITECTURE.md](ARCHITECTURE.md)
  and [BLUEPRINT.md](BLUEPRINT.md).
