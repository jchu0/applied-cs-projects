# Dynamic Graph Execution Runtime - API Documentation

## Core API

### Graph Operations

#### Graph Class
```python
class Graph:
    """Dynamic computation graph."""

    def __init__(self, name: Optional[str] = None)
    def add_node(self, node: Node) -> str
    def remove_node(self, node_id: str) -> None
    def add_edge(self, source: str, target: str, index: int = 0) -> Edge
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

# Create graph
graph = Graph(name="my_model")

# Add nodes
input_node = Node(op_type=OpType.INPUT, name="input")
add_node = Node(op_type=OpType.ADD, name="add")
output_node = Node(op_type=OpType.OUTPUT, name="output")

input_id = graph.add_node(input_node)
add_id = graph.add_node(add_node)
output_id = graph.add_node(output_node)

# Connect nodes
graph.add_edge(input_id, add_id)
graph.add_edge(add_id, output_id)

# Validate and sort
issues = graph.validate()
if not issues:
    order = graph.topological_sort()
```

### Tensor Operations

#### SymbolicTensor Class
```python
class SymbolicTensor:
    """Symbolic tensor representation."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        metadata: Optional[TensorMetadata] = None,
        concrete_value: Optional[np.ndarray] = None
    )

    @property
    def shape(self) -> Optional[Tuple[int, ...]]
    @property
    def dtype(self) -> Optional[np.dtype]
    @property
    def device(self) -> str
    @property
    def requires_grad(self) -> bool

    def detach(self) -> SymbolicTensor
    def clone(self) -> SymbolicTensor
    def to(self, device: str) -> SymbolicTensor
```

**Example:**
```python
from dynamicgraph import SymbolicTensor, TensorFactory
import numpy as np

# Create symbolic tensor
tensor = SymbolicTensor(
    metadata=TensorMetadata(
        shape=(10, 20),
        dtype=np.float32,
        requires_grad=True
    )
)

# Factory methods
zeros = TensorFactory.zeros((5, 5), dtype=np.float32)
ones = TensorFactory.ones((3, 3))
randn = TensorFactory.randn((10, 10), requires_grad=True)

# Operations
result = tensor1 + tensor2
result = tensor1 @ tensor2  # Matrix multiplication
```

### Tracing API

#### BytecodeTracer Class
```python
class BytecodeTracer:
    """Traces Python bytecode."""

    def __init__(self, graph: Optional[Graph] = None)
    def trace_function(self, func: Callable, *args, **kwargs) -> Graph
    def record_graph_break(self, reason: str) -> None
    def reset(self) -> None
```

**Example:**
```python
from dynamicgraph import BytecodeTracer

def my_function(x, y):
    z = x + y
    return z * 2

tracer = BytecodeTracer()
graph = tracer.trace_function(my_function, x, y)

# Inspect captured graph
print(f"Captured {len(graph.nodes)} operations")
print(f"Graph breaks: {tracer.graph_break_reasons}")
```

### Compilation API

#### DynamicCompiler Class
```python
class DynamicCompiler:
    """JIT compiler for dynamic graphs."""

    def __init__(
        self,
        backend: str = "eager",
        optimization_level: int = 1
    )

    def compile(self, func: Callable) -> Callable
    def compile_graph(self, graph: Graph) -> CompiledFunction
    def clear_cache(self) -> None
```

**Example:**
```python
from dynamicgraph import DynamicCompiler

compiler = DynamicCompiler(
    backend="cuda",
    optimization_level=2
)

@compiler.compile
def optimized_function(x, y):
    return x @ y + x

# Or explicit compilation
compiled = compiler.compile(my_function)
result = compiled(input1, input2)
```

### Context Management

#### ExecutionContext
```python
class ExecutionContext:
    """Execution context for graph evaluation."""

    def push_graph(self, graph: Graph) -> None
    def pop_graph(self) -> Optional[Graph]
    def get_tensor(self, node_id: str) -> Optional[SymbolicTensor]
    def set_tensor(self, node_id: str, tensor: SymbolicTensor) -> None
    def set_parameter(self, name: str, value: Any) -> None
```

**Example:**
```python
from dynamicgraph import ExecutionContext, no_grad, enable_grad

context = ExecutionContext()

# Gradient control
with no_grad():
    # Operations without gradient tracking
    output = model(input)

with enable_grad():
    # Operations with gradient tracking
    loss = criterion(output, target)
```

### Optimization API

#### GraphOptimizer Class
```python
class GraphOptimizer:
    """Graph optimization orchestrator."""

    def __init__(self, optimization_level: int = 1)
    def add_pass(self, pass_cls: Type[OptimizationPass]) -> None
    def optimize(self, graph: Graph) -> Graph
```

**Example:**
```python
from dynamicgraph import GraphOptimizer, ConstantFolding, OperatorFusion

optimizer = GraphOptimizer(optimization_level=2)
optimizer.add_pass(ConstantFolding)
optimizer.add_pass(OperatorFusion)

optimized_graph = optimizer.optimize(original_graph)
```

### Backend API

#### Backend Class
```python
class Backend:
    """Abstract backend interface."""

    @abstractmethod
    def compile(self, graph: Graph) -> CompiledCode

    @abstractmethod
    def execute(self, compiled: CompiledCode, inputs: Dict) -> Any

    @abstractmethod
    def supports_op(self, op_type: OpType) -> bool
```

**Example:**
```python
from dynamicgraph import Backend, BackendRegistry

class CUDABackend(Backend):
    def compile(self, graph):
        # Compile to CUDA kernels
        pass

    def execute(self, compiled, inputs):
        # Execute on GPU
        pass

    def supports_op(self, op_type):
        return op_type in self.supported_ops

# Register backend
BackendRegistry.register("cuda", CUDABackend)
```

## Advanced Features

### Custom Operations

```python
from dynamicgraph import register_op, OpType

@register_op("my_custom_op")
def custom_op_impl(inputs, attributes):
    """Custom operation implementation."""
    x, y = inputs
    scale = attributes.get("scale", 1.0)
    return (x + y) * scale

# Use in graph
node = Node(
    op_type=OpType.CUSTOM,
    name="my_custom_op",
    attributes={"scale": 2.0}
)
```

### Graph Transformations

```python
from dynamicgraph import GraphTransform

class MyTransform(GraphTransform):
    def apply(self, graph: Graph) -> Graph:
        # Transform graph
        for node_id, node in graph.nodes.items():
            if node.op_type == OpType.ADD:
                # Transform ADD operations
                pass
        return graph

transform = MyTransform()
transformed = transform.apply(graph)
```

### Profiling

```python
from dynamicgraph import Profiler

profiler = Profiler()

with profiler:
    result = compiled_func(input)

# Get profiling results
stats = profiler.get_stats()
print(f"Execution time: {stats['execution_time_ms']}ms")
print(f"Memory used: {stats['memory_bytes']} bytes")
```

## Error Handling

```python
from dynamicgraph import CompilationError, GraphBreakError

try:
    graph = tracer.trace_function(func, *args)
except GraphBreakError as e:
    print(f"Graph break: {e.reason}")
    # Handle partial compilation

try:
    compiled = compiler.compile_graph(graph)
except CompilationError as e:
    print(f"Compilation failed: {e.message}")
    # Fall back to eager execution
```

## Best Practices

1. **Graph Size Management**
   - Keep graphs under 10,000 nodes for optimal performance
   - Use graph breaks strategically for large models

2. **Optimization Levels**
   - Level 0: No optimization (debugging)
   - Level 1: Basic optimizations (default)
   - Level 2: Aggressive optimizations

3. **Backend Selection**
   - Use "eager" for debugging
   - Use "cuda" for GPU acceleration
   - Use "cpu" for optimized CPU execution

4. **Memory Management**
   - Call `clear_cache()` periodically for long-running applications
   - Use context managers for automatic cleanup
   - Monitor memory usage with profiling tools