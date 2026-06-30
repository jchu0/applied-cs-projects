"""Backend implementations for graph execution."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..core.graph import Graph, Node, OpType


@dataclass
class CompiledFunction:
    """A compiled function from a graph."""
    execute_fn: Callable
    graph: Graph
    backend_name: str
    input_names: List[str]
    output_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the compiled function."""
        return self.execute_fn(*args, **kwargs)


class Backend(ABC):
    """Base class for execution backends."""

    @abstractmethod
    def name(self) -> str:
        """Get backend name."""
        pass

    @abstractmethod
    def compile(self, graph: Graph) -> CompiledFunction:
        """Compile graph to executable function."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass

    def supports_op(self, op_type: OpType) -> bool:
        """Check if backend supports an operation type."""
        return True  # Default: support all ops


class EagerBackend(Backend):
    """
    Execute graph using NumPy eager mode.

    This backend interprets the graph at runtime, executing each
    operation using NumPy functions.
    """

    def __init__(self):
        self._op_map = self._build_op_map()

    def name(self) -> str:
        return "eager_numpy"

    def is_available(self) -> bool:
        return True  # NumPy is always available

    def compile(self, graph: Graph) -> CompiledFunction:
        """Generate function that executes operations eagerly."""

        # Get topological order
        try:
            exec_order = graph.topological_sort()
        except ValueError as e:
            raise RuntimeError(f"Cannot compile graph with cycles: {e}")

        # Extract input and output info
        input_names = []
        for node_id in graph.input_nodes:
            node = graph.nodes[node_id]
            input_names.append(node.name or node_id)

        output_names = []
        for node_id in graph.output_nodes:
            node = graph.nodes[node_id]
            output_names.append(node.name or node_id)

        # Create execution function
        op_map = self._op_map

        def execute(*args, **kwargs):
            # Build value storage
            values: Dict[str, Any] = {}

            # Map inputs
            input_idx = 0
            for node_id in exec_order:
                node = graph.nodes[node_id]
                if node.op_type == OpType.INPUT:
                    if input_idx < len(args):
                        values[node_id] = args[input_idx]
                    elif node.name and node.name in kwargs:
                        values[node_id] = kwargs[node.name]
                    else:
                        raise ValueError(f"Missing input for {node.name or node_id}")
                    input_idx += 1

            # Execute in topological order
            for node_id in exec_order:
                node = graph.nodes[node_id]

                if node.op_type == OpType.INPUT:
                    continue  # Already handled

                if node.op_type == OpType.OUTPUT:
                    # Output just passes through
                    if node.inputs:
                        values[node_id] = values[node.inputs[0]]
                    continue

                if node.op_type == OpType.CONSTANT:
                    values[node_id] = node.attributes.get("value")
                    continue

                if node.op_type == OpType.PARAMETER:
                    values[node_id] = node.attributes.get("value")
                    continue

                if node.op_type == OpType.BUFFER:
                    values[node_id] = node.attributes.get("value")
                    continue

                # Get input values
                input_values = [values.get(inp) for inp in node.inputs]

                # Execute operation
                op_fn = op_map.get(node.op_type)
                if op_fn is None:
                    raise RuntimeError(f"Unsupported operation: {node.op_type}")

                try:
                    result = op_fn(input_values, node.attributes)
                    values[node_id] = result
                except Exception as e:
                    raise RuntimeError(f"Error executing {node.op_type} at {node_id}: {e}")

            # Gather outputs
            outputs = []
            for node_id in graph.output_nodes:
                if node_id in values:
                    outputs.append(values[node_id])
                else:
                    # Try to find the output's input
                    node = graph.nodes[node_id]
                    if node.inputs and node.inputs[0] in values:
                        outputs.append(values[node.inputs[0]])

            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

        return CompiledFunction(
            execute_fn=execute,
            graph=graph,
            backend_name=self.name(),
            input_names=input_names,
            output_names=output_names,
        )

    def _build_op_map(self) -> Dict[OpType, Callable]:
        """Build mapping from operations to NumPy functions."""
        return {
            # Arithmetic
            OpType.ADD: lambda inputs, attrs: np.add(inputs[0], inputs[1]),
            OpType.SUB: lambda inputs, attrs: np.subtract(inputs[0], inputs[1]),
            OpType.MUL: lambda inputs, attrs: np.multiply(inputs[0], inputs[1]),
            OpType.DIV: lambda inputs, attrs: np.divide(inputs[0], inputs[1]),
            OpType.MATMUL: lambda inputs, attrs: np.matmul(inputs[0], inputs[1]),

            # Activations
            OpType.RELU: lambda inputs, attrs: np.maximum(inputs[0], 0),
            OpType.SIGMOID: lambda inputs, attrs: 1 / (1 + np.exp(-inputs[0])),
            OpType.SOFTMAX: self._softmax,

            # Shape ops
            OpType.RESHAPE: lambda inputs, attrs: np.reshape(inputs[0], attrs.get("shape")),
            OpType.TRANSPOSE: lambda inputs, attrs: np.transpose(inputs[0], attrs.get("axes")),
            OpType.PERMUTE: lambda inputs, attrs: np.transpose(inputs[0], attrs.get("dims")),
            OpType.SQUEEZE: lambda inputs, attrs: np.squeeze(inputs[0], attrs.get("dim")),
            OpType.UNSQUEEZE: lambda inputs, attrs: np.expand_dims(inputs[0], attrs.get("dim", 0)),

            # Reductions
            OpType.SUM: self._reduce_sum,
            OpType.MEAN: self._reduce_mean,
            OpType.MAX: self._reduce_max,
            OpType.MIN: self._reduce_min,

            # Memory ops
            OpType.COPY: lambda inputs, attrs: np.copy(inputs[0]),
            OpType.CLONE: lambda inputs, attrs: np.copy(inputs[0]),
            OpType.DETACH: lambda inputs, attrs: inputs[0],  # No-op for numpy

            # Neural network
            OpType.LINEAR: self._linear,
            OpType.CONV2D: self._conv2d,
            OpType.BATCHNORM: self._batchnorm,
            OpType.DROPOUT: lambda inputs, attrs: inputs[0],  # Identity at inference

            # Custom
            OpType.CUSTOM: self._custom_op,
        }

    def _softmax(self, inputs: List[Any], attrs: Dict[str, Any]) -> np.ndarray:
        """Softmax with numerical stability."""
        x = inputs[0]
        dim = attrs.get("dim", -1)
        x_max = np.max(x, axis=dim, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=dim, keepdims=True)

    def _reduce_sum(self, inputs: List[Any], attrs: Dict[str, Any]) -> np.ndarray:
        x = inputs[0]
        dim = attrs.get("dim")
        keepdim = attrs.get("keepdim", False)
        return np.sum(x, axis=dim, keepdims=keepdim)

    def _reduce_mean(self, inputs: List[Any], attrs: Dict[str, Any]) -> np.ndarray:
        x = inputs[0]
        dim = attrs.get("dim")
        keepdim = attrs.get("keepdim", False)
        return np.mean(x, axis=dim, keepdims=keepdim)

    def _reduce_max(self, inputs: List[Any], attrs: Dict[str, Any]) -> np.ndarray:
        x = inputs[0]
        dim = attrs.get("dim")
        keepdim = attrs.get("keepdim", False)
        return np.max(x, axis=dim, keepdims=keepdim)

    def _reduce_min(self, inputs: List[Any], attrs: Dict[str, Any]) -> np.ndarray:
        x = inputs[0]
        dim = attrs.get("dim")
        keepdim = attrs.get("keepdim", False)
        return np.min(x, axis=dim, keepdims=keepdim)

    def _linear(self, inputs: List[Any], attrs: Dict[str, Any]) -> np.ndarray:
        """Linear layer: y = x @ W^T + b."""
        x = inputs[0]
        weight = inputs[1] if len(inputs) > 1 else attrs.get("weight")
        bias = inputs[2] if len(inputs) > 2 else attrs.get("bias")

        result = np.matmul(x, weight.T if weight is not None else x)
        if bias is not None:
            result = result + bias
        return result

    def _conv2d(self, inputs: List[Any], attrs: Dict[str, Any]) -> np.ndarray:
        """Basic 2D convolution (simplified implementation)."""
        x = inputs[0]  # NCHW format
        weight = inputs[1] if len(inputs) > 1 else attrs.get("weight")

        if weight is None:
            return x

        # Get parameters
        stride = attrs.get("stride", 1)
        padding = attrs.get("padding", 0)

        # Simple implementation using numpy (not optimized)
        n, c_in, h_in, w_in = x.shape
        c_out, _, kh, kw = weight.shape

        # Apply padding
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

        # Output dimensions
        h_out = (h_in + 2 * padding - kh) // stride + 1
        w_out = (w_in + 2 * padding - kw) // stride + 1

        # Initialize output
        output = np.zeros((n, c_out, h_out, w_out))

        # Convolve
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride
                patch = x[:, :, h_start:h_start+kh, w_start:w_start+kw]
                output[:, :, i, j] = np.tensordot(patch, weight, axes=([1,2,3], [1,2,3]))

        # Add bias if present
        bias = attrs.get("bias")
        if bias is not None:
            output += bias.reshape(1, -1, 1, 1)

        return output

    def _batchnorm(self, inputs: List[Any], attrs: Dict[str, Any]) -> np.ndarray:
        """Batch normalization."""
        x = inputs[0]
        eps = attrs.get("eps", 1e-5)

        # Get parameters
        gamma = attrs.get("weight") or attrs.get("gamma")
        beta = attrs.get("bias") or attrs.get("beta")
        running_mean = attrs.get("running_mean")
        running_var = attrs.get("running_var")

        if running_mean is not None and running_var is not None:
            # Inference mode: use running statistics
            mean = running_mean
            var = running_var
        else:
            # Training mode: compute batch statistics
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + eps)

        # Scale and shift
        if gamma is not None:
            x_norm = x_norm * gamma.reshape(1, -1, 1, 1)
        if beta is not None:
            x_norm = x_norm + beta.reshape(1, -1, 1, 1)

        return x_norm

    def _custom_op(self, inputs: List[Any], attrs: Dict[str, Any]) -> np.ndarray:
        """Handle custom/fused operations."""
        op_name = attrs.get("name", "")

        if op_name == "fused_conv_bn_relu":
            # Execute fused conv + batchnorm + relu
            x = self._conv2d(inputs, attrs)
            x = self._batchnorm([x], {
                "weight": attrs.get("bn_weight"),
                "bias": attrs.get("bn_bias"),
                "running_mean": attrs.get("bn_running_mean"),
                "running_var": attrs.get("bn_running_var"),
                "eps": attrs.get("bn_eps", 1e-5),
            })
            return np.maximum(x, 0)  # ReLU

        # Unknown custom op - just return first input
        return inputs[0] if inputs else None


class TorchBackend(Backend):
    """
    Execute graph using PyTorch.

    This backend compiles to PyTorch operations for GPU acceleration.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._torch = None

    def name(self) -> str:
        return f"torch_{self.device}"

    def is_available(self) -> bool:
        try:
            import torch
            self._torch = torch
            if self.device.startswith("cuda"):
                return torch.cuda.is_available()
            return True
        except ImportError:
            return False

    def compile(self, graph: Graph) -> CompiledFunction:
        """Generate function using PyTorch."""
        if not self.is_available():
            raise RuntimeError("PyTorch backend not available")

        torch = self._torch

        # Build op map
        op_map = {
            OpType.ADD: torch.add,
            OpType.SUB: torch.sub,
            OpType.MUL: torch.mul,
            OpType.DIV: torch.div,
            OpType.MATMUL: torch.matmul,
            OpType.RELU: torch.relu,
            OpType.SIGMOID: torch.sigmoid,
            OpType.SOFTMAX: lambda x, dim=-1: torch.softmax(x, dim=dim),
        }

        exec_order = graph.topological_sort()
        device = self.device

        def execute(*args):
            # Convert inputs to tensors
            values = {}
            input_idx = 0

            for node_id in exec_order:
                node = graph.nodes[node_id]
                if node.op_type == OpType.INPUT:
                    arg = args[input_idx]
                    if not isinstance(arg, torch.Tensor):
                        arg = torch.tensor(arg, device=device)
                    else:
                        arg = arg.to(device)
                    values[node_id] = arg
                    input_idx += 1

            for node_id in exec_order:
                node = graph.nodes[node_id]

                if node.op_type in (OpType.INPUT, OpType.OUTPUT):
                    continue

                if node.op_type == OpType.CONSTANT:
                    val = node.attributes.get("value")
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor(val, device=device)
                    values[node_id] = val
                    continue

                input_tensors = [values[inp] for inp in node.inputs]
                op_fn = op_map.get(node.op_type)

                if op_fn:
                    values[node_id] = op_fn(*input_tensors)
                else:
                    raise RuntimeError(f"Unsupported op: {node.op_type}")

            outputs = [values[o] for o in graph.output_nodes]
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        input_names = [graph.nodes[n].name or n for n in graph.input_nodes]
        output_names = [graph.nodes[n].name or n for n in graph.output_nodes]

        return CompiledFunction(
            execute_fn=execute,
            graph=graph,
            backend_name=self.name(),
            input_names=input_names,
            output_names=output_names,
        )


class BackendRegistry:
    """Registry of available backends."""

    _backends: Dict[str, Backend] = {}
    _default_backend: Optional[str] = None

    @classmethod
    def register(cls, backend: Backend, set_default: bool = False):
        """Register a backend."""
        cls._backends[backend.name()] = backend
        if set_default or cls._default_backend is None:
            cls._default_backend = backend.name()

    @classmethod
    def get(cls, name: str) -> Optional[Backend]:
        """Get a backend by name."""
        return cls._backends.get(name)

    @classmethod
    def get_default(cls) -> Optional[Backend]:
        """Get the default backend."""
        if cls._default_backend:
            return cls._backends.get(cls._default_backend)
        return None

    @classmethod
    def set_default(cls, name: str):
        """Set the default backend."""
        if name not in cls._backends:
            raise ValueError(f"Backend '{name}' not registered")
        cls._default_backend = name

    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backends."""
        return list(cls._backends.keys())

    @classmethod
    def list_available(cls) -> List[str]:
        """List available backends."""
        return [name for name, backend in cls._backends.items()
                if backend.is_available()]


# Register default backends
BackendRegistry.register(EagerBackend(), set_default=True)

# Try to register PyTorch backend
try:
    torch_backend = TorchBackend()
    if torch_backend.is_available():
        BackendRegistry.register(torch_backend)

        # Also try CUDA
        cuda_backend = TorchBackend(device="cuda")
        if cuda_backend.is_available():
            BackendRegistry.register(cuda_backend)
except Exception:
    pass
