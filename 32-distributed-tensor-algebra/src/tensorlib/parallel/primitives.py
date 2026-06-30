"""Parallel execution primitives: pmap, vmap, collectives."""

from enum import Enum
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

from ..core.tensor import LazyTensor, Device, DeviceType, array


class CollectiveOp(Enum):
    """Collective communication operations."""
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"


# Axis name context for pmap
_axis_name_stack: List[Tuple[str, int, int]] = []


class _axis_name_context:
    """Context manager for axis names in pmap."""

    def __init__(self, name: str, idx: int, size: int):
        self.name = name
        self.idx = idx
        self.size = size

    def __enter__(self):
        _axis_name_stack.append((self.name, self.idx, self.size))

    def __exit__(self, *args):
        _axis_name_stack.pop()


def get_axis_size(axis_name: str) -> int:
    """Get size of named axis."""
    for name, idx, size in reversed(_axis_name_stack):
        if name == axis_name:
            return size
    raise ValueError(f"Unknown axis name: {axis_name}")


def get_axis_index(axis_name: str) -> int:
    """Get current index along named axis."""
    for name, idx, size in reversed(_axis_name_stack):
        if name == axis_name:
            return idx
    raise ValueError(f"Unknown axis name: {axis_name}")


# Collective operations

def psum(x: LazyTensor, axis_name: str) -> LazyTensor:
    """Sum tensor across devices along named axis."""
    # In real implementation, this would be NCCL all-reduce
    # For now, simulate by accumulating results
    return x  # Placeholder - actual implementation needs communication


def pmean(x: LazyTensor, axis_name: str) -> LazyTensor:
    """Mean across devices along named axis."""
    size = get_axis_size(axis_name)
    return psum(x, axis_name) / array(float(size))


def pmax(x: LazyTensor, axis_name: str) -> LazyTensor:
    """Max across devices along named axis."""
    return x  # Placeholder


def all_gather(x: LazyTensor, axis_name: str, axis: int = 0) -> LazyTensor:
    """Gather tensor from all devices along axis."""
    return x  # Placeholder


def broadcast(x: LazyTensor, axis_name: str, root: int = 0) -> LazyTensor:
    """Broadcast from root device to all devices."""
    return x  # Placeholder


# Parallel map

def pmap(
    fun: Callable,
    axis_name: str = 'batch',
    in_axes: Union[int, Tuple] = 0,
    out_axes: Union[int, Tuple] = 0,
    devices: List[Device] = None
) -> Callable:
    """Parallel map: execute function in parallel across devices.

    Args:
        fun: Function to parallelize
        axis_name: Name for the parallel axis
        in_axes: Which input axes to split across devices
        out_axes: Which output axes are split across devices
        devices: Devices to run on

    Returns:
        Parallelized function
    """
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    def pmapped_fun(*args):
        # Determine number of devices
        if devices:
            n_devices = len(devices)
        else:
            # Find from input shape
            for arg, in_axis in zip(args, in_axes):
                if in_axis is not None and isinstance(arg, LazyTensor):
                    n_devices = arg.shape[in_axis]
                    break
            else:
                n_devices = 8  # Default

        # Split inputs across devices
        split_args = []
        for arg, in_axis in zip(args, in_axes):
            if in_axis is None:
                # Replicate
                if isinstance(arg, LazyTensor):
                    split_args.append([arg] * n_devices)
                else:
                    split_args.append([arg] * n_devices)
            else:
                # Split along axis
                if isinstance(arg, LazyTensor):
                    splits = np.array_split(arg.numpy(), n_devices, axis=in_axis)
                    split_args.append([array(s) for s in splits])
                else:
                    split_args.append([arg] * n_devices)

        # Execute on each device
        results = []
        for device_idx in range(n_devices):
            device_args = [split[device_idx] for split in split_args]

            with _axis_name_context(axis_name, device_idx, n_devices):
                result = fun(*device_args)

            results.append(result)

        # Concatenate outputs
        if isinstance(out_axes, int):
            if isinstance(results[0], LazyTensor):
                return array(np.concatenate(
                    [r.numpy() for r in results],
                    axis=out_axes
                ))
            return results
        else:
            # Multiple outputs
            return tuple(
                array(np.concatenate(
                    [r[i].numpy() for r in results],
                    axis=out_axes[i]
                ))
                for i in range(len(out_axes))
            )

    return pmapped_fun


# Vectorizing map

def vmap(
    fun: Callable,
    in_axes: Union[int, Tuple] = 0,
    out_axes: Union[int, Tuple] = 0
) -> Callable:
    """Vectorizing map: automatically batch a function.

    Args:
        fun: Function to vectorize
        in_axes: Which axes of inputs are the batch dimension
        out_axes: Which axes of outputs are the batch dimension

    Returns:
        Vectorized function
    """
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    def vmapped_fun(*args):
        # Get batch size
        batch_size = None
        for arg, in_axis in zip(args, in_axes):
            if in_axis is not None and isinstance(arg, LazyTensor):
                batch_size = arg.shape[in_axis]
                break

        if batch_size is None:
            return fun(*args)

        # Simple implementation: loop over batch
        # Full implementation would use batching rules
        results = []
        for i in range(batch_size):
            batch_args = []
            for arg, in_axis in zip(args, in_axes):
                if in_axis is None:
                    batch_args.append(arg)
                elif isinstance(arg, LazyTensor):
                    slices = [slice(None)] * len(arg.shape)
                    slices[in_axis] = i
                    batch_args.append(array(arg.numpy()[tuple(slices)]))
                else:
                    batch_args.append(arg)

            result = fun(*batch_args)
            results.append(result)

        # Stack results
        if isinstance(out_axes, int):
            return array(np.stack([r.numpy() for r in results], axis=out_axes))
        else:
            return tuple(
                array(np.stack([r[i].numpy() for r in results], axis=out_axes[i]))
                for i in range(len(out_axes))
            )

    return vmapped_fun


def scan(
    fun: Callable,
    init: LazyTensor,
    xs: LazyTensor,
    length: Optional[int] = None
) -> Tuple[LazyTensor, LazyTensor]:
    """Scan operation (like fold but returns all intermediates).

    Args:
        fun: Function (carry, x) -> (carry, y)
        init: Initial carry value
        xs: Input sequence
        length: Optional length override

    Returns:
        (final_carry, stacked_outputs)
    """
    if length is None:
        length = xs.shape[0]

    carry = init
    ys = []

    for i in range(length):
        x = array(xs.numpy()[i])
        carry, y = fun(carry, x)
        ys.append(y)

    stacked_ys = array(np.stack([y.numpy() for y in ys], axis=0))
    return carry, stacked_ys


def checkpoint(fun: Callable) -> Callable:
    """Gradient checkpointing: trade compute for memory.

    Recomputes forward pass during backward instead of storing activations.

    Args:
        fun: Function to checkpoint

    Returns:
        Checkpointed function
    """
    def checkpointed_fun(*args):
        # Simple implementation: just call the function
        # Full implementation would not store intermediates
        return fun(*args)

    return checkpointed_fun
