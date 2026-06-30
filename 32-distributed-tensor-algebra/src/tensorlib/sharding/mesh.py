"""Device mesh and sharding for distributed tensors."""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Set
import numpy as np

from ..core.tensor import LazyTensor, Device, DeviceType, array


@dataclass
class Mesh:
    """Device mesh for distributed computation."""
    devices: np.ndarray
    axis_names: Tuple[str, ...]

    def __post_init__(self):
        if len(self.axis_names) != self.devices.ndim:
            raise ValueError("Axis names must match device array dimensions")

    @property
    def shape(self) -> Dict[str, int]:
        return {name: self.devices.shape[i] for i, name in enumerate(self.axis_names)}

    @property
    def size(self) -> int:
        return self.devices.size

    def __getitem__(self, key):
        if isinstance(key, str):
            idx = self.axis_names.index(key)
            return self.devices.shape[idx]
        return self.devices[key]


def create_device_mesh(
    shape: Tuple[int, ...],
    axis_names: Tuple[str, ...],
    device_type: DeviceType = DeviceType.GPU
) -> Mesh:
    """Create a device mesh with given shape.

    Args:
        shape: Mesh dimensions
        axis_names: Names for each dimension
        device_type: Type of devices

    Returns:
        Device mesh
    """
    total_devices = int(np.prod(shape))
    devices = np.array([
        Device(device_type, i) for i in range(total_devices)
    ]).reshape(shape)
    return Mesh(devices, axis_names)


class PartitionSpec(NamedTuple):
    """Specification for how to partition a tensor across mesh axes."""
    partitions: Tuple[Optional[str], ...]

    @staticmethod
    def create(*args) -> 'PartitionSpec':
        return PartitionSpec(args)


# Alias for convenience
P = PartitionSpec.create


@dataclass
class ShardingSpec:
    """Full sharding specification for a tensor."""
    mesh: Mesh
    partition_spec: PartitionSpec

    def get_shard_shape(self, global_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute local shard shape from global shape."""
        shard_shape = list(global_shape)

        for i, axis_name in enumerate(self.partition_spec.partitions):
            if axis_name is not None and i < len(shard_shape):
                mesh_dim = self.mesh[axis_name]
                if shard_shape[i] % mesh_dim != 0:
                    raise ValueError(
                        f"Dimension {i} (size {shard_shape[i]}) not divisible "
                        f"by mesh axis {axis_name} (size {mesh_dim})"
                    )
                shard_shape[i] //= mesh_dim

        return tuple(shard_shape)

    def get_device_for_index(self, shard_index: Tuple[int, ...]) -> Device:
        """Get device that holds a particular shard."""
        mesh_coords = []
        for i, axis_name in enumerate(self.partition_spec.partitions):
            if axis_name is not None:
                mesh_axis = self.mesh.axis_names.index(axis_name)
                mesh_coords.append(shard_index[i])

        if mesh_coords:
            return self.mesh.devices[tuple(mesh_coords)]
        return self.mesh.devices.flat[0]


class ShardedTensor:
    """Tensor distributed across devices according to sharding spec."""

    def __init__(
        self,
        global_shape: Tuple[int, ...],
        dtype: np.dtype,
        sharding: ShardingSpec
    ):
        self.global_shape = global_shape
        self.dtype = dtype
        self.sharding = sharding
        self.local_shape = sharding.get_shard_shape(global_shape)
        self._shards: Dict[Device, np.ndarray] = {}

    def set_shard(self, device: Device, data: np.ndarray):
        """Set local shard on a device."""
        if data.shape != self.local_shape:
            raise ValueError(
                f"Shard shape {data.shape} doesn't match expected {self.local_shape}"
            )
        self._shards[device] = data

    def get_shard(self, device: Device) -> np.ndarray:
        """Get local shard from a device."""
        return self._shards.get(device)

    def to_global(self) -> np.ndarray:
        """Gather all shards into global tensor."""
        result = np.zeros(self.global_shape, dtype=self.dtype)

        mesh = self.sharding.mesh
        partition_spec = self.sharding.partition_spec

        for device in mesh.devices.flat:
            shard = self._shards.get(device)
            if shard is None:
                continue

            device_idx = np.where(mesh.devices == device)
            slices = []

            for i, axis_name in enumerate(partition_spec.partitions):
                if axis_name is None or i >= len(self.global_shape):
                    slices.append(slice(None))
                else:
                    mesh_axis = mesh.axis_names.index(axis_name)
                    mesh_idx = device_idx[mesh_axis][0]
                    chunk_size = self.global_shape[i] // mesh[axis_name]
                    start = mesh_idx * chunk_size
                    end = start + chunk_size
                    slices.append(slice(start, end))

            result[tuple(slices)] = shard

        return result


def shard_tensor(tensor: LazyTensor, sharding: ShardingSpec) -> ShardedTensor:
    """Shard a tensor according to specification.

    Args:
        tensor: Tensor to shard
        sharding: Sharding specification

    Returns:
        Sharded tensor
    """
    data = tensor.numpy()
    sharded = ShardedTensor(data.shape, data.dtype, sharding)

    mesh = sharding.mesh
    partition_spec = sharding.partition_spec

    for device in mesh.devices.flat:
        slices = []
        device_idx = np.where(mesh.devices == device)

        for i, axis_name in enumerate(partition_spec.partitions):
            if axis_name is None or i >= len(data.shape):
                slices.append(slice(None))
            else:
                mesh_axis = mesh.axis_names.index(axis_name)
                mesh_idx = device_idx[mesh_axis][0]
                chunk_size = data.shape[i] // mesh[axis_name]
                start = mesh_idx * chunk_size
                end = start + chunk_size
                slices.append(slice(start, end))

        sharded.set_shard(device, data[tuple(slices)].copy())

    return sharded


def unshard_tensor(sharded: ShardedTensor) -> LazyTensor:
    """Gather sharded tensor back to single tensor.

    Args:
        sharded: Sharded tensor

    Returns:
        Gathered lazy tensor
    """
    return array(sharded.to_global())


class DeviceFailover:
    """Handle device failures gracefully."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.healthy_devices: Set[Device] = set(mesh.devices.flat)
        self.failed_devices: Set[Device] = set()

    def mark_failed(self, device: Device):
        """Mark a device as failed."""
        self.healthy_devices.discard(device)
        self.failed_devices.add(device)

    def get_replacement(self, failed: Device) -> Optional[Device]:
        """Find replacement device for failed one."""
        for device in self.healthy_devices:
            if device not in self.failed_devices:
                return device
        return None

    def redistribute_shards(self, tensor: ShardedTensor) -> ShardedTensor:
        """Redistribute shards after device failure."""
        new_shards = {}

        for device, shard in tensor._shards.items():
            if device in self.failed_devices:
                replacement = self.get_replacement(device)
                if replacement:
                    new_shards[replacement] = shard
            else:
                new_shards[device] = shard

        tensor._shards = new_shards
        return tensor


class SPMDPartitioner:
    """Single Program Multiple Data partitioner."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def partition_function(
        self,
        fun,
        in_shardings: List[ShardingSpec],
        out_shardings: List[ShardingSpec]
    ):
        """Partition a function for SPMD execution.

        Args:
            fun: Function to partition
            in_shardings: Input sharding specs
            out_shardings: Output sharding specs

        Returns:
            Partitioned function
        """
        def partitioned_fun(*args):
            # Convert inputs to sharded tensors
            sharded_inputs = []
            for arg, sharding in zip(args, in_shardings):
                if sharding and isinstance(arg, LazyTensor):
                    sharded_inputs.append(shard_tensor(arg, sharding))
                else:
                    sharded_inputs.append(arg)

            # Execute on each device
            results = {}
            for device in self.mesh.devices.flat:
                local_args = []
                for inp in sharded_inputs:
                    if isinstance(inp, ShardedTensor):
                        shard = inp.get_shard(device)
                        if shard is not None:
                            local_args.append(array(shard))
                        else:
                            local_args.append(None)
                    else:
                        local_args.append(inp)

                if all(a is not None for a in local_args):
                    results[device] = fun(*local_args)

            return results

        return partitioned_fun


# Utility functions

def replicate(tensor: LazyTensor, mesh: Mesh) -> ShardedTensor:
    """Replicate tensor across all devices.

    Args:
        tensor: Tensor to replicate
        mesh: Device mesh

    Returns:
        Replicated sharded tensor
    """
    # Create partition spec with all None (replicated)
    partition_spec = PartitionSpec(tuple(None for _ in range(tensor.ndim)))
    sharding = ShardingSpec(mesh, partition_spec)

    sharded = ShardedTensor(tensor.shape, tensor.dtype, sharding)
    data = tensor.numpy()

    for device in mesh.devices.flat:
        sharded.set_shard(device, data.copy())

    return sharded


def with_sharding_constraint(tensor: LazyTensor, sharding: ShardingSpec) -> LazyTensor:
    """Apply sharding constraint to tensor.

    This is a hint for the compiler to shard the tensor.

    Args:
        tensor: Tensor to constrain
        sharding: Desired sharding

    Returns:
        Tensor with sharding metadata
    """
    # For now, just return the tensor
    # Full implementation would add metadata
    return tensor
