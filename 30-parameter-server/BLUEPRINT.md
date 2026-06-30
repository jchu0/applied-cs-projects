# Project 30: Large-Scale Parameter Server with Model Sharding

## Executive Summary

A distributed parameter server for large-scale machine learning training with model sharding. Implements async distributed optimization with push/pull pipelines, multiple consistency models (Hogwild!, SSP, BSP), gradient compression, and fault-tolerant checkpointing. Designed for training models too large to fit on a single machine.

> **Concepts covered:** [§03 Data parallelism](../../03-machine-learning-engineering/05-distributed-training/data-parallelism/data-parallelism.md) · [§03 Model parallelism](../../03-machine-learning-engineering/05-distributed-training/model-parallelism/model-parallelism.md) (sharding side). Compare to [Project 40 (DDP/FSDP-style autograd)](../40-distributed-autograd/) for the AllReduce alternative; see [Project 04](../04-ml-training-orchestrator/) for the orchestration layer that drives both. Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Parameter Sharding](#parameter-sharding)
3. [Worker Architecture](#worker-architecture)
4. [Update Engine](#update-engine)
5. [Consistency Models](#consistency-models)
6. [Failure Handling](#failure-handling)
7. [Enterprise Features](#enterprise-features)
8. [Implementation Phases](#implementation-phases)
9. [Stretch Goals](#stretch-goals)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Large-Scale Parameter Server                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Parameter Server Cluster                          │  │
│  │                                                                        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │  Shard 0    │  │  Shard 1    │  │  Shard 2    │  │  Shard N    │  │  │
│  │  │  Server     │  │  Server     │  │  Server     │  │  Server     │  │  │
│  │  │             │  │             │  │             │  │             │  │  │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │  │
│  │  │ │ Params  │ │  │ │ Params  │ │  │ │ Params  │ │  │ │ Params  │ │  │  │
│  │  │ │ [0:1M]  │ │  │ │ [1M:2M] │ │  │ │ [2M:3M] │ │  │ │ [NM:∞]  │ │  │  │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │  │
│  │  │             │  │             │  │             │  │             │  │  │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │  │
│  │  │ │ Update  │ │  │ │ Update  │ │  │ │ Update  │ │  │ │ Update  │ │  │  │
│  │  │ │ Engine  │ │  │ │ Engine  │ │  │ │ Engine  │ │  │ │ Engine  │ │  │  │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                       ↑↓                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Worker Cluster                                 │  │
│  │                                                                        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │  Worker 0   │  │  Worker 1   │  │  Worker 2   │  │  Worker M   │  │  │
│  │  │             │  │             │  │             │  │             │  │  │
│  │  │ - Compute   │  │ - Compute   │  │ - Compute   │  │ - Compute   │  │  │
│  │  │   Gradients │  │   Gradients │  │   Gradients │  │   Gradients │  │  │
│  │  │ - Push/Pull │  │ - Push/Pull │  │ - Push/Pull │  │ - Push/Pull │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Coordination & Fault Tolerance                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │ Coordinator │  │ Checkpoint  │  │  Replica    │  │   Health    │  │  │
│  │  │   Service   │  │   Manager   │  │   Manager   │  │   Monitor   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Communication Pattern

```
Worker Training Loop:
┌─────────────────────────────────────────────────────┐
│                                                      │
│   1. Pull Parameters                                 │
│      Worker ───PULL──▶ Parameter Servers             │
│              ◀─PARAMS─                               │
│                                                      │
│   2. Forward/Backward Pass                           │
│      Compute gradients locally                       │
│                                                      │
│   3. Push Gradients                                  │
│      Worker ───PUSH──▶ Parameter Servers             │
│              (gradients)                             │
│                                                      │
│   4. Servers Apply Updates                           │
│      Parameters = Parameters - lr * gradients        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Parameter Sharding

### Sharding Strategies

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class ShardInfo:
    shard_id: int
    server_address: str
    param_ranges: List[Tuple[int, int]]  # (start, end) indices
    total_params: int
    memory_bytes: int

class ShardingStrategy(ABC):
    @abstractmethod
    def compute_shards(
        self,
        model_params: Dict[str, np.ndarray],
        num_servers: int
    ) -> List[ShardInfo]:
        pass

    @abstractmethod
    def get_shard_for_param(
        self,
        param_name: str,
        param_index: int
    ) -> int:
        pass

class UniformSharding(ShardingStrategy):
    """Distribute parameters uniformly across servers."""

    def compute_shards(
        self,
        model_params: Dict[str, np.ndarray],
        num_servers: int
    ) -> List[ShardInfo]:
        # Flatten all parameters
        total_params = sum(p.size for p in model_params.values())
        params_per_shard = total_params // num_servers

        shards = []
        current_start = 0

        for i in range(num_servers):
            if i == num_servers - 1:
                # Last shard gets remainder
                end = total_params
            else:
                end = current_start + params_per_shard

            shards.append(ShardInfo(
                shard_id=i,
                server_address=f"server-{i}",
                param_ranges=[(current_start, end)],
                total_params=end - current_start,
                memory_bytes=(end - current_start) * 4  # float32
            ))

            current_start = end

        return shards

    def get_shard_for_param(
        self,
        param_name: str,
        param_index: int
    ) -> int:
        # Simple hash-based assignment
        global_idx = self._get_global_index(param_name, param_index)
        return global_idx % self.num_servers

class LayerWiseSharding(ShardingStrategy):
    """Assign entire layers to servers."""

    def compute_shards(
        self,
        model_params: Dict[str, np.ndarray],
        num_servers: int
    ) -> List[ShardInfo]:
        # Group parameters by layer
        layers = self._group_by_layer(model_params)

        # Assign layers to servers using bin packing
        assignments = self._bin_pack_layers(layers, num_servers)

        shards = []
        for server_id, layer_names in assignments.items():
            param_ranges = []
            total = 0
            for name in layer_names:
                size = model_params[name].size
                param_ranges.append((total, total + size))
                total += size

            shards.append(ShardInfo(
                shard_id=server_id,
                server_address=f"server-{server_id}",
                param_ranges=param_ranges,
                total_params=total,
                memory_bytes=total * 4
            ))

        return shards

class AdaptiveSharding(ShardingStrategy):
    """Adapt sharding based on access patterns."""

    def __init__(self):
        self.access_counts = {}

    def update_access_pattern(
        self,
        param_name: str,
        worker_id: int
    ):
        """Track which workers access which parameters."""
        key = (param_name, worker_id)
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

    def compute_shards(
        self,
        model_params: Dict[str, np.ndarray],
        num_servers: int
    ) -> List[ShardInfo]:
        # Co-locate frequently co-accessed parameters
        affinity_matrix = self._compute_affinity(model_params)

        # Cluster parameters using spectral clustering
        clusters = self._spectral_cluster(affinity_matrix, num_servers)

        return self._clusters_to_shards(clusters, model_params)
```

### Parameter Server Implementation

```python
class ParameterServer:
    """A single parameter server shard."""

    def __init__(
        self,
        shard_id: int,
        update_engine: UpdateEngine,
        consistency_model: ConsistencyModel
    ):
        self.shard_id = shard_id
        self.update_engine = update_engine
        self.consistency = consistency_model

        # Parameter storage
        self.params: Dict[str, np.ndarray] = {}
        self.param_versions: Dict[str, int] = {}

        # Gradient accumulation
        self.gradient_buffer: Dict[str, List[np.ndarray]] = {}

        # Locks for thread safety
        self.locks: Dict[str, asyncio.Lock] = {}

    async def initialize(self, params: Dict[str, np.ndarray]):
        """Initialize parameters for this shard."""
        for name, value in params.items():
            self.params[name] = value.copy()
            self.param_versions[name] = 0
            self.locks[name] = asyncio.Lock()
            self.gradient_buffer[name] = []

    async def pull(
        self,
        param_names: List[str],
        worker_id: int
    ) -> Dict[str, Tuple[np.ndarray, int]]:
        """Pull parameters to worker."""
        result = {}

        for name in param_names:
            if name in self.params:
                async with self.locks[name]:
                    result[name] = (
                        self.params[name].copy(),
                        self.param_versions[name]
                    )

        return result

    async def push(
        self,
        gradients: Dict[str, np.ndarray],
        worker_id: int,
        clock: int
    ):
        """Push gradients from worker."""
        for name, grad in gradients.items():
            if name not in self.params:
                continue

            async with self.locks[name]:
                # Check consistency model
                if not self.consistency.can_apply(
                    param_version=self.param_versions[name],
                    worker_clock=clock
                ):
                    self.gradient_buffer[name].append((grad, worker_id, clock))
                    continue

                # Apply update
                self.params[name] = self.update_engine.apply(
                    self.params[name],
                    grad
                )
                self.param_versions[name] += 1

                # Process buffered gradients if any
                await self._process_buffer(name)

    async def _process_buffer(self, param_name: str):
        """Process buffered gradients."""
        buffer = self.gradient_buffer[param_name]
        processed = []

        for i, (grad, worker_id, clock) in enumerate(buffer):
            if self.consistency.can_apply(
                param_version=self.param_versions[param_name],
                worker_clock=clock
            ):
                self.params[param_name] = self.update_engine.apply(
                    self.params[param_name],
                    grad
                )
                self.param_versions[param_name] += 1
                processed.append(i)

        # Remove processed gradients
        for i in reversed(processed):
            buffer.pop(i)

class ParameterServerCluster:
    """Manages the cluster of parameter servers."""

    def __init__(
        self,
        servers: List[ParameterServer],
        sharding_strategy: ShardingStrategy
    ):
        self.servers = {s.shard_id: s for s in servers}
        self.sharding = sharding_strategy

    async def initialize(self, model_params: Dict[str, np.ndarray]):
        """Initialize all servers with model parameters."""
        shards = self.sharding.compute_shards(
            model_params,
            len(self.servers)
        )

        for shard in shards:
            server = self.servers[shard.shard_id]
            shard_params = self._extract_shard_params(
                model_params,
                shard
            )
            await server.initialize(shard_params)

    async def pull(
        self,
        param_names: List[str],
        worker_id: int
    ) -> Dict[str, np.ndarray]:
        """Pull parameters from appropriate servers."""
        # Group params by server
        server_params = self._group_by_server(param_names)

        # Pull from each server in parallel
        tasks = []
        for server_id, names in server_params.items():
            task = self.servers[server_id].pull(names, worker_id)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Merge results
        merged = {}
        for result in results:
            for name, (value, version) in result.items():
                merged[name] = value

        return merged

    async def push(
        self,
        gradients: Dict[str, np.ndarray],
        worker_id: int,
        clock: int
    ):
        """Push gradients to appropriate servers."""
        # Group gradients by server
        server_grads = self._group_by_server(gradients)

        # Push to each server in parallel
        tasks = []
        for server_id, grads in server_grads.items():
            task = self.servers[server_id].push(grads, worker_id, clock)
            tasks.append(task)

        await asyncio.gather(*tasks)
```

---

## Worker Architecture

### Worker Implementation

```python
class Worker:
    """Training worker that computes gradients."""

    def __init__(
        self,
        worker_id: int,
        model: nn.Module,
        ps_cluster: ParameterServerCluster,
        data_loader: DataLoader,
        optimizer_config: dict
    ):
        self.worker_id = worker_id
        self.model = model
        self.ps_cluster = ps_cluster
        self.data_loader = data_loader
        self.optimizer_config = optimizer_config

        self.clock = 0
        self.param_names = [name for name, _ in model.named_parameters()]

    async def train_step(self) -> float:
        """Execute one training step."""
        # 1. Pull latest parameters
        params = await self.ps_cluster.pull(
            self.param_names,
            self.worker_id
        )
        self._load_params(params)

        # 2. Get batch
        batch = next(iter(self.data_loader))

        # 3. Forward pass
        loss = self._forward(batch)

        # 4. Backward pass
        gradients = self._backward(loss)

        # 5. Push gradients
        await self.ps_cluster.push(
            gradients,
            self.worker_id,
            self.clock
        )

        self.clock += 1
        return loss.item()

    def _load_params(self, params: Dict[str, np.ndarray]):
        """Load parameters into model."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.copy_(torch.from_numpy(params[name]))

    def _forward(self, batch) -> torch.Tensor:
        """Forward pass."""
        inputs, targets = batch
        outputs = self.model(inputs)
        return F.cross_entropy(outputs, targets)

    def _backward(self, loss: torch.Tensor) -> Dict[str, np.ndarray]:
        """Backward pass and extract gradients."""
        self.model.zero_grad()
        loss.backward()

        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().numpy()

        return gradients

class AsyncWorker(Worker):
    """Worker with async parameter updates."""

    async def train_loop(self, num_steps: int):
        """Run training loop asynchronously."""
        for step in range(num_steps):
            # Don't wait for push to complete
            loss = await self._async_train_step()

            if step % 100 == 0:
                logger.info(f"Worker {self.worker_id}, Step {step}, Loss {loss:.4f}")

    async def _async_train_step(self) -> float:
        # Pull in background
        pull_task = asyncio.create_task(
            self.ps_cluster.pull(self.param_names, self.worker_id)
        )

        # Get batch while pulling
        batch = next(iter(self.data_loader))

        # Wait for params
        params = await pull_task
        self._load_params(params)

        # Compute
        loss = self._forward(batch)
        gradients = self._backward(loss)

        # Push without waiting
        asyncio.create_task(
            self.ps_cluster.push(gradients, self.worker_id, self.clock)
        )

        self.clock += 1
        return loss.item()
```

---

## Update Engine

### Optimizer Implementations

```python
class UpdateEngine(ABC):
    """Base class for parameter update engines."""

    @abstractmethod
    def apply(
        self,
        params: np.ndarray,
        gradients: np.ndarray
    ) -> np.ndarray:
        pass

class SGDEngine(UpdateEngine):
    """Stochastic Gradient Descent."""

    def __init__(self, lr: float, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def apply(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        param_id: str = None
    ) -> np.ndarray:
        if self.momentum > 0 and param_id:
            if param_id not in self.velocity:
                self.velocity[param_id] = np.zeros_like(params)

            v = self.momentum * self.velocity[param_id] + gradients
            self.velocity[param_id] = v
            return params - self.lr * v

        return params - self.lr * gradients

class AdamEngine(UpdateEngine):
    """Adam optimizer."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = {}  # Timestep

    def apply(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        param_id: str = None
    ) -> np.ndarray:
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(params)
            self.v[param_id] = np.zeros_like(params)
            self.t[param_id] = 0

        self.t[param_id] += 1
        t = self.t[param_id]

        # Update biased first and second moment estimates
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * gradients
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (gradients ** 2)

        # Bias correction
        m_hat = self.m[param_id] / (1 - self.beta1 ** t)
        v_hat = self.v[param_id] / (1 - self.beta2 ** t)

        # Update parameters
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class LARSEngine(UpdateEngine):
    """Layer-wise Adaptive Rate Scaling."""

    def __init__(
        self,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        trust_coefficient: float = 0.001
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.trust_coeff = trust_coefficient
        self.velocity = {}

    def apply(
        self,
        params: np.ndarray,
        gradients: np.ndarray,
        param_id: str = None
    ) -> np.ndarray:
        # Compute local learning rate
        param_norm = np.linalg.norm(params)
        grad_norm = np.linalg.norm(gradients + self.weight_decay * params)

        if param_norm > 0 and grad_norm > 0:
            local_lr = self.trust_coeff * param_norm / grad_norm
        else:
            local_lr = 1.0

        # Apply weight decay
        update = gradients + self.weight_decay * params

        # Momentum
        if param_id not in self.velocity:
            self.velocity[param_id] = np.zeros_like(params)

        self.velocity[param_id] = (
            self.momentum * self.velocity[param_id] +
            self.lr * local_lr * update
        )

        return params - self.velocity[param_id]
```

---

## Consistency Models

### Consistency Implementations

```python
class ConsistencyModel(ABC):
    """Base class for consistency models."""

    @abstractmethod
    def can_apply(
        self,
        param_version: int,
        worker_clock: int
    ) -> bool:
        pass

class HogwildConsistency(ConsistencyModel):
    """No synchronization - apply immediately."""

    def can_apply(
        self,
        param_version: int,
        worker_clock: int
    ) -> bool:
        return True  # Always apply

class BSPConsistency(ConsistencyModel):
    """Bulk Synchronous Parallel - wait for all workers."""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.barriers: Dict[int, Set[int]] = {}  # clock -> workers arrived

    def worker_arrived(self, worker_id: int, clock: int):
        if clock not in self.barriers:
            self.barriers[clock] = set()
        self.barriers[clock].add(worker_id)

    def can_apply(
        self,
        param_version: int,
        worker_clock: int
    ) -> bool:
        # Apply only when all workers have reached this clock
        if worker_clock in self.barriers:
            return len(self.barriers[worker_clock]) >= self.num_workers
        return False

class SSPConsistency(ConsistencyModel):
    """Stale Synchronous Parallel - bounded staleness."""

    def __init__(self, staleness_threshold: int = 3):
        self.threshold = staleness_threshold
        self.worker_clocks: Dict[int, int] = {}

    def update_worker_clock(self, worker_id: int, clock: int):
        self.worker_clocks[worker_id] = clock

    def can_apply(
        self,
        param_version: int,
        worker_clock: int
    ) -> bool:
        if not self.worker_clocks:
            return True

        min_clock = min(self.worker_clocks.values())

        # Allow update if staleness is within threshold
        return worker_clock - min_clock <= self.threshold

class ConsistencyManager:
    """Manages consistency across the cluster."""

    def __init__(
        self,
        model: str,
        num_workers: int,
        **kwargs
    ):
        if model == "hogwild":
            self.consistency = HogwildConsistency()
        elif model == "bsp":
            self.consistency = BSPConsistency(num_workers)
        elif model == "ssp":
            self.consistency = SSPConsistency(**kwargs)
        else:
            raise ValueError(f"Unknown consistency model: {model}")

    def can_apply_gradient(
        self,
        param_version: int,
        worker_clock: int
    ) -> bool:
        return self.consistency.can_apply(param_version, worker_clock)
```

---

## Failure Handling

### Checkpointing

```python
@dataclass
class Checkpoint:
    epoch: int
    global_step: int
    params: Dict[str, np.ndarray]
    optimizer_state: Dict[str, Any]
    worker_clocks: Dict[int, int]
    timestamp: float

class CheckpointManager:
    """Manages distributed checkpointing."""

    def __init__(
        self,
        storage_path: str,
        checkpoint_interval: int = 1000,
        max_checkpoints: int = 5
    ):
        self.storage_path = storage_path
        self.interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[str] = []

    async def save_checkpoint(
        self,
        ps_cluster: ParameterServerCluster,
        epoch: int,
        global_step: int,
        worker_clocks: Dict[int, int]
    ) -> str:
        # Collect parameters from all servers
        all_params = {}
        for server in ps_cluster.servers.values():
            all_params.update(server.params)

        # Collect optimizer state
        optimizer_state = {}
        for server in ps_cluster.servers.values():
            optimizer_state[server.shard_id] = server.update_engine.get_state()

        checkpoint = Checkpoint(
            epoch=epoch,
            global_step=global_step,
            params=all_params,
            optimizer_state=optimizer_state,
            worker_clocks=worker_clocks,
            timestamp=time.time()
        )

        # Save to storage
        checkpoint_path = f"{self.storage_path}/checkpoint_{global_step}.pt"
        await self._save(checkpoint_path, checkpoint)

        # Manage checkpoint rotation
        self.checkpoints.append(checkpoint_path)
        await self._rotate_checkpoints()

        return checkpoint_path

    async def load_checkpoint(
        self,
        checkpoint_path: str,
        ps_cluster: ParameterServerCluster
    ) -> Checkpoint:
        """Load checkpoint and restore state."""
        checkpoint = await self._load(checkpoint_path)

        # Restore parameters to servers
        await ps_cluster.initialize(checkpoint.params)

        # Restore optimizer state
        for server in ps_cluster.servers.values():
            state = checkpoint.optimizer_state.get(server.shard_id, {})
            server.update_engine.load_state(state)

        return checkpoint

    async def _rotate_checkpoints(self):
        """Keep only recent checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            await self._delete(oldest)

class ReplicaManager:
    """Manages parameter server replicas for fault tolerance."""

    def __init__(
        self,
        num_replicas: int = 2,
        replication_strategy: str = "sync"
    ):
        self.num_replicas = num_replicas
        self.strategy = replication_strategy
        self.replicas: Dict[int, List[ParameterServer]] = {}

    async def replicate_update(
        self,
        primary_shard_id: int,
        param_name: str,
        new_value: np.ndarray
    ):
        """Replicate update to replicas."""
        if primary_shard_id not in self.replicas:
            return

        if self.strategy == "sync":
            # Wait for all replicas
            tasks = [
                replica.update_param(param_name, new_value)
                for replica in self.replicas[primary_shard_id]
            ]
            await asyncio.gather(*tasks)
        else:
            # Async replication
            for replica in self.replicas[primary_shard_id]:
                asyncio.create_task(replica.update_param(param_name, new_value))

    async def failover(self, failed_shard_id: int) -> ParameterServer:
        """Promote replica to primary on failure."""
        if failed_shard_id not in self.replicas:
            raise ValueError(f"No replicas for shard {failed_shard_id}")

        # Promote first replica
        new_primary = self.replicas[failed_shard_id].pop(0)
        new_primary.is_primary = True

        logger.info(f"Promoted replica to primary for shard {failed_shard_id}")

        return new_primary
```

### Health Monitoring

```python
class HealthMonitor:
    """Monitors health of parameter servers and workers."""

    def __init__(
        self,
        heartbeat_interval: int = 5,
        failure_threshold: int = 3
    ):
        self.heartbeat_interval = heartbeat_interval
        self.failure_threshold = failure_threshold
        self.last_heartbeats: Dict[str, float] = {}
        self.failure_counts: Dict[str, int] = {}

    async def start_monitoring(
        self,
        ps_cluster: ParameterServerCluster,
        replica_manager: ReplicaManager,
        checkpoint_manager: CheckpointManager
    ):
        while True:
            # Check servers
            for shard_id, server in ps_cluster.servers.items():
                key = f"server-{shard_id}"
                is_healthy = await self._check_health(server)

                if not is_healthy:
                    self.failure_counts[key] = self.failure_counts.get(key, 0) + 1

                    if self.failure_counts[key] >= self.failure_threshold:
                        # Handle failure
                        await self._handle_server_failure(
                            shard_id,
                            ps_cluster,
                            replica_manager,
                            checkpoint_manager
                        )
                else:
                    self.failure_counts[key] = 0
                    self.last_heartbeats[key] = time.time()

            await asyncio.sleep(self.heartbeat_interval)

    async def _handle_server_failure(
        self,
        shard_id: int,
        ps_cluster: ParameterServerCluster,
        replica_manager: ReplicaManager,
        checkpoint_manager: CheckpointManager
    ):
        logger.error(f"Server {shard_id} failed, initiating recovery")

        # Try failover to replica
        try:
            new_primary = await replica_manager.failover(shard_id)
            ps_cluster.servers[shard_id] = new_primary
            logger.info(f"Failover successful for shard {shard_id}")
        except ValueError:
            # No replicas, restore from checkpoint
            logger.info(f"Restoring shard {shard_id} from checkpoint")
            await checkpoint_manager.restore_shard(shard_id, ps_cluster)
```

---

## Enterprise Features

### Gradient Compression

```python
class GradientCompressor:
    """Compresses gradients to reduce communication."""

    def __init__(
        self,
        compression_type: str = "none",
        bits: int = 8,
        top_k_ratio: float = 0.01
    ):
        self.compression_type = compression_type
        self.bits = bits
        self.top_k_ratio = top_k_ratio

        # Error feedback for accumulated errors
        self.error_feedback: Dict[str, np.ndarray] = {}

    def compress(
        self,
        gradient: np.ndarray,
        param_id: str = None
    ) -> Tuple[np.ndarray, dict]:
        """Compress gradient."""
        if self.compression_type == "none":
            return gradient, {}

        elif self.compression_type == "quantize":
            return self._quantize(gradient)

        elif self.compression_type == "top_k":
            return self._top_k_sparsify(gradient)

        elif self.compression_type == "random_k":
            return self._random_k_sparsify(gradient)

    def decompress(
        self,
        compressed: np.ndarray,
        metadata: dict
    ) -> np.ndarray:
        """Decompress gradient."""
        if self.compression_type == "none":
            return compressed

        elif self.compression_type == "quantize":
            return self._dequantize(compressed, metadata)

        elif self.compression_type in ["top_k", "random_k"]:
            return self._desparsify(compressed, metadata)

    def _quantize(self, gradient: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Quantize to lower precision."""
        # Get range
        min_val = gradient.min()
        max_val = gradient.max()

        # Normalize to [0, 2^bits - 1]
        scale = (2 ** self.bits - 1) / (max_val - min_val + 1e-8)
        quantized = ((gradient - min_val) * scale).astype(np.uint8)

        return quantized, {
            "min": min_val,
            "max": max_val,
            "scale": scale
        }

    def _dequantize(
        self,
        compressed: np.ndarray,
        metadata: dict
    ) -> np.ndarray:
        """Dequantize back to float."""
        return compressed / metadata["scale"] + metadata["min"]

    def _top_k_sparsify(
        self,
        gradient: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """Keep only top-k gradients by magnitude."""
        k = int(gradient.size * self.top_k_ratio)

        flat = gradient.flatten()
        indices = np.argpartition(np.abs(flat), -k)[-k:]
        values = flat[indices]

        return np.stack([indices, values]), {
            "shape": gradient.shape,
            "k": k
        }

class MixedPrecisionStorage:
    """Store parameters in mixed precision."""

    def __init__(self, master_dtype=np.float32, storage_dtype=np.float16):
        self.master_dtype = master_dtype
        self.storage_dtype = storage_dtype

    def to_storage(self, params: np.ndarray) -> np.ndarray:
        """Convert to storage precision."""
        return params.astype(self.storage_dtype)

    def to_compute(self, params: np.ndarray) -> np.ndarray:
        """Convert to compute precision."""
        return params.astype(self.master_dtype)
```

### Staleness Control

```python
class StalenessController:
    """Controls and monitors staleness in SSP."""

    def __init__(
        self,
        initial_threshold: int = 3,
        min_threshold: int = 1,
        max_threshold: int = 10
    ):
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.staleness_history: List[float] = []

    def get_threshold(self) -> int:
        return self.threshold

    def record_staleness(self, staleness: int):
        """Record observed staleness."""
        self.staleness_history.append(staleness)

        # Adapt threshold based on history
        if len(self.staleness_history) >= 100:
            avg_staleness = np.mean(self.staleness_history[-100:])

            if avg_staleness > self.threshold * 0.8:
                # Too much staleness, increase threshold
                self.threshold = min(self.threshold + 1, self.max_threshold)
            elif avg_staleness < self.threshold * 0.3:
                # Low staleness, can decrease threshold
                self.threshold = max(self.threshold - 1, self.min_threshold)

    def get_stats(self) -> dict:
        """Get staleness statistics."""
        if not self.staleness_history:
            return {}

        recent = self.staleness_history[-100:]
        return {
            "current_threshold": self.threshold,
            "avg_staleness": np.mean(recent),
            "max_staleness": np.max(recent),
            "p99_staleness": np.percentile(recent, 99)
        }
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-4)

**Deliverables:**
- [ ] Parameter server base implementation
- [ ] Uniform sharding strategy
- [ ] Basic worker with push/pull
- [ ] SGD update engine
- [ ] Hogwild consistency

### Phase 2: Consistency Models (Weeks 5-7)

**Deliverables:**
- [ ] BSP implementation
- [ ] SSP implementation
- [ ] Consistency manager
- [ ] Barrier synchronization

### Phase 3: Advanced Optimizers (Weeks 8-10)

**Deliverables:**
- [ ] Adam optimizer
- [ ] LARS optimizer
- [ ] Momentum support
- [ ] Learning rate scheduling

### Phase 4: Fault Tolerance (Weeks 11-14)

**Deliverables:**
- [ ] Checkpointing system
- [ ] Replica management
- [ ] Health monitoring
- [ ] Automatic recovery

### Phase 5: Enterprise Features (Weeks 15-18)

**Deliverables:**
- [ ] Gradient compression
- [ ] Mixed precision storage
- [ ] Staleness control
- [ ] Performance monitoring

---

## Stretch Goals

### Integration with Distributed Training

```python
class DistributedTrainer:
    """Integrates parameter server with distributed training."""

    def __init__(
        self,
        ps_cluster: ParameterServerCluster,
        num_workers: int,
        data_parallel: bool = True
    ):
        self.ps_cluster = ps_cluster
        self.num_workers = num_workers
        self.data_parallel = data_parallel

    async def train(
        self,
        model: nn.Module,
        train_data: Dataset,
        num_epochs: int
    ):
        # Initialize parameter servers
        params = {name: p.data.numpy() for name, p in model.named_parameters()}
        await self.ps_cluster.initialize(params)

        # Partition data across workers
        partitions = self._partition_data(train_data, self.num_workers)

        # Launch workers
        workers = []
        for i in range(self.num_workers):
            worker = AsyncWorker(
                worker_id=i,
                model=copy.deepcopy(model),
                ps_cluster=self.ps_cluster,
                data_loader=DataLoader(partitions[i])
            )
            workers.append(worker)

        # Run training
        tasks = [w.train_loop(len(train_data) // self.num_workers) for w in workers]
        await asyncio.gather(*tasks)
```

### Layer-wise Adaptive Sharding

```python
class AdaptiveShardingController:
    """Dynamically adjusts sharding based on load."""

    async def rebalance(
        self,
        ps_cluster: ParameterServerCluster,
        load_stats: Dict[int, float]
    ):
        """Rebalance shards based on load."""
        avg_load = np.mean(list(load_stats.values()))

        # Find overloaded and underloaded servers
        overloaded = [s for s, l in load_stats.items() if l > avg_load * 1.3]
        underloaded = [s for s, l in load_stats.items() if l < avg_load * 0.7]

        if not overloaded or not underloaded:
            return

        # Migrate parameters from overloaded to underloaded
        for src_id in overloaded:
            for dst_id in underloaded:
                await self._migrate_params(ps_cluster, src_id, dst_id)
```

---

## File Structure

```
30-parameter-server/
├── src/
│   ├── __init__.py
│   ├── server/
│   │   ├── __init__.py
│   │   ├── parameter_server.py
│   │   ├── sharding.py
│   │   └── cluster.py
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── worker.py
│   │   └── async_worker.py
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── sgd.py
│   │   ├── adam.py
│   │   └── lars.py
│   ├── consistency/
│   │   ├── __init__.py
│   │   ├── hogwild.py
│   │   ├── bsp.py
│   │   └── ssp.py
│   ├── fault_tolerance/
│   │   ├── __init__.py
│   │   ├── checkpoint.py
│   │   ├── replica.py
│   │   └── health.py
│   └── enterprise/
│       ├── __init__.py
│       ├── compression.py
│       └── mixed_precision.py
├── config/
├── tests/
├── docs/
├── BLUEPRINT.md
├── PROGRESS.md
└── SESSION_CONTEXT.md
```

---

## Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Communication Efficiency | > 90% | With compression |
| Convergence Speed | Linear scaling | With worker count |
| Checkpoint Time | < 30s | For 10B params |
| Recovery Time | < 60s | From checkpoint |
| GPU Utilization | > 80% | During training |

---

## References

- [Parameter Server for Distributed ML](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)
- [Hogwild! Parallel SGD](https://arxiv.org/abs/1106.5730)
- [SSP Consistency](https://www.cs.cmu.edu/~seunghak/SSPTable_NIPS2013.pdf)
- [LARS Optimizer](https://arxiv.org/abs/1708.03888)
- [Gradient Compression](https://arxiv.org/abs/1712.01887)
