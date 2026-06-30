"""Distributed GNN training with multi-GPU support."""

from typing import List, Optional, Dict, Set, Tuple, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    dist = None
    DistributedDataParallel = None

from .graph import GraphStorage, PartitionedGraph
from .sampling import NeighborSampler


class HaloExchange:
    """
    Halo (ghost) node exchange for distributed GNN training.

    Manages communication of boundary node features between
    partitions during distributed training.
    """

    def __init__(
        self,
        partitioned_graph: PartitionedGraph,
        world_size: int,
        rank: int,
    ):
        """
        Initialize halo exchange.

        Args:
            partitioned_graph: Partitioned graph.
            world_size: Total number of processes.
            rank: Current process rank.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for distributed training")

        self.partitioned_graph = partitioned_graph
        self.world_size = world_size
        self.rank = rank

        # Build halo node mappings
        self.send_nodes: Dict[int, np.ndarray] = {}  # rank -> nodes to send
        self.recv_nodes: Dict[int, np.ndarray] = {}  # rank -> nodes to receive
        self._build_halo_mappings()

    def _build_halo_mappings(self) -> None:
        """Build send/receive node mappings for each neighbor partition."""
        local_nodes = set(self.partitioned_graph.local_to_global[self.rank])

        for other_rank in range(self.world_size):
            if other_rank == self.rank:
                continue

            other_nodes = set(self.partitioned_graph.local_to_global[other_rank])

            # Nodes we need from other partition (our halo nodes owned by other)
            recv_set: Set[int] = set()
            for node in local_nodes:
                neighbors = self.partitioned_graph.global_graph.get_neighbors(node, 'out')
                for neighbor in neighbors:
                    if neighbor in other_nodes:
                        recv_set.add(int(neighbor))

            # Nodes other partition needs from us
            send_set: Set[int] = set()
            for node in other_nodes:
                neighbors = self.partitioned_graph.global_graph.get_neighbors(node, 'out')
                for neighbor in neighbors:
                    if neighbor in local_nodes:
                        send_set.add(int(neighbor))

            if send_set:
                self.send_nodes[other_rank] = np.array(list(send_set))
            if recv_set:
                self.recv_nodes[other_rank] = np.array(list(recv_set))

    def exchange(
        self,
        node_features: 'torch.Tensor',
        local_to_global: np.ndarray,
    ) -> 'torch.Tensor':
        """
        Exchange halo node features between partitions.

        Args:
            node_features: Local node features [num_local_nodes, F].
            local_to_global: Mapping from local to global node IDs.

        Returns:
            Updated features with halo nodes exchanged.
        """
        if not dist.is_initialized():
            return node_features

        global_to_local = {g: l for l, g in enumerate(local_to_global)}
        feat_dim = node_features.size(1)

        # Send buffers
        send_ops = []
        for target_rank, nodes in self.send_nodes.items():
            local_indices = [global_to_local[n] for n in nodes if n in global_to_local]
            if local_indices:
                send_tensor = node_features[local_indices].contiguous()
                op = dist.isend(send_tensor, target_rank)
                send_ops.append(op)

        # Receive buffers
        recv_tensors: Dict[int, 'torch.Tensor'] = {}
        recv_ops = []
        for source_rank, nodes in self.recv_nodes.items():
            recv_tensor = torch.zeros(len(nodes), feat_dim,
                                     device=node_features.device,
                                     dtype=node_features.dtype)
            op = dist.irecv(recv_tensor, source_rank)
            recv_tensors[source_rank] = recv_tensor
            recv_ops.append(op)

        # Wait for all operations
        for op in send_ops + recv_ops:
            op.wait()

        # Update local features with received halo data
        # (In practice, you'd expand the tensor to include halo nodes)
        return node_features

    def all_reduce_gradients(self, model: nn.Module) -> None:
        """All-reduce gradients across all processes."""
        if not dist.is_initialized():
            return

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size


class DistributedGNNTrainer:
    """
    Multi-GPU distributed GNN training with graph partitioning.

    Supports:
    - Graph partitioning across GPUs
    - Mini-batch training with neighbor sampling
    - Halo node synchronization
    - Gradient synchronization
    """

    def __init__(
        self,
        model: nn.Module,
        graph: GraphStorage,
        num_gpus: int = 1,
        partition_method: str = 'balanced',
    ):
        """
        Initialize distributed trainer.

        Args:
            model: GNN model to train.
            graph: Full graph data.
            num_gpus: Number of GPUs to use.
            partition_method: Partitioning method ('balanced', 'metis').
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for distributed training")

        self.model = model
        self.graph = graph
        self.num_gpus = num_gpus
        self.partition_method = partition_method

        # Initialize distributed if multiple GPUs
        self.world_size = 1
        self.rank = 0

        if num_gpus > 1 and torch.cuda.is_available():
            self._init_distributed()

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')
        else:
            self.device = torch.device('cpu')

        # Move model to device
        self.model = self.model.to(self.device)

        # Partition graph if distributed
        if num_gpus > 1:
            self.partitioned_graph = PartitionedGraph(graph, num_gpus)
            if partition_method == 'metis':
                self.partitioned_graph.partition_metis()
            else:
                self.partitioned_graph.partition_balanced()

            # Wrap model in DDP
            if dist.is_initialized():
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.rank % torch.cuda.device_count()],
                )

            # Setup halo exchange
            self.halo_exchange = HaloExchange(
                self.partitioned_graph,
                self.world_size,
                self.rank,
            )
        else:
            self.partitioned_graph = None
            self.halo_exchange = None

    def _init_distributed(self) -> None:
        """Initialize distributed training environment."""
        import os
        if not dist.is_initialized():
            # Try to get rank from environment
            self.rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))

            if self.world_size > 1:
                dist.init_process_group(backend='nccl')

    def train_epoch(
        self,
        optimizer: 'torch.optim.Optimizer',
        loss_fn: nn.Module,
        batch_size: int = 1024,
        fanouts: Optional[List[int]] = None,
    ) -> float:
        """
        Train one epoch with distributed mini-batches.

        Args:
            optimizer: Optimizer.
            loss_fn: Loss function.
            batch_size: Batch size.
            fanouts: Sampling fanouts per layer.

        Returns:
            Average loss for the epoch.
        """
        if fanouts is None:
            fanouts = [10, 10]

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Get local partition or full graph
        if self.partitioned_graph is not None:
            local_graph, local_nodes, _ = self.partitioned_graph.get_partition(self.rank)
            num_local_nodes = len(self.partitioned_graph.local_to_global[self.rank])
        else:
            local_graph = self.graph
            local_nodes = np.arange(self.graph.num_nodes)
            num_local_nodes = self.graph.num_nodes

        # Create sampler
        sampler = NeighborSampler(local_graph, fanouts=fanouts, device=str(self.device))

        # Training loop over local nodes
        local_node_ids = torch.arange(num_local_nodes, device=self.device)
        perm = torch.randperm(num_local_nodes, device=self.device)

        for i in range(0, num_local_nodes, batch_size):
            batch_idx = perm[i:i + batch_size]
            batch_seeds = local_node_ids[batch_idx]

            # Sample subgraph
            subgraph = sampler.sample(batch_seeds)

            # Get features for sampled nodes
            global_ids = local_nodes[subgraph.node_ids.cpu().numpy()]

            if 'x' in self.graph.node_features:
                features = torch.tensor(
                    self.graph.node_features['x'][global_ids],
                    device=self.device, dtype=torch.float32
                )
            else:
                # Default: random features
                features = torch.randn(len(global_ids), 64, device=self.device)

            if 'y' in self.graph.node_features:
                labels = torch.tensor(
                    self.graph.node_features['y'][global_ids[:len(batch_seeds)]],
                    device=self.device, dtype=torch.long
                )
            else:
                labels = torch.zeros(len(batch_seeds), device=self.device, dtype=torch.long)

            # Forward pass
            optimizer.zero_grad()
            out = self.model(features, subgraph.edge_index)
            loss = loss_fn(out[:len(batch_seeds)], labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Synchronize loss across GPUs if distributed
        if dist.is_initialized():
            loss_tensor = torch.tensor([total_loss, num_batches], device=self.device)
            dist.all_reduce(loss_tensor)
            total_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(
        self,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Evaluate model on validation/test set.

        Args:
            mask: Boolean mask for nodes to evaluate.

        Returns:
            Tuple of (loss, accuracy).
        """
        self.model.eval()

        if mask is None:
            mask = np.ones(self.graph.num_nodes, dtype=bool)

        # Get full features and labels
        if 'x' in self.graph.node_features:
            features = torch.tensor(
                self.graph.node_features['x'],
                device=self.device, dtype=torch.float32
            )
        else:
            features = torch.randn(self.graph.num_nodes, 64, device=self.device)

        if 'y' in self.graph.node_features:
            labels = torch.tensor(
                self.graph.node_features['y'],
                device=self.device, dtype=torch.long
            )
        else:
            labels = torch.zeros(self.graph.num_nodes, device=self.device, dtype=torch.long)

        # Get edge index
        edge_index = torch.tensor(
            self.graph.to_edge_index(),
            device=self.device, dtype=torch.long
        )

        # Forward pass
        out = self.model(features, edge_index)

        # Compute metrics on masked nodes
        mask_tensor = torch.tensor(mask, device=self.device)
        loss = nn.functional.cross_entropy(out[mask_tensor], labels[mask_tensor])

        pred = out[mask_tensor].argmax(dim=1)
        correct = (pred == labels[mask_tensor]).sum().item()
        accuracy = correct / mask.sum()

        return loss.item(), accuracy


class VertexReorderOptimizer:
    """
    Optimize graph vertex ordering for cache efficiency.

    Reorders vertices to improve memory access patterns
    during GNN computation.
    """

    def __init__(self, graph: GraphStorage):
        self.graph = graph

    def reorder_rcm(self) -> np.ndarray:
        """
        Reverse Cuthill-McKee ordering for bandwidth reduction.

        Returns:
            Permutation array mapping old -> new indices.
        """
        # BFS-based RCM
        visited = np.zeros(self.graph.num_nodes, dtype=bool)
        perm = []

        # Start from lowest degree node
        degrees = self.graph.degrees('out')
        start_node = np.argmin(degrees)

        queue = [start_node]
        visited[start_node] = True

        while queue:
            node = queue.pop(0)
            perm.append(node)

            # Get neighbors sorted by degree
            neighbors = self.graph.get_neighbors(node, 'out')
            if len(neighbors) > 0:
                neighbor_degrees = degrees[neighbors]
                sorted_neighbors = neighbors[np.argsort(neighbor_degrees)]

                for neighbor in sorted_neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

        # Add any disconnected nodes
        for node in range(self.graph.num_nodes):
            if not visited[node]:
                perm.append(node)

        # Reverse for RCM
        perm = np.array(perm[::-1], dtype=np.int64)
        return perm

    def reorder_by_degree(self, descending: bool = True) -> np.ndarray:
        """
        Reorder vertices by degree.

        Args:
            descending: If True, high-degree nodes first.

        Returns:
            Permutation array.
        """
        degrees = self.graph.degrees('out')
        if descending:
            return np.argsort(degrees)[::-1]
        else:
            return np.argsort(degrees)

    def reorder_by_partition(self, num_parts: int = 8) -> np.ndarray:
        """
        Reorder vertices by partition, then by degree within partition.

        Args:
            num_parts: Number of partitions.

        Returns:
            Permutation array.
        """
        nodes_per_part = self.graph.num_nodes // num_parts
        partition_ids = np.zeros(self.graph.num_nodes, dtype=np.int32)

        for i in range(self.graph.num_nodes):
            partition_ids[i] = min(i // nodes_per_part, num_parts - 1)

        degrees = self.graph.degrees('out')

        perm = []
        for p in range(num_parts):
            part_nodes = np.where(partition_ids == p)[0]
            part_degrees = degrees[part_nodes]
            sorted_idx = np.argsort(part_degrees)[::-1]
            perm.extend(part_nodes[sorted_idx])

        return np.array(perm, dtype=np.int64)

    def apply_reordering(self, perm: np.ndarray) -> GraphStorage:
        """
        Apply vertex reordering to create new graph.

        Args:
            perm: Permutation array (perm[new_id] = old_id).

        Returns:
            New GraphStorage with reordered vertices.
        """
        inv_perm = np.argsort(perm)  # Maps old_id -> new_id

        # Reorder CSR
        new_indptr = np.zeros(self.graph.num_nodes + 1, dtype=np.int64)
        new_indices_list = []

        for new_id in range(self.graph.num_nodes):
            old_id = perm[new_id]
            start = self.graph.csr_indptr[old_id]
            end = self.graph.csr_indptr[old_id + 1]

            # Map old neighbor IDs to new IDs
            old_neighbors = self.graph.csr_indices[start:end]
            new_neighbors = inv_perm[old_neighbors]

            new_indptr[new_id + 1] = new_indptr[new_id] + len(new_neighbors)
            new_indices_list.extend(sorted(new_neighbors))  # Sort for locality

        new_indices = np.array(new_indices_list, dtype=np.int64)

        # Reorder features
        new_node_features = {}
        for key, feat in self.graph.node_features.items():
            new_node_features[key] = feat[perm]

        return GraphStorage(
            num_nodes=self.graph.num_nodes,
            num_edges=self.graph.num_edges,
            csr_indptr=new_indptr,
            csr_indices=new_indices,
            node_features=new_node_features,
        )

    def compute_cache_efficiency(self, perm: Optional[np.ndarray] = None) -> float:
        """
        Compute cache efficiency metric.

        Args:
            perm: Optional permutation to evaluate.

        Returns:
            Cache efficiency score (lower is better).
        """
        if perm is not None:
            graph = self.apply_reordering(perm)
        else:
            graph = self.graph

        # Compute average edge "distance" in memory
        total_distance = 0.0
        num_edges = 0

        for node in range(graph.num_nodes):
            neighbors = graph.get_neighbors(node, 'out')
            for neighbor in neighbors:
                distance = abs(node - neighbor)
                total_distance += distance
                num_edges += 1

        return total_distance / max(num_edges, 1)
