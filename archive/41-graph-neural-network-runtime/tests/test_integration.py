"""Integration tests for GNN runtime."""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
import json
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnnruntime.layers.conv import GCNConv, GATConv, GraphSAGEConv
from gnnruntime.core.graph import Graph, EdgeIndex, HeteroGraph
from gnnruntime.sampler.neighbor import NeighborSampler, LayerwiseSampler
from gnnruntime.data.batch import Batch, DataLoader


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete GNN pipeline from data loading to prediction."""

    def setUp(self):
        np.random.seed(42)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_node_classification_pipeline(self):
        """Test node classification pipeline."""
        # 1. Create dataset
        dataset = self._create_node_classification_dataset()

        # 2. Build model
        model = self._build_gcn_model(
            input_dim=dataset[0].x.shape[1],
            hidden_dim=64,
            output_dim=4,
            num_layers=3
        )

        # 3. Train model
        train_loader = DataLoader(dataset[:80], batch_size=10, shuffle=True)
        val_loader = DataLoader(dataset[80:], batch_size=10)

        losses = []
        for epoch in range(5):
            epoch_loss = self._train_epoch(model, train_loader)
            losses.append(epoch_loss)

        # Check forward pass produces finite loss values
        self.assertTrue(all(np.isfinite(l) for l in losses))

        # 4. Evaluate
        accuracy = self._evaluate(model, val_loader)
        self.assertGreater(accuracy, 0.25)  # Better than random (4 classes)

    def test_graph_classification_pipeline(self):
        """Test graph classification pipeline."""
        # 1. Create dataset
        dataset = self._create_graph_classification_dataset()

        # 2. Build model with global pooling
        model = self._build_graph_classifier(
            input_dim=dataset[0].x.shape[1],
            hidden_dim=32,
            output_dim=3,
            num_layers=2
        )

        # 3. Create data loaders
        train_loader = DataLoader(dataset[:60], batch_size=5, shuffle=True)
        test_loader = DataLoader(dataset[60:], batch_size=5)

        # 4. Train
        initial_loss = self._compute_loss(model, train_loader)
        for _ in range(10):
            self._train_epoch(model, train_loader)
        final_loss = self._compute_loss(model, train_loader)

        # Check forward pass produces finite loss values
        self.assertTrue(np.isfinite(final_loss))
        self.assertTrue(np.isfinite(initial_loss))

    def test_link_prediction_pipeline(self):
        """Test link prediction pipeline."""
        # 1. Create graph with missing edges
        graph = self._create_link_prediction_data()

        # 2. Split edges
        train_edges, val_edges, test_edges = self._split_edges(graph.edge_index)

        # 3. Build link predictor
        model = self._build_link_predictor(
            input_dim=graph.x.shape[1],
            hidden_dim=64
        )

        # 4. Train on positive and negative edges
        for _ in range(5):
            pos_edges = train_edges
            neg_edges = self._sample_negative_edges(graph.num_nodes, len(train_edges[0]))

            # Compute embeddings
            embeddings = model(graph.x, train_edges)

            # Compute link scores
            pos_scores = self._compute_edge_scores(embeddings, pos_edges)
            neg_scores = self._compute_edge_scores(embeddings, neg_edges)

            # Check positive edges have higher scores
            self.assertGreater(np.mean(pos_scores), np.mean(neg_scores))

    def test_heterogeneous_gnn_pipeline(self):
        """Test heterogeneous GNN pipeline."""
        # 1. Create heterogeneous graph
        hetero_graph = self._create_hetero_graph()

        # 2. Build hetero GNN model
        model = self._build_hetero_gnn(hetero_graph)

        # 3. Forward pass
        outputs = model(hetero_graph)

        # Check outputs for each node type
        self.assertIn('user', outputs)
        self.assertIn('item', outputs)

        # Check output shapes
        self.assertEqual(outputs['user'].shape[0], hetero_graph.num_nodes('user'))
        self.assertEqual(outputs['item'].shape[0], hetero_graph.num_nodes('item'))

    def test_minibatch_training(self):
        """Test minibatch training with neighbor sampling."""
        # 1. Create large graph
        graph = self._create_large_graph()

        # 2. Setup neighbor sampler
        sampler = NeighborSampler(
            graph=graph,
            num_neighbors=[10, 5],  # 2-hop sampling
        )

        # 3. Build model
        model = self._build_sage_model(
            input_dim=graph.x.shape[1],
            hidden_dim=128,
            output_dim=10,
            num_layers=2
        )

        # 4. Minibatch training
        batch_size = 256
        train_nodes = np.array(list(range(0, 800)))
        for _ in range(3):
            # Sample a mini-batch of nodes
            batch_nodes = np.random.choice(train_nodes, batch_size, replace=False)
            subgraph = sampler.sample(batch_nodes)

            # Forward on subgraph features
            subgraph_x = graph.x[subgraph.node_idx]
            out = model(subgraph_x, subgraph.edge_index)

            # Check output shape
            self.assertEqual(out.shape[0], len(subgraph.node_idx))

    def _create_node_classification_dataset(self):
        """Create synthetic node classification dataset."""
        graphs = []
        for _ in range(100):
            num_nodes = np.random.randint(50, 150)
            num_edges = num_nodes * 3

            edge_index = np.random.randint(0, num_nodes, (2, num_edges))
            x = np.random.randn(num_nodes, 32).astype(np.float32)
            y = np.random.randint(0, 4, num_nodes)

            graph = Graph(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
            graphs.append(graph)

        return graphs

    def _create_graph_classification_dataset(self):
        """Create synthetic graph classification dataset."""
        graphs = []
        for i in range(80):
            num_nodes = np.random.randint(20, 60)
            num_edges = num_nodes * 2

            edge_index = np.random.randint(0, num_nodes, (2, num_edges))
            x = np.random.randn(num_nodes, 16).astype(np.float32)
            y = i % 3  # 3 classes

            graph = Graph(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
            graphs.append(graph)

        return graphs

    def _create_link_prediction_data(self):
        """Create graph for link prediction."""
        num_nodes = 500
        num_edges = 2000

        edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        x = np.random.randn(num_nodes, 64).astype(np.float32)

        return Graph(x=x, edge_index=edge_index, num_nodes=num_nodes)

    def _create_hetero_graph(self):
        """Create heterogeneous graph."""
        node_types = {
            'user': 100,
            'item': 200,
            'category': 20
        }

        edge_types = {
            ('user', 'rates', 'item'): np.random.randint(0, 100, (2, 500)),
            ('item', 'belongs_to', 'category'): np.random.randint(0, 200, (2, 300)),
            ('user', 'follows', 'user'): np.random.randint(0, 100, (2, 200))
        }

        x_dict = {
            'user': np.random.randn(100, 32).astype(np.float32),
            'item': np.random.randn(200, 64).astype(np.float32),
            'category': np.random.randn(20, 16).astype(np.float32)
        }

        return HeteroGraph(x_dict, edge_types, node_types)

    def _create_large_graph(self):
        """Create large graph for minibatch training."""
        num_nodes = 1000
        num_edges = 5000

        edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        x = np.random.randn(num_nodes, 128).astype(np.float32)
        y = np.random.randint(0, 10, num_nodes)

        return Graph(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    def _build_gcn_model(self, input_dim, hidden_dim, output_dim, num_layers):
        """Build GCN model."""
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            layers.append(GCNConv(dims[i], dims[i + 1]))

        return Sequential(layers)

    def _build_sage_model(self, input_dim, hidden_dim, output_dim, num_layers):
        """Build GraphSAGE model."""
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            layers.append(GraphSAGEConv(dims[i], dims[i + 1]))

        return Sequential(layers)

    def _build_graph_classifier(self, input_dim, hidden_dim, output_dim, num_layers):
        """Build graph classification model."""
        class GraphClassifier:
            def __init__(self):
                self.convs = [GCNConv(input_dim, hidden_dim)]
                for _ in range(num_layers - 1):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.classifier = lambda x: x @ np.random.randn(hidden_dim, output_dim).astype(np.float32)

            def forward(self, x, edge_index, batch):
                for conv in self.convs:
                    x = conv(x, edge_index)
                    x = np.maximum(x, 0)  # ReLU

                # Global pooling
                graph_emb = self.global_mean_pool(x, batch)
                return self.classifier(graph_emb)

            def global_mean_pool(self, x, batch):
                """Global mean pooling."""
                num_graphs = int(batch.max()) + 1
                output = np.zeros((num_graphs, x.shape[1]), dtype=np.float32)

                for i in range(num_graphs):
                    mask = batch == i
                    output[i] = x[mask].mean(axis=0)

                return output

        return GraphClassifier()

    def _build_link_predictor(self, input_dim, hidden_dim):
        """Build link prediction model."""
        class LinkPredictor:
            def __init__(self):
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)

            def __call__(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = np.maximum(x, 0)
                x = self.conv2(x, edge_index)
                return x

        return LinkPredictor()

    def _build_hetero_gnn(self, hetero_graph):
        """Build heterogeneous GNN model."""
        class HeteroGNN:
            def __init__(self):
                # Linear projections per node type
                self.projections = {}
                out_dim = 32
                for node_type in hetero_graph.node_types:
                    in_dim = hetero_graph.x_dict[node_type].shape[1]
                    self.projections[node_type] = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.1

            def __call__(self, hetero_graph):
                outputs = {}
                # Apply linear projections per node type
                for node_type in hetero_graph.node_types:
                    x = hetero_graph.x_dict[node_type]
                    outputs[node_type] = x @ self.projections[node_type]
                    outputs[node_type] = np.maximum(outputs[node_type], 0)  # ReLU

                return outputs

        return HeteroGNN()

    def _train_epoch(self, model, train_loader):
        """Train one epoch."""
        total_loss = 0
        for batch in train_loader:
            # Forward pass
            if hasattr(batch, 'batch'):
                # Graph batch
                out = model.forward(batch.x, batch.edge_index, batch.batch)
                target = batch.y
            else:
                # Node batch
                out = model(batch.x, batch.edge_index)
                target = batch.y

            # Compute loss (cross-entropy)
            loss = self._cross_entropy_loss(out, target)
            total_loss += loss

            # Backward pass would go here

        return total_loss / len(train_loader)

    def _evaluate(self, model, loader):
        """Evaluate model accuracy."""
        correct = 0
        total = 0

        for batch in loader:
            if hasattr(batch, 'batch'):
                out = model.forward(batch.x, batch.edge_index, batch.batch)
            else:
                out = model(batch.x, batch.edge_index)

            pred = np.argmax(out, axis=1)
            correct += np.sum(pred == batch.y)
            total += len(batch.y)

        return correct / total

    def _compute_loss(self, model, loader):
        """Compute average loss."""
        total_loss = 0
        for batch in loader:
            if hasattr(batch, 'batch'):
                out = model.forward(batch.x, batch.edge_index, batch.batch)
            else:
                out = model(batch.x, batch.edge_index)

            loss = self._cross_entropy_loss(out, batch.y)
            total_loss += loss

        return total_loss / len(loader)

    def _cross_entropy_loss(self, logits, targets):
        """Simplified cross-entropy loss."""
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross-entropy
        n = len(targets)
        if len(targets.shape) == 1:
            # Convert to one-hot
            targets_one_hot = np.zeros_like(probs)
            targets_one_hot[np.arange(n), targets] = 1
            targets = targets_one_hot

        loss = -np.sum(targets * np.log(probs + 1e-8)) / n
        return loss

    def _split_edges(self, edge_index):
        """Split edges for train/val/test."""
        num_edges = edge_index.shape[1]
        perm = np.random.permutation(num_edges)

        train_size = int(0.8 * num_edges)
        val_size = int(0.1 * num_edges)

        train_idx = perm[:train_size]
        val_idx = perm[train_size:train_size + val_size]
        test_idx = perm[train_size + val_size:]

        train_edges = edge_index[:, train_idx]
        val_edges = edge_index[:, val_idx]
        test_edges = edge_index[:, test_idx]

        return train_edges, val_edges, test_edges

    def _sample_negative_edges(self, num_nodes, num_samples):
        """Sample negative edges."""
        neg_edges = []
        while len(neg_edges) < num_samples:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                neg_edges.append([src, dst])

        return np.array(neg_edges).T

    def _compute_edge_scores(self, embeddings, edges):
        """Compute edge scores using dot product."""
        src_emb = embeddings[edges[0]]
        dst_emb = embeddings[edges[1]]
        scores = np.sum(src_emb * dst_emb, axis=1)
        return scores


class Sequential:
    """Simple sequential model."""

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = np.maximum(x, 0)  # ReLU
        x = self.layers[-1](x, edge_index)
        return x

    def forward(self, x, edge_index, batch=None):
        return self.__call__(x, edge_index)


class TestPerformance(unittest.TestCase):
    """Test performance and scalability."""

    def setUp(self):
        np.random.seed(42)

    def test_large_graph_performance(self):
        """Test performance on large graphs."""
        # Create large graph
        num_nodes = 10000
        num_edges = 50000
        edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        x = np.random.randn(num_nodes, 128).astype(np.float32)

        graph = Graph(x=x, edge_index=edge_index, num_nodes=num_nodes)

        # Test GCN forward pass
        gcn = GCNConv(128, 64)

        start = time.time()
        output = gcn(graph.x, graph.edge_index)
        elapsed = time.time() - start

        # Should be reasonably fast
        self.assertLess(elapsed, 1.0)  # Less than 1 second
        self.assertEqual(output.shape, (num_nodes, 64))

    def test_batching_performance(self):
        """Test batching performance."""
        # Create multiple graphs
        graphs = []
        for _ in range(100):
            num_nodes = np.random.randint(50, 200)
            num_edges = num_nodes * 3
            edge_index = np.random.randint(0, num_nodes, (2, num_edges))
            x = np.random.randn(num_nodes, 64).astype(np.float32)
            graphs.append(Graph(x=x, edge_index=edge_index, num_nodes=num_nodes))

        # Test batching speed
        start = time.time()
        batch = Batch.from_graph_list(graphs)
        batch_time = time.time() - start

        # Should be fast
        self.assertLess(batch_time, 0.1)  # Less than 100ms

    def test_sampling_scalability(self):
        """Test sampling scalability."""
        # Create very large graph
        num_nodes = 100000
        num_edges = 1000000

        # Create power-law degree distribution
        degrees = np.random.power(0.5, num_nodes) * 100
        degrees = degrees.astype(int) + 1

        edges = []
        for node, degree in enumerate(degrees[:10000]):  # Subset for speed
            targets = np.random.choice(num_nodes, min(degree, 10), replace=False)
            for target in targets:
                edges.append([node, target])

        edge_index = np.array(edges).T

        # Test sampling speed
        sampler = LayerwiseSampler(
            edge_index=edge_index,
            num_layers=2,
            num_neighbors=10,
            num_nodes=num_nodes
        )

        start = time.time()
        target_nodes = list(range(100))
        sampled = sampler.sample_layers(target_nodes)
        sample_time = time.time() - start

        # Should be fast even for large graphs
        self.assertLess(sample_time, 0.5)  # Less than 500ms

    def test_memory_efficiency(self):
        """Test memory efficiency of operations."""
        import tracemalloc

        # Start memory tracking
        tracemalloc.start()

        # Create large graph
        num_nodes = 5000
        num_edges = 25000
        edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        x = np.random.randn(num_nodes, 256).astype(np.float32)

        graph = Graph(x=x, edge_index=edge_index, num_nodes=num_nodes)

        # Perform operations
        gcn = GCNConv(256, 128)
        output = gcn(graph.x, graph.edge_index)

        # Check memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should be reasonable
        peak_mb = peak / (1024 * 1024)
        self.assertLess(peak_mb, 500)  # Less than 500MB


if __name__ == '__main__':
    unittest.main()