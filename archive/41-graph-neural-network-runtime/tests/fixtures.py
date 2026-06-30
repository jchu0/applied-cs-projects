"""Test fixtures and helpers for GNN runtime tests."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os


class GraphFactory:
    """Factory for creating test graphs."""

    @staticmethod
    def create_simple_graph(num_nodes: int = 10, num_edges: int = 20) -> Dict:
        """Create a simple random graph."""
        edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        x = np.random.randn(num_nodes, 16).astype(np.float32)

        return {
            "x": x,
            "edge_index": edge_index,
            "num_nodes": num_nodes
        }

    @staticmethod
    def create_grid_graph(width: int = 10, height: int = 10) -> Dict:
        """Create a grid graph."""
        num_nodes = width * height
        edges = []

        # Create grid edges
        for i in range(height):
            for j in range(width):
                node = i * width + j

                # Right edge
                if j < width - 1:
                    edges.append([node, node + 1])
                    edges.append([node + 1, node])

                # Down edge
                if i < height - 1:
                    edges.append([node, node + width])
                    edges.append([node + width, node])

        edge_index = np.array(edges).T
        x = np.random.randn(num_nodes, 32).astype(np.float32)

        # Add spatial features
        positions = np.zeros((num_nodes, 2))
        for i in range(height):
            for j in range(width):
                node = i * width + j
                positions[node] = [i / height, j / width]

        return {
            "x": x,
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "positions": positions
        }

    @staticmethod
    def create_tree_graph(depth: int = 4, branching_factor: int = 2) -> Dict:
        """Create a tree graph."""
        edges = []
        num_nodes = sum(branching_factor ** d for d in range(depth + 1))

        node_id = 0
        queue = [0]
        next_id = 1

        while queue and next_id < num_nodes:
            parent = queue.pop(0)
            for _ in range(branching_factor):
                if next_id >= num_nodes:
                    break
                edges.append([parent, next_id])
                edges.append([next_id, parent])
                queue.append(next_id)
                next_id += 1

        edge_index = np.array(edges).T
        x = np.random.randn(num_nodes, 16).astype(np.float32)

        # Add depth features
        depths = np.zeros(num_nodes)
        for d in range(depth + 1):
            start = sum(branching_factor ** i for i in range(d))
            end = sum(branching_factor ** i for i in range(d + 1))
            depths[start:end] = d

        return {
            "x": x,
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "depths": depths
        }

    @staticmethod
    def create_community_graph(
        num_communities: int = 4,
        community_size: int = 25,
        p_intra: float = 0.3,
        p_inter: float = 0.01
    ) -> Dict:
        """Create a graph with community structure."""
        num_nodes = num_communities * community_size
        edges = []

        # Intra-community edges
        for c in range(num_communities):
            start = c * community_size
            end = (c + 1) * community_size

            for i in range(start, end):
                for j in range(i + 1, end):
                    if np.random.rand() < p_intra:
                        edges.append([i, j])
                        edges.append([j, i])

        # Inter-community edges
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                c1 = i // community_size
                c2 = j // community_size
                if c1 != c2 and np.random.rand() < p_inter:
                    edges.append([i, j])
                    edges.append([j, i])

        edge_index = np.array(edges).T if edges else np.zeros((2, 0), dtype=int)
        x = np.random.randn(num_nodes, 64).astype(np.float32)

        # Add community labels
        communities = np.array([i // community_size for i in range(num_nodes)])

        return {
            "x": x,
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "communities": communities
        }

    @staticmethod
    def create_scale_free_graph(num_nodes: int = 1000, m: int = 5) -> Dict:
        """Create a scale-free graph using preferential attachment."""
        edges = []

        # Start with a complete graph of m+1 nodes
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                edges.append([i, j])
                edges.append([j, i])

        degrees = np.zeros(num_nodes)
        for i in range(m + 1):
            degrees[i] = m

        # Add remaining nodes with preferential attachment
        for new_node in range(m + 1, num_nodes):
            # Compute probabilities based on degree
            probs = degrees[:new_node] / np.sum(degrees[:new_node])

            # Select m nodes to connect to
            targets = np.random.choice(new_node, m, replace=False, p=probs)

            for target in targets:
                edges.append([new_node, target])
                edges.append([target, new_node])
                degrees[new_node] += 1
                degrees[target] += 1

        edge_index = np.array(edges).T
        x = np.random.randn(num_nodes, 128).astype(np.float32)

        return {
            "x": x,
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "degrees": degrees
        }


class DatasetFactory:
    """Factory for creating test datasets."""

    @staticmethod
    def create_node_classification_dataset(
        num_graphs: int = 100,
        num_classes: int = 4,
        min_nodes: int = 50,
        max_nodes: int = 150
    ) -> List[Dict]:
        """Create node classification dataset."""
        dataset = []

        for _ in range(num_graphs):
            num_nodes = np.random.randint(min_nodes, max_nodes)
            graph = GraphFactory.create_community_graph(
                num_communities=num_classes,
                community_size=num_nodes // num_classes
            )

            # Use community labels as node labels
            graph["y"] = graph["communities"]
            dataset.append(graph)

        return dataset

    @staticmethod
    def create_graph_classification_dataset(
        num_graphs: int = 200,
        num_classes: int = 3
    ) -> List[Dict]:
        """Create graph classification dataset."""
        dataset = []

        for i in range(num_graphs):
            # Different graph types for different classes
            class_label = i % num_classes

            if class_label == 0:
                # Tree graphs
                graph = GraphFactory.create_tree_graph(
                    depth=np.random.randint(3, 6),
                    branching_factor=2
                )
            elif class_label == 1:
                # Grid graphs
                size = np.random.randint(5, 10)
                graph = GraphFactory.create_grid_graph(size, size)
            else:
                # Community graphs
                graph = GraphFactory.create_community_graph(
                    num_communities=np.random.randint(2, 5)
                )

            graph["y"] = class_label
            dataset.append(graph)

        return dataset

    @staticmethod
    def create_link_prediction_dataset(
        num_nodes: int = 1000,
        num_edges: int = 5000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Dict:
        """Create link prediction dataset."""
        # Create full graph
        all_edges = set()
        while len(all_edges) < num_edges:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                all_edges.add((min(src, dst), max(src, dst)))

        all_edges = list(all_edges)
        np.random.shuffle(all_edges)

        # Split edges
        train_size = int(train_ratio * len(all_edges))
        val_size = int(val_ratio * len(all_edges))

        train_edges = all_edges[:train_size]
        val_edges = all_edges[train_size:train_size + val_size]
        test_edges = all_edges[train_size + val_size:]

        # Create edge indices
        train_edge_index = np.array([[e[0] for e in train_edges],
                                     [e[1] for e in train_edges]])

        x = np.random.randn(num_nodes, 128).astype(np.float32)

        return {
            "x": x,
            "train_edges": train_edge_index,
            "val_edges": np.array([[e[0] for e in val_edges],
                                   [e[1] for e in val_edges]]),
            "test_edges": np.array([[e[0] for e in test_edges],
                                    [e[1] for e in test_edges]]),
            "num_nodes": num_nodes
        }


class HeteroGraphFactory:
    """Factory for creating heterogeneous graphs."""

    @staticmethod
    def create_user_item_graph(
        num_users: int = 100,
        num_items: int = 200,
        num_categories: int = 20,
        num_ratings: int = 1000
    ) -> Dict:
        """Create user-item heterogeneous graph."""
        # Node features
        x_dict = {
            "user": np.random.randn(num_users, 32).astype(np.float32),
            "item": np.random.randn(num_items, 64).astype(np.float32),
            "category": np.random.randn(num_categories, 16).astype(np.float32)
        }

        # Edge indices
        edge_index_dict = {}

        # User rates item
        user_ids = np.random.randint(0, num_users, num_ratings)
        item_ids = np.random.randint(0, num_items, num_ratings)
        edge_index_dict[("user", "rates", "item")] = np.array([user_ids, item_ids])

        # Item belongs to category
        item_ids = np.arange(num_items)
        category_ids = np.random.randint(0, num_categories, num_items)
        edge_index_dict[("item", "belongs_to", "category")] = np.array([item_ids, category_ids])

        # User follows user
        num_follows = num_users * 2
        src_users = np.random.randint(0, num_users, num_follows)
        dst_users = np.random.randint(0, num_users, num_follows)
        # Remove self-loops
        mask = src_users != dst_users
        edge_index_dict[("user", "follows", "user")] = np.array([src_users[mask], dst_users[mask]])

        return {
            "x_dict": x_dict,
            "edge_index_dict": edge_index_dict,
            "node_types": {"user": num_users, "item": num_items, "category": num_categories}
        }

    @staticmethod
    def create_citation_network(
        num_papers: int = 500,
        num_authors: int = 200,
        num_venues: int = 30
    ) -> Dict:
        """Create academic citation network."""
        x_dict = {
            "paper": np.random.randn(num_papers, 128).astype(np.float32),
            "author": np.random.randn(num_authors, 64).astype(np.float32),
            "venue": np.random.randn(num_venues, 32).astype(np.float32)
        }

        edge_index_dict = {}

        # Paper cites paper
        num_citations = num_papers * 3
        src_papers = np.random.randint(0, num_papers, num_citations)
        dst_papers = np.random.randint(0, num_papers, num_citations)
        mask = src_papers != dst_papers
        edge_index_dict[("paper", "cites", "paper")] = np.array([src_papers[mask], dst_papers[mask]])

        # Author writes paper
        papers_per_author = 5
        author_ids = []
        paper_ids = []
        for author in range(num_authors):
            papers = np.random.choice(num_papers, papers_per_author, replace=False)
            author_ids.extend([author] * papers_per_author)
            paper_ids.extend(papers)
        edge_index_dict[("author", "writes", "paper")] = np.array([author_ids, paper_ids])

        # Paper published in venue
        paper_ids = np.arange(num_papers)
        venue_ids = np.random.randint(0, num_venues, num_papers)
        edge_index_dict[("paper", "published_in", "venue")] = np.array([paper_ids, venue_ids])

        return {
            "x_dict": x_dict,
            "edge_index_dict": edge_index_dict,
            "node_types": {"paper": num_papers, "author": num_authors, "venue": num_venues}
        }


class TemporalGraphFactory:
    """Factory for creating temporal graphs."""

    @staticmethod
    def create_temporal_graph(
        num_nodes: int = 100,
        num_timestamps: int = 10,
        edges_per_timestamp: int = 50
    ) -> Dict:
        """Create temporal graph with discrete timestamps."""
        all_edges = []
        all_timestamps = []

        for t in range(num_timestamps):
            # Generate edges for this timestamp
            src = np.random.randint(0, num_nodes, edges_per_timestamp)
            dst = np.random.randint(0, num_nodes, edges_per_timestamp)

            for s, d in zip(src, dst):
                if s != d:
                    all_edges.append([s, d])
                    all_timestamps.append(t)

        edge_index = np.array(all_edges).T
        timestamps = np.array(all_timestamps)

        # Node features can change over time
        x = np.random.randn(num_nodes, 32, num_timestamps).astype(np.float32)

        return {
            "edge_index": edge_index,
            "timestamps": timestamps,
            "x": x,
            "num_nodes": num_nodes,
            "num_timestamps": num_timestamps
        }

    @staticmethod
    def create_continuous_temporal_graph(
        num_nodes: int = 200,
        duration: float = 100.0,
        num_edges: int = 1000
    ) -> Dict:
        """Create temporal graph with continuous timestamps."""
        edges = []
        timestamps = []

        for _ in range(num_edges):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            t = np.random.uniform(0, duration)

            if src != dst:
                edges.append([src, dst])
                timestamps.append(t)

        # Sort by time
        sorted_idx = np.argsort(timestamps)
        edge_index = np.array(edges)[sorted_idx].T
        timestamps = np.array(timestamps)[sorted_idx]

        x = np.random.randn(num_nodes, 64).astype(np.float32)

        return {
            "edge_index": edge_index,
            "timestamps": timestamps,
            "x": x,
            "num_nodes": num_nodes,
            "duration": duration
        }


class TestMetrics:
    """Metrics for evaluating GNN models."""

    @staticmethod
    def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(predictions == labels)

    @staticmethod
    def f1_score(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
        """Compute macro F1 score."""
        f1_scores = []

        for c in range(num_classes):
            true_positives = np.sum((predictions == c) & (labels == c))
            false_positives = np.sum((predictions == c) & (labels != c))
            false_negatives = np.sum((predictions != c) & (labels == c))

            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)

            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores.append(f1)

        return np.mean(f1_scores)

    @staticmethod
    def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute AUC score for binary classification."""
        # Sort by scores
        sorted_idx = np.argsort(scores)[::-1]
        sorted_labels = labels[sorted_idx]

        # Compute AUC
        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos

        tpr = np.cumsum(sorted_labels) / n_pos
        fpr = np.cumsum(1 - sorted_labels) / n_neg

        auc = np.trapz(tpr, fpr)
        return auc

    @staticmethod
    def mean_reciprocal_rank(rankings: List[int]) -> float:
        """Compute mean reciprocal rank."""
        reciprocal_ranks = [1.0 / (r + 1) for r in rankings]
        return np.mean(reciprocal_ranks)

    @staticmethod
    def hit_rate(rankings: List[int], k: int = 10) -> float:
        """Compute hit rate at k."""
        hits = sum(1 for r in rankings if r < k)
        return hits / len(rankings)


# Export commonly used test data
SAMPLE_GRAPHS = {
    "small": GraphFactory.create_simple_graph(num_nodes=10, num_edges=20),
    "medium": GraphFactory.create_simple_graph(num_nodes=100, num_edges=500),
    "large": GraphFactory.create_simple_graph(num_nodes=1000, num_edges=5000),
    "tree": GraphFactory.create_tree_graph(depth=4, branching_factor=3),
    "grid": GraphFactory.create_grid_graph(width=10, height=10),
    "community": GraphFactory.create_community_graph(num_communities=4),
}

SAMPLE_HETERO_GRAPH = HeteroGraphFactory.create_user_item_graph()
SAMPLE_TEMPORAL_GRAPH = TemporalGraphFactory.create_temporal_graph()

SAMPLE_NODE_CLASSIFICATION = DatasetFactory.create_node_classification_dataset(num_graphs=10)
SAMPLE_GRAPH_CLASSIFICATION = DatasetFactory.create_graph_classification_dataset(num_graphs=20)
SAMPLE_LINK_PREDICTION = DatasetFactory.create_link_prediction_dataset()