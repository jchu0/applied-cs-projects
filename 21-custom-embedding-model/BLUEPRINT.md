# Custom Embedding Model Training System

## Executive Summary

Production-grade embedding model training infrastructure supporting contrastive learning with hard negative mining, multi-GPU distributed training, and comprehensive evaluation pipelines. Designed for domain adaptation and enterprise deployment with MLOps integration.

> **Concepts covered:** [§04 Embeddings](../../04-ai-engineering/03-vector-databases/embeddings/embeddings.md) · [§04 Vector stores](../../04-ai-engineering/03-vector-databases/vector-stores/vector-stores.md) · [§03 Distributed training — data parallelism](../../03-machine-learning-engineering/05-distributed-training/data-parallelism/). See also the [concept-to-project map](../CONCEPT_TO_PROJECT_MAP.md).

## System Architecture

```
+------------------------------------------------------------------+
|                    Embedding Training System                      |
+------------------------------------------------------------------+
|                                                                   |
|  +-----------------+    +------------------+    +---------------+ |
|  | Dataset Engine  |    | Training Engine  |    | Eval Engine   | |
|  |-----------------|    |------------------|    |---------------| |
|  | - Pair Mining   |    | - Bi-Encoder     |    | - Recall@k    | |
|  | - Hard Negatives|--->| - Triplet Loss   |--->| - MRR/MAP     | |
|  | - Batch Sampler |    | - DDP Training   |    | - NDCG        | |
|  | - Augmentation  |    | - Grad Accum     |    | - t-SNE Viz   | |
|  +-----------------+    +------------------+    +---------------+ |
|           |                      |                      |         |
|           v                      v                      v         |
|  +------------------------------------------------------------------+
|  |                    MLOps Infrastructure                        |
|  |----------------------------------------------------------------|
|  | MLflow Registry | DVC Dataset | Drift Monitor | ONNX Export   |
|  +------------------------------------------------------------------+
|                                                                   |
+------------------------------------------------------------------+
```

## Core Components

### 1. Dataset Pipeline

#### Data Schema

```python
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class EmbeddingExample:
    """Single embedding training example."""
    anchor_id: str
    anchor_text: str
    positive_id: str
    positive_text: str
    negative_ids: list[str]
    negative_texts: list[str]
    domain: Optional[str] = None
    difficulty: float = 0.5  # 0=easy, 1=hard

@dataclass
class EmbeddingBatch:
    """Collated batch for training."""
    anchor_input_ids: torch.Tensor      # [B, seq_len]
    anchor_attention_mask: torch.Tensor
    positive_input_ids: torch.Tensor
    positive_attention_mask: torch.Tensor
    negative_input_ids: torch.Tensor    # [B, num_neg, seq_len]
    negative_attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None
```

#### Hard Negative Mining

```python
import numpy as np
from typing import Protocol
import faiss

class NegativeMiner(Protocol):
    def mine(self, anchor_emb: np.ndarray, corpus_embs: np.ndarray,
             k: int) -> np.ndarray: ...

class HardNegativeMiner:
    """Mine hard negatives using approximate nearest neighbor search."""

    def __init__(self, index_type: str = "IVF1024,Flat", nprobe: int = 64):
        self.index_type = index_type
        self.nprobe = nprobe
        self.index = None

    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index from corpus embeddings."""
        dim = embeddings.shape[1]

        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(dim)
        else:
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 1024)
            self.index.train(embeddings)

        self.index.add(embeddings)
        self.index.nprobe = self.nprobe

    def mine(self, anchor_embs: np.ndarray, positive_ids: np.ndarray,
             k: int = 10, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """
        Mine hard negatives that are close to anchor but not positives.

        Returns:
            negative_ids: [batch_size, k] indices of hard negatives
            scores: [batch_size, k] similarity scores
        """
        # Search for candidates (get more than k to filter positives)
        scores, indices = self.index.search(anchor_embs, k * 2)

        # Filter out positives and select top-k
        batch_negatives = []
        batch_scores = []

        for i, (pos_id, cand_ids, cand_scores) in enumerate(
            zip(positive_ids, indices, scores)
        ):
            # Remove positive from candidates
            mask = cand_ids != pos_id
            neg_ids = cand_ids[mask][:k]
            neg_scores = cand_scores[mask][:k]

            batch_negatives.append(neg_ids)
            batch_scores.append(neg_scores)

        return np.array(batch_negatives), np.array(batch_scores)


class InBatchNegativeSampler:
    """Use other batch elements as negatives with memory bank."""

    def __init__(self, memory_size: int = 65536, embedding_dim: int = 768):
        self.memory_bank = np.zeros((memory_size, embedding_dim), dtype=np.float32)
        self.memory_ptr = 0
        self.memory_size = memory_size
        self.is_full = False

    def update_memory(self, embeddings: np.ndarray):
        """Add new embeddings to memory bank (FIFO)."""
        batch_size = embeddings.shape[0]

        if self.memory_ptr + batch_size <= self.memory_size:
            self.memory_bank[self.memory_ptr:self.memory_ptr + batch_size] = embeddings
            self.memory_ptr += batch_size
        else:
            # Wrap around
            remaining = self.memory_size - self.memory_ptr
            self.memory_bank[self.memory_ptr:] = embeddings[:remaining]
            self.memory_bank[:batch_size - remaining] = embeddings[remaining:]
            self.memory_ptr = batch_size - remaining
            self.is_full = True

    def get_negatives(self, exclude_indices: np.ndarray = None) -> np.ndarray:
        """Get all memory bank embeddings as potential negatives."""
        if self.is_full:
            return self.memory_bank
        return self.memory_bank[:self.memory_ptr]
```

### 2. Training Engine

#### Bi-Encoder Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class BiEncoder(nn.Module):
    """Bi-encoder for contrastive embedding learning."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        pooling: str = "mean",  # mean, cls, max
        normalize: bool = True,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.normalize = normalize

        hidden_size = self.encoder.config.hidden_size

        # Optional projection head
        if projection_dim:
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, projection_dim),
            )
        else:
            self.projection = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text to embeddings."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Pool token embeddings
        if self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        elif self.pooling == "mean":
            token_embs = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
            sum_embs = torch.sum(token_embs * mask, dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = sum_embs / sum_mask
        elif self.pooling == "max":
            token_embs = outputs.last_hidden_state
            token_embs[attention_mask == 0] = -1e9
            embeddings = torch.max(token_embs, dim=1)[0]

        # Project if configured
        if self.projection:
            embeddings = self.projection(embeddings)

        # Normalize for cosine similarity
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class MultipleNegativesRankingLoss(nn.Module):
    """
    MNRL loss: treat in-batch samples as negatives.
    Efficient and effective for contrastive learning.
    """

    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        anchor_embs: torch.Tensor,  # [B, D]
        positive_embs: torch.Tensor,  # [B, D]
        negative_embs: Optional[torch.Tensor] = None,  # [B, N, D]
    ) -> torch.Tensor:
        # Compute similarity matrix: [B, B] or [B, B+N]
        scores = torch.matmul(anchor_embs, positive_embs.T) * self.scale

        if negative_embs is not None:
            # Add explicit negatives
            neg_scores = torch.bmm(
                anchor_embs.unsqueeze(1),
                negative_embs.transpose(1, 2)
            ).squeeze(1) * self.scale
            scores = torch.cat([scores, neg_scores], dim=1)

        # Labels: diagonal (positive pairs)
        labels = torch.arange(len(anchor_embs), device=scores.device)

        return self.cross_entropy(scores, labels)


class TripletMarginLoss(nn.Module):
    """Triplet loss with margin for hard negative training."""

    def __init__(self, margin: float = 0.5, distance: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        if self.distance == "cosine":
            # Cosine distance = 1 - cosine_similarity
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
```

#### Distributed Training Loop

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

class EmbeddingTrainer:
    """Distributed trainer for embedding models."""

    def __init__(
        self,
        model: BiEncoder,
        train_dataset,
        eval_dataset,
        config: TrainingConfig,
    ):
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")

        # Setup distributed
        if config.distributed:
            dist.init_process_group(backend="nccl")
            self.model = DDP(
                model.to(self.device),
                device_ids=[config.local_rank],
                find_unused_parameters=False,
            )
            self.is_main = config.local_rank == 0
        else:
            self.model = model.to(self.device)
            self.is_main = True

        # Data loaders
        sampler = DistributedSampler(train_dataset) if config.distributed else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.fp16 else None

        # Loss function
        self.loss_fn = MultipleNegativesRankingLoss(scale=config.temperature)

        # Memory bank for in-batch negatives
        self.memory_bank = InBatchNegativeSampler(
            memory_size=config.memory_bank_size,
            embedding_dim=model.encoder.config.hidden_size,
        )

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(self.train_loader, disable=not self.is_main)

        for step, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast(enabled=self.scaler is not None):
                anchor_embs = self.model(
                    batch["anchor_input_ids"],
                    batch["anchor_attention_mask"],
                )
                positive_embs = self.model(
                    batch["positive_input_ids"],
                    batch["positive_attention_mask"],
                )

                # Get memory bank negatives
                if self.config.use_memory_bank:
                    memory_negs = self.memory_bank.get_negatives()
                    memory_negs = torch.from_numpy(memory_negs).to(self.device)
                else:
                    memory_negs = None

                loss = self.loss_fn(anchor_embs, positive_embs, memory_negs)

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

            # Update memory bank
            if self.config.use_memory_bank:
                with torch.no_grad():
                    self.memory_bank.update_memory(
                        positive_embs.detach().cpu().numpy()
                    )

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"train_loss": total_loss / len(self.train_loader)}
```

### 3. Evaluation Pipeline

```python
import numpy as np
from typing import Any
from collections import defaultdict

class EmbeddingEvaluator:
    """Comprehensive embedding evaluation metrics."""

    def __init__(self, model: BiEncoder, tokenizer, device: str = "cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        self.model.eval()
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            embs = self.model(**inputs).cpu().numpy()
            embeddings.append(embs)

        return np.concatenate(embeddings, axis=0)

    def evaluate_retrieval(
        self,
        queries: list[str],
        corpus: list[str],
        relevance: dict[int, list[int]],  # query_idx -> [relevant_doc_indices]
        k_values: list[int] = [1, 5, 10, 20, 100],
    ) -> dict[str, float]:
        """
        Evaluate retrieval metrics.

        Args:
            queries: Query texts
            corpus: Corpus texts
            relevance: Mapping from query index to relevant document indices
            k_values: K values for Recall@K and Precision@K
        """
        # Encode all texts
        query_embs = self.encode(queries)
        corpus_embs = self.encode(corpus)

        # Build FAISS index
        index = faiss.IndexFlatIP(corpus_embs.shape[1])
        index.add(corpus_embs)

        # Search
        max_k = max(k_values)
        scores, indices = index.search(query_embs, max_k)

        # Compute metrics
        metrics = defaultdict(list)

        for q_idx, (retrieved, rel_docs) in enumerate(zip(indices, relevance.values())):
            rel_set = set(rel_docs)

            # Recall@K
            for k in k_values:
                retrieved_k = set(retrieved[:k])
                recall = len(retrieved_k & rel_set) / len(rel_set) if rel_set else 0
                metrics[f"Recall@{k}"].append(recall)

            # Precision@K
            for k in k_values:
                retrieved_k = retrieved[:k]
                precision = sum(1 for r in retrieved_k if r in rel_set) / k
                metrics[f"Precision@{k}"].append(precision)

            # MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for rank, doc_idx in enumerate(retrieved, 1):
                if doc_idx in rel_set:
                    mrr = 1.0 / rank
                    break
            metrics["MRR"].append(mrr)

            # NDCG@K
            for k in k_values:
                dcg = sum(
                    1 / np.log2(rank + 2)
                    for rank, doc_idx in enumerate(retrieved[:k])
                    if doc_idx in rel_set
                )
                idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(rel_set))))
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics[f"NDCG@{k}"].append(ndcg)

            # MAP (Mean Average Precision)
            precisions = []
            num_relevant = 0
            for rank, doc_idx in enumerate(retrieved, 1):
                if doc_idx in rel_set:
                    num_relevant += 1
                    precisions.append(num_relevant / rank)
            ap = sum(precisions) / len(rel_set) if rel_set else 0
            metrics["MAP"].append(ap)

        # Average all metrics
        return {k: np.mean(v) for k, v in metrics.items()}

    def evaluate_clustering(
        self,
        texts: list[str],
        labels: list[int],
    ) -> dict[str, float]:
        """Evaluate clustering quality of embeddings."""
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
        )
        from sklearn.cluster import KMeans

        embeddings = self.encode(texts)
        n_clusters = len(set(labels))

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        pred_labels = kmeans.fit_predict(embeddings)

        # Clustering metrics
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
        )

        return {
            "Silhouette": silhouette_score(embeddings, labels),
            "Calinski-Harabasz": calinski_harabasz_score(embeddings, labels),
            "Davies-Bouldin": davies_bouldin_score(embeddings, labels),
            "ARI": adjusted_rand_score(labels, pred_labels),
            "NMI": normalized_mutual_info_score(labels, pred_labels),
        }
```

### 4. Deployment Pipeline

```python
import onnx
import onnxruntime as ort
from pathlib import Path

class EmbeddingModelExporter:
    """Export trained models for production deployment."""

    def __init__(self, model: BiEncoder, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def export_onnx(
        self,
        output_path: Path,
        opset_version: int = 14,
        optimize: bool = True,
    ):
        """Export model to ONNX format."""
        self.model.eval()

        # Create dummy inputs
        dummy_input = self.tokenizer(
            "dummy input text",
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        # Export
        torch.onnx.export(
            self.model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size"},
            },
            opset_version=opset_version,
        )

        if optimize:
            self._optimize_onnx(output_path)

    def _optimize_onnx(self, model_path: Path):
        """Optimize ONNX model for inference."""
        from onnxruntime.transformers import optimizer

        optimized_model = optimizer.optimize_model(
            str(model_path),
            model_type="bert",
            num_heads=12,
            hidden_size=768,
        )
        optimized_model.save_model_to_file(str(model_path))


class ONNXEmbeddingModel:
    """ONNX runtime inference wrapper."""

    def __init__(self, model_path: str, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Configure ONNX Runtime
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=providers,
        )

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts using ONNX model."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            outputs = self.session.run(
                ["embeddings"],
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
            )
            embeddings.append(outputs[0])

        return np.concatenate(embeddings, axis=0)
```

## Enterprise Features

### MLflow Model Registry Integration

```python
import mlflow
from mlflow.tracking import MlflowClient

class EmbeddingModelRegistry:
    """MLflow integration for model versioning and deployment."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

    def log_training_run(
        self,
        model: BiEncoder,
        tokenizer,
        metrics: dict,
        params: dict,
        artifacts: dict[str, str] = None,
    ):
        """Log training run with model and metrics."""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name="embedding-model",
            )

            # Log tokenizer
            tokenizer.save_pretrained("tokenizer")
            mlflow.log_artifacts("tokenizer", "tokenizer")

            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)

    def promote_model(
        self,
        model_name: str,
        version: int,
        stage: str = "Production",
    ):
        """Promote model version to production."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )

    def load_production_model(self, model_name: str) -> BiEncoder:
        """Load the current production model."""
        model_uri = f"models:/{model_name}/Production"
        return mlflow.pytorch.load_model(model_uri)
```

### DVC Dataset Versioning

```yaml
# dvc.yaml
stages:
  prepare_data:
    cmd: python scripts/prepare_data.py
    deps:
      - data/raw/
      - scripts/prepare_data.py
    outs:
      - data/processed/train.parquet
      - data/processed/eval.parquet
    metrics:
      - data/processed/stats.json:
          cache: false

  mine_negatives:
    cmd: python scripts/mine_negatives.py
    deps:
      - data/processed/train.parquet
      - models/pretrained/
    outs:
      - data/processed/train_with_negatives.parquet
    params:
      - negative_mining.k
      - negative_mining.strategy

  train:
    cmd: python scripts/train.py
    deps:
      - data/processed/train_with_negatives.parquet
      - src/model.py
    outs:
      - models/trained/
    metrics:
      - metrics/train_metrics.json:
          cache: false
    plots:
      - metrics/loss_curve.csv:
          x: step
          y: loss
```

### Embedding Drift Monitoring

```python
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import numpy as np

class EmbeddingDriftMonitor:
    """Monitor embedding distribution drift in production."""

    def __init__(self, reference_embeddings: np.ndarray):
        self.reference = reference_embeddings
        self.reference_stats = self._compute_stats(reference_embeddings)

    def _compute_stats(self, embeddings: np.ndarray) -> dict:
        """Compute statistical properties of embedding distribution."""
        return {
            "mean": embeddings.mean(axis=0),
            "std": embeddings.std(axis=0),
            "norms": np.linalg.norm(embeddings, axis=1),
            "pairwise_cosine_mean": self._mean_pairwise_cosine(embeddings),
        }

    def _mean_pairwise_cosine(self, embeddings: np.ndarray, sample_size: int = 1000) -> float:
        """Compute mean pairwise cosine similarity."""
        if len(embeddings) > sample_size:
            idx = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings = embeddings[idx]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        cosine_sim = normalized @ normalized.T

        # Exclude diagonal
        mask = ~np.eye(len(cosine_sim), dtype=bool)
        return cosine_sim[mask].mean()

    def detect_drift(
        self,
        current_embeddings: np.ndarray,
        threshold: float = 0.05,
    ) -> dict:
        """
        Detect distribution drift between reference and current embeddings.

        Returns:
            Dictionary with drift metrics and alerts
        """
        current_stats = self._compute_stats(current_embeddings)

        results = {
            "drift_detected": False,
            "metrics": {},
            "alerts": [],
        }

        # KS test on embedding norms
        ks_stat, ks_pvalue = ks_2samp(
            self.reference_stats["norms"],
            current_stats["norms"],
        )
        results["metrics"]["norm_ks_statistic"] = ks_stat
        results["metrics"]["norm_ks_pvalue"] = ks_pvalue

        if ks_pvalue < threshold:
            results["alerts"].append(f"Norm distribution drift detected (p={ks_pvalue:.4f})")
            results["drift_detected"] = True

        # Mean vector drift (cosine distance)
        mean_cosine = np.dot(
            self.reference_stats["mean"],
            current_stats["mean"]
        ) / (
            np.linalg.norm(self.reference_stats["mean"]) *
            np.linalg.norm(current_stats["mean"])
        )
        mean_drift = 1 - mean_cosine
        results["metrics"]["mean_vector_drift"] = mean_drift

        if mean_drift > 0.1:
            results["alerts"].append(f"Mean vector drift: {mean_drift:.4f}")
            results["drift_detected"] = True

        # Pairwise similarity drift
        sim_diff = abs(
            self.reference_stats["pairwise_cosine_mean"] -
            current_stats["pairwise_cosine_mean"]
        )
        results["metrics"]["pairwise_sim_diff"] = sim_diff

        if sim_diff > 0.05:
            results["alerts"].append(f"Pairwise similarity drift: {sim_diff:.4f}")
            results["drift_detected"] = True

        return results
```

## API Design

### Training API

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Embedding Training Service")

class TrainingRequest(BaseModel):
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    dataset_path: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    use_hard_negatives: bool = True
    num_negatives: int = 5

class TrainingStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float
    metrics: dict | None = None
    error: str | None = None

@app.post("/train", response_model=TrainingStatus)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
):
    """Start embedding model training job."""
    job_id = generate_job_id()

    background_tasks.add_task(
        run_training_job,
        job_id=job_id,
        config=request,
    )

    return TrainingStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
    )

@app.get("/train/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get training job status."""
    return get_job_status(job_id)
```

### Inference API

```python
class EmbeddingRequest(BaseModel):
    texts: list[str]
    normalize: bool = True
    batch_size: int = 32

class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model_version: str
    latency_ms: float

@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """Generate embeddings for input texts."""
    start_time = time.time()

    embeddings = embedding_service.encode(
        request.texts,
        batch_size=request.batch_size,
        normalize=request.normalize,
    )

    latency_ms = (time.time() - start_time) * 1000

    return EmbeddingResponse(
        embeddings=embeddings.tolist(),
        model_version=embedding_service.model_version,
        latency_ms=latency_ms,
    )

class SimilarityRequest(BaseModel):
    query: str
    candidates: list[str]
    top_k: int = 10

class SimilarityResponse(BaseModel):
    results: list[dict]  # [{"text": str, "score": float, "rank": int}]

@app.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest):
    """Compute similarity between query and candidates."""
    query_emb = embedding_service.encode([request.query])
    cand_embs = embedding_service.encode(request.candidates)

    scores = (query_emb @ cand_embs.T)[0]
    top_indices = np.argsort(scores)[::-1][:request.top_k]

    results = [
        {
            "text": request.candidates[i],
            "score": float(scores[i]),
            "rank": rank + 1,
        }
        for rank, i in enumerate(top_indices)
    ]

    return SimilarityResponse(results=results)
```

## Performance Optimization

### Training Throughput

| Optimization | Throughput | Memory | Notes |
|-------------|-----------|--------|-------|
| Baseline (fp32) | 1x | 16GB | Single GPU, batch=32 |
| Mixed Precision | 1.8x | 10GB | AMP with GradScaler |
| Gradient Checkpointing | 0.9x | 8GB | Trade compute for memory |
| DDP (8 GPU) | 7.2x | 16GB each | Near-linear scaling |
| Memory Bank | 1.1x | +2GB | More effective negatives |

### Inference Optimization

```python
# Benchmark results for encoding 10,000 texts
# Model: all-mpnet-base-v2, Batch size: 64, GPU: A100

# PyTorch FP32:     45.2 ms/batch, 14,200 texts/sec
# PyTorch FP16:     28.1 ms/batch, 22,800 texts/sec
# ONNX Runtime:     18.5 ms/batch, 34,600 texts/sec
# ONNX + TensorRT:  12.3 ms/batch, 52,000 texts/sec
```

## Implementation Phases

### Phase 1: Core Training (Weeks 1-2)
- [ ] Dataset pipeline with positive pair loading
- [ ] Bi-encoder architecture with multiple pooling strategies
- [ ] MNRL loss implementation
- [ ] Basic training loop with evaluation
- [ ] Unit tests for all components

### Phase 2: Hard Negative Mining (Weeks 3-4)
- [ ] FAISS-based hard negative miner
- [ ] In-batch negative sampling with memory bank
- [ ] Triplet loss with margin
- [ ] Curriculum learning with difficulty scores
- [ ] Mining strategy benchmarks

### Phase 3: Distributed Training (Weeks 5-6)
- [ ] DDP setup with NCCL backend
- [ ] Gradient accumulation for large effective batch
- [ ] Mixed precision training
- [ ] Checkpointing and resume
- [ ] Multi-node training support

### Phase 4: Evaluation Pipeline (Weeks 7-8)
- [ ] Retrieval metrics (Recall@k, MRR, MAP, NDCG)
- [ ] Clustering evaluation
- [ ] Embedding visualization (t-SNE, UMAP)
- [ ] Benchmark suite (BEIR, MTEB)
- [ ] Automated evaluation in CI/CD

### Phase 5: MLOps Integration (Weeks 9-10)
- [ ] MLflow model registry
- [ ] DVC dataset versioning
- [ ] Experiment tracking dashboard
- [ ] Automated hyperparameter search
- [ ] A/B testing framework

### Phase 6: Production Deployment (Weeks 11-12)
- [ ] ONNX export and optimization
- [ ] TensorRT integration
- [ ] REST API service
- [ ] Drift monitoring
- [ ] Load testing and scaling

## Testing Strategy

### Unit Tests

```python
import pytest
import torch

class TestBiEncoder:
    @pytest.fixture
    def model(self):
        return BiEncoder(
            model_name="prajjwal1/bert-tiny",
            pooling="mean",
            normalize=True,
        )

    def test_forward_shape(self, model):
        input_ids = torch.randint(0, 1000, (4, 32))
        attention_mask = torch.ones_like(input_ids)

        embeddings = model(input_ids, attention_mask)

        assert embeddings.shape == (4, 128)  # bert-tiny hidden size

    def test_normalized_output(self, model):
        input_ids = torch.randint(0, 1000, (4, 32))
        attention_mask = torch.ones_like(input_ids)

        embeddings = model(input_ids, attention_mask)
        norms = torch.norm(embeddings, dim=1)

        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

class TestLossFunctions:
    def test_mnrl_loss(self):
        loss_fn = MultipleNegativesRankingLoss(scale=20.0)

        anchor = torch.randn(8, 64)
        positive = torch.randn(8, 64)

        loss = loss_fn(anchor, positive)

        assert loss.item() > 0
        assert not torch.isnan(loss)
```

### Integration Tests

```python
class TestTrainingPipeline:
    def test_end_to_end_training(self, tmp_path):
        # Create small dataset
        dataset = create_dummy_dataset(num_samples=100)

        # Configure training
        config = TrainingConfig(
            num_epochs=1,
            batch_size=8,
            learning_rate=1e-4,
            output_dir=tmp_path,
        )

        # Run training
        trainer = EmbeddingTrainer(
            model=BiEncoder("prajjwal1/bert-tiny"),
            train_dataset=dataset,
            eval_dataset=dataset,
            config=config,
        )

        metrics = trainer.train()

        assert "train_loss" in metrics
        assert metrics["train_loss"] > 0
        assert (tmp_path / "model.pt").exists()
```

### Benchmarks

```python
@pytest.mark.benchmark
class TestPerformance:
    def test_encoding_throughput(self, benchmark):
        model = ONNXEmbeddingModel("model.onnx", "tokenizer")
        texts = ["sample text"] * 1000

        result = benchmark(model.encode, texts, batch_size=64)

        # Assert minimum throughput
        texts_per_second = 1000 / result.stats["mean"]
        assert texts_per_second > 5000
```

## Stretch Goals

### Cross-Encoder Distillation

```python
class CrossEncoderDistillation:
    """Distill cross-encoder scores to bi-encoder."""

    def __init__(self, cross_encoder, bi_encoder, temperature: float = 2.0):
        self.cross_encoder = cross_encoder
        self.bi_encoder = bi_encoder
        self.temperature = temperature

    def compute_distillation_loss(
        self,
        queries: list[str],
        documents: list[str],
    ) -> torch.Tensor:
        # Get cross-encoder scores (teacher)
        with torch.no_grad():
            teacher_scores = self.cross_encoder.predict(
                [(q, d) for q, d in zip(queries, documents)]
            )
            teacher_probs = F.softmax(
                teacher_scores / self.temperature, dim=-1
            )

        # Get bi-encoder scores (student)
        query_embs = self.bi_encoder.encode(queries)
        doc_embs = self.bi_encoder.encode(documents)
        student_scores = (query_embs * doc_embs).sum(dim=-1)
        student_probs = F.softmax(
            student_scores / self.temperature, dim=-1
        )

        # KL divergence loss
        loss = F.kl_div(
            student_probs.log(),
            teacher_probs,
            reduction="batchmean",
        )

        return loss * (self.temperature ** 2)
```

### Multilingual Support

```python
class MultilingualEmbeddingModel(BiEncoder):
    """Multilingual embedding model with language-specific adapters."""

    def __init__(self, base_model: str, languages: list[str]):
        super().__init__(base_model)

        # Language-specific adapter layers
        self.language_adapters = nn.ModuleDict({
            lang: nn.Sequential(
                nn.Linear(768, 768),
                nn.GELU(),
                nn.Linear(768, 768),
            )
            for lang in languages
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        language: str = "en",
    ) -> torch.Tensor:
        embeddings = super().forward(input_ids, attention_mask)

        if language in self.language_adapters:
            embeddings = embeddings + self.language_adapters[language](embeddings)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
```

## References

- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [SimCSE](https://arxiv.org/abs/2104.08821)
- [BEIR Benchmark](https://arxiv.org/abs/2104.08663)
- [Approximate Nearest Neighbor Negative Contrastive Learning](https://arxiv.org/abs/2009.13835)
- [Multi-Stage Training for Dense Retrieval](https://arxiv.org/abs/2201.05438)
