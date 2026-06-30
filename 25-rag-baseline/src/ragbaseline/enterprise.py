"""Enterprise features: multi-tenancy, logging, and analytics."""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .schemas import Document, SearchResult, RAGConfig
from .embeddings import EmbeddingModel, get_embedding_model
from .vectorstore import VectorStore, get_vector_store
from .chunking import ChunkingStrategy, SentenceChunker
from .index import RAGIndex
from .pipeline import RAGPipeline, LLMProvider


class TenantManager:
    """Manage per-tenant document collections with isolation."""

    def __init__(
        self,
        base_directory: str = "./tenants",
        default_embedding_model: str = "sentence-transformers",
        default_embedding_name: str = "BAAI/bge-small-en-v1.5",
        default_vector_store: str = "chroma",
    ):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

        self.default_embedding_model = default_embedding_model
        self.default_embedding_name = default_embedding_name
        self.default_vector_store = default_vector_store

        # Cache of tenant indices
        self._indices: dict[str, RAGIndex] = {}
        self._tenant_configs: dict[str, dict] = {}

    def get_tenant_index(
        self,
        tenant_id: str,
        embedding_model: EmbeddingModel = None,
        vector_store: VectorStore = None,
        chunker: ChunkingStrategy = None,
    ) -> RAGIndex:
        """Get or create tenant-specific index.

        Args:
            tenant_id: Unique tenant identifier
            embedding_model: Optional custom embedding model
            vector_store: Optional custom vector store
            chunker: Optional custom chunking strategy

        Returns:
            RAGIndex for the tenant
        """
        if tenant_id not in self._indices:
            self._create_tenant(
                tenant_id,
                embedding_model,
                vector_store,
                chunker,
            )

        return self._indices[tenant_id]

    def _create_tenant(
        self,
        tenant_id: str,
        embedding_model: EmbeddingModel = None,
        vector_store: VectorStore = None,
        chunker: ChunkingStrategy = None,
    ):
        """Create new tenant with isolated storage."""
        # Create tenant directory
        tenant_dir = self.base_directory / tenant_id
        tenant_dir.mkdir(parents=True, exist_ok=True)

        # Create embedding model
        if embedding_model is None:
            embedding_model = get_embedding_model(
                model_type=self.default_embedding_model,
                model_name=self.default_embedding_name,
            )

        # Create vector store
        if vector_store is None:
            if self.default_vector_store == "chroma":
                vector_store = get_vector_store(
                    store_type="chroma",
                    collection_name=f"tenant_{tenant_id}",
                    persist_directory=str(tenant_dir / "chroma"),
                )
            else:
                vector_store = get_vector_store(store_type="simple")

        # Create chunker
        if chunker is None:
            chunker = SentenceChunker()

        # Create index
        self._indices[tenant_id] = RAGIndex(
            embedding_model=embedding_model,
            vector_store=vector_store,
            chunker=chunker,
        )

        # Store tenant config
        self._tenant_configs[tenant_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "directory": str(tenant_dir),
        }

        # Save tenant metadata
        self._save_tenant_metadata(tenant_id)

    def _save_tenant_metadata(self, tenant_id: str):
        """Save tenant metadata to disk."""
        tenant_dir = self.base_directory / tenant_id
        metadata_path = tenant_dir / "metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(self._tenant_configs[tenant_id], f, indent=2)

    def delete_tenant(self, tenant_id: str):
        """Delete tenant and all associated data.

        Args:
            tenant_id: Tenant to delete

        Warning: This permanently deletes all tenant data!
        """
        # Remove from cache
        if tenant_id in self._indices:
            del self._indices[tenant_id]

        if tenant_id in self._tenant_configs:
            del self._tenant_configs[tenant_id]

        # Delete tenant directory
        tenant_dir = self.base_directory / tenant_id
        if tenant_dir.exists():
            shutil.rmtree(tenant_dir)

    def list_tenants(self) -> list[str]:
        """List all tenant IDs."""
        tenants = []
        for path in self.base_directory.iterdir():
            if path.is_dir():
                tenants.append(path.name)
        return sorted(tenants)

    def get_tenant_info(self, tenant_id: str) -> dict:
        """Get tenant information."""
        if tenant_id in self._tenant_configs:
            info = self._tenant_configs[tenant_id].copy()
        else:
            tenant_dir = self.base_directory / tenant_id
            metadata_path = tenant_dir / "metadata.json"

            if metadata_path.exists():
                with open(metadata_path) as f:
                    info = json.load(f)
            else:
                info = {}

        # Add document count if index exists
        if tenant_id in self._indices:
            info["document_count"] = self._indices[tenant_id].count

        return info

    def get_tenant_pipeline(
        self,
        tenant_id: str,
        llm_provider: LLMProvider,
        config: RAGConfig = None,
    ) -> RAGPipeline:
        """Get RAG pipeline for tenant.

        Args:
            tenant_id: Tenant ID
            llm_provider: LLM provider for generation
            config: Optional RAG config

        Returns:
            Configured RAG pipeline for tenant
        """
        index = self.get_tenant_index(tenant_id)
        return RAGPipeline(index, llm_provider, config)


class RetrievalLogger:
    """Log retrieval queries and results for analysis and debugging."""

    def __init__(
        self,
        log_directory: str = "./logs",
        log_file: str = "retrieval.jsonl",
        enable_analytics: bool = True,
        rotation_size_mb: int = 100,
    ):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_directory / log_file
        self.enable_analytics = enable_analytics
        self.rotation_size_mb = rotation_size_mb

        # In-memory analytics cache
        self._query_count = 0
        self._latencies: list[float] = []
        self._tenant_queries: dict[str, int] = {}

    def log_query(
        self,
        query: str,
        results: list[SearchResult],
        response: str = None,
        tenant_id: str = None,
        user_id: str = None,
        latency_ms: float = None,
        metadata: dict = None,
    ):
        """Log a retrieval query.

        Args:
            query: User query
            results: Search results
            response: LLM response (if generated)
            tenant_id: Tenant identifier
            user_id: User identifier
            latency_ms: Query latency in milliseconds
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "num_results": len(results),
            "result_ids": [r.id for r in results],
            "result_scores": [r.score for r in results],
            "response_length": len(response) if response else 0,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "latency_ms": latency_ms,
            "metadata": metadata or {},
        }

        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Update analytics cache
        if self.enable_analytics:
            self._update_analytics(tenant_id, latency_ms)

        # Check for rotation
        self._check_rotation()

    def _update_analytics(self, tenant_id: str, latency_ms: float):
        """Update in-memory analytics."""
        self._query_count += 1

        if latency_ms:
            self._latencies.append(latency_ms)

        if tenant_id:
            self._tenant_queries[tenant_id] = (
                self._tenant_queries.get(tenant_id, 0) + 1
            )

    def _check_rotation(self):
        """Rotate log file if it exceeds size limit."""
        if not self.log_file.exists():
            return

        size_mb = self.log_file.stat().st_size / (1024 * 1024)
        if size_mb >= self.rotation_size_mb:
            # Rotate file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            rotated_path = self.log_file.with_suffix(f".{timestamp}.jsonl")
            self.log_file.rename(rotated_path)

    def get_analytics(self) -> dict:
        """Get analytics summary.

        Returns:
            Dictionary with analytics metrics
        """
        analytics = {
            "total_queries": self._query_count,
            "tenants": len(self._tenant_queries),
            "queries_by_tenant": self._tenant_queries.copy(),
        }

        if self._latencies:
            latencies = np.array(self._latencies)
            analytics["latency"] = {
                "avg_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
            }
        else:
            analytics["latency"] = None

        return analytics

    def analyze_logs(
        self,
        start_date: str = None,
        end_date: str = None,
        tenant_id: str = None,
    ) -> dict:
        """Analyze retrieval logs from disk.

        Args:
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            tenant_id: Filter by tenant

        Returns:
            Detailed analytics
        """
        if not self.log_file.exists():
            return {"error": "No logs found"}

        queries = []
        latencies = []
        result_counts = []
        tenants = {}
        users = {}

        with open(self.log_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply filters
                if start_date and entry["timestamp"] < start_date:
                    continue
                if end_date and entry["timestamp"] > end_date:
                    continue
                if tenant_id and entry.get("tenant_id") != tenant_id:
                    continue

                queries.append(entry["query"])
                result_counts.append(entry["num_results"])

                if entry.get("latency_ms"):
                    latencies.append(entry["latency_ms"])

                if entry.get("tenant_id"):
                    tid = entry["tenant_id"]
                    tenants[tid] = tenants.get(tid, 0) + 1

                if entry.get("user_id"):
                    uid = entry["user_id"]
                    users[uid] = users.get(uid, 0) + 1

        analytics = {
            "total_queries": len(queries),
            "unique_queries": len(set(queries)),
            "avg_results": np.mean(result_counts) if result_counts else 0,
            "tenants": tenants,
            "top_users": dict(sorted(
                users.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
        }

        if latencies:
            lat_array = np.array(latencies)
            analytics["latency"] = {
                "avg_ms": float(np.mean(lat_array)),
                "p50_ms": float(np.percentile(lat_array, 50)),
                "p95_ms": float(np.percentile(lat_array, 95)),
                "p99_ms": float(np.percentile(lat_array, 99)),
            }

        return analytics

    def export_logs(
        self,
        output_path: str,
        format: str = "jsonl",
        start_date: str = None,
        end_date: str = None,
    ):
        """Export logs to file.

        Args:
            output_path: Output file path
            format: Output format (jsonl, csv)
            start_date: Filter by start date
            end_date: Filter by end date
        """
        if not self.log_file.exists():
            return

        output_path = Path(output_path)

        if format == "jsonl":
            # Copy with optional filtering
            with open(self.log_file) as f_in, open(output_path, "w") as f_out:
                for line in f_in:
                    entry = json.loads(line)
                    if start_date and entry["timestamp"] < start_date:
                        continue
                    if end_date and entry["timestamp"] > end_date:
                        continue
                    f_out.write(line)

        elif format == "csv":
            import csv

            with open(self.log_file) as f_in, open(output_path, "w", newline="") as f_out:
                writer = None

                for line in f_in:
                    entry = json.loads(line)
                    if start_date and entry["timestamp"] < start_date:
                        continue
                    if end_date and entry["timestamp"] > end_date:
                        continue

                    if writer is None:
                        fieldnames = list(entry.keys())
                        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                        writer.writeheader()

                    # Flatten complex fields
                    flat_entry = entry.copy()
                    flat_entry["result_ids"] = ",".join(entry.get("result_ids", []))
                    flat_entry["result_scores"] = ",".join(
                        str(s) for s in entry.get("result_scores", [])
                    )
                    flat_entry["metadata"] = json.dumps(entry.get("metadata", {}))

                    writer.writerow(flat_entry)


class UsageTracker:
    """Track usage metrics for billing and quotas."""

    def __init__(
        self,
        storage_path: str = "./usage",
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._usage: dict[str, dict] = {}
        self._load_usage()

    def _load_usage(self):
        """Load usage data from disk."""
        usage_file = self.storage_path / "usage.json"
        if usage_file.exists():
            with open(usage_file) as f:
                self._usage = json.load(f)

    def _save_usage(self):
        """Save usage data to disk."""
        usage_file = self.storage_path / "usage.json"
        with open(usage_file, "w") as f:
            json.dump(self._usage, f, indent=2)

    def track_query(
        self,
        tenant_id: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ):
        """Track a query for billing.

        Args:
            tenant_id: Tenant ID
            tokens_in: Input tokens
            tokens_out: Output tokens
        """
        if tenant_id not in self._usage:
            self._usage[tenant_id] = {
                "queries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "documents": 0,
            }

        self._usage[tenant_id]["queries"] += 1
        self._usage[tenant_id]["tokens_in"] += tokens_in
        self._usage[tenant_id]["tokens_out"] += tokens_out

        self._save_usage()

    def track_document(
        self,
        tenant_id: str,
        count: int = 1,
    ):
        """Track document ingestion.

        Args:
            tenant_id: Tenant ID
            count: Number of documents
        """
        if tenant_id not in self._usage:
            self._usage[tenant_id] = {
                "queries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "documents": 0,
            }

        self._usage[tenant_id]["documents"] += count
        self._save_usage()

    def get_usage(self, tenant_id: str) -> dict:
        """Get usage for tenant."""
        return self._usage.get(tenant_id, {
            "queries": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "documents": 0,
        })

    def get_all_usage(self) -> dict:
        """Get usage for all tenants."""
        return self._usage.copy()

    def reset_usage(self, tenant_id: str):
        """Reset usage for tenant (e.g., for new billing period)."""
        if tenant_id in self._usage:
            self._usage[tenant_id] = {
                "queries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "documents": 0,
            }
            self._save_usage()
