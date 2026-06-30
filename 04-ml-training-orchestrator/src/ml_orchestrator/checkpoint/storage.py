"""Checkpoint storage backends."""

import asyncio
import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from uuid import uuid4
import structlog

from ml_orchestrator.core.models import Checkpoint, CheckpointStatus, StorageBackend
from ml_orchestrator.core.exceptions import CheckpointError, CheckpointNotFoundError


logger = structlog.get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a stored checkpoint."""

    checkpoint_id: str
    job_id: str
    epoch: int
    step: int
    path: str
    size_bytes: int
    checksum: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class CheckpointStorage(ABC):
    """Abstract base class for checkpoint storage backends."""

    @property
    @abstractmethod
    def backend_type(self) -> StorageBackend:
        """Get the storage backend type."""
        pass

    @abstractmethod
    async def save(
        self,
        checkpoint: Checkpoint,
        data: bytes,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Save checkpoint data.

        Args:
            checkpoint: Checkpoint metadata
            data: Checkpoint binary data
            metadata: Additional metadata

        Returns:
            Storage path/URL of the saved checkpoint
        """
        pass

    @abstractmethod
    async def load(self, path: str) -> bytes:
        """
        Load checkpoint data.

        Args:
            path: Storage path

        Returns:
            Checkpoint binary data

        Raises:
            CheckpointNotFoundError: If checkpoint not found
        """
        pass

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            path: Storage path

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if checkpoint exists."""
        pass

    @abstractmethod
    async def list_checkpoints(
        self,
        job_id: str,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """
        List checkpoints for a job.

        Args:
            job_id: Job ID
            limit: Maximum number to return

        Returns:
            List of checkpoint metadata
        """
        pass

    @abstractmethod
    async def get_metadata(self, path: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a checkpoint."""
        pass

    def _compute_checksum(self, data: bytes) -> str:
        """Compute MD5 checksum of data."""
        return hashlib.md5(data).hexdigest()


class LocalStorage(CheckpointStorage):
    """Local filesystem checkpoint storage."""

    def __init__(self, base_path: str = "/checkpoints"):
        self._base_path = Path(base_path)
        self._metadata_cache: dict[str, CheckpointMetadata] = {}

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.LOCAL

    def _get_checkpoint_dir(self, job_id: str) -> Path:
        """Get directory for job's checkpoints."""
        return self._base_path / job_id

    def _get_checkpoint_path(self, job_id: str, checkpoint_id: str) -> Path:
        """Get full path for a checkpoint."""
        return self._get_checkpoint_dir(job_id) / f"{checkpoint_id}.ckpt"

    def _get_metadata_path(self, job_id: str, checkpoint_id: str) -> Path:
        """Get path for checkpoint metadata file."""
        return self._get_checkpoint_dir(job_id) / f"{checkpoint_id}.meta.json"

    async def save(
        self,
        checkpoint: Checkpoint,
        data: bytes,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save checkpoint to local filesystem."""
        job_dir = self._get_checkpoint_dir(checkpoint.job_id)

        # Create directory if needed
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: job_dir.mkdir(parents=True, exist_ok=True)
        )

        checkpoint_path = self._get_checkpoint_path(checkpoint.job_id, checkpoint.id)
        metadata_path = self._get_metadata_path(checkpoint.job_id, checkpoint.id)

        # Write checkpoint data
        def write_data():
            with open(checkpoint_path, "wb") as f:
                f.write(data)

        await asyncio.get_event_loop().run_in_executor(None, write_data)

        # Create metadata
        checksum = self._compute_checksum(data)
        meta = CheckpointMetadata(
            checkpoint_id=checkpoint.id,
            job_id=checkpoint.job_id,
            epoch=checkpoint.epoch,
            step=checkpoint.step,
            path=str(checkpoint_path),
            size_bytes=len(data),
            checksum=checksum,
            metrics=checkpoint.metrics,
            extra=metadata or {},
        )

        # Write metadata
        import json

        def write_meta():
            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "checkpoint_id": meta.checkpoint_id,
                        "job_id": meta.job_id,
                        "epoch": meta.epoch,
                        "step": meta.step,
                        "path": meta.path,
                        "size_bytes": meta.size_bytes,
                        "checksum": meta.checksum,
                        "created_at": meta.created_at.isoformat(),
                        "metrics": meta.metrics,
                        "extra": meta.extra,
                    },
                    f,
                )

        await asyncio.get_event_loop().run_in_executor(None, write_meta)

        self._metadata_cache[str(checkpoint_path)] = meta

        logger.info(
            "checkpoint_saved_local",
            checkpoint_id=checkpoint.id,
            path=str(checkpoint_path),
            size_bytes=len(data),
        )

        return str(checkpoint_path)

    async def load(self, path: str) -> bytes:
        """Load checkpoint from local filesystem."""

        def read_data():
            if not os.path.exists(path):
                raise CheckpointNotFoundError(path)
            with open(path, "rb") as f:
                return f.read()

        try:
            data = await asyncio.get_event_loop().run_in_executor(None, read_data)
            logger.debug("checkpoint_loaded_local", path=path, size_bytes=len(data))
            return data
        except FileNotFoundError:
            raise CheckpointNotFoundError(path)

    async def delete(self, path: str) -> bool:
        """Delete checkpoint from local filesystem."""

        def do_delete():
            if os.path.exists(path):
                os.remove(path)
                # Also remove metadata file
                meta_path = path.replace(".ckpt", ".meta.json")
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                return True
            return False

        result = await asyncio.get_event_loop().run_in_executor(None, do_delete)
        if result:
            self._metadata_cache.pop(path, None)
            logger.info("checkpoint_deleted_local", path=path)
        return result

    async def exists(self, path: str) -> bool:
        """Check if checkpoint exists locally."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: os.path.exists(path)
        )

    async def list_checkpoints(
        self,
        job_id: str,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """List checkpoints for a job."""
        job_dir = self._get_checkpoint_dir(job_id)

        def list_files():
            if not job_dir.exists():
                return []
            return sorted(job_dir.glob("*.meta.json"), key=lambda p: p.stat().st_mtime)

        meta_files = await asyncio.get_event_loop().run_in_executor(None, list_files)

        results = []
        import json

        for meta_file in meta_files[-limit:]:
            try:

                def read_meta():
                    with open(meta_file) as f:
                        return json.load(f)

                data = await asyncio.get_event_loop().run_in_executor(None, read_meta)
                meta = CheckpointMetadata(
                    checkpoint_id=data["checkpoint_id"],
                    job_id=data["job_id"],
                    epoch=data["epoch"],
                    step=data["step"],
                    path=data["path"],
                    size_bytes=data["size_bytes"],
                    checksum=data.get("checksum"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    metrics=data.get("metrics", {}),
                    extra=data.get("extra", {}),
                )
                results.append(meta)
            except Exception as e:
                logger.warning("checkpoint_meta_read_error", file=str(meta_file), error=str(e))

        return results

    async def get_metadata(self, path: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a checkpoint."""
        if path in self._metadata_cache:
            return self._metadata_cache[path]

        meta_path = path.replace(".ckpt", ".meta.json")

        if not await self.exists(meta_path):
            return None

        import json

        def read_meta():
            with open(meta_path) as f:
                return json.load(f)

        try:
            data = await asyncio.get_event_loop().run_in_executor(None, read_meta)
            meta = CheckpointMetadata(
                checkpoint_id=data["checkpoint_id"],
                job_id=data["job_id"],
                epoch=data["epoch"],
                step=data["step"],
                path=data["path"],
                size_bytes=data["size_bytes"],
                checksum=data.get("checksum"),
                created_at=datetime.fromisoformat(data["created_at"]),
                metrics=data.get("metrics", {}),
                extra=data.get("extra", {}),
            )
            self._metadata_cache[path] = meta
            return meta
        except Exception:
            return None

    async def cleanup_old_checkpoints(
        self,
        job_id: str,
        keep_last_n: int = 3,
    ) -> int:
        """Delete old checkpoints, keeping the last N."""
        checkpoints = await self.list_checkpoints(job_id, limit=1000)

        if len(checkpoints) <= keep_last_n:
            return 0

        # Sort by step (descending) and keep last N
        sorted_ckpts = sorted(checkpoints, key=lambda c: (c.epoch, c.step), reverse=True)
        to_delete = sorted_ckpts[keep_last_n:]

        deleted = 0
        for ckpt in to_delete:
            if await self.delete(ckpt.path):
                deleted += 1

        logger.info(
            "old_checkpoints_cleaned",
            job_id=job_id,
            deleted=deleted,
            kept=keep_last_n,
        )

        return deleted


class S3Storage(CheckpointStorage):
    """S3 checkpoint storage backend."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "checkpoints",
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
    ):
        self._bucket = bucket
        self._prefix = prefix
        self._region = region
        self._endpoint_url = endpoint_url
        self._client = None

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.S3

    def _get_s3_key(self, job_id: str, checkpoint_id: str, suffix: str = ".ckpt") -> str:
        """Get S3 key for a checkpoint."""
        return f"{self._prefix}/{job_id}/{checkpoint_id}{suffix}"

    async def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            try:
                import aioboto3

                session = aioboto3.Session()
                self._client = await session.client(
                    "s3",
                    region_name=self._region,
                    endpoint_url=self._endpoint_url,
                ).__aenter__()
            except ImportError:
                raise CheckpointError("aioboto3 not installed for S3 storage")
        return self._client

    async def save(
        self,
        checkpoint: Checkpoint,
        data: bytes,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save checkpoint to S3."""
        client = await self._get_client()

        key = self._get_s3_key(checkpoint.job_id, checkpoint.id)
        checksum = self._compute_checksum(data)

        # Prepare metadata for S3
        s3_metadata = {
            "checkpoint_id": checkpoint.id,
            "job_id": checkpoint.job_id,
            "epoch": str(checkpoint.epoch),
            "step": str(checkpoint.step),
            "checksum": checksum,
        }

        try:
            await client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                Metadata=s3_metadata,
            )

            # Also save detailed metadata as JSON
            import json

            meta = CheckpointMetadata(
                checkpoint_id=checkpoint.id,
                job_id=checkpoint.job_id,
                epoch=checkpoint.epoch,
                step=checkpoint.step,
                path=f"s3://{self._bucket}/{key}",
                size_bytes=len(data),
                checksum=checksum,
                metrics=checkpoint.metrics,
                extra=metadata or {},
            )

            meta_key = self._get_s3_key(checkpoint.job_id, checkpoint.id, ".meta.json")
            await client.put_object(
                Bucket=self._bucket,
                Key=meta_key,
                Body=json.dumps(
                    {
                        "checkpoint_id": meta.checkpoint_id,
                        "job_id": meta.job_id,
                        "epoch": meta.epoch,
                        "step": meta.step,
                        "path": meta.path,
                        "size_bytes": meta.size_bytes,
                        "checksum": meta.checksum,
                        "created_at": meta.created_at.isoformat(),
                        "metrics": meta.metrics,
                        "extra": meta.extra,
                    }
                ).encode(),
            )

            path = f"s3://{self._bucket}/{key}"
            logger.info(
                "checkpoint_saved_s3",
                checkpoint_id=checkpoint.id,
                path=path,
                size_bytes=len(data),
            )

            return path

        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint to S3: {e}")

    async def load(self, path: str) -> bytes:
        """Load checkpoint from S3."""
        client = await self._get_client()

        # Parse S3 path
        if path.startswith("s3://"):
            path = path[5:]
            bucket, key = path.split("/", 1)
        else:
            bucket = self._bucket
            key = path

        try:
            response = await client.get_object(Bucket=bucket, Key=key)
            data = await response["Body"].read()
            logger.debug("checkpoint_loaded_s3", path=path, size_bytes=len(data))
            return data
        except client.exceptions.NoSuchKey:
            raise CheckpointNotFoundError(path)
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint from S3: {e}")

    async def delete(self, path: str) -> bool:
        """Delete checkpoint from S3."""
        client = await self._get_client()

        if path.startswith("s3://"):
            path = path[5:]
            bucket, key = path.split("/", 1)
        else:
            bucket = self._bucket
            key = path

        try:
            await client.delete_object(Bucket=bucket, Key=key)
            # Also delete metadata
            meta_key = key.replace(".ckpt", ".meta.json")
            await client.delete_object(Bucket=bucket, Key=meta_key)
            logger.info("checkpoint_deleted_s3", path=path)
            return True
        except Exception as e:
            logger.error("checkpoint_delete_failed_s3", path=path, error=str(e))
            return False

    async def exists(self, path: str) -> bool:
        """Check if checkpoint exists in S3."""
        client = await self._get_client()

        if path.startswith("s3://"):
            path = path[5:]
            bucket, key = path.split("/", 1)
        else:
            bucket = self._bucket
            key = path

        try:
            await client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    async def list_checkpoints(
        self,
        job_id: str,
        limit: int = 100,
    ) -> list[CheckpointMetadata]:
        """List checkpoints for a job from S3."""
        client = await self._get_client()
        prefix = f"{self._prefix}/{job_id}/"

        try:
            response = await client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix,
                MaxKeys=limit * 2,  # Account for .ckpt and .meta.json files
            )

            results = []
            import json

            for obj in response.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".meta.json"):
                    continue

                try:
                    meta_response = await client.get_object(
                        Bucket=self._bucket, Key=key
                    )
                    data = json.loads(await meta_response["Body"].read())
                    meta = CheckpointMetadata(
                        checkpoint_id=data["checkpoint_id"],
                        job_id=data["job_id"],
                        epoch=data["epoch"],
                        step=data["step"],
                        path=data["path"],
                        size_bytes=data["size_bytes"],
                        checksum=data.get("checksum"),
                        created_at=datetime.fromisoformat(data["created_at"]),
                        metrics=data.get("metrics", {}),
                        extra=data.get("extra", {}),
                    )
                    results.append(meta)
                except Exception:
                    continue

            return results[:limit]

        except Exception as e:
            logger.error("list_checkpoints_failed_s3", job_id=job_id, error=str(e))
            return []

    async def get_metadata(self, path: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a checkpoint from S3."""
        if path.startswith("s3://"):
            path = path[5:]
            bucket, key = path.split("/", 1)
        else:
            bucket = self._bucket
            key = path

        meta_key = key.replace(".ckpt", ".meta.json")

        try:
            client = await self._get_client()
            import json

            response = await client.get_object(Bucket=bucket, Key=meta_key)
            data = json.loads(await response["Body"].read())
            return CheckpointMetadata(
                checkpoint_id=data["checkpoint_id"],
                job_id=data["job_id"],
                epoch=data["epoch"],
                step=data["step"],
                path=data["path"],
                size_bytes=data["size_bytes"],
                checksum=data.get("checksum"),
                created_at=datetime.fromisoformat(data["created_at"]),
                metrics=data.get("metrics", {}),
                extra=data.get("extra", {}),
            )
        except Exception:
            return None
