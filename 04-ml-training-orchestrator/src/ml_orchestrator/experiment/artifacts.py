"""Artifact storage for experiment tracking."""

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
import structlog


logger = structlog.get_logger(__name__)


@dataclass
class Artifact:
    """An artifact stored for a run."""

    id: str = field(default_factory=lambda: str(uuid4()))
    run_id: str = ""
    name: str = ""
    artifact_type: str = "file"  # file, model, dataset, image, etc.
    path: str = ""
    size_bytes: int = 0
    checksum: Optional[str] = None
    content_type: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactStore:
    """
    Stores and manages artifacts for experiment runs.

    Supports:
    - File-based artifacts (models, configs, logs)
    - Metadata tracking
    - Deduplication via checksums
    - Listing and retrieval
    """

    def __init__(self, base_path: str = "/artifacts"):
        self._base_path = Path(base_path)
        self._artifacts: dict[str, Artifact] = {}
        self._run_artifacts: dict[str, list[str]] = {}  # run_id -> artifact_ids
        self._checksum_index: dict[str, str] = {}  # checksum -> artifact_id
        self._lock = asyncio.Lock()

    def _get_artifact_dir(self, run_id: str) -> Path:
        """Get directory for run's artifacts."""
        return self._base_path / run_id

    def _get_artifact_path(self, run_id: str, name: str) -> Path:
        """Get full path for an artifact."""
        # Sanitize name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        return self._get_artifact_dir(run_id) / safe_name

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum."""
        return hashlib.sha256(data).hexdigest()

    def _detect_content_type(self, name: str) -> str:
        """Detect content type from filename."""
        ext = Path(name).suffix.lower()
        content_types = {
            ".pt": "application/x-pytorch",
            ".pth": "application/x-pytorch",
            ".onnx": "application/x-onnx",
            ".h5": "application/x-hdf5",
            ".pkl": "application/x-pickle",
            ".json": "application/json",
            ".yaml": "application/x-yaml",
            ".yml": "application/x-yaml",
            ".txt": "text/plain",
            ".log": "text/plain",
            ".csv": "text/csv",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }
        return content_types.get(ext, "application/octet-stream")

    def _detect_artifact_type(self, name: str) -> str:
        """Detect artifact type from filename."""
        ext = Path(name).suffix.lower()
        name_lower = name.lower()

        if ext in (".pt", ".pth", ".onnx", ".h5"):
            return "model"
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".svg"):
            return "image"
        elif ext in (".csv", ".parquet", ".json") and "data" in name_lower:
            return "dataset"
        elif ext in (".yaml", ".yml", ".json") and "config" in name_lower:
            return "config"
        elif ext in (".txt", ".log"):
            return "log"
        else:
            return "file"

    async def save_artifact(
        self,
        run_id: str,
        name: str,
        data: bytes,
        artifact_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Artifact:
        """
        Save an artifact for a run.

        Args:
            run_id: Run ID
            name: Artifact name
            data: Artifact binary data
            artifact_type: Type of artifact
            metadata: Additional metadata

        Returns:
            Created Artifact
        """
        checksum = self._compute_checksum(data)

        async with self._lock:
            # Check for duplicate
            if checksum in self._checksum_index:
                existing = self._artifacts.get(self._checksum_index[checksum])
                if existing:
                    logger.debug(
                        "artifact_deduplicated",
                        run_id=run_id,
                        name=name,
                        existing_id=existing.id,
                    )
                    # Create reference to existing artifact
                    artifact = Artifact(
                        run_id=run_id,
                        name=name,
                        artifact_type=artifact_type or existing.artifact_type,
                        path=existing.path,
                        size_bytes=existing.size_bytes,
                        checksum=checksum,
                        content_type=existing.content_type,
                        metadata=metadata or {},
                    )
                    self._artifacts[artifact.id] = artifact
                    if run_id not in self._run_artifacts:
                        self._run_artifacts[run_id] = []
                    self._run_artifacts[run_id].append(artifact.id)
                    return artifact

        # Create new artifact
        artifact_dir = self._get_artifact_dir(run_id)
        artifact_path = self._get_artifact_path(run_id, name)

        # Ensure directory exists
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: artifact_dir.mkdir(parents=True, exist_ok=True)
        )

        # Write file
        def write_file():
            with open(artifact_path, "wb") as f:
                f.write(data)

        await asyncio.get_event_loop().run_in_executor(None, write_file)

        artifact = Artifact(
            run_id=run_id,
            name=name,
            artifact_type=artifact_type or self._detect_artifact_type(name),
            path=str(artifact_path),
            size_bytes=len(data),
            checksum=checksum,
            content_type=self._detect_content_type(name),
            metadata=metadata or {},
        )

        async with self._lock:
            self._artifacts[artifact.id] = artifact
            self._checksum_index[checksum] = artifact.id
            if run_id not in self._run_artifacts:
                self._run_artifacts[run_id] = []
            self._run_artifacts[run_id].append(artifact.id)

        logger.info(
            "artifact_saved",
            artifact_id=artifact.id,
            run_id=run_id,
            name=name,
            size_bytes=len(data),
        )

        return artifact

    async def load_artifact(self, artifact_id: str) -> Optional[tuple[Artifact, bytes]]:
        """
        Load an artifact.

        Returns:
            Tuple of (Artifact, data) or None if not found
        """
        async with self._lock:
            artifact = self._artifacts.get(artifact_id)
            if not artifact:
                return None

        def read_file():
            with open(artifact.path, "rb") as f:
                return f.read()

        try:
            data = await asyncio.get_event_loop().run_in_executor(None, read_file)
            return artifact, data
        except FileNotFoundError:
            return None

    async def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get artifact metadata by ID."""
        async with self._lock:
            return self._artifacts.get(artifact_id)

    async def list_artifacts(
        self,
        run_id: str,
        artifact_type: Optional[str] = None,
    ) -> list[Artifact]:
        """List artifacts for a run."""
        async with self._lock:
            artifact_ids = self._run_artifacts.get(run_id, [])
            artifacts = [
                self._artifacts[aid]
                for aid in artifact_ids
                if aid in self._artifacts
            ]

            if artifact_type:
                artifacts = [a for a in artifacts if a.artifact_type == artifact_type]

            return artifacts

    async def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact."""
        async with self._lock:
            artifact = self._artifacts.pop(artifact_id, None)
            if not artifact:
                return False

            # Remove from run artifacts
            if artifact.run_id in self._run_artifacts:
                self._run_artifacts[artifact.run_id] = [
                    aid for aid in self._run_artifacts[artifact.run_id]
                    if aid != artifact_id
                ]

            # Remove from checksum index
            if artifact.checksum:
                self._checksum_index.pop(artifact.checksum, None)

        # Delete file
        def delete_file():
            if os.path.exists(artifact.path):
                os.remove(artifact.path)

        await asyncio.get_event_loop().run_in_executor(None, delete_file)

        logger.info("artifact_deleted", artifact_id=artifact_id)
        return True

    async def delete_run_artifacts(self, run_id: str) -> int:
        """Delete all artifacts for a run."""
        async with self._lock:
            artifact_ids = self._run_artifacts.pop(run_id, [])

        deleted = 0
        for artifact_id in artifact_ids:
            if await self.delete_artifact(artifact_id):
                deleted += 1

        # Also try to remove the directory
        artifact_dir = self._get_artifact_dir(run_id)

        def remove_dir():
            if artifact_dir.exists():
                import shutil

                shutil.rmtree(artifact_dir, ignore_errors=True)

        await asyncio.get_event_loop().run_in_executor(None, remove_dir)

        logger.info("run_artifacts_deleted", run_id=run_id, count=deleted)
        return deleted

    async def get_total_size(self, run_id: Optional[str] = None) -> int:
        """Get total size of artifacts."""
        async with self._lock:
            if run_id:
                artifact_ids = self._run_artifacts.get(run_id, [])
                artifacts = [
                    self._artifacts[aid]
                    for aid in artifact_ids
                    if aid in self._artifacts
                ]
            else:
                artifacts = list(self._artifacts.values())

            return sum(a.size_bytes for a in artifacts)

    async def get_stats(self) -> dict[str, Any]:
        """Get artifact store statistics."""
        async with self._lock:
            total_artifacts = len(self._artifacts)
            total_runs = len(self._run_artifacts)
            total_size = sum(a.size_bytes for a in self._artifacts.values())

            by_type: dict[str, int] = {}
            for artifact in self._artifacts.values():
                by_type[artifact.artifact_type] = (
                    by_type.get(artifact.artifact_type, 0) + 1
                )

            return {
                "total_artifacts": total_artifacts,
                "total_runs": total_runs,
                "total_size_bytes": total_size,
                "by_type": by_type,
                "deduplicated_count": len(self._checksum_index),
            }
