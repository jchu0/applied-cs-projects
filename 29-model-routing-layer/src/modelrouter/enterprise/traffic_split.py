"""Traffic splitting for canary deployments."""

import time
import hashlib
from typing import Any

from ..schemas import InferenceRequest, CanaryConfig, generate_id


class TrafficSplitter:
    """Splits traffic for canary deployments."""

    def __init__(self, config_store=None):
        """Initialize traffic splitter.

        Args:
            config_store: Configuration storage
        """
        self.config_store = config_store or InMemoryCanaryStore()

    async def get_target(
        self,
        request: InferenceRequest
    ) -> str:
        """Determine which deployment to route to.

        Args:
            request: Inference request

        Returns:
            "stable" or "canary"
        """
        # Get canary config
        canary = await self.config_store.get_canary(request.model)

        if not canary or not canary.active:
            return "stable"

        # Consistent hashing for tenant
        hash_input = f"{request.tenant_id}:{canary.id}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 100

        if hash_val < canary.traffic_percentage:
            return "canary"
        else:
            return "stable"

    async def create_canary(
        self,
        model: str,
        canary_workers: list[str],
        traffic_percentage: float
    ) -> str:
        """Create a canary deployment.

        Args:
            model: Model name
            canary_workers: Worker IDs for canary
            traffic_percentage: Percentage of traffic

        Returns:
            Canary ID
        """
        canary_id = generate_id()

        canary = CanaryConfig(
            id=canary_id,
            model=model,
            workers=canary_workers,
            traffic_percentage=traffic_percentage,
            active=True
        )

        await self.config_store.set_canary(model, canary)

        return canary_id

    async def update_traffic(
        self,
        model: str,
        traffic_percentage: float
    ):
        """Update canary traffic percentage.

        Args:
            model: Model name
            traffic_percentage: New percentage
        """
        canary = await self.config_store.get_canary(model)
        if canary:
            canary.traffic_percentage = traffic_percentage
            await self.config_store.set_canary(model, canary)

    async def complete_canary(self, model: str):
        """Complete canary rollout (100% to canary).

        Args:
            model: Model name
        """
        await self.update_traffic(model, 100.0)

    async def rollback_canary(self, model: str):
        """Rollback canary (0% to canary).

        Args:
            model: Model name
        """
        canary = await self.config_store.get_canary(model)
        if canary:
            canary.active = False
            await self.config_store.set_canary(model, canary)

    async def get_canary_workers(self, model: str) -> list[str]:
        """Get canary worker IDs.

        Args:
            model: Model name

        Returns:
            List of worker IDs
        """
        canary = await self.config_store.get_canary(model)
        return canary.workers if canary else []

    async def get_canary_config(self, model: str) -> CanaryConfig | None:
        """Get canary configuration.

        Args:
            model: Model name

        Returns:
            Canary config or None
        """
        return await self.config_store.get_canary(model)


class InMemoryCanaryStore:
    """In-memory canary configuration storage."""

    def __init__(self):
        self._canaries: dict[str, CanaryConfig] = {}

    async def get_canary(self, model: str) -> CanaryConfig | None:
        """Get canary config."""
        return self._canaries.get(model)

    async def set_canary(self, model: str, canary: CanaryConfig):
        """Set canary config."""
        self._canaries[model] = canary

    async def delete_canary(self, model: str):
        """Delete canary config."""
        self._canaries.pop(model, None)
