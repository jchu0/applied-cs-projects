"""Rate limiting for model routing."""

import time
from typing import Any

from ..schemas import InferenceRequest, generate_id


class RateLimitExceeded(Exception):
    """Rate limit exceeded exception."""
    pass


class RateLimiter:
    """Token bucket rate limiter with per-tenant quotas."""

    def __init__(self, storage=None):
        """Initialize rate limiter.

        Args:
            storage: Storage backend (default: in-memory)
        """
        self.storage = storage or InMemoryStorage()
        self._tenant_limits: dict[str, dict[str, int]] = {}

    def set_limits(
        self,
        tenant_id: str,
        rps_limit: int,
        tpm_limit: int
    ):
        """Set rate limits for tenant.

        Args:
            tenant_id: Tenant identifier
            rps_limit: Requests per second limit
            tpm_limit: Tokens per minute limit
        """
        self._tenant_limits[tenant_id] = {
            "rps": rps_limit,
            "tpm": tpm_limit
        }

    async def check(
        self,
        tenant_id: str,
        request: InferenceRequest
    ) -> bool:
        """Check if request is within rate limits.

        Args:
            tenant_id: Tenant identifier
            request: Inference request

        Returns:
            True if allowed

        Raises:
            RateLimitExceeded: If limit exceeded
        """
        # Check requests per second
        rps_key = f"ratelimit:{tenant_id}:rps"
        rps_allowed = await self._check_bucket(
            rps_key,
            limit=self._get_rps_limit(tenant_id),
            window=1
        )

        if not rps_allowed:
            raise RateLimitExceeded("Requests per second limit exceeded")

        # Check tokens per minute
        tpm_key = f"ratelimit:{tenant_id}:tpm"
        tpm_allowed = await self._check_bucket(
            tpm_key,
            limit=self._get_tpm_limit(tenant_id),
            window=60,
            cost=request.estimated_tokens or 1
        )

        if not tpm_allowed:
            raise RateLimitExceeded("Tokens per minute limit exceeded")

        return True

    async def _check_bucket(
        self,
        key: str,
        limit: int,
        window: int,
        cost: int = 1
    ) -> bool:
        """Check and consume from token bucket.

        Args:
            key: Bucket key
            limit: Maximum allowed in window
            window: Time window in seconds
            cost: Cost of this request

        Returns:
            True if allowed
        """
        now = time.time()
        window_start = now - window

        # Get current entries
        entries = await self.storage.get_range(key, window_start, now)

        if len(entries) + cost > limit:
            return False

        # Add new entries
        for _ in range(cost):
            await self.storage.add_entry(key, now, f"{now}:{generate_id()}")

        # Set expiry
        await self.storage.set_expiry(key, window * 2)

        return True

    def _get_rps_limit(self, tenant_id: str) -> int:
        """Get RPS limit for tenant."""
        if tenant_id in self._tenant_limits:
            return self._tenant_limits[tenant_id].get("rps", 100)
        return 100  # Default

    def _get_tpm_limit(self, tenant_id: str) -> int:
        """Get TPM limit for tenant."""
        if tenant_id in self._tenant_limits:
            return self._tenant_limits[tenant_id].get("tpm", 100000)
        return 100000  # Default


class InMemoryStorage:
    """In-memory storage for rate limiting."""

    def __init__(self):
        self._data: dict[str, list[tuple[float, str]]] = {}
        self._expiry: dict[str, float] = {}

    async def get_range(
        self,
        key: str,
        start: float,
        end: float
    ) -> list:
        """Get entries in time range."""
        if key not in self._data:
            return []

        # Clean expired
        self._clean_expired(key)

        return [
            (ts, val) for ts, val in self._data.get(key, [])
            if start <= ts <= end
        ]

    async def add_entry(self, key: str, timestamp: float, value: str):
        """Add entry to bucket."""
        if key not in self._data:
            self._data[key] = []
        self._data[key].append((timestamp, value))

    async def set_expiry(self, key: str, ttl: int):
        """Set expiry for key."""
        self._expiry[key] = time.time() + ttl

    def _clean_expired(self, key: str):
        """Clean expired entries."""
        if key in self._expiry and time.time() > self._expiry[key]:
            self._data.pop(key, None)
            self._expiry.pop(key, None)
