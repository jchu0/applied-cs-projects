"""Tenant quota management."""

import time
from typing import Any

from ..schemas import TenantQuota, InferenceRequest, UsageReport


class QuotaExceeded(Exception):
    """Quota exceeded exception."""
    pass


class QuotaManager:
    """Manages per-tenant quotas."""

    def __init__(self, storage=None):
        """Initialize quota manager.

        Args:
            storage: Storage backend
        """
        self.storage = storage or InMemoryQuotaStorage()

    async def set_quota(self, quota: TenantQuota):
        """Set quota for tenant.

        Args:
            quota: Tenant quota
        """
        await self.storage.set_quota(quota.tenant_id, quota)

    async def check_quota(
        self,
        tenant_id: str,
        request: InferenceRequest
    ) -> bool:
        """Check if request is within quota.

        Args:
            tenant_id: Tenant ID
            request: Inference request

        Returns:
            True if allowed

        Raises:
            QuotaExceeded: If quota exceeded
        """
        quota = await self._get_quota(tenant_id)
        usage = await self._get_usage(tenant_id)

        # Check monthly budget
        if usage.tokens_used + request.estimated_tokens > quota.monthly_token_budget:
            raise QuotaExceeded("Monthly token budget exceeded")

        return True

    async def record_usage(
        self,
        tenant_id: str,
        tokens_used: int
    ):
        """Record token usage for tenant.

        Args:
            tenant_id: Tenant ID
            tokens_used: Tokens used
        """
        await self.storage.incr_usage(tenant_id, tokens_used)

    async def get_usage_report(
        self,
        tenant_id: str
    ) -> UsageReport:
        """Get usage report for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Usage report
        """
        usage = await self._get_usage(tenant_id)
        quota = await self._get_quota(tenant_id)

        remaining = quota.monthly_token_budget - usage.tokens_used

        return UsageReport(
            tenant_id=tenant_id,
            tokens_used=usage.tokens_used,
            tokens_remaining=remaining,
            utilization=usage.tokens_used / quota.monthly_token_budget if quota.monthly_token_budget > 0 else 0
        )

    async def reset_usage(self, tenant_id: str):
        """Reset usage for tenant (monthly reset).

        Args:
            tenant_id: Tenant ID
        """
        await self.storage.reset_usage(tenant_id)

    async def _get_quota(self, tenant_id: str) -> TenantQuota:
        """Get quota for tenant."""
        quota = await self.storage.get_quota(tenant_id)
        if not quota:
            # Default quota
            quota = TenantQuota(
                tenant_id=tenant_id,
                requests_per_second=100,
                tokens_per_minute=100000,
                monthly_token_budget=10000000
            )
        return quota

    async def _get_usage(self, tenant_id: str) -> UsageReport:
        """Get usage for tenant."""
        return await self.storage.get_usage(tenant_id)


class InMemoryQuotaStorage:
    """In-memory storage for quotas."""

    def __init__(self):
        self._quotas: dict[str, TenantQuota] = {}
        self._usage: dict[str, int] = {}

    async def set_quota(self, tenant_id: str, quota: TenantQuota):
        """Set quota."""
        self._quotas[tenant_id] = quota

    async def get_quota(self, tenant_id: str) -> TenantQuota | None:
        """Get quota."""
        return self._quotas.get(tenant_id)

    async def incr_usage(self, tenant_id: str, tokens: int):
        """Increment usage."""
        self._usage[tenant_id] = self._usage.get(tenant_id, 0) + tokens

    async def get_usage(self, tenant_id: str) -> UsageReport:
        """Get usage."""
        tokens = self._usage.get(tenant_id, 0)
        return UsageReport(
            tenant_id=tenant_id,
            tokens_used=tokens,
            tokens_remaining=0,  # Calculated by manager
            utilization=0
        )

    async def reset_usage(self, tenant_id: str):
        """Reset usage."""
        self._usage[tenant_id] = 0
