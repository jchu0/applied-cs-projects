"""Main gateway for model routing layer."""

from typing import Any
import logging

from ..schemas import (
    InferenceRequest,
    InferenceResponse,
    Tenant,
    Priority,
    generate_id,
)
from .rate_limiter import RateLimiter
from .token_estimator import TokenEstimator

logger = logging.getLogger(__name__)


class TenantAuthenticator:
    """Authenticates tenants from API keys."""

    def __init__(self, tenants: dict[str, Tenant] = None):
        """Initialize authenticator.

        Args:
            tenants: Tenant configurations
        """
        self._tenants = tenants or {}
        self._keys: dict[str, str] = {}

        for tenant in self._tenants.values():
            self._keys[tenant.api_key] = tenant.id

    def register_tenant(self, tenant: Tenant):
        """Register a tenant."""
        self._tenants[tenant.id] = tenant
        self._keys[tenant.api_key] = tenant.id

    async def authenticate(self, request: dict[str, Any]) -> Tenant:
        """Authenticate request and return tenant.

        Args:
            request: Raw request with API key

        Returns:
            Tenant configuration

        Raises:
            AuthenticationError: If authentication fails
        """
        api_key = request.get("api_key") or request.get("headers", {}).get("x-api-key")

        if not api_key:
            raise AuthenticationError("Missing API key")

        tenant_id = self._keys.get(api_key)
        if not tenant_id:
            raise AuthenticationError("Invalid API key")

        return self._tenants[tenant_id]


class AuthenticationError(Exception):
    """Authentication failed."""
    pass


class Gateway:
    """Main gateway for model routing layer."""

    def __init__(
        self,
        rate_limiter: RateLimiter,
        authenticator: TenantAuthenticator,
        token_estimator: TokenEstimator,
        scheduler
    ):
        """Initialize gateway.

        Args:
            rate_limiter: Rate limiter
            authenticator: Tenant authenticator
            token_estimator: Token estimator
            scheduler: Request scheduler
        """
        self.rate_limiter = rate_limiter
        self.auth = authenticator
        self.token_estimator = token_estimator
        self.scheduler = scheduler
        self._metrics: list[dict] = []

    async def handle_request(
        self,
        raw_request: dict[str, Any]
    ) -> InferenceResponse:
        """Handle incoming inference request.

        Args:
            raw_request: Raw request data

        Returns:
            Inference response
        """
        # Authenticate and get tenant config
        tenant = await self.auth.authenticate(raw_request)

        # Create typed request
        request = self._create_request(raw_request, tenant)

        # Check rate limits
        await self.rate_limiter.check(tenant.id, request)

        # Estimate token cost
        request.estimated_tokens = self.token_estimator.estimate(
            request.prompt,
            request.max_tokens,
            request.model
        )

        logger.info(
            f"Processing request {request.request_id} "
            f"for tenant {tenant.id} "
            f"(~{request.estimated_tokens} tokens)"
        )

        # Submit to scheduler
        result = await self.scheduler.submit(request)

        # Update metrics
        await self._update_metrics(request, result)

        return result

    def _create_request(
        self,
        raw: dict[str, Any],
        tenant: Tenant
    ) -> InferenceRequest:
        """Create typed request from raw data.

        Args:
            raw: Raw request data
            tenant: Tenant configuration

        Returns:
            Inference request
        """
        # Determine priority
        priority = self._compute_priority(raw, tenant)

        # Compute SLA deadline
        sla_deadline = self._compute_sla(priority, tenant)

        return InferenceRequest(
            request_id=generate_id(),
            tenant_id=tenant.id,
            model=raw["model"],
            prompt=raw["prompt"],
            max_tokens=raw.get("max_tokens", 500),
            temperature=raw.get("temperature", 0.7),
            priority=priority,
            sla_deadline_ms=sla_deadline,
            metadata=raw.get("metadata", {})
        )

    def _compute_priority(
        self,
        raw: dict[str, Any],
        tenant: Tenant
    ) -> Priority:
        """Compute request priority.

        Args:
            raw: Raw request data
            tenant: Tenant configuration

        Returns:
            Priority level
        """
        # Check explicit priority
        if "priority" in raw:
            priority_str = raw["priority"].upper()
            return Priority[priority_str]

        # Use tenant default
        return tenant.default_priority

    def _compute_sla(
        self,
        priority: Priority,
        tenant: Tenant
    ) -> int:
        """Compute SLA deadline in milliseconds.

        Args:
            priority: Request priority
            tenant: Tenant configuration

        Returns:
            SLA deadline in ms
        """
        sla_map = {
            Priority.CRITICAL: 1000,   # 1s
            Priority.HIGH: 3000,       # 3s
            Priority.NORMAL: 10000,    # 10s
            Priority.LOW: 30000,       # 30s
            Priority.BATCH: 300000     # 5min
        }

        return tenant.sla_overrides.get(
            priority.name,
            sla_map[priority]
        )

    async def _update_metrics(
        self,
        request: InferenceRequest,
        result: InferenceResponse
    ):
        """Update gateway metrics.

        Args:
            request: Original request
            result: Response
        """
        self._metrics.append({
            "request_id": request.request_id,
            "tenant_id": request.tenant_id,
            "model": request.model,
            "tokens": result.tokens_used,
            "latency_ms": result.latency_ms,
            "priority": request.priority.name
        })

    def get_metrics(self) -> list[dict]:
        """Get gateway metrics."""
        return self._metrics.copy()


def create_gateway(scheduler, tenants: dict[str, Tenant] = None) -> Gateway:
    """Create configured gateway.

    Args:
        scheduler: Request scheduler
        tenants: Tenant configurations

    Returns:
        Configured gateway
    """
    rate_limiter = RateLimiter()
    authenticator = TenantAuthenticator(tenants)
    token_estimator = TokenEstimator()

    return Gateway(
        rate_limiter=rate_limiter,
        authenticator=authenticator,
        token_estimator=token_estimator,
        scheduler=scheduler
    )
