"""Tests for gateway, authentication, and rate limiting."""

import pytest
import asyncio
import time

from modelrouter import (
    Priority,
    InferenceRequest,
    InferenceResponse,
    Tenant,
    TenantQuota,
    Gateway,
    TenantAuthenticator,
    AuthenticationError,
    RateLimiter,
    RateLimitExceeded,
    TokenEstimator,
    ModelPricing,
    generate_id,
    create_gateway,
)
from modelrouter.gateway.token_estimator import MockTokenizer

# Import factory functions from conftest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from conftest import make_request, make_tenant, make_raw_request


class TestTenantAuthenticator:
    """Tests for tenant authentication."""

    def test_authenticate_valid_api_key(self, tenant_authenticator, sample_tenant):
        """Test successful authentication with valid API key."""
        async def run():
            request = {"api_key": "test-api-key-123"}
            tenant = await tenant_authenticator.authenticate(request)
            assert tenant.id == sample_tenant.id
            assert tenant.name == sample_tenant.name

        asyncio.run(run())

    def test_authenticate_header_api_key(self, tenant_authenticator):
        """Test authentication with API key in headers."""
        async def run():
            request = {
                "headers": {"x-api-key": "test-api-key-123"}
            }
            tenant = await tenant_authenticator.authenticate(request)
            assert tenant.id == "tenant-1"

        asyncio.run(run())

    def test_authenticate_missing_api_key(self, tenant_authenticator):
        """Test authentication fails without API key."""
        async def run():
            request = {"model": "gpt-4"}
            with pytest.raises(AuthenticationError) as exc_info:
                await tenant_authenticator.authenticate(request)
            assert "Missing API key" in str(exc_info.value)

        asyncio.run(run())

    def test_authenticate_invalid_api_key(self, tenant_authenticator):
        """Test authentication fails with invalid API key."""
        async def run():
            request = {"api_key": "invalid-key"}
            with pytest.raises(AuthenticationError) as exc_info:
                await tenant_authenticator.authenticate(request)
            assert "Invalid API key" in str(exc_info.value)

        asyncio.run(run())

    def test_register_tenant(self):
        """Test registering a new tenant."""
        auth = TenantAuthenticator()
        tenant = Tenant(
            id="new-tenant",
            name="New Tenant",
            api_key="new-api-key"
        )
        auth.register_tenant(tenant)
        assert "new-api-key" in auth._keys
        assert auth._keys["new-api-key"] == "new-tenant"

    def test_register_multiple_tenants(self):
        """Test registering multiple tenants."""
        auth = TenantAuthenticator()
        for i in range(5):
            tenant = Tenant(
                id=f"tenant-{i}",
                name=f"Tenant {i}",
                api_key=f"key-{i}"
            )
            auth.register_tenant(tenant)
        assert len(auth._tenants) == 5
        assert len(auth._keys) == 5


class TestRateLimiter:
    """Tests for rate limiting functionality."""

    def test_rate_limit_allows_within_limit(self, rate_limiter):
        """Test requests within rate limit are allowed."""
        async def run():
            rate_limiter.set_limits("tenant-1", rps_limit=10, tpm_limit=10000)
            request = make_request(estimated_tokens=60)
            result = await rate_limiter.check("tenant-1", request)
            assert result is True

        asyncio.run(run())

    def test_rate_limit_rps_exceeded(self, rate_limiter):
        """Test RPS limit is enforced."""
        async def run():
            rate_limiter.set_limits("tenant-1", rps_limit=5, tpm_limit=100000)
            request = make_request(estimated_tokens=60)

            # Make requests up to limit
            for i in range(5):
                await rate_limiter.check("tenant-1", request)

            # Next request should fail
            with pytest.raises(RateLimitExceeded) as exc_info:
                await rate_limiter.check("tenant-1", request)
            assert "Requests per second limit exceeded" in str(exc_info.value)

        asyncio.run(run())

    def test_rate_limit_tpm_exceeded(self, rate_limiter):
        """Test tokens per minute limit is enforced."""
        async def run():
            rate_limiter.set_limits("tenant-1", rps_limit=100, tpm_limit=500)
            request = make_request(estimated_tokens=200)

            # Make 2 requests (400 tokens)
            await rate_limiter.check("tenant-1", request)
            await rate_limiter.check("tenant-1", request)

            # Third request would exceed 500 TPM
            with pytest.raises(RateLimitExceeded) as exc_info:
                await rate_limiter.check("tenant-1", request)
            assert "Tokens per minute limit exceeded" in str(exc_info.value)

        asyncio.run(run())

    def test_rate_limit_default_limits(self, rate_limiter):
        """Test default limits are applied for unknown tenants."""
        async def run():
            request = make_request(tenant_id="unknown-tenant", estimated_tokens=60)
            # Should use default limits (100 RPS, 100000 TPM)
            result = await rate_limiter.check("unknown-tenant", request)
            assert result is True

        asyncio.run(run())

    def test_rate_limit_separate_tenants(self, rate_limiter):
        """Test rate limits are tracked separately per tenant."""
        async def run():
            rate_limiter.set_limits("tenant-1", rps_limit=2, tpm_limit=100000)
            rate_limiter.set_limits("tenant-2", rps_limit=2, tpm_limit=100000)

            request1 = make_request(tenant_id="tenant-1", estimated_tokens=60)
            request2 = make_request(tenant_id="tenant-2", estimated_tokens=60)

            # Exhaust tenant-1's limit
            await rate_limiter.check("tenant-1", request1)
            await rate_limiter.check("tenant-1", request1)

            # Tenant-1 should be rate limited
            with pytest.raises(RateLimitExceeded):
                await rate_limiter.check("tenant-1", request1)

            # Tenant-2 should still work
            result = await rate_limiter.check("tenant-2", request2)
            assert result is True

        asyncio.run(run())


class TestTokenEstimator:
    """Tests for token estimation."""

    def test_estimate_without_tokenizer(self, token_estimator):
        """Test token estimation without a tokenizer uses character count."""
        # ~4 chars per token
        prompt = "x" * 400  # Should be ~100 tokens
        estimate = token_estimator.estimate(prompt, max_tokens=50)
        # 100 input + 50 output
        assert estimate == 150

    def test_estimate_with_tokenizer(self):
        """Test token estimation with a custom tokenizer."""
        tokenizer = MockTokenizer(chars_per_token=5)
        estimator = TokenEstimator(tokenizers={"gpt-4": tokenizer})
        prompt = "x" * 500  # 100 tokens at 5 chars/token
        estimate = estimator.estimate(prompt, max_tokens=50, model="gpt-4")
        # 100 input + 50 output
        assert estimate == 150

    def test_estimate_input_tokens_only(self, token_estimator):
        """Test estimating input tokens only."""
        prompt = "x" * 200  # ~50 tokens
        input_tokens = token_estimator.estimate_input_tokens(prompt)
        assert input_tokens == 50

    def test_estimate_cost(self, token_estimator, sample_request, sample_model_pricing):
        """Test cost estimation in dollars."""
        cost = token_estimator.estimate_cost(sample_request, sample_model_pricing)
        # input: (prompt_tokens * 0.03/1000) + output: (100 * 0.06/1000)
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_empty_prompt(self, token_estimator):
        """Test estimation with empty prompt."""
        estimate = token_estimator.estimate("", max_tokens=100)
        # 0 input + 100 output
        assert estimate == 100

    def test_estimate_fallback_tokenizer(self):
        """Test fallback to default tokenizer."""
        default_tokenizer = MockTokenizer(chars_per_token=4)
        estimator = TokenEstimator(tokenizers={"default": default_tokenizer})
        prompt = "x" * 400  # 100 tokens
        # Request for unknown model should use default
        estimate = estimator.estimate(prompt, max_tokens=50, model="unknown-model")
        assert estimate == 150


class TestGateway:
    """Tests for the main gateway."""

    def _make_mock_scheduler(self):
        """Create a mock scheduler."""
        class MockScheduler:
            def __init__(self):
                self.submitted = []

            async def submit(self, request):
                self.submitted.append(request)
                return InferenceResponse(
                    request_id=request.request_id,
                    text="Mock response",
                    tokens_used=request.estimated_tokens,
                    latency_ms=100,
                    worker_id="mock-worker",
                    model=request.model
                )

        return MockScheduler()

    def test_gateway_handles_request(
        self,
        rate_limiter,
        tenant_authenticator,
        token_estimator,
        raw_request
    ):
        """Test gateway handles a complete request flow."""
        async def run():
            mock_scheduler = self._make_mock_scheduler()
            gateway = Gateway(
                rate_limiter=rate_limiter,
                authenticator=tenant_authenticator,
                token_estimator=token_estimator,
                scheduler=mock_scheduler
            )
            response = await gateway.handle_request(raw_request)
            assert response is not None
            assert response.model == "gpt-4"
            assert len(mock_scheduler.submitted) == 1

        asyncio.run(run())

    def test_gateway_computes_priority(
        self,
        rate_limiter,
        tenant_authenticator,
        token_estimator
    ):
        """Test gateway computes priority from request."""
        async def run():
            mock_scheduler = self._make_mock_scheduler()
            gateway = Gateway(
                rate_limiter=rate_limiter,
                authenticator=tenant_authenticator,
                token_estimator=token_estimator,
                scheduler=mock_scheduler
            )
            # Request with explicit priority
            request = {
                "api_key": "test-api-key-123",
                "model": "gpt-4",
                "prompt": "Test",
                "priority": "high"
            }
            await gateway.handle_request(request)
            submitted = mock_scheduler.submitted[-1]
            assert submitted.priority == Priority.HIGH

        asyncio.run(run())

    def test_gateway_computes_sla(
        self,
        rate_limiter,
        tenant_authenticator,
        token_estimator
    ):
        """Test gateway computes SLA deadline."""
        async def run():
            mock_scheduler = self._make_mock_scheduler()
            gateway = Gateway(
                rate_limiter=rate_limiter,
                authenticator=tenant_authenticator,
                token_estimator=token_estimator,
                scheduler=mock_scheduler
            )
            request = {
                "api_key": "test-api-key-123",
                "model": "gpt-4",
                "prompt": "Test",
                "priority": "critical"
            }
            await gateway.handle_request(request)
            submitted = mock_scheduler.submitted[-1]
            # CRITICAL should have 1000ms SLA
            assert submitted.sla_deadline_ms == 1000

        asyncio.run(run())

    def test_gateway_estimates_tokens(
        self,
        rate_limiter,
        tenant_authenticator,
        token_estimator
    ):
        """Test gateway estimates token count."""
        async def run():
            mock_scheduler = self._make_mock_scheduler()
            gateway = Gateway(
                rate_limiter=rate_limiter,
                authenticator=tenant_authenticator,
                token_estimator=token_estimator,
                scheduler=mock_scheduler
            )
            request = {
                "api_key": "test-api-key-123",
                "model": "gpt-4",
                "prompt": "x" * 400,  # ~100 input tokens
                "max_tokens": 200
            }
            await gateway.handle_request(request)
            submitted = mock_scheduler.submitted[-1]
            assert submitted.estimated_tokens == 300  # 100 input + 200 output

        asyncio.run(run())

    def test_gateway_auth_failure(
        self,
        rate_limiter,
        tenant_authenticator,
        token_estimator
    ):
        """Test gateway handles authentication failure."""
        async def run():
            mock_scheduler = self._make_mock_scheduler()
            gateway = Gateway(
                rate_limiter=rate_limiter,
                authenticator=tenant_authenticator,
                token_estimator=token_estimator,
                scheduler=mock_scheduler
            )
            request = {
                "api_key": "invalid-key",
                "model": "gpt-4",
                "prompt": "Test"
            }
            with pytest.raises(AuthenticationError):
                await gateway.handle_request(request)

        asyncio.run(run())

    def test_gateway_rate_limit_failure(
        self,
        tenant_authenticator,
        token_estimator
    ):
        """Test gateway handles rate limit exceeded."""
        async def run():
            rate_limiter = RateLimiter()
            rate_limiter.set_limits("tenant-1", rps_limit=1, tpm_limit=100000)
            mock_scheduler = self._make_mock_scheduler()
            gateway = Gateway(
                rate_limiter=rate_limiter,
                authenticator=tenant_authenticator,
                token_estimator=token_estimator,
                scheduler=mock_scheduler
            )
            request = {
                "api_key": "test-api-key-123",
                "model": "gpt-4",
                "prompt": "Test"
            }
            # First request succeeds
            await gateway.handle_request(request)
            # Second request should be rate limited
            with pytest.raises(RateLimitExceeded):
                await gateway.handle_request(request)

        asyncio.run(run())

    def test_gateway_metrics_collection(
        self,
        rate_limiter,
        tenant_authenticator,
        token_estimator,
        raw_request
    ):
        """Test gateway collects metrics."""
        async def run():
            mock_scheduler = self._make_mock_scheduler()
            gateway = Gateway(
                rate_limiter=rate_limiter,
                authenticator=tenant_authenticator,
                token_estimator=token_estimator,
                scheduler=mock_scheduler
            )
            await gateway.handle_request(raw_request)
            await gateway.handle_request(raw_request)
            metrics = gateway.get_metrics()
            assert len(metrics) == 2
            assert all("request_id" in m for m in metrics)
            assert all("latency_ms" in m for m in metrics)

        asyncio.run(run())

    def test_create_gateway_factory(self):
        """Test gateway factory function."""
        class MockScheduler:
            async def submit(self, request):
                pass

        tenants = {
            "tenant-1": Tenant(
                id="tenant-1",
                name="Test",
                api_key="key-1"
            )
        }
        gateway = create_gateway(MockScheduler(), tenants)
        assert gateway is not None
        assert gateway.rate_limiter is not None
        assert gateway.auth is not None


class TestSLAComputation:
    """Tests for SLA deadline computation."""

    def _make_gateway_with_scheduler(
        self,
        rate_limiter,
        tenant_authenticator,
        token_estimator
    ):
        """Create gateway with mock scheduler."""
        class MockScheduler:
            def __init__(self):
                self.submitted = []

            async def submit(self, request):
                self.submitted.append(request)
                return InferenceResponse(
                    request_id=request.request_id,
                    text="Mock",
                    tokens_used=100,
                    latency_ms=50,
                    worker_id="mock",
                    model=request.model
                )

        scheduler = MockScheduler()
        gateway = Gateway(
            rate_limiter=rate_limiter,
            authenticator=tenant_authenticator,
            token_estimator=token_estimator,
            scheduler=scheduler
        )
        return gateway, scheduler

    def test_sla_critical(self, rate_limiter, tenant_authenticator, token_estimator):
        """Test SLA for CRITICAL priority."""
        async def run():
            gateway, scheduler = self._make_gateway_with_scheduler(
                rate_limiter, tenant_authenticator, token_estimator
            )
            request = {
                "api_key": "test-api-key-123",
                "model": "gpt-4",
                "prompt": "Test",
                "priority": "critical"
            }
            await gateway.handle_request(request)
            assert scheduler.submitted[-1].sla_deadline_ms == 1000

        asyncio.run(run())

    def test_sla_high(self, rate_limiter, tenant_authenticator, token_estimator):
        """Test SLA for HIGH priority."""
        async def run():
            gateway, scheduler = self._make_gateway_with_scheduler(
                rate_limiter, tenant_authenticator, token_estimator
            )
            request = {
                "api_key": "test-api-key-123",
                "model": "gpt-4",
                "prompt": "Test",
                "priority": "high"
            }
            await gateway.handle_request(request)
            assert scheduler.submitted[-1].sla_deadline_ms == 3000

        asyncio.run(run())

    def test_sla_normal(self, rate_limiter, tenant_authenticator, token_estimator):
        """Test SLA for NORMAL priority."""
        async def run():
            gateway, scheduler = self._make_gateway_with_scheduler(
                rate_limiter, tenant_authenticator, token_estimator
            )
            request = {
                "api_key": "test-api-key-123",
                "model": "gpt-4",
                "prompt": "Test",
                "priority": "normal"
            }
            await gateway.handle_request(request)
            assert scheduler.submitted[-1].sla_deadline_ms == 10000

        asyncio.run(run())

    def test_sla_low(self, rate_limiter, tenant_authenticator, token_estimator):
        """Test SLA for LOW priority."""
        async def run():
            gateway, scheduler = self._make_gateway_with_scheduler(
                rate_limiter, tenant_authenticator, token_estimator
            )
            request = {
                "api_key": "test-api-key-123",
                "model": "gpt-4",
                "prompt": "Test",
                "priority": "low"
            }
            await gateway.handle_request(request)
            assert scheduler.submitted[-1].sla_deadline_ms == 30000

        asyncio.run(run())

    def test_sla_batch(self, rate_limiter, tenant_authenticator, token_estimator):
        """Test SLA for BATCH priority."""
        async def run():
            gateway, scheduler = self._make_gateway_with_scheduler(
                rate_limiter, tenant_authenticator, token_estimator
            )
            request = {
                "api_key": "test-api-key-123",
                "model": "gpt-4",
                "prompt": "Test",
                "priority": "batch"
            }
            await gateway.handle_request(request)
            assert scheduler.submitted[-1].sla_deadline_ms == 300000

        asyncio.run(run())

    def test_sla_tenant_override(self, rate_limiter, token_estimator):
        """Test SLA override from tenant configuration."""
        async def run():
            # Create tenant with SLA override
            tenant = Tenant(
                id="tenant-custom",
                name="Custom SLA Tenant",
                api_key="custom-key",
                sla_overrides={"CRITICAL": 500, "HIGH": 1500}
            )

            auth = TenantAuthenticator()
            auth.register_tenant(tenant)

            class MockScheduler:
                def __init__(self):
                    self.submitted = []

                async def submit(self, request):
                    self.submitted.append(request)
                    return InferenceResponse(
                        request_id=request.request_id,
                        text="Mock",
                        tokens_used=100,
                        latency_ms=50,
                        worker_id="mock",
                        model=request.model
                    )

            scheduler = MockScheduler()
            gateway = Gateway(
                rate_limiter=rate_limiter,
                authenticator=auth,
                token_estimator=token_estimator,
                scheduler=scheduler
            )

            request = {
                "api_key": "custom-key",
                "model": "gpt-4",
                "prompt": "Test",
                "priority": "critical"
            }
            await gateway.handle_request(request)
            # Should use tenant's custom SLA
            assert scheduler.submitted[-1].sla_deadline_ms == 500

        asyncio.run(run())
