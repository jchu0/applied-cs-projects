"""Tests for routing logic and model selection."""

import pytest
import asyncio
import time

from modelrouter import (
    Priority,
    InferenceRequest,
    WorkerInfo,
    GPUInfo,
    ModelRouter,
    create_router,
    RoutingEngine,
    WorkerRegistry,
    CostComputer,
    NoWorkersAvailable,
    LocalityAwareStrategy,
    generate_id,
)

# Import factory functions from conftest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from conftest import make_gpu_info, make_worker, make_workers, make_request


class TestRoutingEngine:
    """Tests for RoutingEngine routing strategies."""

    def test_least_loaded_strategy(self, worker_registry, cost_computer, sample_workers):
        """Test least loaded routing strategy selects worker with lowest load."""
        async def run():
            # Register workers with different loads
            for worker in sample_workers:
                await worker_registry.register(worker)

            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="least_loaded"
            )

            request = make_request()
            selected = await engine.route(request)

            # Worker-1 has lowest load (0.2)
            assert selected.worker_id == "worker-1"
            assert selected.current_load == 0.2

        asyncio.run(run())

    def test_round_robin_strategy(self, worker_registry, cost_computer, sample_workers):
        """Test round robin strategy cycles through workers."""
        async def run():
            for worker in sample_workers:
                await worker_registry.register(worker)

            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="round_robin"
            )

            request = make_request()

            # Route multiple requests and check cycling
            selected_ids = []
            for _ in range(6):  # 2 full cycles
                selected = await engine.route(request)
                selected_ids.append(selected.worker_id)

            # Should cycle through workers (order depends on registration)
            # Check that we see all workers and they repeat in the same pattern
            first_cycle = selected_ids[:3]
            second_cycle = selected_ids[3:]
            assert first_cycle == second_cycle
            assert set(first_cycle) == {"worker-1", "worker-2", "worker-3"}

        asyncio.run(run())

    def test_token_based_strategy(self, worker_registry, cost_computer, sample_gpu_info):
        """Test token-based routing selects worker with most available capacity."""
        async def run():
            # Create workers with different token capacities
            workers = [
                WorkerInfo(
                    worker_id="worker-low",
                    host="localhost",
                    port=8080,
                    models=["gpt-4"],
                    gpu_info=sample_gpu_info,
                    current_load=0.3,
                    queue_depth=5,
                    tokens_in_flight=90000,  # Low availability
                    token_budget=100000,
                    status="healthy",
                    last_heartbeat=time.time(),
                ),
                WorkerInfo(
                    worker_id="worker-high",
                    host="localhost",
                    port=8081,
                    models=["gpt-4"],
                    gpu_info=sample_gpu_info,
                    current_load=0.5,
                    queue_depth=10,
                    tokens_in_flight=10000,  # High availability
                    token_budget=100000,
                    status="healthy",
                    last_heartbeat=time.time(),
                ),
            ]

            for worker in workers:
                await worker_registry.register(worker)

            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="token_based"
            )

            request = make_request()
            selected = await engine.route(request)

            # Should select worker with highest available tokens
            assert selected.worker_id == "worker-high"

        asyncio.run(run())

    def test_sla_based_strategy(self, worker_registry, cost_computer, sample_gpu_info):
        """Test SLA-based routing considers latency predictions."""
        async def run():
            # Create workers with different performance factors
            workers = [
                WorkerInfo(
                    worker_id="worker-fast",
                    host="localhost",
                    port=8080,
                    models=["gpt-4"],
                    gpu_info=sample_gpu_info,
                    current_load=0.5,
                    queue_depth=0,
                    tokens_in_flight=0,
                    token_budget=100000,
                    status="healthy",
                    last_heartbeat=time.time(),
                    performance_factor=2.0,  # Fast worker
                    worker_type="gpu-optimized",
                ),
                WorkerInfo(
                    worker_id="worker-slow",
                    host="localhost",
                    port=8081,
                    models=["gpt-4"],
                    gpu_info=sample_gpu_info,
                    current_load=0.1,  # Lower load but slower
                    queue_depth=0,
                    tokens_in_flight=0,
                    token_budget=100000,
                    status="healthy",
                    last_heartbeat=time.time(),
                    performance_factor=0.5,  # Slow worker
                    worker_type="standard",
                ),
            ]

            for worker in workers:
                await worker_registry.register(worker)

            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="sla_based"
            )

            # Create request with tight SLA
            request = InferenceRequest(
                request_id=generate_id(),
                tenant_id="tenant-1",
                model="gpt-4",
                prompt="Test prompt",
                max_tokens=100,
                temperature=0.7,
                priority=Priority.HIGH,
                sla_deadline_ms=2000,  # Tight SLA
                estimated_tokens=125
            )

            selected = await engine.route(request)

            # Should select a worker
            assert selected is not None

        asyncio.run(run())

    def test_weighted_strategy(self, worker_registry, cost_computer, sample_workers):
        """Test weighted routing considers multiple factors."""
        async def run():
            for worker in sample_workers:
                await worker_registry.register(worker)

            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="weighted"
            )

            request = InferenceRequest(
                request_id=generate_id(),
                tenant_id="tenant-1",
                model="gpt-4",
                prompt="Test prompt",
                max_tokens=100,
                temperature=0.7,
                priority=Priority.NORMAL,
                sla_deadline_ms=10000,
                estimated_tokens=125
            )

            selected = await engine.route(request)

            # Should select based on weighted combination of factors
            assert selected is not None
            assert selected.worker_id in ["worker-1", "worker-2", "worker-3"]

        asyncio.run(run())

    def test_no_workers_available(self, worker_registry, cost_computer):
        """Test NoWorkersAvailable exception when no workers for model."""
        async def run():
            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="least_loaded"
            )

            request = InferenceRequest(
                request_id=generate_id(),
                tenant_id="tenant-1",
                model="nonexistent-model",
                prompt="Test prompt",
                max_tokens=100,
                temperature=0.7,
                priority=Priority.NORMAL,
                estimated_tokens=125
            )

            with pytest.raises(NoWorkersAvailable) as exc_info:
                await engine.route(request)

            assert "nonexistent-model" in str(exc_info.value)

        asyncio.run(run())

    def test_set_strategy(self, worker_registry, cost_computer, sample_workers):
        """Test dynamic strategy switching."""
        async def run():
            for worker in sample_workers:
                await worker_registry.register(worker)

            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="least_loaded"
            )

            assert engine.strategy == "least_loaded"

            engine.set_strategy("round_robin")
            assert engine.strategy == "round_robin"

            engine.set_strategy("token_based")
            assert engine.strategy == "token_based"

        asyncio.run(run())

    def test_unknown_strategy_raises(self, worker_registry, cost_computer, sample_workers):
        """Test that unknown strategy raises ValueError."""
        async def run():
            for worker in sample_workers:
                await worker_registry.register(worker)

            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="unknown_strategy"
            )

            request = make_request()

            with pytest.raises(ValueError) as exc_info:
                await engine.route(request)

            assert "Unknown strategy" in str(exc_info.value)

        asyncio.run(run())


class TestLocalityAwareStrategy:
    """Tests for locality-aware routing strategy."""

    def test_select_worker_with_cached_context(self, sample_workers):
        """Test that worker with cached context is preferred."""
        strategy = LocalityAwareStrategy()

        # Add context to first worker
        strategy.add_context("worker-1", "conversation-123")

        # Create request with context
        request = InferenceRequest(
            request_id=generate_id(),
            tenant_id="tenant-1",
            model="gpt-4",
            prompt="Continue our conversation",
            max_tokens=100,
            temperature=0.7,
            priority=Priority.NORMAL,
            metadata={"context_key": "conversation-123"}
        )

        selected = strategy.select(request, sample_workers)

        # Should select worker with cached context
        assert selected.worker_id == "worker-1"

    def test_fallback_to_least_loaded_without_context(self, sample_workers):
        """Test fallback to least loaded when no context match."""
        strategy = LocalityAwareStrategy()

        request = InferenceRequest(
            request_id=generate_id(),
            tenant_id="tenant-1",
            model="gpt-4",
            prompt="New conversation",
            max_tokens=100,
            temperature=0.7,
            priority=Priority.NORMAL,
            metadata={"context_key": "new-context"}
        )

        selected = strategy.select(request, sample_workers)

        # Should select least loaded (worker-1 with 0.2 load)
        assert selected.worker_id == "worker-1"

    def test_fallback_without_context_key(self, sample_workers):
        """Test fallback when no context key in request."""
        strategy = LocalityAwareStrategy()
        strategy.add_context("worker-1", "some-context")

        request = InferenceRequest(
            request_id=generate_id(),
            tenant_id="tenant-1",
            model="gpt-4",
            prompt="Request without context",
            max_tokens=100,
            temperature=0.7,
            priority=Priority.NORMAL,
            metadata={}
        )

        selected = strategy.select(request, sample_workers)

        # Should select least loaded
        assert selected.worker_id == "worker-1"

    def test_add_multiple_contexts(self, sample_workers):
        """Test adding multiple contexts to same worker."""
        strategy = LocalityAwareStrategy()

        strategy.add_context("worker-1", "context-a")
        strategy.add_context("worker-1", "context-b")
        strategy.add_context("worker-2", "context-c")

        assert "context-a" in strategy.cache["worker-1"]
        assert "context-b" in strategy.cache["worker-1"]
        assert "context-c" in strategy.cache["worker-2"]


class TestModelRouter:
    """Tests for the main ModelRouter class."""

    def test_router_creation(self):
        """Test creating a model router."""
        router = create_router(
            routing_strategy="least_loaded",
            max_queue_depth=500
        )

        assert router is not None
        assert router.max_queue_depth == 500

    def test_register_worker(self):
        """Test worker registration."""
        async def run():
            router = create_router()

            worker_id = await router.register_worker(
                worker_id="worker-test",
                host="localhost",
                port=8080,
                models=["gpt-4", "gpt-3.5"],
                token_budget=50000
            )

            assert worker_id == "worker-test"

            # Verify worker is registered
            workers = await router.registry.get_workers(model="gpt-4")
            assert len(workers) == 1
            assert workers[0].worker_id == "worker-test"

        asyncio.run(run())

    def test_register_tenant(self):
        """Test tenant registration."""
        async def run():
            router = create_router()

            router.register_tenant(
                tenant_id="test-tenant",
                name="Test Tenant",
                api_key="api-key-123",
                monthly_budget=5000000
            )

            # Verify tenant can authenticate
            tenant = await router.authenticator.authenticate({"api_key": "api-key-123"})
            assert tenant.id == "test-tenant"

        asyncio.run(run())

    def test_submit_request(self):
        """Test submitting a request through the router."""
        async def run():
            router = create_router(routing_strategy="least_loaded")

            await router.register_worker(
                worker_id="worker-1",
                host="localhost",
                port=8080,
                models=["gpt-4", "gpt-3.5-turbo"],
                token_budget=100000
            )

            router.register_tenant(
                tenant_id="tenant-1",
                name="Test Tenant",
                api_key="test-api-key-123"
            )

            request = InferenceRequest(
                request_id=generate_id(),
                tenant_id="tenant-1",
                model="gpt-4",
                prompt="What is 2 + 2?",
                max_tokens=50,
                temperature=0.7,
                priority=Priority.NORMAL,
                estimated_tokens=65
            )

            response = await router.submit(request)

            assert response is not None
            assert response.request_id == request.request_id
            assert response.tokens_used == request.estimated_tokens
            assert response.worker_id == "worker-1"

        asyncio.run(run())

    def test_handle_request_through_gateway(self):
        """Test handling request through the gateway."""
        async def run():
            router = create_router(routing_strategy="least_loaded")

            await router.register_worker(
                worker_id="worker-1",
                host="localhost",
                port=8080,
                models=["gpt-4", "gpt-3.5-turbo"],
                token_budget=100000
            )

            router.register_tenant(
                tenant_id="tenant-1",
                name="Test Tenant",
                api_key="test-api-key-123"
            )

            raw_request = {
                "api_key": "test-api-key-123",
                "model": "gpt-4",
                "prompt": "What is the capital of France?",
                "max_tokens": 100,
                "temperature": 0.7
            }

            response = await router.handle_request(raw_request)

            assert response is not None
            assert response.model == "gpt-4"
            assert response.worker_id == "worker-1"

        asyncio.run(run())

    def test_set_routing_strategy(self):
        """Test changing routing strategy."""
        router = create_router()

        router.set_routing_strategy("round_robin")
        assert router.routing_engine.strategy == "round_robin"

        router.set_routing_strategy("token_based")
        assert router.routing_engine.strategy == "token_based"

    def test_get_metrics(self):
        """Test metrics collection."""
        async def run():
            router = create_router()

            await router.register_worker(
                worker_id="worker-1",
                host="localhost",
                port=8080,
                models=["gpt-4"],
                token_budget=100000
            )

            router.register_tenant(
                tenant_id="tenant-1",
                name="Test Tenant",
                api_key="test-key"
            )

            request = InferenceRequest(
                request_id=generate_id(),
                tenant_id="tenant-1",
                model="gpt-4",
                prompt="Test prompt",
                max_tokens=50,
                temperature=0.7,
                priority=Priority.NORMAL,
                estimated_tokens=65
            )

            await router.submit(request)

            metrics = router.get_metrics()
            assert len(metrics) > 0
            assert metrics[-1]["request_id"] == request.request_id
            assert "latency_ms" in metrics[-1]

        asyncio.run(run())

    def test_get_capacity(self):
        """Test getting capacity information."""
        async def run():
            router = create_router()

            await router.register_worker(
                worker_id="worker-1",
                host="localhost",
                port=8080,
                models=["gpt-4"],
                token_budget=100000
            )

            capacity = await router.get_capacity("gpt-4")

            assert capacity is not None
            assert capacity.total_workers == 1
            assert capacity.healthy_workers == 1

        asyncio.run(run())

    def test_get_queue_stats(self):
        """Test getting queue statistics."""
        async def run():
            router = create_router()

            request = make_request()
            await router.queue_manager.enqueue(request)

            stats = await router.get_queue_stats()

            assert "gpt-4" in stats
            assert stats["gpt-4"]["depth"] >= 0

        asyncio.run(run())


class TestFailoverScenarios:
    """Tests for failover and resilience scenarios."""

    def test_failover_to_healthy_worker(self, worker_registry, cost_computer, sample_gpu_info):
        """Test failover when primary worker is unhealthy."""
        async def run():
            # Create two workers, one unhealthy
            workers = [
                WorkerInfo(
                    worker_id="worker-unhealthy",
                    host="localhost",
                    port=8080,
                    models=["gpt-4"],
                    gpu_info=sample_gpu_info,
                    current_load=0.1,  # Would be preferred by least_loaded
                    queue_depth=0,
                    tokens_in_flight=0,
                    token_budget=100000,
                    status="unhealthy",  # But is unhealthy
                    last_heartbeat=time.time() - 60,
                ),
                WorkerInfo(
                    worker_id="worker-healthy",
                    host="localhost",
                    port=8081,
                    models=["gpt-4"],
                    gpu_info=sample_gpu_info,
                    current_load=0.5,  # Higher load
                    queue_depth=5,
                    tokens_in_flight=5000,
                    token_budget=100000,
                    status="healthy",  # But is healthy
                    last_heartbeat=time.time(),
                ),
            ]

            for worker in workers:
                await worker_registry.register(worker)

            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="least_loaded"
            )

            request = make_request()
            selected = await engine.route(request)

            # Should select healthy worker even though it has higher load
            assert selected.worker_id == "worker-healthy"

        asyncio.run(run())

    def test_no_healthy_workers_raises(self, worker_registry, cost_computer, sample_gpu_info):
        """Test exception when all workers are unhealthy."""
        async def run():
            # Create only unhealthy workers
            worker = WorkerInfo(
                worker_id="worker-down",
                host="localhost",
                port=8080,
                models=["gpt-4"],
                gpu_info=sample_gpu_info,
                current_load=0.1,
                queue_depth=0,
                tokens_in_flight=0,
                token_budget=100000,
                status="unhealthy",
                last_heartbeat=time.time() - 120,
            )

            await worker_registry.register(worker)

            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="least_loaded"
            )

            request = make_request()

            with pytest.raises(NoWorkersAvailable):
                await engine.route(request)

        asyncio.run(run())

    def test_model_specific_routing(self, worker_registry, cost_computer, sample_gpu_info):
        """Test routing respects model availability on workers."""
        async def run():
            # Create workers with different model support
            workers = [
                WorkerInfo(
                    worker_id="worker-gpt4-only",
                    host="localhost",
                    port=8080,
                    models=["gpt-4"],
                    gpu_info=sample_gpu_info,
                    current_load=0.1,
                    queue_depth=0,
                    tokens_in_flight=0,
                    token_budget=100000,
                    status="healthy",
                    last_heartbeat=time.time(),
                ),
                WorkerInfo(
                    worker_id="worker-claude-only",
                    host="localhost",
                    port=8081,
                    models=["claude-3"],
                    gpu_info=sample_gpu_info,
                    current_load=0.1,
                    queue_depth=0,
                    tokens_in_flight=0,
                    token_budget=100000,
                    status="healthy",
                    last_heartbeat=time.time(),
                ),
            ]

            for worker in workers:
                await worker_registry.register(worker)

            engine = RoutingEngine(
                worker_registry=worker_registry,
                cost_computer=cost_computer,
                routing_strategy="least_loaded"
            )

            # Request for gpt-4
            gpt4_request = make_request(model="gpt-4")
            selected = await engine.route(gpt4_request)
            assert selected.worker_id == "worker-gpt4-only"

            # Request for claude-3
            claude_request = make_request(model="claude-3")
            selected = await engine.route(claude_request)
            assert selected.worker_id == "worker-claude-only"

        asyncio.run(run())
