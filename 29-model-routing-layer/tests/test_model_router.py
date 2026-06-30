"""Tests for ModelRouter orchestrator, GPUMonitor, HealthChecker, and create_router."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from modelrouter import (
    Priority,
    InferenceRequest,
    InferenceResponse,
    GPUInfo,
    WorkerInfo,
    ModelRouter,
    create_router,
    GPUMonitor,
    HealthChecker,
    WorkerRegistry,
    CongestionThrottled,
    generate_id,
)

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from conftest import make_request, make_worker, make_workers


# ---------------------------------------------------------------------------
# TestCreateRouter — factory function
# ---------------------------------------------------------------------------

class TestCreateRouter:
    """Tests for the create_router factory function."""

    def test_factory_defaults(self):
        """Default create_router produces a working ModelRouter."""
        router = create_router()
        assert isinstance(router, ModelRouter)
        assert router.max_queue_depth == 1000
        assert router.routing_engine.strategy == "least_loaded"

    def test_custom_strategy(self):
        """Factory respects custom routing strategy."""
        router = create_router(routing_strategy="round_robin")
        assert router.routing_engine.strategy == "round_robin"

    def test_custom_queue_depth(self):
        """Factory respects custom max queue depth."""
        router = create_router(max_queue_depth=250)
        assert router.max_queue_depth == 250


# ---------------------------------------------------------------------------
# TestGPUMonitor
# ---------------------------------------------------------------------------

class TestGPUMonitor:
    """Tests for the GPUMonitor class."""

    def test_poll_interval_stored(self):
        """Constructor stores poll_interval."""
        monitor = GPUMonitor(poll_interval=15)
        assert monitor.poll_interval == 15

    def test_collect_metrics_returns_gpu_info(self):
        """collect_metrics returns a GPUInfo dataclass."""
        async def run():
            monitor = GPUMonitor()
            info = await monitor.collect_metrics("worker-1")
            assert isinstance(info, GPUInfo)
            assert info.device_name == "NVIDIA A100"
            assert info.memory_total_mb == 40960
            assert info.utilization_percent == 45.0

        asyncio.run(run())

    def test_start_monitoring_calls_callback(self):
        """start_monitoring invokes the callback with metrics."""
        async def run():
            monitor = GPUMonitor(poll_interval=0)
            received = []

            async def cb(wid, metrics):
                received.append((wid, metrics))
                if len(received) >= 2:
                    raise asyncio.CancelledError  # stop the loop

            try:
                await monitor.start_monitoring("w-1", cb)
            except asyncio.CancelledError:
                pass

            assert len(received) >= 2
            assert received[0][0] == "w-1"
            assert isinstance(received[0][1], GPUInfo)

        asyncio.run(run())

    def test_monitoring_handles_errors(self):
        """start_monitoring continues after a callback error."""
        async def run():
            monitor = GPUMonitor(poll_interval=0)
            call_count = 0

            async def failing_cb(wid, metrics):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("boom")
                if call_count >= 3:
                    raise asyncio.CancelledError

            try:
                await monitor.start_monitoring("w-1", failing_cb)
            except asyncio.CancelledError:
                pass

            # At least 3 calls: 1 failure + 2 more
            assert call_count >= 3

        asyncio.run(run())


# ---------------------------------------------------------------------------
# TestHealthCheckerAsync
# ---------------------------------------------------------------------------

class TestHealthCheckerAsync:
    """Tests for HealthChecker start/stop lifecycle."""

    def test_start_stop_lifecycle(self):
        """start() runs until stop() is called."""
        async def run():
            registry = WorkerRegistry()
            checker = HealthChecker(registry, check_interval=0)

            task = asyncio.create_task(checker.start())
            await asyncio.sleep(0.05)
            assert checker._running is True

            checker.stop()
            await asyncio.sleep(0.05)
            assert checker._running is False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())

    def test_stop_terminates_loop(self):
        """Calling stop() sets _running to False."""
        registry = WorkerRegistry()
        checker = HealthChecker(registry)
        checker._running = True
        checker.stop()
        assert checker._running is False

    def test_check_once_delegation(self):
        """check_once returns health mapping for registered workers."""
        async def run():
            registry = WorkerRegistry()
            worker = make_worker("w-1")
            await registry.register(worker)

            checker = HealthChecker(registry)
            results = await checker.check_once()

            assert "w-1" in results
            assert results["w-1"] is True

        asyncio.run(run())

    def test_failure_count_accumulation(self):
        """Repeated failures accumulate and mark worker unhealthy."""
        async def run():
            registry = WorkerRegistry()
            worker = make_worker("w-1")
            # Make heartbeat stale so _check_worker returns False
            worker = WorkerInfo(
                worker_id="w-1",
                host="localhost",
                port=8080,
                models=["gpt-4"],
                gpu_info=GPUInfo(
                    device_name="NVIDIA A100",
                    memory_total_mb=40960,
                    memory_used_mb=8192,
                    utilization_percent=45.0,
                    temperature_celsius=65,
                ),
                current_load=0.3,
                queue_depth=5,
                tokens_in_flight=1000,
                token_budget=100000,
                status="healthy",
                last_heartbeat=time.time() - 60,  # stale
            )
            await registry.register(worker)

            checker = HealthChecker(registry, unhealthy_threshold=2)

            await checker.check_once()
            assert checker.failure_counts.get("w-1", 0) == 1

            await checker.check_once()
            assert checker.failure_counts["w-1"] >= 2

            # Worker should now be marked unhealthy
            w = await registry.get_worker("w-1")
            assert w.status == "unhealthy"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# TestModelRouterEdgeCases
# ---------------------------------------------------------------------------

class TestModelRouterEdgeCases:
    """Edge-case tests for ModelRouter submit path."""

    @staticmethod
    async def _setup_router():
        """Helper: create a router with one worker and tenant."""
        router = create_router()
        await router.register_worker(
            worker_id="worker-1",
            host="localhost",
            port=8080,
            models=["gpt-4"],
            token_budget=100000,
        )
        router.register_tenant(
            tenant_id="tenant-1",
            name="Test",
            api_key="key-1",
        )
        return router

    def test_submit_high_priority_triggers_preemption_check(self):
        """HIGH-priority submit invokes preemption path."""
        async def run():
            router = await self._setup_router()
            request = InferenceRequest(
                request_id=generate_id(),
                tenant_id="tenant-1",
                model="gpt-4",
                prompt="Urgent",
                max_tokens=50,
                temperature=0.5,
                priority=Priority.HIGH,
                sla_deadline_ms=3000,
                estimated_tokens=65,
            )

            with patch.object(
                router.preemption_manager, "maybe_preempt", new_callable=AsyncMock
            ) as mock_preempt:
                await router.submit(request)
                mock_preempt.assert_called_once()

        asyncio.run(run())

    def test_submit_records_rl_reward(self):
        """submit() feeds a reward to the RL router."""
        async def run():
            router = await self._setup_router()
            request = make_request(priority=Priority.NORMAL)

            with patch.object(router.rl_router, "update") as mock_update:
                await router.submit(request)
                mock_update.assert_called_once()
                # reward arg should be a float
                _, reward = mock_update.call_args[0]
                assert isinstance(reward, float)

        asyncio.run(run())

    def test_submit_records_quota_usage(self):
        """submit() calls quota_manager.record_usage."""
        async def run():
            router = await self._setup_router()
            request = make_request(priority=Priority.NORMAL)

            with patch.object(
                router.quota_manager, "record_usage", new_callable=AsyncMock
            ) as mock_record:
                await router.submit(request)
                mock_record.assert_called_once_with(
                    request.tenant_id, request.estimated_tokens
                )

        asyncio.run(run())

    def test_metrics_limit_parameter(self):
        """get_metrics(limit=N) returns at most N entries."""
        async def run():
            router = await self._setup_router()
            for _ in range(5):
                req = make_request(priority=Priority.NORMAL)
                await router.submit(req)

            assert len(router.get_metrics(limit=3)) == 3
            assert len(router.get_metrics(limit=100)) == 5

        asyncio.run(run())

    def test_submit_congestion_throttles_low_priority(self):
        """LOW-priority request is rejected when congestion predictor fires."""
        async def run():
            router = await self._setup_router()

            with patch.object(
                router.congestion_predictor, "should_throttle", return_value=True
            ):
                request = make_request(priority=Priority.LOW)
                with pytest.raises(CongestionThrottled):
                    await router.submit(request)

        asyncio.run(run())

    def test_submit_congestion_does_not_throttle_normal(self):
        """NORMAL-priority request bypasses congestion throttle."""
        async def run():
            router = await self._setup_router()

            with patch.object(
                router.congestion_predictor, "should_throttle", return_value=True
            ):
                request = make_request(priority=Priority.NORMAL)
                # Should not raise — NORMAL is below the LOW threshold
                response = await router.submit(request)
                assert response is not None

        asyncio.run(run())

    def test_get_capacity_all_models(self):
        """get_capacity() without model returns dict keyed by model."""
        async def run():
            router = await self._setup_router()
            capacity = await router.get_capacity()
            assert isinstance(capacity, dict)
            assert "gpt-4" in capacity

        asyncio.run(run())

    def test_get_capacity_specific_model(self):
        """get_capacity(model) returns CapacityInfo for that model."""
        async def run():
            router = await self._setup_router()
            capacity = await router.get_capacity("gpt-4")
            assert capacity.total_workers == 1
            assert capacity.healthy_workers == 1

        asyncio.run(run())

    def test_get_queue_stats(self):
        """get_queue_stats returns queue info dict."""
        async def run():
            router = await self._setup_router()
            stats = await router.get_queue_stats()
            assert isinstance(stats, dict)

        asyncio.run(run())

    def test_set_routing_strategy(self):
        """set_routing_strategy updates the engine strategy."""
        router = create_router()
        router.set_routing_strategy("weighted")
        assert router.routing_engine.strategy == "weighted"
