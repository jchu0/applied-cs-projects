"""Tests for scheduling, queue management, and cost optimization."""

import pytest
import asyncio
import time

from modelrouter import (
    Priority,
    InferenceRequest,
    WorkerInfo,
    GPUInfo,
    TenantQuota,
    ModelPricing,
    RequestCost,
    QueueManager,
    CostComputer,
    LatencyPredictor,
    QuotaManager,
    QuotaExceeded,
    PreemptionManager,
    TrafficSplitter,
    CapacityTracker,
    HealthChecker,
    WorkerRegistry,
    generate_id,
)

# Import factory functions from conftest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from conftest import make_request, make_gpu_info, make_worker, make_workers


class TestQueueManager:
    """Tests for priority queue management."""

    def test_enqueue_request(self, queue_manager, sample_request):
        """Test enqueueing a request."""
        async def run():
            request_id = await queue_manager.enqueue(sample_request)
            assert request_id == sample_request.request_id
            assert sample_request.model in queue_manager.queues
            assert len(queue_manager.queues[sample_request.model]) == 1

        asyncio.run(run())

    def test_dequeue_request(self, queue_manager, sample_request):
        """Test dequeueing a request."""
        async def run():
            await queue_manager.enqueue(sample_request)
            dequeued = await queue_manager.dequeue(sample_request.model)
            assert dequeued is not None
            assert dequeued.request_id == sample_request.request_id

        asyncio.run(run())

    def test_dequeue_empty_queue(self, queue_manager):
        """Test dequeueing from empty queue returns None."""
        async def run():
            result = await queue_manager.dequeue("nonexistent-model")
            assert result is None

        asyncio.run(run())

    def test_priority_ordering(self, queue_manager):
        """Test requests are dequeued in priority order."""
        async def run():
            # Create requests with different priorities
            requests = []
            for priority in [Priority.LOW, Priority.NORMAL, Priority.HIGH, Priority.CRITICAL]:
                req = InferenceRequest(
                    request_id=generate_id(),
                    tenant_id="tenant-1",
                    model="gpt-4",
                    prompt=f"Priority {priority.name}",
                    max_tokens=100,
                    temperature=0.7,
                    priority=priority,
                    estimated_tokens=125
                )
                requests.append(req)

            # Enqueue in mixed order (LOW first)
            for req in requests:
                await queue_manager.enqueue(req)

            # Dequeue should return highest priority first
            dequeued = []
            for _ in range(4):
                req = await queue_manager.dequeue("gpt-4")
                dequeued.append(req.priority)

            # CRITICAL (0) should come first, LOW (3) last
            assert dequeued[0] == Priority.CRITICAL
            assert dequeued[-1] == Priority.LOW

        asyncio.run(run())

    def test_peek_request(self, queue_manager, sample_request):
        """Test peeking at queue without removing."""
        async def run():
            await queue_manager.enqueue(sample_request)
            peeked = await queue_manager.peek(sample_request.model)
            assert peeked is not None
            assert peeked.request_id == sample_request.request_id
            # Request should still be in queue
            assert len(queue_manager.queues[sample_request.model]) == 1

        asyncio.run(run())

    def test_remove_specific_request(self, queue_manager, sample_request):
        """Test removing a specific request from queue."""
        async def run():
            await queue_manager.enqueue(sample_request)
            removed = await queue_manager.remove(sample_request.request_id, sample_request.model)
            assert removed is True
            assert len(queue_manager.queues[sample_request.model]) == 0

        asyncio.run(run())

    def test_remove_nonexistent_request(self, queue_manager, sample_request):
        """Test removing nonexistent request returns False."""
        async def run():
            await queue_manager.enqueue(sample_request)
            removed = await queue_manager.remove("nonexistent-id", sample_request.model)
            assert removed is False

        asyncio.run(run())

    def test_queue_stats(self, queue_manager, many_requests):
        """Test getting queue statistics."""
        async def run():
            # Enqueue multiple requests
            for req in many_requests[:10]:
                await queue_manager.enqueue(req)

            stats = await queue_manager.get_queue_stats("gpt-4")
            assert stats["depth"] == 10
            assert "oldest_ms" in stats
            assert "by_priority" in stats

        asyncio.run(run())

    def test_get_all_stats(self, queue_manager):
        """Test getting stats for all queues."""
        async def run():
            # Create requests for different models
            for model in ["gpt-4", "gpt-3.5", "claude-3"]:
                req = make_request(model=model)
                await queue_manager.enqueue(req)

            stats = await queue_manager.get_all_stats()
            assert "gpt-4" in stats
            assert "gpt-3.5" in stats
            assert "claude-3" in stats

        asyncio.run(run())

    def test_priority_score_aging(self, queue_manager):
        """Test that older requests get priority boost."""
        # Create request with old timestamp
        old_request = InferenceRequest(
            request_id=generate_id(),
            tenant_id="tenant-1",
            model="gpt-4",
            prompt="Old request",
            max_tokens=100,
            temperature=0.7,
            priority=Priority.LOW,
            created_at=time.time() - 120,  # 2 minutes old
            estimated_tokens=125
        )

        new_request = InferenceRequest(
            request_id=generate_id(),
            tenant_id="tenant-1",
            model="gpt-4",
            prompt="New request",
            max_tokens=100,
            temperature=0.7,
            priority=Priority.LOW,
            created_at=time.time(),
            estimated_tokens=125
        )

        # Older request should have lower score (higher priority)
        old_score = queue_manager._compute_priority_score(old_request)
        new_score = queue_manager._compute_priority_score(new_request)
        assert old_score < new_score


class TestLatencyPredictor:
    """Tests for latency prediction."""

    def test_predict_without_history(self, latency_predictor, sample_request, sample_worker):
        """Test prediction without historical data uses formula."""
        prediction = latency_predictor.predict(sample_request, sample_worker)
        # Should use formula: tokens * 10 * load_factor * queue_factor
        assert prediction > 0
        assert isinstance(prediction, int)

    def test_predict_with_history(self, sample_request, sample_worker):
        """Test prediction with historical data uses median."""
        predictor = LatencyPredictor()
        # Record historical latencies
        for i in range(10):
            predictor.record(
                model="gpt-4",
                worker_type="standard",
                tokens=125,
                latency_ms=1000 + i * 10  # 1000-1090ms
            )
        prediction = predictor.predict(sample_request, sample_worker)
        # Should be around median adjusted for load
        assert prediction > 0

    def test_record_latency(self, latency_predictor):
        """Test recording observed latency."""
        latency_predictor.record(
            model="gpt-4",
            worker_type="standard",
            tokens=100,
            latency_ms=500
        )
        key = "gpt-4:standard"
        assert key in latency_predictor.history
        assert len(latency_predictor.history[key]) == 1
        assert latency_predictor.history[key][0] == (100, 500)

    def test_history_limit(self):
        """Test history is limited to 1000 entries."""
        predictor = LatencyPredictor()
        # Record more than 1000 entries
        for i in range(1100):
            predictor.record("gpt-4", "standard", 100, 500)
        key = "gpt-4:standard"
        assert len(predictor.history[key]) == 1000


class TestCostComputer:
    """Tests for cost computation."""

    def test_compute_cost(self, cost_computer, sample_request, sample_worker):
        """Test computing request cost."""
        cost = cost_computer.compute(sample_request, sample_worker)
        assert isinstance(cost, RequestCost)
        assert cost.tokens == sample_request.estimated_tokens
        assert cost.compute_units > 0
        assert cost.estimated_latency_ms > 0
        assert cost.dollar_cost > 0

    def test_compute_cost_without_pricing(self, sample_request, sample_worker):
        """Test cost computation with default pricing."""
        computer = CostComputer()
        cost = computer.compute(sample_request, sample_worker)
        # Should use default pricing
        assert cost.dollar_cost > 0

    def test_set_pricing(self, cost_computer, sample_model_pricing):
        """Test setting pricing for a model."""
        new_pricing = ModelPricing(
            model="claude-3",
            input_cost_per_1k=0.02,
            output_cost_per_1k=0.04,
            compute_cost_per_unit=0.002
        )
        cost_computer.set_pricing("claude-3", new_pricing)
        assert "claude-3" in cost_computer.pricing
        assert cost_computer.pricing["claude-3"].input_cost_per_1k == 0.02

    def test_compute_units_estimation(self, cost_computer, sample_request, sample_worker):
        """Test compute units are properly estimated."""
        cost = cost_computer.compute(sample_request, sample_worker)
        # Compute units = tokens * 0.001 / performance_factor
        expected_units = sample_request.estimated_tokens * 0.001 / sample_worker.performance_factor
        assert cost.compute_units == pytest.approx(expected_units, rel=0.01)


class TestQuotaManager:
    """Tests for tenant quota management."""

    def test_check_quota_within_limit(self, quota_manager, sample_request, sample_tenant_quota):
        """Test quota check passes when within limit."""
        async def run():
            await quota_manager.set_quota(sample_tenant_quota)
            result = await quota_manager.check_quota(
                sample_request.tenant_id,
                sample_request
            )
            assert result is True

        asyncio.run(run())

    def test_check_quota_exceeded(self, quota_manager, sample_request):
        """Test quota check fails when budget exceeded."""
        async def run():
            # Set a very small budget
            quota = TenantQuota(
                tenant_id=sample_request.tenant_id,
                requests_per_second=100,
                tokens_per_minute=100000,
                monthly_token_budget=100  # Very small
            )
            await quota_manager.set_quota(quota)

            with pytest.raises(QuotaExceeded) as exc_info:
                await quota_manager.check_quota(sample_request.tenant_id, sample_request)
            assert "Monthly token budget exceeded" in str(exc_info.value)

        asyncio.run(run())

    def test_record_usage(self, quota_manager, sample_tenant_quota):
        """Test recording token usage."""
        async def run():
            await quota_manager.set_quota(sample_tenant_quota)
            await quota_manager.record_usage("tenant-1", 1000)
            await quota_manager.record_usage("tenant-1", 500)
            report = await quota_manager.get_usage_report("tenant-1")
            assert report.tokens_used == 1500

        asyncio.run(run())

    def test_usage_report(self, quota_manager, sample_tenant_quota):
        """Test getting usage report."""
        async def run():
            await quota_manager.set_quota(sample_tenant_quota)
            await quota_manager.record_usage("tenant-1", 1000000)
            report = await quota_manager.get_usage_report("tenant-1")
            assert report.tenant_id == "tenant-1"
            assert report.tokens_used == 1000000
            assert report.tokens_remaining == 9000000  # 10M - 1M
            assert report.utilization == pytest.approx(0.1, rel=0.01)

        asyncio.run(run())

    def test_reset_usage(self, quota_manager, sample_tenant_quota):
        """Test resetting usage (monthly reset)."""
        async def run():
            await quota_manager.set_quota(sample_tenant_quota)
            await quota_manager.record_usage("tenant-1", 5000000)
            await quota_manager.reset_usage("tenant-1")
            report = await quota_manager.get_usage_report("tenant-1")
            assert report.tokens_used == 0

        asyncio.run(run())

    def test_default_quota(self, quota_manager, sample_request):
        """Test default quota is applied for unknown tenants."""
        async def run():
            # No quota set - should use default
            result = await quota_manager.check_quota("unknown-tenant", sample_request)
            assert result is True

        asyncio.run(run())


class TestPreemptionManager:
    """Tests for request preemption."""

    def test_register_in_flight(self, preemption_manager, sample_request):
        """Test registering in-flight request."""
        preemption_manager.register_in_flight(sample_request)
        assert sample_request.request_id in preemption_manager._in_flight
        assert preemption_manager.get_in_flight_count() == 1

    def test_complete_request(self, preemption_manager, sample_request):
        """Test completing a request removes from tracking."""
        preemption_manager.register_in_flight(sample_request)
        preemption_manager.complete_request(sample_request.request_id)
        assert sample_request.request_id not in preemption_manager._in_flight
        assert preemption_manager.get_in_flight_count() == 0

    def test_preempt_low_priority_for_critical(
        self,
        preemption_manager,
        critical_request,
        batch_request
    ):
        """Test critical request preempts batch request."""
        async def run():
            # Register batch request as in-flight
            preemption_manager.register_in_flight(batch_request)

            # Critical request should trigger preemption
            preempted = await preemption_manager.maybe_preempt(
                critical_request,
                critical_request.model
            )
            assert preempted is True
            assert batch_request.request_id in preemption_manager._preempted

        asyncio.run(run())

    def test_no_preemption_for_same_priority(
        self,
        preemption_manager,
        sample_request
    ):
        """Test no preemption for same priority requests."""
        async def run():
            # Register normal priority request
            preemption_manager.register_in_flight(sample_request)

            # Another normal priority should not preempt
            another_normal = InferenceRequest(
                request_id=generate_id(),
                tenant_id="tenant-1",
                model="gpt-4",
                prompt="Another normal",
                max_tokens=100,
                temperature=0.7,
                priority=Priority.NORMAL,
                estimated_tokens=125
            )

            preempted = await preemption_manager.maybe_preempt(
                another_normal,
                another_normal.model
            )
            assert preempted is False

        asyncio.run(run())

    def test_no_preemption_for_low_priority(
        self,
        preemption_manager,
        sample_request,
        batch_request
    ):
        """Test low priority requests don't preempt."""
        async def run():
            preemption_manager.register_in_flight(sample_request)

            # Batch should not preempt normal
            preempted = await preemption_manager.maybe_preempt(
                batch_request,
                batch_request.model
            )
            assert preempted is False

        asyncio.run(run())

    def test_in_flight_count_by_model(self, preemption_manager):
        """Test counting in-flight requests by model."""
        for i, model in enumerate(["gpt-4", "gpt-4", "claude-3"]):
            req = make_request(model=model)
            preemption_manager.register_in_flight(req)

        assert preemption_manager.get_in_flight_count("gpt-4") == 2
        assert preemption_manager.get_in_flight_count("claude-3") == 1
        assert preemption_manager.get_in_flight_count() == 3


class TestTrafficSplitter:
    """Tests for canary traffic splitting."""

    def test_no_canary_returns_stable(self, traffic_splitter, sample_request):
        """Test returns stable when no canary configured."""
        async def run():
            target = await traffic_splitter.get_target(sample_request)
            assert target == "stable"

        asyncio.run(run())

    def test_canary_traffic_split(self, traffic_splitter, sample_request):
        """Test traffic is split between stable and canary."""
        async def run():
            await traffic_splitter.create_canary(
                model="gpt-4",
                canary_workers=["canary-1", "canary-2"],
                traffic_percentage=50.0
            )

            # Run multiple requests and check distribution
            stable_count = 0
            canary_count = 0

            for i in range(100):
                req = make_request(tenant_id=f"tenant-{i}")
                target = await traffic_splitter.get_target(req)
                if target == "stable":
                    stable_count += 1
                else:
                    canary_count += 1

            # Should be roughly 50/50 (allow for hash distribution variance)
            assert 30 < canary_count < 70
            assert 30 < stable_count < 70

        asyncio.run(run())

    def test_update_traffic_percentage(self, traffic_splitter):
        """Test updating canary traffic percentage."""
        async def run():
            await traffic_splitter.create_canary(
                model="gpt-4",
                canary_workers=["canary-1"],
                traffic_percentage=10.0
            )
            await traffic_splitter.update_traffic("gpt-4", 50.0)
            config = await traffic_splitter.get_canary_config("gpt-4")
            assert config.traffic_percentage == 50.0

        asyncio.run(run())

    def test_complete_canary(self, traffic_splitter):
        """Test completing canary rollout."""
        async def run():
            await traffic_splitter.create_canary(
                model="gpt-4",
                canary_workers=["canary-1"],
                traffic_percentage=10.0
            )
            await traffic_splitter.complete_canary("gpt-4")
            config = await traffic_splitter.get_canary_config("gpt-4")
            assert config.traffic_percentage == 100.0

        asyncio.run(run())

    def test_rollback_canary(self, traffic_splitter, sample_request):
        """Test rolling back canary."""
        async def run():
            await traffic_splitter.create_canary(
                model="gpt-4",
                canary_workers=["canary-1"],
                traffic_percentage=50.0
            )
            await traffic_splitter.rollback_canary("gpt-4")
            # Should return stable after rollback
            target = await traffic_splitter.get_target(sample_request)
            assert target == "stable"

        asyncio.run(run())

    def test_get_canary_workers(self, traffic_splitter):
        """Test getting canary worker list."""
        async def run():
            await traffic_splitter.create_canary(
                model="gpt-4",
                canary_workers=["canary-1", "canary-2"],
                traffic_percentage=20.0
            )
            workers = await traffic_splitter.get_canary_workers("gpt-4")
            assert workers == ["canary-1", "canary-2"]

        asyncio.run(run())

    def test_consistent_hashing_same_tenant(self, traffic_splitter):
        """Test same tenant always goes to same target."""
        async def run():
            await traffic_splitter.create_canary(
                model="gpt-4",
                canary_workers=["canary-1"],
                traffic_percentage=50.0
            )

            # Same tenant should always get same result
            request = make_request(tenant_id="consistent-tenant")
            first_target = await traffic_splitter.get_target(request)

            for _ in range(10):
                target = await traffic_splitter.get_target(request)
                assert target == first_target

        asyncio.run(run())


class TestCapacityTracker:
    """Tests for capacity tracking."""

    def test_get_capacity_no_workers(self, capacity_tracker):
        """Test capacity with no workers."""
        async def run():
            capacity = await capacity_tracker.get_capacity("gpt-4")
            assert capacity.total_workers == 0
            assert capacity.healthy_workers == 0
            assert capacity.utilization == 0

        asyncio.run(run())

    def test_get_capacity_with_workers(self, worker_registry, sample_workers):
        """Test capacity with registered workers."""
        async def run():
            for worker in sample_workers:
                await worker_registry.register(worker)

            tracker = CapacityTracker(worker_registry)
            capacity = await tracker.get_capacity("gpt-4")

            assert capacity.total_workers == 3
            assert capacity.healthy_workers == 3
            assert capacity.total_token_budget == 300000  # 3 * 100000
            assert capacity.gpu_memory_total_mb > 0

        asyncio.run(run())

    def test_capacity_utilization(self, worker_registry, sample_gpu_info):
        """Test capacity utilization calculation."""
        async def run():
            worker = WorkerInfo(
                worker_id="worker-1",
                host="localhost",
                port=8080,
                models=["gpt-4"],
                gpu_info=sample_gpu_info,
                current_load=0.5,
                queue_depth=10,
                tokens_in_flight=50000,  # 50% utilization
                token_budget=100000,
                status="healthy",
                last_heartbeat=time.time(),
            )
            await worker_registry.register(worker)

            tracker = CapacityTracker(worker_registry)
            capacity = await tracker.get_capacity("gpt-4")

            assert capacity.utilization == pytest.approx(0.5, rel=0.01)
            assert capacity.available_tokens == 50000

        asyncio.run(run())

    def test_get_all_capacity(self, worker_registry, sample_gpu_info):
        """Test getting capacity for all models."""
        async def run():
            models = ["gpt-4", "gpt-3.5", "claude-3"]

            for i, model in enumerate(models):
                worker = WorkerInfo(
                    worker_id=f"worker-{model}",
                    host="localhost",
                    port=8080 + i,
                    models=[model],
                    gpu_info=sample_gpu_info,
                    current_load=0.3,
                    queue_depth=5,
                    tokens_in_flight=10000,
                    token_budget=100000,
                    status="healthy",
                    last_heartbeat=time.time(),
                )
                await worker_registry.register(worker)

            tracker = CapacityTracker(worker_registry)
            all_capacity = await tracker.get_all_capacity()

            assert "gpt-4" in all_capacity
            assert "gpt-3.5" in all_capacity
            assert "claude-3" in all_capacity

        asyncio.run(run())


class TestHealthChecker:
    """Tests for worker health checking."""

    def test_check_once(self, health_checker, worker_registry, sample_worker):
        """Test running one health check cycle."""
        async def run():
            await worker_registry.register(sample_worker)
            results = await health_checker.check_once()
            assert sample_worker.worker_id in results
            assert results[sample_worker.worker_id] is True

        asyncio.run(run())

    def test_stale_heartbeat_marks_unhealthy(
        self,
        worker_registry,
        sample_gpu_info
    ):
        """Test worker with stale heartbeat is marked unhealthy."""
        async def run():
            stale_worker = WorkerInfo(
                worker_id="stale-worker",
                host="localhost",
                port=8080,
                models=["gpt-4"],
                gpu_info=sample_gpu_info,
                current_load=0.3,
                queue_depth=5,
                tokens_in_flight=1000,
                token_budget=100000,
                status="healthy",
                last_heartbeat=time.time() - 60,  # 60 seconds old
            )
            await worker_registry.register(stale_worker)

            checker = HealthChecker(worker_registry, unhealthy_threshold=1)
            results = await checker.check_once()
            assert results["stale-worker"] is False

        asyncio.run(run())

    def test_failure_threshold(self, worker_registry, sample_gpu_info):
        """Test unhealthy threshold before marking unhealthy."""
        async def run():
            # Create worker with stale heartbeat from the start
            stale_worker = WorkerInfo(
                worker_id="threshold-worker",
                host="localhost",
                port=8080,
                models=["gpt-4"],
                gpu_info=sample_gpu_info,
                current_load=0.3,
                queue_depth=5,
                tokens_in_flight=1000,
                token_budget=100000,
                status="healthy",
                last_heartbeat=time.time() - 60,  # Already stale
            )
            await worker_registry.register(stale_worker)

            # Checker requires 3 failures before marking unhealthy
            checker = HealthChecker(worker_registry, unhealthy_threshold=3)

            # Run 3 health checks - only after 3 should it be unhealthy
            for i in range(3):
                await checker.check_once()

            # After 3 failures, should be marked unhealthy
            worker = await worker_registry.get_worker(stale_worker.worker_id)
            assert worker.status == "unhealthy"

        asyncio.run(run())

    def test_recovery_resets_failure_count(
        self,
        worker_registry,
        sample_gpu_info
    ):
        """Test recovery resets failure count."""
        async def run():
            # Create worker with stale heartbeat
            stale_worker = WorkerInfo(
                worker_id="recovery-worker",
                host="localhost",
                port=8080,
                models=["gpt-4"],
                gpu_info=sample_gpu_info,
                current_load=0.3,
                queue_depth=5,
                tokens_in_flight=1000,
                token_budget=100000,
                status="healthy",
                last_heartbeat=time.time() - 60,  # Stale
            )
            await worker_registry.register(stale_worker)

            checker = HealthChecker(worker_registry, unhealthy_threshold=3)

            # Cause some failures
            await checker.check_once()
            await checker.check_once()

            # Now worker recovers - update heartbeat
            await worker_registry.heartbeat(stale_worker.worker_id, {
                "last_heartbeat": time.time()
            })

            # After fresh heartbeat, check should pass and reset counter
            await checker.check_once()

            # Failure count should be 0 after successful check
            assert checker.failure_counts.get(stale_worker.worker_id, 0) == 0

        asyncio.run(run())


class TestWorkerRegistry:
    """Tests for worker registry."""

    def test_register_worker(self, worker_registry, sample_worker):
        """Test registering a worker."""
        async def run():
            worker_id = await worker_registry.register(sample_worker)
            assert worker_id == sample_worker.worker_id

            worker = await worker_registry.get_worker(worker_id)
            assert worker is not None
            assert worker.host == sample_worker.host

        asyncio.run(run())

    def test_deregister_worker(self, worker_registry, sample_worker):
        """Test deregistering a worker."""
        async def run():
            await worker_registry.register(sample_worker)
            await worker_registry.deregister(sample_worker.worker_id)
            worker = await worker_registry.get_worker(sample_worker.worker_id)
            assert worker is None

        asyncio.run(run())

    def test_get_workers_by_model(self, worker_registry, sample_workers):
        """Test filtering workers by model."""
        async def run():
            for worker in sample_workers:
                await worker_registry.register(worker)
            workers = await worker_registry.get_workers(model="gpt-4")
            assert len(workers) == 3

        asyncio.run(run())

    def test_get_workers_by_status(self, worker_registry, sample_workers):
        """Test filtering workers by status."""
        async def run():
            for worker in sample_workers:
                await worker_registry.register(worker)

            # Mark one worker unhealthy
            await worker_registry.update_status("worker-1", "unhealthy")

            healthy = await worker_registry.get_workers(status="healthy")
            unhealthy = await worker_registry.get_workers(status="unhealthy")

            assert len(healthy) == 2
            assert len(unhealthy) == 1

        asyncio.run(run())

    def test_heartbeat_updates_worker(self, worker_registry, sample_worker):
        """Test heartbeat updates worker status."""
        async def run():
            await worker_registry.register(sample_worker)
            new_load = 0.8
            await worker_registry.heartbeat(sample_worker.worker_id, {
                "current_load": new_load
            })
            worker = await worker_registry.get_worker(sample_worker.worker_id)
            assert worker.current_load == new_load

        asyncio.run(run())

    def test_update_load(self, worker_registry, sample_worker):
        """Test updating worker load metrics."""
        async def run():
            await worker_registry.register(sample_worker)
            await worker_registry.update_load(
                sample_worker.worker_id,
                current_load=0.9,
                queue_depth=20,
                tokens_in_flight=50000
            )
            worker = await worker_registry.get_worker(sample_worker.worker_id)
            assert worker.current_load == 0.9
            assert worker.queue_depth == 20
            assert worker.tokens_in_flight == 50000

        asyncio.run(run())
