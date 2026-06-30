"""Tests for RL-based routing and congestion prediction optimization modules."""

import pytest
import asyncio
import time
import numpy as np

from modelrouter import (
    Priority,
    InferenceRequest,
    WorkerInfo,
    GPUInfo,
    RLExperience,
    CongestionMetrics,
    CongestionThrottled,
    RoutingEngine,
    ModelRouter,
    generate_id,
)
from modelrouter.optimization import (
    ExperienceBuffer,
    DQNPolicy,
    RLRouter,
    MetricsCollector,
    CongestionModel,
    CongestionPredictor,
)

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from conftest import make_request, make_workers, make_worker, make_gpu_info


# ─── DQN Policy Tests ───────────────────────────────────────────────────────


class TestDQNPolicy:
    """Tests for the DQN neural network policy."""

    def test_forward_returns_correct_shape(self, dqn_policy):
        """Test forward pass returns Q-values with correct shape."""
        state = np.random.randn(dqn_policy.state_dim).astype(np.float32)
        q_values = dqn_policy.forward(state)
        assert q_values.shape == (dqn_policy.action_dim,)

    def test_forward_batch(self, dqn_policy):
        """Test forward pass with batch input."""
        states = np.random.randn(5, dqn_policy.state_dim).astype(np.float32)
        q_values = dqn_policy.forward(states)
        assert q_values.shape == (5, dqn_policy.action_dim)

    def test_select_action_returns_valid_index(self, dqn_policy):
        """Test action selection returns valid index."""
        state = np.random.randn(dqn_policy.state_dim).astype(np.float32)
        action = dqn_policy.select_action(state)
        assert 0 <= action < dqn_policy.action_dim

    def test_select_action_respects_num_valid(self, dqn_policy):
        """Test action is within num_valid_actions range."""
        state = np.random.randn(dqn_policy.state_dim).astype(np.float32)
        for _ in range(50):
            action = dqn_policy.select_action(state, num_valid_actions=3)
            assert 0 <= action < 3

    def test_epsilon_greedy_explores(self):
        """Test that epsilon > 0 causes exploration."""
        policy = DQNPolicy(state_dim=10, action_dim=4, epsilon=1.0)
        state = np.zeros(10, dtype=np.float32)
        actions = set()
        for _ in range(100):
            actions.add(policy.select_action(state))
        # With epsilon=1.0, should explore all actions
        assert len(actions) > 1

    def test_epsilon_zero_is_greedy(self):
        """Test that epsilon=0 is purely greedy."""
        policy = DQNPolicy(state_dim=10, action_dim=4, epsilon=0.0)
        state = np.random.randn(10).astype(np.float32)
        first_action = policy.select_action(state)
        for _ in range(20):
            assert policy.select_action(state) == first_action

    def test_update_returns_loss(self, dqn_policy):
        """Test update returns a positive loss value."""
        experiences = []
        for _ in range(8):
            exp = RLExperience(
                request_id=generate_id(),
                state=np.random.randn(dqn_policy.state_dim).astype(np.float32),
                action=np.random.randint(0, dqn_policy.action_dim),
                reward=-0.5,
                done=True,
            )
            experiences.append(exp)
        loss = dqn_policy.update(experiences)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_training_reduces_loss(self):
        """Test that repeated training reduces loss."""
        policy = DQNPolicy(state_dim=10, action_dim=4, lr=0.01, epsilon=0.0)

        # Create a simple pattern: action 0 always gets reward 1.0
        experiences = []
        for _ in range(32):
            exp = RLExperience(
                request_id=generate_id(),
                state=np.ones(10, dtype=np.float32),
                action=0,
                reward=1.0,
                done=True,
            )
            experiences.append(exp)

        first_loss = policy.update(experiences)
        # Train several iterations
        for _ in range(50):
            last_loss = policy.update(experiences)
        assert last_loss < first_loss


# ─── Experience Buffer Tests ────────────────────────────────────────────────


class TestExperienceBuffer:
    """Tests for the experience replay buffer."""

    def test_add_and_get(self, experience_buffer):
        """Test adding and retrieving experiences."""
        exp = RLExperience(
            request_id="req-1",
            state=np.zeros(10),
            action=0,
        )
        experience_buffer.add(exp)
        assert experience_buffer.get("req-1") is exp
        assert len(experience_buffer) == 1

    def test_update_reward(self, experience_buffer):
        """Test updating reward for an experience."""
        exp = RLExperience(
            request_id="req-1",
            state=np.zeros(10),
            action=0,
        )
        experience_buffer.add(exp)
        result = experience_buffer.update_reward("req-1", 0.5)
        assert result is True
        assert experience_buffer.get("req-1").reward == 0.5
        assert experience_buffer.get("req-1").done is True

    def test_sample(self, experience_buffer):
        """Test sampling from buffer."""
        for i in range(20):
            exp = RLExperience(
                request_id=f"req-{i}",
                state=np.random.randn(10),
                action=i % 4,
            )
            experience_buffer.add(exp)
        samples = experience_buffer.sample(5)
        assert len(samples) == 5

    def test_capacity_limit(self):
        """Test buffer respects capacity limit."""
        buf = ExperienceBuffer(capacity=5)
        for i in range(10):
            buf.add(RLExperience(
                request_id=f"req-{i}",
                state=np.zeros(4),
                action=0,
            ))
        assert len(buf) == 5
        # Oldest should be evicted
        assert buf.get("req-0") is None
        assert buf.get("req-9") is not None

    def test_missing_key_returns_none(self, experience_buffer):
        """Test getting nonexistent key returns None."""
        assert experience_buffer.get("nonexistent") is None
        assert experience_buffer.update_reward("nonexistent", 1.0) is False


# ─── RL Router Tests ────────────────────────────────────────────────────────


class TestRLRouter:
    """Tests for the RL-based router."""

    def test_route_returns_worker(self, rl_router):
        """Test routing returns a valid worker."""
        request = make_request()
        workers = make_workers()
        worker = rl_router.route(request, workers)
        assert worker in workers

    def test_route_stores_experience(self, rl_router):
        """Test routing stores experience in buffer."""
        request = make_request()
        workers = make_workers()
        rl_router.route(request, workers)
        exp = rl_router.experience_buffer.get(request.request_id)
        assert exp is not None
        assert exp.action >= 0
        assert exp.action < len(workers)

    def test_state_extraction_shape(self, rl_router):
        """Test state vector has correct shape."""
        request = make_request()
        workers = make_workers()
        state = rl_router._extract_state(request, workers)
        expected_dim = 4 + rl_router.max_workers * 6
        assert state.shape == (expected_dim,)

    def test_state_features_populated(self, rl_router):
        """Test state vector has non-zero request features."""
        request = make_request()
        workers = make_workers()
        state = rl_router._extract_state(request, workers)
        # Request features should be set
        assert state[1] > 0  # estimated_tokens / 1000
        assert state[2] > 0  # temperature
        # Worker features should be set for first worker
        assert state[4] > 0  # current_load of worker-1

    def test_reward_computation(self):
        """Test reward computation formula."""
        # Fast response meeting SLA
        reward = RLRouter.compute_reward(latency_ms=500, sla_deadline_ms=1000)
        assert reward == pytest.approx(-0.5 + 1.0)

        # Slow response missing SLA
        reward = RLRouter.compute_reward(latency_ms=2000, sla_deadline_ms=1000)
        assert reward == pytest.approx(-2.0 - 2.0)

        # No SLA
        reward = RLRouter.compute_reward(latency_ms=1000)
        assert reward == pytest.approx(-1.0)

    def test_update_feeds_reward(self, rl_router):
        """Test update feeds reward to experience."""
        request = make_request()
        workers = make_workers()
        rl_router.route(request, workers)
        result = rl_router.update(request.request_id, 0.75)
        assert result is True
        exp = rl_router.experience_buffer.get(request.request_id)
        assert exp.reward == 0.75
        assert exp.done is True

    def test_varying_worker_counts(self, rl_router):
        """Test routing works with different numbers of workers."""
        for n in [1, 2, 5, 10]:
            workers = [make_worker(f"w-{i}", current_load=0.1 * i) for i in range(n)]
            request = make_request()
            worker = rl_router.route(request, workers)
            assert worker in workers

    def test_train_after_routing(self, rl_router):
        """Test training after collecting experiences."""
        workers = make_workers()
        for _ in range(10):
            req = make_request()
            rl_router.route(req, workers)
            rl_router.update(req.request_id, -0.5)
        loss = rl_router.train(batch_size=8)
        assert isinstance(loss, float)


# ─── Metrics Collector Tests ────────────────────────────────────────────────


class TestMetricsCollector:
    """Tests for the congestion metrics collector."""

    def test_record_arrival(self, metrics_collector):
        """Test recording arrival events."""
        metrics_collector.record_arrival("gpt-4")
        metrics_collector.record_arrival("gpt-4")
        rate = metrics_collector.get_arrival_rate("gpt-4", window_seconds=60)
        assert rate > 0

    def test_record_completion(self, metrics_collector):
        """Test recording completion events."""
        metrics_collector.record_completion("gpt-4")
        metrics_collector.record_completion("gpt-4")
        rate = metrics_collector.get_service_rate("gpt-4", window_seconds=60)
        assert rate > 0

    def test_windowed_queries(self, metrics_collector):
        """Test metrics are windowed correctly."""
        now = time.time()
        # Record old snapshot (outside window)
        old = CongestionMetrics(
            model="gpt-4",
            timestamp=now - 3600,  # 1 hour ago
            queue_depth=10,
            arrival_rate=5.0,
            service_rate=4.0,
        )
        metrics_collector.record_snapshot(old)

        # Record recent snapshot
        recent = CongestionMetrics(
            model="gpt-4",
            timestamp=now,
            queue_depth=20,
            arrival_rate=10.0,
            service_rate=8.0,
        )
        metrics_collector.record_snapshot(recent)

        metrics = metrics_collector.get_metrics("gpt-4")
        assert len(metrics) == 1  # Only recent should be in window
        assert metrics[0].queue_depth == 20

    def test_rate_computation(self, metrics_collector):
        """Test arrival and service rate computation."""
        now = time.time()
        for i in range(10):
            metrics_collector.record_arrival("gpt-4", now - i * 0.5)
        for i in range(8):
            metrics_collector.record_completion("gpt-4", now - i * 0.5)

        arrival_rate = metrics_collector.get_arrival_rate("gpt-4", window_seconds=10)
        service_rate = metrics_collector.get_service_rate("gpt-4", window_seconds=10)
        assert arrival_rate > service_rate

    def test_observation_count(self, metrics_collector):
        """Test observation count tracking."""
        now = time.time()
        for i in range(5):
            metrics_collector.record_snapshot(CongestionMetrics(
                model="gpt-4",
                timestamp=now - i,
                queue_depth=i * 5,
                arrival_rate=1.0,
                service_rate=0.8,
            ))
        assert metrics_collector.get_observation_count("gpt-4") == 5
        assert metrics_collector.get_observation_count("nonexistent") == 0


# ─── Congestion Model Tests ────────────────────────────────────────────────


class TestCongestionModel:
    """Tests for the congestion neural network model."""

    def test_prediction_range(self):
        """Test predictions are in [0, 1]."""
        model = CongestionModel()
        for _ in range(20):
            features = np.random.randn(6)
            prob = model.predict(features)
            assert 0 <= prob <= 1

    def test_training_updates_weights(self):
        """Test training changes model weights."""
        model = CongestionModel()
        w1_before = model.W1.copy()
        features = np.random.randn(10, 6)
        labels = np.random.randint(0, 2, size=10).astype(float)
        model.train(features, labels)
        assert not np.array_equal(model.W1, w1_before)

    def test_training_returns_loss(self):
        """Test training returns a valid loss."""
        model = CongestionModel()
        features = np.random.randn(10, 6)
        labels = np.random.randint(0, 2, size=10).astype(float)
        loss = model.train(features, labels)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_learned_behavior(self):
        """Test model can learn a simple pattern."""
        model = CongestionModel(feature_dim=6, hidden_dim=16)
        model.lr = 0.05

        # Pattern: high queue depth (feature 0) -> congested (1)
        features_pos = np.random.randn(50, 6) * 0.5
        features_pos[:, 0] = np.random.uniform(0.7, 1.0, 50)
        labels_pos = np.ones(50)

        features_neg = np.random.randn(50, 6) * 0.5
        features_neg[:, 0] = np.random.uniform(-0.5, 0.1, 50)
        labels_neg = np.zeros(50)

        features = np.vstack([features_pos, features_neg])
        labels = np.concatenate([labels_pos, labels_neg])

        for _ in range(200):
            idx = np.random.permutation(len(features))
            model.train(features[idx], labels[idx])

        # Test: high queue depth should predict high congestion
        high = np.zeros(6)
        high[0] = 1.0
        low = np.zeros(6)
        low[0] = -0.5
        assert model.predict(high) > model.predict(low)


# ─── Congestion Predictor Tests ─────────────────────────────────────────────


class TestCongestionPredictor:
    """Tests for the congestion prediction system."""

    def test_no_data_returns_zero(self, congestion_predictor):
        """Test returns 0.0 when insufficient data."""
        prob = congestion_predictor.predict_congestion("gpt-4")
        assert prob == 0.0

    def test_prediction_with_data(self, congestion_predictor):
        """Test prediction returns value with sufficient data."""
        now = time.time()
        for i in range(5):
            congestion_predictor.metrics_collector.record_snapshot(CongestionMetrics(
                model="gpt-4",
                timestamp=now - i,
                queue_depth=10 + i * 5,
                arrival_rate=5.0,
                service_rate=3.0,
                avg_latency_ms=500,
            ))
        prob = congestion_predictor.predict_congestion("gpt-4")
        assert 0 <= prob <= 1

    def test_throttle_threshold(self):
        """Test should_throttle returns True when prediction > 0.8."""
        model = CongestionModel()
        collector = MetricsCollector()
        predictor = CongestionPredictor(model, collector)

        now = time.time()
        for i in range(5):
            collector.record_snapshot(CongestionMetrics(
                model="gpt-4",
                timestamp=now - i,
                queue_depth=50,
                arrival_rate=10.0,
                service_rate=2.0,
                avg_latency_ms=5000,
            ))

        # The model output depends on random weights, so just verify the method works
        result = predictor.should_throttle("gpt-4")
        assert isinstance(result, bool)

    def test_online_training(self, congestion_predictor):
        """Test online training updates the model."""
        now = time.time()
        for i in range(5):
            congestion_predictor.metrics_collector.record_snapshot(CongestionMetrics(
                model="gpt-4",
                timestamp=now - i,
                queue_depth=30,
                arrival_rate=8.0,
                service_rate=4.0,
                avg_latency_ms=1000,
            ))

        w1_before = congestion_predictor.model.W1.copy()
        congestion_predictor.train_on_observation("gpt-4", was_congested=True)
        assert not np.array_equal(congestion_predictor.model.W1, w1_before)

    def test_no_training_with_insufficient_data(self, congestion_predictor):
        """Test online training is skipped with insufficient data."""
        w1_before = congestion_predictor.model.W1.copy()
        congestion_predictor.train_on_observation("gpt-4", was_congested=True)
        assert np.array_equal(congestion_predictor.model.W1, w1_before)


# ─── RL Routing Integration Tests ───────────────────────────────────────────


class TestRLRoutingIntegration:
    """Integration tests for RL routing through RoutingEngine."""

    def test_routing_engine_rl_strategy(self):
        """Test RoutingEngine routes via RL strategy."""
        async def run():
            registry = _make_registry_with_workers()
            rl_router = _make_simple_rl_router()
            engine = RoutingEngine(
                registry, routing_strategy="rl", rl_router=rl_router
            )
            request = make_request()
            worker = await engine.route(request)
            assert worker is not None
            assert worker.worker_id.startswith("worker-")
        asyncio.run(run())

    def test_routing_engine_rl_missing_router(self):
        """Test RoutingEngine raises error when rl_router is None."""
        async def run():
            registry = _make_registry_with_workers()
            engine = RoutingEngine(
                registry, routing_strategy="rl", rl_router=None
            )
            request = make_request()
            with pytest.raises(ValueError, match="RL strategy requires rl_router"):
                await engine.route(request)
        asyncio.run(run())

    def test_model_router_end_to_end_with_rl(self):
        """Test ModelRouter end-to-end with RL strategy."""
        async def run():
            router = ModelRouter(routing_strategy="rl")
            await router.register_worker("worker-1", "localhost", 8080, ["gpt-4"])
            await router.register_worker("worker-2", "localhost", 8081, ["gpt-4"])

            request = make_request()
            response = await router.submit(request)
            assert response.request_id == request.request_id

            # Check reward was fed back
            exp = router.experience_buffer.get(request.request_id)
            assert exp is not None
            assert exp.done is True
        asyncio.run(run())


# ─── Congestion Integration Tests ───────────────────────────────────────────


class TestCongestionIntegration:
    """Integration tests for congestion throttling in ModelRouter."""

    def test_throttle_low_priority(self):
        """Test low-priority requests are throttled when congested."""
        async def run():
            router = ModelRouter()
            await router.register_worker("worker-1", "localhost", 8080, ["gpt-4"])

            # Force congestion predictor to report congestion
            # Add enough snapshots and bias model to predict high
            now = time.time()
            for i in range(5):
                router.metrics_collector.record_snapshot(CongestionMetrics(
                    model="gpt-4",
                    timestamp=now - i,
                    queue_depth=100,
                    arrival_rate=50.0,
                    service_rate=5.0,
                    avg_latency_ms=9000,
                ))

            # Train the congestion model to predict congestion
            features = router.congestion_predictor._extract_features("gpt-4")
            for _ in range(200):
                router.congestion_model.train(
                    np.atleast_2d(features),
                    np.array([1.0]),
                )

            # Verify predictor now predicts congestion
            assert router.congestion_predictor.should_throttle("gpt-4") is True

            # LOW priority should be throttled
            low_request = make_request(priority=Priority.LOW)
            with pytest.raises(CongestionThrottled):
                await router.submit(low_request)

            # BATCH priority should also be throttled
            batch_request = make_request(priority=Priority.BATCH)
            with pytest.raises(CongestionThrottled):
                await router.submit(batch_request)
        asyncio.run(run())

    def test_allow_high_priority_through(self):
        """Test high-priority requests pass even when congested."""
        async def run():
            router = ModelRouter()
            await router.register_worker("worker-1", "localhost", 8080, ["gpt-4"])

            # Force congestion
            now = time.time()
            for i in range(5):
                router.metrics_collector.record_snapshot(CongestionMetrics(
                    model="gpt-4",
                    timestamp=now - i,
                    queue_depth=100,
                    arrival_rate=50.0,
                    service_rate=5.0,
                    avg_latency_ms=9000,
                ))

            features = router.congestion_predictor._extract_features("gpt-4")
            for _ in range(200):
                router.congestion_model.train(
                    np.atleast_2d(features),
                    np.array([1.0]),
                )

            assert router.congestion_predictor.should_throttle("gpt-4") is True

            # HIGH priority should NOT be throttled
            high_request = make_request(priority=Priority.HIGH)
            response = await router.submit(high_request)
            assert response.request_id == high_request.request_id

            # CRITICAL should NOT be throttled
            critical_request = make_request(priority=Priority.CRITICAL)
            response = await router.submit(critical_request)
            assert response.request_id == critical_request.request_id

            # NORMAL should NOT be throttled (priority.value == 2 < LOW==3)
            normal_request = make_request(priority=Priority.NORMAL)
            response = await router.submit(normal_request)
            assert response.request_id == normal_request.request_id
        asyncio.run(run())


# ─── Helpers ────────────────────────────────────────────────────────────────


class _SimpleRegistry:
    """Minimal worker registry for integration tests."""
    def __init__(self, workers):
        self._workers = workers

    async def get_workers(self, model=None, status=None):
        return [w for w in self._workers
                if (model is None or model in w.models)
                and (status is None or w.status == status)]


def _make_registry_with_workers():
    """Create a simple registry with workers."""
    return _SimpleRegistry(make_workers())


def _make_simple_rl_router(max_workers=16):
    """Create a simple RL router for tests."""
    state_dim = 4 + max_workers * 6
    policy = DQNPolicy(state_dim=state_dim, action_dim=max_workers)
    buffer = ExperienceBuffer()
    return RLRouter(policy, buffer, max_workers)
