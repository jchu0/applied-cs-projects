"""Tests for Worker."""

import pytest
import numpy as np

from paramserver.worker.worker import (
    Worker,
    MockGradientComputer,
    DataBatchGenerator,
)
from paramserver.schemas import WorkerStatus


class TestWorkerInit:
    """Tests for Worker initialization."""

    @pytest.mark.asyncio
    async def test_create_worker(self, initialized_cluster, small_params):
        """Test worker creation."""
        await initialized_cluster.initialize(small_params)

        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
        )

        assert worker.worker_id == 0
        assert worker.clock == 0
        assert worker.info.status == WorkerStatus.IDLE

    @pytest.mark.asyncio
    async def test_worker_with_gradient_fn(self, initialized_cluster, small_params):
        """Test worker with gradient function."""
        await initialized_cluster.initialize(small_params)

        shapes = {name: p.shape for name, p in small_params.items()}
        grad_fn = MockGradientComputer(shapes)

        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
            compute_gradients=grad_fn,
        )

        assert worker._compute_gradients is not None


class TestWorkerPullPush:
    """Tests for pull and push operations."""

    @pytest.mark.asyncio
    async def test_pull_params(self, initialized_cluster, small_params):
        """Test pulling parameters."""
        await initialized_cluster.initialize(small_params)

        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
        )

        params = await worker.pull_params()

        assert len(params) == len(small_params)
        for name, value in small_params.items():
            np.testing.assert_array_equal(params[name], value)

    @pytest.mark.asyncio
    async def test_push_gradients(self, initialized_cluster, small_params):
        """Test pushing gradients."""
        await initialized_cluster.initialize(small_params)

        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
        )

        grads = {
            name: np.random.randn(*value.shape).astype(np.float32) * 0.01
            for name, value in small_params.items()
        }

        applied = await worker.push_gradients(grads)
        assert applied == len(small_params)
        assert worker.clock == 1

    @pytest.mark.asyncio
    async def test_push_increments_clock(self, initialized_cluster, small_params):
        """Test that push increments worker clock."""
        await initialized_cluster.initialize(small_params)

        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
        )

        assert worker.clock == 0

        grads = {"w1": np.array([0.1, 0.1, 0.1], dtype=np.float32)}
        await worker.push_gradients(grads)
        assert worker.clock == 1

        await worker.push_gradients(grads)
        assert worker.clock == 2


class TestWorkerTrainStep:
    """Tests for training step."""

    @pytest.mark.asyncio
    async def test_train_step(self, initialized_cluster, small_params):
        """Test single training step."""
        await initialized_cluster.initialize(small_params)

        shapes = {name: p.shape for name, p in small_params.items()}
        grad_fn = MockGradientComputer(shapes)

        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
            compute_gradients=grad_fn,
        )

        batch = {"x": np.random.randn(10, 5)}
        loss = await worker.train_step(batch)

        assert loss > 0
        assert worker.clock == 1
        assert worker.info.total_steps == 1

    @pytest.mark.asyncio
    async def test_train_step_no_grad_fn(self, initialized_cluster, small_params):
        """Test train step without gradient function raises error."""
        await initialized_cluster.initialize(small_params)

        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
        )

        with pytest.raises(ValueError, match="Gradient function not set"):
            await worker.train_step({})

    @pytest.mark.asyncio
    async def test_train_step_updates_stats(self, initialized_cluster, small_params):
        """Test that train step updates statistics."""
        await initialized_cluster.initialize(small_params)

        shapes = {name: p.shape for name, p in small_params.items()}
        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
            compute_gradients=MockGradientComputer(shapes),
        )

        assert worker.info.total_steps == 0
        assert worker.info.total_updates == 0

        await worker.train_step({})

        assert worker.info.total_steps == 1
        assert worker.info.total_updates == 1


class TestWorkerTrainLoop:
    """Tests for training loop."""

    @pytest.mark.asyncio
    async def test_train_loop(self, initialized_cluster, small_params):
        """Test training loop execution."""
        await initialized_cluster.initialize(small_params)

        shapes = {name: p.shape for name, p in small_params.items()}
        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
            compute_gradients=MockGradientComputer(shapes),
        )

        data = iter(DataBatchGenerator(num_batches=100))
        stats = await worker.train_loop(data, num_steps=10)

        assert stats["steps"] == 10
        assert stats["avg_loss"] > 0
        assert stats["final_clock"] == 10

    @pytest.mark.asyncio
    async def test_train_loop_with_callback(self, initialized_cluster, small_params):
        """Test training loop with callback."""
        await initialized_cluster.initialize(small_params)

        shapes = {name: p.shape for name, p in small_params.items()}
        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
            compute_gradients=MockGradientComputer(shapes),
        )

        steps_called = []
        def callback(step, loss):
            steps_called.append(step)

        data = iter(DataBatchGenerator(num_batches=100))
        await worker.train_loop(data, num_steps=5, callback=callback)

        assert len(steps_called) == 5
        assert steps_called == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_train_loop_stops_at_data_end(self, initialized_cluster, small_params):
        """Test that loop stops when data is exhausted."""
        await initialized_cluster.initialize(small_params)

        shapes = {name: p.shape for name, p in small_params.items()}
        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
            compute_gradients=MockGradientComputer(shapes),
        )

        # Only 5 batches available
        data = iter(DataBatchGenerator(num_batches=5))
        stats = await worker.train_loop(data, num_steps=100)  # Ask for more

        assert stats["steps"] == 5  # Only did 5

    @pytest.mark.asyncio
    async def test_stop_training(self, initialized_cluster, small_params):
        """Test stopping training manually."""
        await initialized_cluster.initialize(small_params)

        shapes = {name: p.shape for name, p in small_params.items()}
        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
            compute_gradients=MockGradientComputer(shapes),
        )

        # Stop after 3 steps via callback
        def callback(step, loss):
            if step >= 3:
                worker.stop()

        data = iter(DataBatchGenerator(num_batches=100))
        stats = await worker.train_loop(data, callback=callback)

        assert stats["steps"] == 3


class TestWorkerAsyncTrainStep:
    """Tests for async training step."""

    @pytest.mark.asyncio
    async def test_async_train_step(self, initialized_cluster, small_params):
        """Test async training step."""
        await initialized_cluster.initialize(small_params)

        shapes = {name: p.shape for name, p in small_params.items()}
        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
            compute_gradients=MockGradientComputer(shapes),
        )

        batch = {"x": np.random.randn(10, 5)}
        loss = await worker.async_train_step(batch)

        assert loss > 0
        assert worker.info.total_steps == 1


class TestWorkerProperties:
    """Tests for worker properties."""

    @pytest.mark.asyncio
    async def test_avg_loss(self, initialized_cluster, small_params):
        """Test average loss calculation."""
        await initialized_cluster.initialize(small_params)

        shapes = {name: p.shape for name, p in small_params.items()}
        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
            compute_gradients=MockGradientComputer(shapes),
        )

        assert worker.avg_loss == 0.0

        # Do some training
        data = iter(DataBatchGenerator(num_batches=100))
        await worker.train_loop(data, num_steps=10)

        assert worker.avg_loss > 0

    @pytest.mark.asyncio
    async def test_stats(self, initialized_cluster, small_params):
        """Test worker stats."""
        await initialized_cluster.initialize(small_params)

        shapes = {name: p.shape for name, p in small_params.items()}
        worker = Worker(
            worker_id=0,
            ps_cluster=initialized_cluster,
            param_names=list(small_params.keys()),
            compute_gradients=MockGradientComputer(shapes),
        )

        data = iter(DataBatchGenerator(num_batches=100))
        await worker.train_loop(data, num_steps=5)

        stats = worker.stats
        assert stats["worker_id"] == 0
        assert stats["clock"] == 5
        assert stats["total_steps"] == 5
        assert stats["avg_loss"] > 0


class TestMockGradientComputer:
    """Tests for MockGradientComputer."""

    def test_compute_gradients(self, small_params):
        """Test gradient computation."""
        shapes = {name: p.shape for name, p in small_params.items()}
        computer = MockGradientComputer(shapes)

        grads, loss = computer(small_params, {})

        assert len(grads) == len(small_params)
        for name, grad in grads.items():
            assert grad.shape == shapes[name]
        assert loss > 0

    def test_loss_decreases(self, small_params):
        """Test that simulated loss decreases."""
        shapes = {name: p.shape for name, p in small_params.items()}
        computer = MockGradientComputer(shapes)

        losses = []
        for _ in range(10):
            _, loss = computer(small_params, {})
            losses.append(loss)

        # Later losses should be lower
        assert losses[-1] < losses[0]


class TestDataBatchGenerator:
    """Tests for DataBatchGenerator."""

    def test_generate_batches(self):
        """Test batch generation."""
        gen = DataBatchGenerator(num_batches=5, batch_size=32)

        batches = list(gen)
        assert len(batches) == 5
        assert batches[0]["features"].shape == (32, 64)

    def test_reusable(self):
        """Test generator is reusable."""
        gen = DataBatchGenerator(num_batches=3)

        batches1 = list(gen)
        batches2 = list(gen)

        assert len(batches1) == 3
        assert len(batches2) == 3

    def test_custom_dimensions(self):
        """Test custom batch dimensions."""
        gen = DataBatchGenerator(num_batches=1, batch_size=16, feature_dim=128)

        batch = next(iter(gen))
        assert batch["features"].shape == (16, 128)
        assert batch["labels"].shape == (16,)


class TestMultipleWorkers:
    """Tests with multiple workers."""

    @pytest.mark.asyncio
    async def test_multiple_workers_concurrent(self, initialized_cluster, small_params):
        """Test multiple workers training concurrently."""
        import asyncio

        await initialized_cluster.initialize(small_params)
        shapes = {name: p.shape for name, p in small_params.items()}

        workers = []
        for i in range(3):
            worker = Worker(
                worker_id=i,
                ps_cluster=initialized_cluster,
                param_names=list(small_params.keys()),
                compute_gradients=MockGradientComputer(shapes),
            )
            workers.append(worker)

        async def train_worker(worker):
            data = iter(DataBatchGenerator(num_batches=100))
            return await worker.train_loop(data, num_steps=10)

        tasks = [train_worker(w) for w in workers]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert result["steps"] == 10
