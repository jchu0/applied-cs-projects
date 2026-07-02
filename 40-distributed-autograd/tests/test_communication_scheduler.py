"""Comprehensive tests for Communication Scheduler and Overlap features (Phase 4)."""

import pytest
import numpy as np
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from typing import List
from concurrent.futures import Future

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distautograd.distributed.ddp import (
    CommunicationScheduler,
    CommunicationStats,
    PipelineScheduler,
    WorkItem,
    GradientBucket,
    Reducer,
)
from distautograd.core.context import ProcessGroup, Backend


# =============================================================================
# Test Fixtures
# =============================================================================

class MockParameter:
    """Mock parameter for testing."""

    def __init__(self, shape, requires_grad=True):
        self.data = np.random.randn(*shape).astype(np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._hooks = []

    def register_hook(self, hook):
        self._hooks.append(hook)
        return len(self._hooks) - 1


@pytest.fixture
def process_group():
    """Create a process group for testing."""
    return ProcessGroup(ranks=[0, 1, 2, 3], backend=Backend.GLOO, name="test_group")


@pytest.fixture
def comm_scheduler():
    """Create a communication scheduler."""
    scheduler = CommunicationScheduler(num_threads=2, enable_stats=True)
    yield scheduler
    scheduler.stop()


@pytest.fixture
def mock_bucket():
    """Create a mock gradient bucket."""
    params = [MockParameter((10, 10))]
    params[0].grad = np.ones((10, 10))
    bucket = GradientBucket(
        index=0,
        params=params,
        size_bytes=400
    )
    return bucket


# =============================================================================
# CommunicationStats Tests
# =============================================================================

class TestCommunicationStats:
    """Tests for CommunicationStats dataclass."""

    def test_stats_creation(self):
        """Test creating communication stats."""
        stats = CommunicationStats()

        assert stats.total_comm_time_ms == 0.0
        assert stats.total_overlap_time_ms == 0.0
        assert stats.total_compute_time_ms == 0.0
        assert stats.num_operations == 0
        assert stats.total_bytes_transferred == 0

    def test_overlap_ratio_zero_division(self):
        """Test overlap ratio with zero comm time."""
        stats = CommunicationStats()
        assert stats.overlap_ratio == 0.0

    def test_overlap_ratio_calculation(self):
        """Test overlap ratio calculation."""
        stats = CommunicationStats(
            total_comm_time_ms=100.0,
            total_overlap_time_ms=70.0
        )
        assert stats.overlap_ratio == 0.7

    def test_bandwidth_calculation(self):
        """Test bandwidth calculation."""
        stats = CommunicationStats(
            total_comm_time_ms=1000.0,  # 1 second
            total_bytes_transferred=1_000_000_000  # 1 GB
        )
        # 1 GB in 1 second = 8 Gbps
        assert abs(stats.bandwidth_gbps - 8.0) < 0.1

    def test_bandwidth_zero_division(self):
        """Test bandwidth with zero time."""
        stats = CommunicationStats()
        assert stats.bandwidth_gbps == 0.0

    def test_avg_bucket_time(self):
        """Test average bucket time calculation."""
        stats = CommunicationStats(
            bucket_times_ms=[10.0, 20.0, 30.0]
        )
        assert stats.avg_bucket_time_ms == 20.0

    def test_avg_bucket_time_empty(self):
        """Test average bucket time with no buckets."""
        stats = CommunicationStats()
        assert stats.avg_bucket_time_ms == 0.0

    def test_reset(self):
        """Test resetting stats."""
        stats = CommunicationStats(
            total_comm_time_ms=100.0,
            total_overlap_time_ms=50.0,
            num_operations=10,
            bucket_times_ms=[1.0, 2.0, 3.0]
        )

        stats.reset()

        assert stats.total_comm_time_ms == 0.0
        assert stats.total_overlap_time_ms == 0.0
        assert stats.num_operations == 0
        assert len(stats.bucket_times_ms) == 0

    def test_to_dict(self):
        """Test converting stats to dictionary."""
        stats = CommunicationStats(
            total_comm_time_ms=100.0,
            total_overlap_time_ms=70.0,
            num_operations=5
        )

        d = stats.to_dict()

        assert d["total_comm_time_ms"] == 100.0
        assert d["total_overlap_time_ms"] == 70.0
        assert d["overlap_ratio"] == 0.7
        assert d["num_operations"] == 5
        assert "bandwidth_gbps" in d
        assert "avg_bucket_time_ms" in d


# =============================================================================
# WorkItem Tests
# =============================================================================

class TestWorkItem:
    """Tests for WorkItem dataclass."""

    def test_work_item_creation(self, mock_bucket):
        """Test creating a work item."""
        work_item = WorkItem(
            bucket=mock_bucket,
            allreduce_fn=lambda x: x,
            priority=1
        )

        assert work_item.bucket == mock_bucket
        assert work_item.priority == 1
        assert work_item.callback is None
        assert work_item.start_time == 0.0

    def test_work_item_with_callback(self, mock_bucket):
        """Test work item with callback."""
        callback = Mock()
        work_item = WorkItem(
            bucket=mock_bucket,
            allreduce_fn=lambda x: x,
            callback=callback,
            start_time=time.perf_counter()
        )

        assert work_item.callback == callback
        assert work_item.start_time > 0


# =============================================================================
# CommunicationScheduler Tests
# =============================================================================

class TestCommunicationScheduler:
    """Tests for CommunicationScheduler class."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = CommunicationScheduler(num_threads=4, enable_stats=True)

        assert scheduler.num_threads == 4
        assert scheduler.enable_stats == True
        assert not scheduler.is_running

        scheduler.stop()

    def test_scheduler_start_stop(self):
        """Test starting and stopping scheduler."""
        scheduler = CommunicationScheduler()

        scheduler.start()
        assert scheduler.is_running

        scheduler.stop()
        assert not scheduler.is_running

    def test_scheduler_context_manager(self):
        """Test scheduler as context manager."""
        with CommunicationScheduler() as scheduler:
            assert scheduler.is_running

        assert not scheduler.is_running

    def test_schedule_allreduce(self, mock_bucket):
        """Test scheduling an all-reduce operation."""
        scheduler = CommunicationScheduler()
        scheduler.start()

        def simple_allreduce(tensor):
            return tensor / 4.0

        future = scheduler.schedule_allreduce(
            mock_bucket,
            simple_allreduce
        )

        assert isinstance(future, Future)
        result = future.result(timeout=5.0)

        scheduler.stop()

    def test_schedule_with_callback(self, mock_bucket):
        """Test scheduling with completion callback."""
        scheduler = CommunicationScheduler()
        scheduler.start()

        callback_called = threading.Event()
        callback_bucket = [None]

        def callback(bucket):
            callback_bucket[0] = bucket
            callback_called.set()

        future = scheduler.schedule_allreduce(
            mock_bucket,
            lambda x: x,
            callback=callback
        )

        future.result(timeout=5.0)
        callback_called.wait(timeout=5.0)

        assert callback_bucket[0] == mock_bucket

        scheduler.stop()

    def test_global_callback(self, mock_bucket):
        """Test global completion callbacks."""
        scheduler = CommunicationScheduler()
        scheduler.start()

        call_count = [0]

        def global_callback(bucket, time_ms):
            call_count[0] += 1

        scheduler.add_callback(global_callback)

        future = scheduler.schedule_allreduce(mock_bucket, lambda x: x)
        future.result(timeout=5.0)

        time.sleep(0.1)  # Allow callback to execute
        assert call_count[0] >= 1

        scheduler.remove_callback(global_callback)
        scheduler.stop()

    def test_wait_all(self, mock_bucket):
        """Test waiting for all operations."""
        scheduler = CommunicationScheduler()
        scheduler.start()

        futures = []
        for i in range(5):
            bucket = GradientBucket(index=i, size_bytes=100)
            bucket.params = [MockParameter((5, 5))]
            bucket.params[0].grad = np.ones((5, 5))

            future = scheduler.schedule_allreduce(bucket, lambda x: x)
            futures.append(future)

        scheduler.wait_all(timeout=10.0)

        # All should be complete
        for future in futures:
            assert future.done()

        scheduler.stop()

    def test_statistics_collection(self, mock_bucket):
        """Test that statistics are collected."""
        scheduler = CommunicationScheduler(enable_stats=True)
        scheduler.start()

        future = scheduler.schedule_allreduce(mock_bucket, lambda x: x)
        future.result(timeout=5.0)

        stats = scheduler.get_stats()

        assert stats.num_operations >= 1
        assert stats.total_bytes_transferred >= 0

        scheduler.stop()

    def test_statistics_reset(self, mock_bucket):
        """Test resetting statistics."""
        scheduler = CommunicationScheduler(enable_stats=True)
        scheduler.start()

        future = scheduler.schedule_allreduce(mock_bucket, lambda x: x)
        future.result(timeout=5.0)

        scheduler.reset_stats()
        stats = scheduler.get_stats()

        assert stats.num_operations == 0

        scheduler.stop()

    def test_backward_timing(self, mock_bucket):
        """Test backward pass timing."""
        scheduler = CommunicationScheduler(enable_stats=True)
        scheduler.start()

        scheduler.mark_backward_start()
        time.sleep(0.01)

        future = scheduler.schedule_allreduce(mock_bucket, lambda x: x)
        future.result(timeout=5.0)

        scheduler.mark_backward_end()

        stats = scheduler.get_stats()
        assert stats.total_compute_time_ms > 0

        scheduler.stop()

    def test_priority_scheduling(self):
        """Test priority-based scheduling."""
        scheduler = CommunicationScheduler(num_threads=1)
        scheduler.start()

        completion_order = []

        def record_completion(bucket):
            completion_order.append(bucket.index)

        buckets = []
        for i in range(3):
            bucket = GradientBucket(index=i, size_bytes=100)
            bucket.params = [MockParameter((5, 5))]
            bucket.params[0].grad = np.ones((5, 5))
            buckets.append(bucket)

        # Schedule with different priorities
        for i, priority in enumerate([2, 0, 1]):  # Out of order
            scheduler.schedule_allreduce(
                buckets[i],
                lambda x: x,
                callback=record_completion,
                priority=priority
            )

        scheduler.wait_all(timeout=10.0)
        time.sleep(0.1)

        # Operations should complete (order depends on thread scheduling)
        assert len(completion_order) == 3

        scheduler.stop()

    def test_auto_start_on_schedule(self, mock_bucket):
        """Test that scheduler auto-starts when scheduling."""
        scheduler = CommunicationScheduler()
        assert not scheduler.is_running

        future = scheduler.schedule_allreduce(mock_bucket, lambda x: x)
        assert scheduler.is_running

        future.result(timeout=5.0)
        scheduler.stop()


# =============================================================================
# PipelineScheduler Tests
# =============================================================================

class TestPipelineScheduler:
    """Tests for PipelineScheduler class."""

    def test_pipeline_scheduler_creation(self, comm_scheduler):
        """Test creating a pipeline scheduler."""
        pipeline = PipelineScheduler(comm_scheduler, num_pipeline_stages=3)

        assert pipeline.num_pipeline_stages == 3
        assert pipeline.comm_scheduler == comm_scheduler

    def test_schedule_bucket(self, comm_scheduler, mock_bucket):
        """Test scheduling a bucket through pipeline."""
        pipeline = PipelineScheduler(comm_scheduler, num_pipeline_stages=2)

        future = pipeline.schedule_bucket(mock_bucket, lambda x: x)

        assert isinstance(future, Future)
        future.result(timeout=5.0)

    def test_pipeline_stages(self, comm_scheduler):
        """Test that pipeline manages stages correctly."""
        pipeline = PipelineScheduler(comm_scheduler, num_pipeline_stages=2)

        buckets = []
        futures = []

        for i in range(4):
            bucket = GradientBucket(index=i, size_bytes=100)
            bucket.params = [MockParameter((5, 5))]
            bucket.params[0].grad = np.ones((5, 5))
            buckets.append(bucket)

            future = pipeline.schedule_bucket(bucket, lambda x: x)
            futures.append(future)

        # Wait for all
        pipeline.flush()

        for future in futures:
            assert future.done()

    def test_flush(self, comm_scheduler, mock_bucket):
        """Test flushing pipeline."""
        pipeline = PipelineScheduler(comm_scheduler, num_pipeline_stages=2)

        futures = []
        for i in range(3):
            bucket = GradientBucket(index=i, size_bytes=100)
            bucket.params = [MockParameter((5, 5))]
            bucket.params[0].grad = np.ones((5, 5))

            future = pipeline.schedule_bucket(bucket, lambda x: x)
            futures.append(future)

        pipeline.flush()

        for future in futures:
            assert future.done()

    def test_bandwidth_utilization(self, comm_scheduler):
        """Test bandwidth utilization tracking."""
        pipeline = PipelineScheduler(comm_scheduler, num_pipeline_stages=2)
        pipeline.set_target_bandwidth(10.0)

        # Schedule some work
        for i in range(3):
            bucket = GradientBucket(index=i, size_bytes=1000000)  # 1MB
            bucket.params = [MockParameter((500, 500))]
            bucket.params[0].grad = np.ones((500, 500))

            pipeline.schedule_bucket(bucket, lambda x: x)

        pipeline.flush()

        # Utilization should be calculable
        util = pipeline.get_bandwidth_utilization()
        assert isinstance(util, float)
        assert 0.0 <= util <= 1.0

    def test_set_target_bandwidth(self, comm_scheduler):
        """Test setting target bandwidth."""
        pipeline = PipelineScheduler(comm_scheduler)
        pipeline.set_target_bandwidth(25.0)

        assert pipeline._target_bandwidth_gbps == 25.0

    def test_wait_for_stage_logs_failure(self, comm_scheduler, caplog):
        """_wait_for_stage must log (not silently swallow) a stage failure.

        A stage whose AllReduce raised should be surfaced via the logger and
        the scheduler should return rather than hang or hide the error.
        """
        import logging
        pipeline = PipelineScheduler(comm_scheduler, num_pipeline_stages=2)

        # Submit a work item that raises so its future carries an exception.
        def boom(_):
            raise RuntimeError("allreduce exploded")

        bucket = GradientBucket(index=0, size_bytes=100)
        bucket.params = [MockParameter((3, 3))]
        bucket.params[0].grad = np.ones((3, 3))

        future = comm_scheduler.schedule_allreduce(bucket, boom)
        # Let the worker thread run the failing job.
        try:
            future.result(timeout=5.0)
        except RuntimeError:
            pass

        pipeline._stage_futures[0] = future

        with caplog.at_level(logging.ERROR):
            # Should not raise, should not hang; should log the failure.
            pipeline._wait_for_stage()

        assert any("Pipeline stage" in rec.message for rec in caplog.records)

    def test_wait_for_stage_completes_normally(self, comm_scheduler):
        """_wait_for_stage returns cleanly when a stage has finished."""
        pipeline = PipelineScheduler(comm_scheduler, num_pipeline_stages=2)

        bucket = GradientBucket(index=0, size_bytes=100)
        bucket.params = [MockParameter((3, 3))]
        bucket.params[0].grad = np.ones((3, 3))

        future = comm_scheduler.schedule_allreduce(bucket, lambda x: x)
        future.result(timeout=5.0)
        pipeline._stage_futures[0] = future

        # A completed, successful future should simply return.
        pipeline._wait_for_stage()


# =============================================================================
# GradientBucket Enhanced Tests
# =============================================================================

class TestGradientBucketEnhanced:
    """Tests for enhanced GradientBucket methods."""

    def test_bucket_flatten(self):
        """Test flattening gradients in bucket."""
        params = [MockParameter((3, 3)), MockParameter((2, 2))]
        params[0].grad = np.ones((3, 3))
        params[1].grad = np.ones((2, 2)) * 2

        bucket = GradientBucket(index=0, params=params, size_bytes=52)

        flat = bucket.flatten()

        assert len(flat) == 9 + 4  # 3*3 + 2*2
        assert bucket.flat_tensor is not None

    def test_bucket_unflatten(self):
        """Test unflattening gradients in bucket."""
        params = [MockParameter((3, 3)), MockParameter((2, 2))]
        params[0].grad = np.ones((3, 3))
        params[1].grad = np.ones((2, 2))

        bucket = GradientBucket(index=0, params=params, size_bytes=52)

        # Flatten
        bucket.flatten()

        # Modify flat tensor
        bucket.flat_tensor *= 5.0

        # Unflatten
        bucket.unflatten()

        np.testing.assert_array_almost_equal(params[0].grad, np.ones((3, 3)) * 5.0)
        np.testing.assert_array_almost_equal(params[1].grad, np.ones((2, 2)) * 5.0)

    def test_bucket_is_last(self):
        """Test is_last flag."""
        bucket = GradientBucket(index=0, is_last=True)
        assert bucket.is_last


# =============================================================================
# Async Reducer Tests
# =============================================================================

class TestAsyncReducer:
    """Tests for async Reducer functionality."""

    def test_reducer_async_mode(self, process_group):
        """Test reducer in async mode."""
        params = [MockParameter((10, 10))]
        params[0].grad = np.ones((10, 10)) * 4

        reducer = Reducer(
            params,
            process_group,
            use_async=True
        )

        assert reducer.use_async
        assert reducer._comm_scheduler is not None

        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)
        reducer.finalize(timeout=5.0)

        # Gradient should be averaged
        np.testing.assert_array_almost_equal(params[0].grad, np.ones((10, 10)))

        reducer.shutdown()

    def test_reducer_with_external_scheduler(self, process_group):
        """Test reducer with external communication scheduler."""
        scheduler = CommunicationScheduler()
        scheduler.start()

        params = [MockParameter((5, 5))]
        params[0].grad = np.ones((5, 5)) * 4

        reducer = Reducer(
            params,
            process_group,
            use_async=True,
            comm_scheduler=scheduler
        )

        assert reducer._comm_scheduler == scheduler
        assert not reducer._owns_scheduler

        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)
        reducer.finalize(timeout=5.0)

        scheduler.stop()

    def test_reducer_get_stats(self, process_group):
        """Test getting stats from reducer."""
        params = [MockParameter((10, 10))]
        params[0].grad = np.ones((10, 10))

        reducer = Reducer(
            params,
            process_group,
            use_async=True
        )

        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)
        reducer.finalize(timeout=5.0)

        stats = reducer.get_stats()
        assert stats is not None
        assert isinstance(stats, CommunicationStats)

        reducer.shutdown()

    def test_reducer_reset_stats(self, process_group):
        """Test resetting reducer stats."""
        params = [MockParameter((10, 10))]
        params[0].grad = np.ones((10, 10))

        reducer = Reducer(
            params,
            process_group,
            use_async=True
        )

        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)
        reducer.finalize(timeout=5.0)

        reducer.reset_stats()
        stats = reducer.get_stats()

        assert stats.num_operations == 0

        reducer.shutdown()

    def test_reducer_bucket_callback(self, process_group):
        """Test bucket completion callbacks."""
        params = [MockParameter((5, 5))]
        params[0].grad = np.ones((5, 5))

        reducer = Reducer(params, process_group)

        completed_buckets = []

        def callback(bucket):
            completed_buckets.append(bucket.index)

        reducer.add_bucket_callback(callback)
        reducer.prepare_for_backward()
        reducer.mark_grad_ready(0)

        assert 0 in completed_buckets

        reducer.remove_bucket_callback(callback)

    def test_reducer_shutdown(self, process_group):
        """Test reducer shutdown."""
        params = [MockParameter((5, 5))]

        reducer = Reducer(
            params,
            process_group,
            use_async=True
        )

        assert reducer._comm_scheduler is not None

        reducer.shutdown()

        assert reducer._comm_scheduler is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase4Integration:
    """Integration tests for Phase 4 features."""

    def test_full_async_training_step(self, process_group):
        """Test a complete async training step."""
        # Create parameters
        params = [
            MockParameter((100, 100)),
            MockParameter((50, 50)),
            MockParameter((25, 25))
        ]

        # Set gradients
        for p in params:
            p.grad = np.random.randn(*p.data.shape).astype(np.float32)

        # Create async reducer
        reducer = Reducer(
            params,
            process_group,
            bucket_cap_mb=0.01,  # Small buckets
            use_async=True
        )

        # Simulate training step
        reducer.prepare_for_backward()

        # Mark gradients ready
        for i in range(len(params)):
            reducer.mark_grad_ready(i)

        # Finalize
        reducer.finalize(timeout=10.0)

        # Check stats
        stats = reducer.get_stats()
        assert stats.num_operations >= 1

        reducer.shutdown()

    def test_pipeline_with_overlap(self, process_group):
        """Test pipeline scheduling with overlap."""
        scheduler = CommunicationScheduler(num_threads=2)
        scheduler.start()

        pipeline = PipelineScheduler(scheduler, num_pipeline_stages=2)

        # Create buckets
        buckets = []
        for i in range(5):
            bucket = GradientBucket(index=i, size_bytes=10000)
            bucket.params = [MockParameter((50, 50))]
            bucket.params[0].grad = np.random.randn(50, 50).astype(np.float32)
            buckets.append(bucket)

        # Schedule with simulated compute overlap
        scheduler.mark_backward_start()

        for bucket in buckets:
            pipeline.schedule_bucket(bucket, lambda x: x / 4.0)
            time.sleep(0.001)  # Simulate compute

        scheduler.mark_backward_end()
        pipeline.flush()

        # Check stats
        stats = scheduler.get_stats()
        assert stats.num_operations == 5
        assert stats.total_compute_time_ms > 0

        scheduler.stop()

    def test_stats_across_multiple_iterations(self, process_group):
        """Test stats accumulation across iterations."""
        params = [MockParameter((20, 20))]

        reducer = Reducer(
            params,
            process_group,
            use_async=True
        )

        # Multiple training iterations
        for _ in range(5):
            params[0].grad = np.ones((20, 20))
            reducer.prepare_for_backward()
            reducer.mark_grad_ready(0)
            reducer.finalize(timeout=5.0)

        stats = reducer.get_stats()
        assert stats.num_operations == 5

        reducer.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
