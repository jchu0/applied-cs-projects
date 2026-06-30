"""Tests for Phase 6: Enterprise Features."""

import numpy as np
import pytest
import time
import tempfile
import os

from distautograd.distributed.ddp import (
    # Backends
    CommunicationBackend,
    BackendConfig,
    BackendRegistry,
    AbstractBackend,
    NCCLBackend,
    GlooBackend,
    MPIBackend,
    SimulatedBackend,
    create_backend,
    # Compression
    CompressionStats,
    GradientCompressor,
    TopKCompressor,
    QuantizedCompressor,
    ErrorFeedbackCompressor,
    PowerSGDCompressor,
    # Adaptive Communication
    NetworkMetrics,
    AdaptiveCommunicator,
    AllReduceStrategy,
    # Elastic Training
    MembershipChange,
    MembershipEvent,
    ElasticTrainer,
    # Enhanced FSDP
    ShardingStrategy,
    FSDPConfig,
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    EnhancedFSDP,
)
from distautograd.core.context import ProcessGroup


# =============================================================================
# Backend Tests
# =============================================================================

class TestCommunicationBackend:
    """Tests for CommunicationBackend enum."""

    def test_backend_values(self):
        assert CommunicationBackend.NCCL.value == "nccl"
        assert CommunicationBackend.GLOO.value == "gloo"
        assert CommunicationBackend.MPI.value == "mpi"
        assert CommunicationBackend.SIMULATED.value == "simulated"

    def test_backend_members(self):
        assert len(CommunicationBackend) == 4


class TestBackendConfig:
    """Tests for BackendConfig dataclass."""

    def test_default_config(self):
        config = BackendConfig(backend_type=CommunicationBackend.GLOO)
        assert config.is_available is True
        assert config.supports_gpu is False
        assert config.supports_cpu is True
        assert config.requires_init is True

    def test_custom_config(self):
        config = BackendConfig(
            backend_type=CommunicationBackend.NCCL,
            is_available=True,
            supports_gpu=True,
            supports_cpu=False
        )
        assert config.supports_gpu is True
        assert config.supports_cpu is False


class TestBackendRegistry:
    """Tests for BackendRegistry."""

    def test_get_backend(self):
        config = BackendRegistry.get_backend("gloo")
        assert config is not None
        assert config.backend_type == CommunicationBackend.GLOO

    def test_get_backend_case_insensitive(self):
        config = BackendRegistry.get_backend("NCCL")
        assert config is not None
        assert config.backend_type == CommunicationBackend.NCCL

    def test_list_available(self):
        available = BackendRegistry.list_available()
        assert "gloo" in available
        assert "simulated" in available

    def test_register_custom_backend(self):
        custom_config = BackendConfig(
            backend_type=CommunicationBackend.GLOO,
            is_available=True
        )
        BackendRegistry.register_backend("custom", custom_config)
        retrieved = BackendRegistry.get_backend("custom")
        assert retrieved is not None


class TestNCCLBackend:
    """Tests for NCCLBackend."""

    def test_creation(self):
        backend = NCCLBackend(rank=0, world_size=4)
        assert backend.rank == 0
        assert backend.world_size == 4
        assert not backend.is_initialized

    def test_initialize(self):
        backend = NCCLBackend(rank=0, world_size=4)
        result = backend.initialize()
        assert result is True
        assert backend.is_initialized

    def test_all_reduce(self):
        backend = NCCLBackend(rank=0, world_size=4)
        backend.initialize()
        tensor = np.array([1.0, 2.0, 3.0])
        result = backend.all_reduce(tensor)
        assert result.shape == tensor.shape

    def test_all_gather(self):
        backend = NCCLBackend(rank=0, world_size=4)
        backend.initialize()
        tensor = np.array([1.0, 2.0])
        result = backend.all_gather(tensor)
        assert len(result) == 4

    def test_broadcast(self):
        backend = NCCLBackend(rank=0, world_size=4)
        backend.initialize()
        tensor = np.array([1.0, 2.0, 3.0])
        result = backend.broadcast(tensor, src=0)
        np.testing.assert_array_equal(result, tensor)

    def test_reduce_scatter(self):
        backend = NCCLBackend(rank=0, world_size=4)
        backend.initialize()
        tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = backend.reduce_scatter(tensor)
        assert len(result) == 2

    def test_not_initialized_error(self):
        backend = NCCLBackend(rank=0, world_size=4)
        with pytest.raises(RuntimeError, match="not initialized"):
            backend.all_reduce(np.array([1.0]))

    def test_shutdown(self):
        backend = NCCLBackend(rank=0, world_size=4)
        backend.initialize()
        backend.shutdown()
        assert not backend.is_initialized


class TestGlooBackend:
    """Tests for GlooBackend."""

    def test_creation(self):
        backend = GlooBackend(rank=1, world_size=2)
        assert backend.rank == 1
        assert backend.world_size == 2

    def test_initialize_and_operations(self):
        backend = GlooBackend(rank=0, world_size=2)
        backend.initialize()
        assert backend.is_initialized

        tensor = np.array([1.0, 2.0])
        result = backend.all_reduce(tensor)
        assert result.shape == tensor.shape

        gathered = backend.all_gather(tensor)
        assert len(gathered) == 2


class TestMPIBackend:
    """Tests for MPIBackend."""

    def test_creation_and_initialize(self):
        backend = MPIBackend(rank=0, world_size=8)
        backend.initialize()
        assert backend.is_initialized

        tensor = np.array([1.0, 2.0, 3.0])
        result = backend.all_reduce(tensor)
        assert result.shape == tensor.shape


class TestSimulatedBackend:
    """Tests for SimulatedBackend."""

    def test_auto_initialized(self):
        backend = SimulatedBackend()
        assert backend.is_initialized

    def test_operations(self):
        backend = SimulatedBackend(rank=0, world_size=4)

        tensor = np.array([1.0, 2.0, 3.0, 4.0])
        reduced = backend.all_reduce(tensor)
        np.testing.assert_array_equal(reduced, tensor)

        gathered = backend.all_gather(tensor)
        assert len(gathered) == 4

        broadcast = backend.broadcast(tensor)
        np.testing.assert_array_equal(broadcast, tensor)

        scatter = backend.reduce_scatter(tensor)
        assert len(scatter) == 1


class TestCreateBackend:
    """Tests for create_backend factory."""

    def test_create_nccl(self):
        backend = create_backend("nccl", 0, 4)
        assert isinstance(backend, NCCLBackend)

    def test_create_gloo(self):
        backend = create_backend("gloo", 0, 4)
        assert isinstance(backend, GlooBackend)

    def test_create_mpi(self):
        backend = create_backend("mpi", 0, 4)
        assert isinstance(backend, MPIBackend)

    def test_create_simulated(self):
        backend = create_backend("simulated", 0, 1)
        assert isinstance(backend, SimulatedBackend)

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("unknown", 0, 4)


# =============================================================================
# Compression Tests
# =============================================================================

class TestCompressionStats:
    """Tests for CompressionStats."""

    def test_default_values(self):
        stats = CompressionStats()
        assert stats.original_size_bytes == 0
        assert stats.compressed_size_bytes == 0
        assert stats.num_compressions == 0

    def test_compression_ratio(self):
        stats = CompressionStats(
            original_size_bytes=1000,
            compressed_size_bytes=100
        )
        assert stats.compression_ratio == 0.1
        assert stats.space_savings == 0.9

    def test_zero_original_size(self):
        stats = CompressionStats()
        assert stats.compression_ratio == 1.0

    def test_avg_compression_time(self):
        stats = CompressionStats(
            num_compressions=10,
            total_compression_time_ms=100.0
        )
        assert stats.avg_compression_time_ms == 10.0

    def test_reset(self):
        stats = CompressionStats(
            original_size_bytes=1000,
            num_compressions=5
        )
        stats.reset()
        assert stats.original_size_bytes == 0
        assert stats.num_compressions == 0

    def test_to_dict(self):
        stats = CompressionStats(
            original_size_bytes=1000,
            compressed_size_bytes=100
        )
        d = stats.to_dict()
        assert "compression_ratio" in d
        assert "space_savings" in d


class TestTopKCompressor:
    """Tests for TopKCompressor."""

    def test_creation(self):
        compressor = TopKCompressor(compression_ratio=0.1)
        assert compressor.compression_ratio == 0.1

    def test_compress_decompress(self):
        compressor = TopKCompressor(compression_ratio=0.1)
        tensor = np.random.randn(100)

        compressed, metadata = compressor.compress(tensor)
        decompressed = compressor.decompress(compressed, metadata)

        assert decompressed.shape == tensor.shape
        # Top 10% should be preserved
        assert metadata["k"] == 10

    def test_compression_stats(self):
        compressor = TopKCompressor(compression_ratio=0.1)
        tensor = np.random.randn(1000)

        compressor.compress(tensor)

        assert compressor.stats.num_compressions == 1
        assert compressor.stats.original_size_bytes > 0
        # Compressed should be smaller (sparse representation)

    def test_preserves_large_values(self):
        compressor = TopKCompressor(compression_ratio=0.1)
        tensor = np.zeros(100)
        tensor[0] = 100.0  # Large value
        tensor[99] = -50.0  # Another large value

        compressed, metadata = compressor.compress(tensor)
        decompressed = compressor.decompress(compressed, metadata)

        # Large values should be preserved
        assert abs(decompressed[0] - 100.0) < 0.01 or abs(decompressed[99] - (-50.0)) < 0.01


class TestQuantizedCompressor:
    """Tests for QuantizedCompressor."""

    def test_creation(self):
        compressor = QuantizedCompressor(bits=8)
        assert compressor.bits == 8
        assert compressor.max_val == 127

    def test_compress_decompress(self):
        compressor = QuantizedCompressor(bits=8)
        tensor = np.random.randn(100)

        compressed, metadata = compressor.compress(tensor)
        decompressed = compressor.decompress(compressed, metadata)

        assert decompressed.shape == tensor.shape
        assert "scale" in metadata

    def test_compression_ratio(self):
        compressor = QuantizedCompressor(bits=8)
        tensor = np.random.randn(1000).astype(np.float64)

        compressed, _ = compressor.compress(tensor)

        # int8 should be 1/8 the size of float64
        assert compressed.nbytes < tensor.nbytes

    def test_accuracy(self):
        compressor = QuantizedCompressor(bits=8)
        tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        compressed, metadata = compressor.compress(tensor)
        decompressed = compressor.decompress(compressed, metadata)

        # Should be reasonably accurate
        np.testing.assert_allclose(decompressed, tensor, rtol=0.1)


class TestErrorFeedbackCompressor:
    """Tests for ErrorFeedbackCompressor."""

    def test_creation(self):
        base = TopKCompressor(compression_ratio=0.1)
        compressor = ErrorFeedbackCompressor(base)
        assert compressor.base_compressor is base

    def test_compress_decompress(self):
        base = TopKCompressor(compression_ratio=0.1)
        compressor = ErrorFeedbackCompressor(base)

        tensor = np.random.randn(100)
        compressed, metadata = compressor.compress(tensor)
        decompressed = compressor.decompress(compressed, metadata)

        assert decompressed.shape == tensor.shape

    def test_error_accumulation(self):
        base = TopKCompressor(compression_ratio=0.1)
        compressor = ErrorFeedbackCompressor(base)

        tensor1 = np.random.randn(100)
        compressor.compress(tensor1)

        # Error should be stored
        assert compressor._error is not None

    def test_reset_error(self):
        base = TopKCompressor(compression_ratio=0.1)
        compressor = ErrorFeedbackCompressor(base)

        tensor = np.random.randn(100)
        compressor.compress(tensor)
        compressor.reset_error()

        assert compressor._error is None


class TestPowerSGDCompressor:
    """Tests for PowerSGDCompressor."""

    def test_creation(self):
        compressor = PowerSGDCompressor(rank=4)
        assert compressor.matrix_rank == 4

    def test_compress_1d(self):
        compressor = PowerSGDCompressor(rank=2)
        tensor = np.random.randn(100)

        compressed, metadata = compressor.compress(tensor)
        decompressed = compressor.decompress(compressed, metadata)

        assert decompressed.shape == tensor.shape

    def test_compress_2d(self):
        compressor = PowerSGDCompressor(rank=4)
        tensor = np.random.randn(50, 100)

        compressed, metadata = compressor.compress(tensor)
        decompressed = compressor.decompress(compressed, metadata)

        assert decompressed.shape == tensor.shape

    def test_low_rank_compression(self):
        compressor = PowerSGDCompressor(rank=2)
        tensor = np.random.randn(100, 100)  # 10000 elements

        compressed, metadata = compressor.compress(tensor)

        # Low-rank should reduce size
        # rank=2: P is 100x2, Q is 100x2 = 400 elements
        assert compressed.size < tensor.size


# =============================================================================
# Adaptive Communication Tests
# =============================================================================

class TestNetworkMetrics:
    """Tests for NetworkMetrics."""

    def test_default_values(self):
        metrics = NetworkMetrics()
        assert metrics.bandwidth_gbps == 10.0
        assert metrics.latency_us == 1.0

    def test_update_bandwidth(self):
        metrics = NetworkMetrics()
        metrics.update_bandwidth(20.0)
        metrics.update_bandwidth(30.0)

        assert len(metrics.bandwidth_history) == 2
        assert metrics.bandwidth_gbps == 25.0

    def test_update_latency(self):
        metrics = NetworkMetrics()
        metrics.update_latency(2.0)
        metrics.update_latency(4.0)

        assert len(metrics.latency_history) == 2
        assert metrics.latency_us == 3.0

    def test_window_size(self):
        metrics = NetworkMetrics()
        for i in range(15):
            metrics.update_bandwidth(float(i))

        assert len(metrics.bandwidth_history) == 10


class TestAdaptiveCommunicator:
    """Tests for AdaptiveCommunicator."""

    def test_creation(self):
        comm = AdaptiveCommunicator()
        assert comm.world_size == 1

    def test_with_process_group(self):
        pg = ProcessGroup(ranks=[0, 1, 2, 3])
        comm = AdaptiveCommunicator(process_group=pg)
        assert comm.world_size == 4

    def test_select_strategy_small_message(self):
        pg = ProcessGroup(ranks=list(range(8)))
        comm = AdaptiveCommunicator(process_group=pg)

        # Small message should prefer tree (lower latency)
        strategy = comm.select_strategy(1000)
        assert strategy in [AllReduceStrategy.TREE, AllReduceStrategy.RING,
                           AllReduceStrategy.RECURSIVE_HALVING]

    def test_select_strategy_large_message(self):
        pg = ProcessGroup(ranks=list(range(8)))
        comm = AdaptiveCommunicator(process_group=pg)

        # Large message should prefer ring (higher bandwidth)
        strategy = comm.select_strategy(100 * 1024 * 1024)
        assert strategy in [AllReduceStrategy.RING, AllReduceStrategy.TREE,
                           AllReduceStrategy.RECURSIVE_HALVING]

    def test_all_reduce(self):
        comm = AdaptiveCommunicator()
        tensor = np.array([1.0, 2.0, 3.0])

        result = comm.all_reduce(tensor)
        np.testing.assert_array_equal(result, tensor)

    def test_estimate_bandwidth(self):
        comm = AdaptiveCommunicator()
        initial_bw = comm.metrics.bandwidth_gbps

        comm.estimate_bandwidth(1_000_000, 0.001)

        # Bandwidth should be updated
        assert len(comm.metrics.bandwidth_history) == 1

    def test_strategy_stats(self):
        comm = AdaptiveCommunicator()

        for _ in range(5):
            comm.select_strategy(1000)

        stats = comm.get_strategy_stats()
        total = sum(stats.values())
        assert total == 5


# =============================================================================
# Elastic Training Tests
# =============================================================================

class TestMembershipChange:
    """Tests for MembershipChange enum."""

    def test_values(self):
        assert MembershipChange.JOIN is not None
        assert MembershipChange.LEAVE is not None
        assert MembershipChange.FAILURE is not None


class TestMembershipEvent:
    """Tests for MembershipEvent dataclass."""

    def test_creation(self):
        event = MembershipEvent(
            change_type=MembershipChange.JOIN,
            rank=3,
            timestamp=time.perf_counter(),
            new_world_size=4
        )
        assert event.rank == 3
        assert event.new_world_size == 4


class TestElasticTrainer:
    """Tests for ElasticTrainer."""

    def test_creation(self):
        trainer = ElasticTrainer(min_workers=1, max_workers=8)
        assert trainer.min_workers == 1
        assert trainer.max_workers == 8

    def test_initialize(self):
        trainer = ElasticTrainer()
        trainer.initialize(rank=0, world_size=4)

        assert trainer.rank == 0
        assert trainer.world_size == 4

    def test_request_join(self):
        trainer = ElasticTrainer(max_workers=8)
        trainer.initialize(rank=0, world_size=4)

        result = trainer.request_join(new_rank=4)
        assert result is True

    def test_request_join_max_reached(self):
        trainer = ElasticTrainer(max_workers=4)
        trainer.initialize(rank=0, world_size=4)

        result = trainer.request_join(new_rank=4)
        assert result is False

    def test_request_leave(self):
        trainer = ElasticTrainer(min_workers=1)
        trainer.initialize(rank=0, world_size=4)

        result = trainer.request_leave(leaving_rank=3)
        assert result is True

    def test_request_leave_min_reached(self):
        trainer = ElasticTrainer(min_workers=4)
        trainer.initialize(rank=0, world_size=4)

        result = trainer.request_leave(leaving_rank=3)
        assert result is False

    def test_handle_failure(self):
        trainer = ElasticTrainer()
        trainer.initialize(rank=0, world_size=4)

        trainer.handle_failure(failed_rank=2)

        assert trainer.world_size == 3
        assert 2 not in trainer._current_workers

    def test_commit_membership_changes(self):
        trainer = ElasticTrainer()
        trainer.initialize(rank=0, world_size=4)

        trainer.request_join(new_rank=4)
        trainer.request_join(new_rank=5)

        changed = trainer.commit_membership_changes()

        assert changed is True
        assert trainer.world_size == 6

    def test_callbacks(self):
        trainer = ElasticTrainer()
        trainer.initialize(rank=0, world_size=4)

        joined_ranks = []
        trainer.add_join_callback(lambda r: joined_ranks.append(r))

        trainer.request_join(new_rank=4)
        trainer.commit_membership_changes()

        assert 4 in joined_ranks

    def test_get_data_shard(self):
        trainer = ElasticTrainer()
        trainer.initialize(rank=1, world_size=4)

        start, end = trainer.get_data_shard(total_samples=1000)

        assert start == 250
        assert end == 500

    def test_get_data_shard_last_worker(self):
        trainer = ElasticTrainer()
        trainer.initialize(rank=3, world_size=4)

        start, end = trainer.get_data_shard(total_samples=1003)

        assert start == 750
        assert end == 1003  # Gets remaining samples

    def test_should_restart_epoch(self):
        trainer = ElasticTrainer()
        trainer.initialize(rank=0, world_size=4)

        # Initially false
        assert trainer.should_restart_epoch() is False

        # After failure (triggers rebalance)
        trainer.handle_failure(failed_rank=2)

        # Should be true briefly
        assert trainer.should_restart_epoch() is True

    def test_membership_history(self):
        trainer = ElasticTrainer()
        trainer.initialize(rank=0, world_size=4)

        trainer.handle_failure(failed_rank=2)
        trainer.request_join(new_rank=4)
        trainer.commit_membership_changes()

        history = trainer.get_membership_history()
        assert len(history) == 2
        assert history[0].change_type == MembershipChange.FAILURE


# =============================================================================
# Enhanced FSDP Tests
# =============================================================================

class TestShardingStrategy:
    """Tests for ShardingStrategy enum."""

    def test_strategies(self):
        assert ShardingStrategy.FULL_SHARD is not None
        assert ShardingStrategy.SHARD_GRAD_OP is not None
        assert ShardingStrategy.NO_SHARD is not None
        assert ShardingStrategy.HYBRID_SHARD is not None


class TestFSDPConfig:
    """Tests for FSDPConfig."""

    def test_defaults(self):
        config = FSDPConfig()
        assert config.sharding_strategy == ShardingStrategy.FULL_SHARD
        assert config.cpu_offload is False
        assert config.mixed_precision is False

    def test_custom_config(self):
        config = FSDPConfig(
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            cpu_offload=True,
            mixed_precision=True
        )
        assert config.cpu_offload is True
        assert config.mixed_precision is True


class TestCPUOffloadPolicy:
    """Tests for CPUOffloadPolicy."""

    def test_creation(self):
        policy = CPUOffloadPolicy()
        assert policy.offload_params is True
        assert policy.offload_grads is True

    def test_offload_and_prefetch(self):
        policy = CPUOffloadPolicy()
        tensor = np.array([1.0, 2.0, 3.0])

        cpu_tensor = policy.offload_to_cpu(tensor, tensor_id=0)
        np.testing.assert_array_equal(cpu_tensor, tensor)

        prefetched = policy.prefetch_to_gpu(tensor_id=0)
        np.testing.assert_array_equal(prefetched, tensor)

    def test_clear_buffer(self):
        policy = CPUOffloadPolicy()
        tensor = np.array([1.0, 2.0])

        policy.offload_to_cpu(tensor, tensor_id=0)
        policy.clear_buffer(tensor_id=0)

        result = policy.prefetch_to_gpu(tensor_id=0)
        assert result is None

    def test_clear_all(self):
        policy = CPUOffloadPolicy()

        for i in range(5):
            policy.offload_to_cpu(np.array([float(i)]), tensor_id=i)

        policy.clear_all()

        for i in range(5):
            assert policy.prefetch_to_gpu(tensor_id=i) is None


class TestMixedPrecisionPolicy:
    """Tests for MixedPrecisionPolicy."""

    def test_creation(self):
        policy = MixedPrecisionPolicy()
        assert policy.param_dtype == "float32"
        assert policy.reduce_dtype == "float16"

    def test_cast_for_compute(self):
        policy = MixedPrecisionPolicy(param_dtype="float32")
        tensor = np.array([1.0, 2.0], dtype=np.float64)

        result = policy.cast_for_compute(tensor)
        assert result.dtype == np.float32

    def test_cast_for_reduce(self):
        policy = MixedPrecisionPolicy(reduce_dtype="float16")
        tensor = np.array([1.0, 2.0], dtype=np.float32)

        result = policy.cast_for_reduce(tensor)
        assert result.dtype == np.float16

    def test_cast_for_storage(self):
        policy = MixedPrecisionPolicy(buffer_dtype="float64")
        tensor = np.array([1.0, 2.0], dtype=np.float32)

        result = policy.cast_for_storage(tensor)
        assert result.dtype == np.float64


class MockModule:
    """Mock module for testing FSDP."""

    def __init__(self, params=None):
        if params is None:
            self._params = [
                type('Param', (), {'data': np.random.randn(100), 'requires_grad': True})(),
                type('Param', (), {'data': np.random.randn(50), 'requires_grad': True})(),
            ]
        else:
            self._params = params

    def parameters(self):
        return iter(self._params)

    def __call__(self, *args, **kwargs):
        return np.array([1.0])

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)


class TestEnhancedFSDP:
    """Tests for EnhancedFSDP."""

    def test_creation(self):
        module = MockModule()
        fsdp = EnhancedFSDP(module)

        assert fsdp.module is module
        assert fsdp.rank == 0
        assert fsdp.world_size == 1

    def test_with_process_group(self):
        module = MockModule()
        pg = ProcessGroup(ranks=[0, 1, 2, 3])
        fsdp = EnhancedFSDP(module, process_group=pg)

        assert fsdp.world_size == 4

    def test_parameter_sharding(self):
        module = MockModule()
        pg = ProcessGroup(ranks=[0, 1, 2, 3])
        fsdp = EnhancedFSDP(module, process_group=pg)

        # Parameters should be sharded
        assert fsdp._local_shard is not None
        assert len(fsdp._local_shard) < len(fsdp._flat_params)

    def test_forward(self):
        module = MockModule()
        fsdp = EnhancedFSDP(module)

        result = fsdp(np.array([1.0]))
        assert result is not None

    def test_cpu_offload(self):
        module = MockModule()
        config = FSDPConfig(cpu_offload=True)
        fsdp = EnhancedFSDP(module, config=config)

        assert fsdp._cpu_offload is not None

    def test_mixed_precision(self):
        module = MockModule()
        config = FSDPConfig(mixed_precision=True)
        fsdp = EnhancedFSDP(module, config=config)

        assert fsdp._mixed_precision is not None

    def test_state_dict(self):
        module = MockModule()
        fsdp = EnhancedFSDP(module)

        state = fsdp.state_dict()

        assert "flat_params" in state
        assert "param_shapes" in state
        assert "config" in state

    def test_load_state_dict(self):
        module = MockModule()
        fsdp = EnhancedFSDP(module)

        state = fsdp.state_dict()

        # Create new FSDP and load state
        module2 = MockModule()
        fsdp2 = EnhancedFSDP(module2)
        fsdp2.load_state_dict(state)

        assert fsdp2._flat_params is not None

    def test_all_gather_params(self):
        module = MockModule()
        pg = ProcessGroup(ranks=[0, 1])
        fsdp = EnhancedFSDP(module, process_group=pg)

        fsdp._all_gather_params()

        assert fsdp._is_gathered is True
        assert fsdp._gathered_params is not None

    def test_free_gathered_params(self):
        module = MockModule()
        fsdp = EnhancedFSDP(module)

        fsdp._all_gather_params()
        fsdp._free_gathered_params()

        assert fsdp._is_gathered is False
        assert fsdp._gathered_params is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase6Integration:
    """Integration tests for Phase 6 features."""

    def test_backend_with_compression(self):
        """Test combining backend with compression."""
        backend = SimulatedBackend(rank=0, world_size=4)
        compressor = TopKCompressor(compression_ratio=0.1)

        tensor = np.random.randn(1000)

        # Compress
        compressed, metadata = compressor.compress(tensor)

        # Reduce (simulated)
        reduced = backend.all_reduce(compressed)

        # Decompress
        result = compressor.decompress(reduced, metadata)

        assert result.shape == tensor.shape

    def test_adaptive_comm_with_elastic(self):
        """Test adaptive communicator with elastic trainer."""
        trainer = ElasticTrainer()
        trainer.initialize(rank=0, world_size=4)

        pg = ProcessGroup(ranks=[0, 1, 2, 3])
        comm = AdaptiveCommunicator(process_group=pg)

        # Do some communications
        for _ in range(5):
            tensor = np.random.randn(1000)
            comm.all_reduce(tensor)

        # Simulate worker failure
        trainer.handle_failure(failed_rank=2)

        # World size changed
        assert trainer.world_size == 3

    def test_fsdp_with_compression(self):
        """Test FSDP with gradient compression."""
        module = MockModule()
        fsdp = EnhancedFSDP(module)
        compressor = QuantizedCompressor(bits=8)

        # Forward
        _ = fsdp(np.array([1.0]))

        # Simulate gradient compression
        if fsdp._flat_params is not None:
            compressed, metadata = compressor.compress(fsdp._flat_params)
            decompressed = compressor.decompress(compressed, metadata)

            assert decompressed.shape == fsdp._flat_params.shape

    def test_full_enterprise_pipeline(self):
        """Test full enterprise features pipeline."""
        # Create backend
        backend = create_backend("simulated", 0, 4)

        # Create elastic trainer
        trainer = ElasticTrainer()
        trainer.initialize(rank=0, world_size=4)

        # Create adaptive communicator
        comm = AdaptiveCommunicator()

        # Create compressor
        compressor = TopKCompressor(compression_ratio=0.1)

        # Simulate training iteration
        for iteration in range(5):
            # Gradient simulation
            gradient = np.random.randn(1000)

            # Compress
            compressed, metadata = compressor.compress(gradient)

            # Select strategy and communicate
            strategy = comm.select_strategy(compressed.nbytes)
            reduced = backend.all_reduce(compressed)

            # Decompress
            result = compressor.decompress(reduced, metadata)

        # Check stats
        assert compressor.stats.num_compressions == 5
        stats = comm.get_strategy_stats()
        assert sum(stats.values()) == 5
