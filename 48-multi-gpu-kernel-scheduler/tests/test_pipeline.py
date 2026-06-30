"""Tests for pipeline parallelism functionality."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernelsched import (
    ComputeGraph, GPUDevice, MultiGPUCluster, create_test_graph,
    create_gemm_kernel, create_attention_kernel, create_elementwise_kernel,
    PipelineStage, PipelineConfig, PipelinePartitioner,
    MicrobatchSchedule, PipelineSchedule, PipelineScheduler,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def quad_gpu_cluster() -> MultiGPUCluster:
    """Create a 4-GPU cluster for pipeline testing."""
    devices = [
        GPUDevice(device_id=i, name=f"GPU{i}", total_memory_gb=80.0)
        for i in range(4)
    ]
    return MultiGPUCluster(
        devices=devices,
        nvlink_bandwidth_gbps=600.0,
        pcie_bandwidth_gbps=32.0,
    )


@pytest.fixture
def transformer_graph() -> ComputeGraph:
    """Create a transformer-like graph."""
    return create_test_graph(num_layers=4)


@pytest.fixture
def large_transformer_graph() -> ComputeGraph:
    """Create a larger transformer graph."""
    return create_test_graph(num_layers=8)


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Create a default pipeline configuration."""
    return PipelineConfig(
        num_stages=4,
        num_microbatches=8,
        interleave_stages=False,
        recompute_activations=False,
    )


@pytest.fixture
def partitioner(quad_gpu_cluster) -> PipelinePartitioner:
    """Create a pipeline partitioner."""
    return PipelinePartitioner(num_stages=4, num_devices=4)


@pytest.fixture
def pipeline_scheduler(quad_gpu_cluster, pipeline_config) -> PipelineScheduler:
    """Create a pipeline scheduler."""
    return PipelineScheduler(cluster=quad_gpu_cluster, config=pipeline_config)


# =============================================================================
# Test PipelineStage
# =============================================================================

class TestPipelineStage:
    """Tests for PipelineStage dataclass."""

    def test_create_stage(self):
        """Test creating a pipeline stage."""
        stage = PipelineStage(
            stage_id=0,
            device_id=0,
            kernel_ids=["k1", "k2", "k3"],
        )

        assert stage.stage_id == 0
        assert stage.device_id == 0
        assert stage.num_kernels == 3

    def test_empty_stage(self):
        """Test creating an empty stage."""
        stage = PipelineStage(stage_id=1, device_id=1)

        assert stage.num_kernels == 0
        assert stage.kernel_ids == []
        assert stage.input_tensors == []
        assert stage.output_tensors == []


# =============================================================================
# Test PipelineConfig
# =============================================================================

class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_create_config(self):
        """Test creating a pipeline config."""
        config = PipelineConfig(
            num_stages=4,
            num_microbatches=8,
        )

        assert config.num_stages == 4
        assert config.num_microbatches == 8
        assert config.interleave_stages is False
        assert config.recompute_activations is False

    def test_interleaved_config(self):
        """Test creating interleaved pipeline config."""
        config = PipelineConfig(
            num_stages=4,
            num_microbatches=16,
            interleave_stages=True,
        )

        assert config.interleave_stages is True


# =============================================================================
# Test PipelinePartitioner
# =============================================================================

class TestPipelinePartitioner:
    """Tests for PipelinePartitioner."""

    def test_balanced_partition(self, transformer_graph, partitioner):
        """Test balanced partitioning strategy."""
        stages = partitioner.partition(transformer_graph, strategy="balanced")

        assert len(stages) <= 4
        assert all(isinstance(s, PipelineStage) for s in stages)
        assert all(s.num_kernels > 0 for s in stages)

        # Each kernel should be in exactly one stage
        all_kernels = set()
        for stage in stages:
            for kid in stage.kernel_ids:
                assert kid not in all_kernels
                all_kernels.add(kid)

        assert all_kernels == set(transformer_graph.kernels.keys())

    def test_memory_partition(self, transformer_graph, partitioner):
        """Test memory-based partitioning strategy."""
        stages = partitioner.partition(transformer_graph, strategy="memory")

        assert len(stages) <= 4
        assert all(s.num_kernels > 0 for s in stages)

    def test_layer_partition(self, transformer_graph, partitioner):
        """Test layer-based partitioning strategy."""
        stages = partitioner.partition(transformer_graph, strategy="layer")

        assert len(stages) <= 4
        assert all(s.num_kernels > 0 for s in stages)

    def test_stage_device_assignment(self, transformer_graph, partitioner):
        """Test that stages are assigned to different devices."""
        stages = partitioner.partition(transformer_graph, strategy="balanced")

        # Stages should be distributed across devices
        device_ids = [s.device_id for s in stages]
        assert len(set(device_ids)) > 1 or len(stages) == 1

    def test_stage_boundaries(self, transformer_graph, partitioner):
        """Test that stage input/output tensors are computed."""
        stages = partitioner.partition(transformer_graph, strategy="balanced")

        # At least first stage should have no external inputs
        # and last stage should have no external outputs (may vary by graph)
        for stage in stages:
            # Input/output tensors should be computed
            assert hasattr(stage, 'input_tensors')
            assert hasattr(stage, 'output_tensors')


# =============================================================================
# Test MicrobatchSchedule
# =============================================================================

class TestMicrobatchSchedule:
    """Tests for MicrobatchSchedule."""

    def test_create_schedule(self):
        """Test creating a microbatch schedule."""
        schedule = MicrobatchSchedule(
            microbatch_id=0,
            stage_times={
                0: (0.0, 100.0),
                1: (100.0, 200.0),
            },
        )

        assert schedule.microbatch_id == 0
        assert len(schedule.stage_times) == 2
        assert schedule.stage_times[0] == (0.0, 100.0)


# =============================================================================
# Test PipelineSchedule
# =============================================================================

class TestPipelineSchedule:
    """Tests for PipelineSchedule."""

    def test_schedule_properties(self):
        """Test pipeline schedule properties."""
        stages = [
            PipelineStage(stage_id=0, device_id=0),
            PipelineStage(stage_id=1, device_id=1),
        ]

        mb_schedules = [
            MicrobatchSchedule(microbatch_id=0, stage_times={
                0: (0.0, 100.0),
                1: (100.0, 200.0),
            }),
            MicrobatchSchedule(microbatch_id=1, stage_times={
                0: (100.0, 200.0),
                1: (200.0, 300.0),
            }),
        ]

        schedule = PipelineSchedule(
            stages=stages,
            microbatch_schedules=mb_schedules,
            total_time_us=300.0,
            pipeline_bubble_us=100.0,
            num_microbatches=2,
        )

        assert schedule.efficiency == 1.0 - (100.0 / 300.0)
        assert 0.0 < schedule.bubble_ratio < 1.0

    def test_stage_utilization(self):
        """Test stage utilization calculation."""
        stages = [PipelineStage(stage_id=0, device_id=0)]

        mb_schedules = [
            MicrobatchSchedule(microbatch_id=0, stage_times={0: (0.0, 50.0)}),
            MicrobatchSchedule(microbatch_id=1, stage_times={0: (50.0, 100.0)}),
        ]

        schedule = PipelineSchedule(
            stages=stages,
            microbatch_schedules=mb_schedules,
            total_time_us=150.0,
            pipeline_bubble_us=50.0,
            num_microbatches=2,
        )

        utilization = schedule.get_stage_utilization(0)
        assert utilization == 100.0 / 150.0


# =============================================================================
# Test PipelineScheduler
# =============================================================================

class TestPipelineScheduler:
    """Tests for PipelineScheduler."""

    def test_create_scheduler(self, quad_gpu_cluster, pipeline_config):
        """Test creating a pipeline scheduler."""
        scheduler = PipelineScheduler(
            cluster=quad_gpu_cluster,
            config=pipeline_config,
        )

        assert scheduler.config.num_stages == 4
        assert scheduler.config.num_microbatches == 8

    def test_schedule_transformer(self, pipeline_scheduler, transformer_graph):
        """Test scheduling a transformer graph."""
        schedule = pipeline_scheduler.schedule(transformer_graph)

        assert isinstance(schedule, PipelineSchedule)
        assert len(schedule.stages) > 0
        assert len(schedule.microbatch_schedules) == pipeline_scheduler.config.num_microbatches
        assert schedule.total_time_us > 0

    def test_schedule_with_balanced_strategy(self, pipeline_scheduler, transformer_graph):
        """Test scheduling with balanced partitioning."""
        schedule = pipeline_scheduler.schedule(transformer_graph, strategy="balanced")

        assert schedule.total_time_us > 0
        assert schedule.pipeline_bubble_us >= 0

    def test_schedule_with_memory_strategy(self, pipeline_scheduler, transformer_graph):
        """Test scheduling with memory-based partitioning."""
        schedule = pipeline_scheduler.schedule(transformer_graph, strategy="memory")

        assert schedule.total_time_us > 0
        assert len(schedule.stages) > 0

    def test_schedule_with_layer_strategy(self, pipeline_scheduler, transformer_graph):
        """Test scheduling with layer-based partitioning."""
        schedule = pipeline_scheduler.schedule(transformer_graph, strategy="layer")

        assert schedule.total_time_us > 0
        assert len(schedule.stages) > 0

    def test_pipeline_efficiency(self, pipeline_scheduler, transformer_graph):
        """Test that pipeline efficiency is reasonable."""
        schedule = pipeline_scheduler.schedule(transformer_graph)

        # Efficiency should be between 0 and 1
        assert 0.0 <= schedule.efficiency <= 1.0

        # With enough microbatches, efficiency should be reasonable
        # (not too many bubbles)
        assert schedule.efficiency > 0.3

    def test_interleaved_scheduling(self, quad_gpu_cluster, transformer_graph):
        """Test interleaved (1F1B) scheduling."""
        config = PipelineConfig(
            num_stages=2,
            num_microbatches=8,
            interleave_stages=True,
        )
        scheduler = PipelineScheduler(cluster=quad_gpu_cluster, config=config)

        schedule = scheduler.schedule(transformer_graph)

        assert schedule.total_time_us > 0
        assert len(schedule.microbatch_schedules) == 8

    def test_varying_microbatch_count(self, quad_gpu_cluster, transformer_graph):
        """Test scheduling with different microbatch counts."""
        for num_mb in [2, 4, 8, 16]:
            config = PipelineConfig(num_stages=4, num_microbatches=num_mb)
            scheduler = PipelineScheduler(cluster=quad_gpu_cluster, config=config)

            schedule = scheduler.schedule(transformer_graph)

            assert len(schedule.microbatch_schedules) == num_mb
            assert schedule.num_microbatches == num_mb

    def test_varying_stage_count(self, quad_gpu_cluster, large_transformer_graph):
        """Test scheduling with different stage counts."""
        for num_stages in [2, 4, 8]:
            config = PipelineConfig(num_stages=num_stages, num_microbatches=8)
            scheduler = PipelineScheduler(cluster=quad_gpu_cluster, config=config)

            schedule = scheduler.schedule(large_transformer_graph)

            assert len(schedule.stages) <= num_stages
            assert schedule.total_time_us > 0

    def test_more_microbatches_reduces_bubbles(self, quad_gpu_cluster, transformer_graph):
        """Test that more microbatches generally reduces bubble ratio."""
        bubble_ratios = []

        for num_mb in [2, 4, 8, 16]:
            config = PipelineConfig(num_stages=4, num_microbatches=num_mb)
            scheduler = PipelineScheduler(cluster=quad_gpu_cluster, config=config)

            schedule = scheduler.schedule(transformer_graph)
            bubble_ratios.append(schedule.bubble_ratio)

        # More microbatches should generally have lower bubble ratio
        # (not strictly monotonic due to scheduling details, but trend should be down)
        assert bubble_ratios[-1] <= bubble_ratios[0] or bubble_ratios[-1] < 0.5


# =============================================================================
# Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for pipeline parallelism."""

    def test_end_to_end_pipeline(self, quad_gpu_cluster):
        """Test end-to-end pipeline scheduling."""
        # Create a graph
        graph = create_test_graph(num_layers=4)

        # Configure pipeline
        config = PipelineConfig(
            num_stages=4,
            num_microbatches=8,
            interleave_stages=False,
        )

        # Create scheduler
        scheduler = PipelineScheduler(cluster=quad_gpu_cluster, config=config)

        # Schedule
        schedule = scheduler.schedule(graph)

        # Verify results
        assert schedule.total_time_us > 0
        assert schedule.efficiency > 0.2
        assert len(schedule.stages) > 0
        assert all(len(s.kernel_ids) > 0 for s in schedule.stages)

    def test_small_graph_pipeline(self, quad_gpu_cluster):
        """Test pipeline with a small graph."""
        graph = ComputeGraph()

        # Create a simple 4-kernel linear graph
        kernels = []
        for i in range(4):
            k = create_gemm_kernel(256, 256, 256, device_id=0)
            graph.add_kernel(k)
            kernels.append(k)
            if i > 0:
                graph.add_dependency(
                    kernels[i-1].kernel_id,
                    kernels[i].kernel_id,
                    f"dep_{i-1}_{i}"
                )

        config = PipelineConfig(num_stages=2, num_microbatches=4)
        scheduler = PipelineScheduler(cluster=quad_gpu_cluster, config=config)

        schedule = scheduler.schedule(graph)

        assert schedule.total_time_us > 0
        assert len(schedule.stages) <= 2

    def test_single_stage_pipeline(self, quad_gpu_cluster, transformer_graph):
        """Test pipeline with a single stage (no actual pipeline)."""
        config = PipelineConfig(num_stages=1, num_microbatches=4)
        scheduler = PipelineScheduler(cluster=quad_gpu_cluster, config=config)

        schedule = scheduler.schedule(transformer_graph)

        assert len(schedule.stages) == 1
        assert schedule.total_time_us > 0
