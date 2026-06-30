"""Unit tests for GPU scheduling algorithms."""

import pytest
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpusched.core.resources import (
    GPUType, JobState, PriorityClass, GPUResources,
    GPU, Node, Container, Pod, Job, Queue, Tenant, Cluster,
    create_gpu, create_node, create_training_job
)
from gpusched.scheduler.scheduler import (
    SchedulingDecision, SchedulingContext, SchedulingPlugin,
    NodeAffinityPlugin, GPUResourcePlugin, BinPackingPlugin,
    SpreadingPlugin, FairSharePlugin, GPUScheduler,
    PriorityQueue, QueueScheduler, PreemptionScheduler
)


class TestSchedulingDecision:
    """Tests for SchedulingDecision class."""

    def test_initialization(self):
        """Test decision initialization."""
        decision = SchedulingDecision(
            pod_id="pod-001",
            node_id="node-001",
            gpu_ids=["gpu-1", "gpu-2"],
            success=True,
            reason="Scheduled successfully",
            score=85.5
        )

        assert decision.pod_id == "pod-001"
        assert decision.success is True
        assert decision.score == 85.5
        assert len(decision.gpu_ids) == 2


class TestNodeAffinityPlugin:
    """Tests for NodeAffinityPlugin."""

    def test_filter_node_selector(self):
        """Test filtering based on node selector."""
        plugin = NodeAffinityPlugin()
        cluster = Cluster("cluster-001")

        # Create pod with selector
        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[],
            node_selector={"gpu": "a100", "zone": "us-west"}
        )

        # Create matching node
        node = create_node("host-001")
        node.labels = {"gpu": "a100", "zone": "us-west"}

        ctx = SchedulingContext(cluster=cluster, pod=pod)
        assert plugin.filter(ctx, node) is True

        # Non-matching node
        node.labels = {"gpu": "v100", "zone": "us-west"}
        assert plugin.filter(ctx, node) is False

    def test_filter_tolerations(self):
        """Test filtering based on tolerations."""
        plugin = NodeAffinityPlugin()
        cluster = Cluster("cluster-001")

        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[],
            tolerations=["gpu=NoSchedule"]
        )

        # Node with taint
        node = create_node("host-001")
        node.taints = ["gpu=NoSchedule"]

        ctx = SchedulingContext(cluster=cluster, pod=pod)
        assert plugin.filter(ctx, node) is True

        # Pod without toleration
        pod.tolerations = []
        assert plugin.filter(ctx, node) is False

    def test_score_preferred_affinity(self):
        """Test scoring based on preferred affinity."""
        plugin = NodeAffinityPlugin()
        cluster = Cluster("cluster-001")

        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[],
            affinity={
                "nodeAffinity": {
                    "preferredDuringScheduling": [
                        {
                            "weight": 10,
                            "matchExpressions": [{"key": "gpu"}]
                        }
                    ]
                }
            }
        )

        node = create_node("host-001")
        node.labels = {"gpu": "a100"}

        ctx = SchedulingContext(cluster=cluster, pod=pod)
        score = plugin.score(ctx, node)
        assert score == 10.0


class TestGPUResourcePlugin:
    """Tests for GPUResourcePlugin."""

    def test_filter_gpu_availability(self):
        """Test filtering based on GPU availability."""
        plugin = GPUResourcePlugin()
        cluster = Cluster("cluster-001")

        # Pod requiring 2 GPUs
        container = Container(
            name="main",
            image="app:latest",
            gpu_resources=GPUResources(count=2, memory_gb=16.0)
        )
        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[container]
        )

        # Node with sufficient GPUs
        node = create_node("host-001", num_gpus=4, gpu_memory_gb=32.0)
        ctx = SchedulingContext(cluster=cluster, pod=pod)
        assert plugin.filter(ctx, node) is True

        # Node with insufficient GPUs
        node_small = create_node("host-002", num_gpus=1)
        assert plugin.filter(ctx, node_small) is False

    def test_filter_gpu_type(self):
        """Test filtering based on GPU type."""
        plugin = GPUResourcePlugin()
        cluster = Cluster("cluster-001")

        # Pod requiring specific GPU type
        container = Container(
            name="main",
            image="app:latest",
            gpu_resources=GPUResources(
                count=1,
                memory_gb=16.0,
                gpu_type=GPUType.A100
            )
        )
        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[container]
        )

        # Node with matching GPUs
        node = create_node("host-001", gpu_type=GPUType.A100)
        ctx = SchedulingContext(cluster=cluster, pod=pod)
        assert plugin.filter(ctx, node) is True

        # Node with different GPU type
        node_v100 = create_node("host-002", gpu_type=GPUType.V100)
        assert plugin.filter(ctx, node_v100) is False

    def test_score_utilization(self):
        """Test scoring based on utilization."""
        plugin = GPUResourcePlugin()
        cluster = Cluster("cluster-001")

        container = Container(
            name="main",
            image="app:latest",
            gpu_resources=GPUResources(count=1, memory_gb=20.0)
        )
        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[container]
        )

        node = create_node("host-001", num_gpus=2, gpu_memory_gb=80.0)
        ctx = SchedulingContext(cluster=cluster, pod=pod)

        # Calculate expected score
        total_capacity = 160.0  # 2 * 80
        total_available = 160.0
        utilization_after = (total_capacity - total_available + 20.0) / total_capacity
        expected_score = utilization_after * 100

        score = plugin.score(ctx, node)
        assert score == pytest.approx(expected_score, rel=1e-2)


class TestBinPackingPlugin:
    """Tests for BinPackingPlugin."""

    def test_score_minimal_waste(self):
        """Test bin packing scoring."""
        plugin = BinPackingPlugin()
        cluster = Cluster("cluster-001")

        # Pod requiring 30GB
        container = Container(
            name="main",
            image="app:latest",
            gpu_resources=GPUResources(count=1, memory_gb=30.0)
        )
        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[container]
        )

        # Node with 40GB GPU (less waste)
        node1 = create_node("host-001", num_gpus=1, gpu_memory_gb=40.0)

        # Node with 80GB GPU (more waste)
        node2 = create_node("host-002", num_gpus=1, gpu_memory_gb=80.0)

        ctx = SchedulingContext(cluster=cluster, pod=pod)

        score1 = plugin.score(ctx, node1)
        score2 = plugin.score(ctx, node2)

        # Node1 should have higher score (less waste)
        assert score1 > score2


class TestSpreadingPlugin:
    """Tests for SpreadingPlugin."""

    def test_score_spreading(self):
        """Test spreading scoring."""
        plugin = SpreadingPlugin()
        cluster = Cluster("cluster-001")

        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[]
        )

        # Empty node (preferred for spreading)
        node1 = create_node("host-001", num_gpus=4)

        # Partially used node
        node2 = create_node("host-002", num_gpus=4)
        node2.gpus[0].allocated_jobs = ["job1"]
        node2.gpus[1].allocated_jobs = ["job2"]

        ctx = SchedulingContext(cluster=cluster, pod=pod)

        score1 = plugin.score(ctx, node1)
        score2 = plugin.score(ctx, node2)

        # Node1 should have higher score (less utilized)
        assert score1 > score2
        assert score1 == 100.0  # Fully available
        assert score2 == 50.0   # 50% available


class TestFairSharePlugin:
    """Tests for FairSharePlugin."""

    def test_score_fairness(self):
        """Test fair share scoring."""
        cluster = Cluster("cluster-001")

        # Create tenants
        tenant1 = Tenant("tenant-001", "Team A", total_gpu_quota=100)
        tenant1.gpu_used = 20  # 20% utilization

        tenant2 = Tenant("tenant-002", "Team B", total_gpu_quota=100)
        tenant2.gpu_used = 80  # 80% utilization

        cluster.tenants["tenant-001"] = tenant1
        cluster.tenants["tenant-002"] = tenant2

        plugin = FairSharePlugin(cluster)

        # Job from tenant with lower utilization
        job1 = create_training_job("job1", tenant_id="tenant-001")
        pod1 = job1.pods[0]

        # Job from tenant with higher utilization
        job2 = create_training_job("job2", tenant_id="tenant-002")
        pod2 = job2.pods[0]

        node = create_node("host-001")

        ctx1 = SchedulingContext(cluster=cluster, pod=pod1, job=job1)
        ctx2 = SchedulingContext(cluster=cluster, pod=pod2, job=job2)

        score1 = plugin.score(ctx1, node)
        score2 = plugin.score(ctx2, node)

        # Tenant1 should have higher score (lower utilization)
        assert score1 > score2
        assert score1 == pytest.approx(80.0, rel=1e-6)  # (1 - 0.2) * 100
        assert score2 == pytest.approx(20.0, rel=1e-6)  # (1 - 0.8) * 100


class TestGPUScheduler:
    """Tests for main GPUScheduler."""

    def test_schedule_pod_success(self):
        """Test successful pod scheduling."""
        cluster = Cluster("cluster-001")

        # Add nodes
        node1 = create_node("host-001", num_gpus=4, gpu_type=GPUType.A100)
        node2 = create_node("host-002", num_gpus=2, gpu_type=GPUType.V100)
        cluster.add_node(node1)
        cluster.add_node(node2)

        scheduler = GPUScheduler(cluster)

        # Create pod requiring A100
        container = Container(
            name="main",
            image="app:latest",
            gpu_resources=GPUResources(
                count=2,
                memory_gb=40.0,
                gpu_type=GPUType.A100
            )
        )
        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[container]
        )

        decision = scheduler.schedule_pod(pod)

        assert decision.success is True
        assert decision.node_id == node1.node_id
        assert len(decision.gpu_ids) == 2

    def test_schedule_pod_no_resources(self):
        """Test scheduling when no resources available."""
        cluster = Cluster("cluster-001")

        # Add small node
        node = create_node("host-001", num_gpus=1, gpu_memory_gb=16.0)
        cluster.add_node(node)

        scheduler = GPUScheduler(cluster)

        # Pod requiring more resources than available
        container = Container(
            name="main",
            image="app:latest",
            gpu_resources=GPUResources(count=4, memory_gb=80.0)
        )
        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[container]
        )

        decision = scheduler.schedule_pod(pod)

        assert decision.success is False
        assert decision.reason == "No feasible nodes"

    def test_schedule_gang(self):
        """Test gang scheduling."""
        cluster = Cluster("cluster-001")

        # Add large node
        node = create_node("host-001", num_gpus=8)
        cluster.add_node(node)

        scheduler = GPUScheduler(cluster)

        # Create gang job with 3 pods
        job = create_training_job("gang-job", num_gpus=2, parallelism=3)
        job.gang_schedule = True

        decisions = scheduler.schedule_gang(job)

        assert len(decisions) == 3
        assert all(d.success for d in decisions)

    def test_run_scheduling_cycle(self):
        """Test full scheduling cycle."""
        cluster = Cluster("cluster-001")

        # Add nodes
        node1 = create_node("host-001", num_gpus=4)
        node2 = create_node("host-002", num_gpus=4)
        cluster.add_node(node1)
        cluster.add_node(node2)

        # Submit jobs
        job1 = create_training_job("job1", num_gpus=1, priority=PriorityClass.HIGH)
        job2 = create_training_job("job2", num_gpus=2, priority=PriorityClass.NORMAL)

        cluster.submit_job(job1)
        cluster.submit_job(job2)

        scheduler = GPUScheduler(cluster)
        decisions = scheduler.run_scheduling_cycle()

        # High priority job should be scheduled first
        assert len(decisions) == 2
        assert decisions[0].pod_id == job1.pods[0].pod_id
        assert decisions[0].success is True


class TestPriorityQueue:
    """Tests for PriorityQueue."""

    def test_push_pop(self):
        """Test priority queue operations."""
        pq = PriorityQueue()

        # Add pods with different priorities
        pq.push("pod-1", priority=1, timestamp=100.0)
        pq.push("pod-2", priority=3, timestamp=101.0)
        pq.push("pod-3", priority=2, timestamp=102.0)

        # Should pop in priority order (highest first)
        assert pq.pop() == "pod-2"  # Priority 3
        assert pq.pop() == "pod-3"  # Priority 2
        assert pq.pop() == "pod-1"  # Priority 1
        assert pq.pop() is None

    def test_remove(self):
        """Test removing from queue."""
        pq = PriorityQueue()

        pq.push("pod-1", priority=1, timestamp=100.0)
        pq.push("pod-2", priority=2, timestamp=101.0)

        assert len(pq) == 2

        pq.remove("pod-1")
        assert len(pq) == 1
        assert pq.pop() == "pod-2"


class TestQueueScheduler:
    """Tests for QueueScheduler."""

    def test_enqueue(self):
        """Test enqueueing pods."""
        cluster = Cluster("cluster-001")
        scheduler = QueueScheduler(cluster)

        pod = Pod(
            pod_id="pod-001",
            name="test-pod",
            namespace="default",
            containers=[],
            priority=PriorityClass.HIGH
        )

        assert scheduler.enqueue(pod, "default") is True
        assert "default" in scheduler.queue_priorities
        assert len(scheduler.queue_priorities["default"]) == 1

    def test_fair_share_scheduling(self):
        """Test fair share across queues."""
        cluster = Cluster("cluster-001")

        # Add node
        node = create_node("host-001", num_gpus=8)
        cluster.add_node(node)

        # Create queues with different weights
        queue1 = Queue("high", "tenant-001", priority_weight=2.0)
        queue2 = Queue("low", "tenant-002", priority_weight=1.0)
        cluster.queues["high"] = queue1
        cluster.queues["low"] = queue2

        scheduler = QueueScheduler(cluster)

        # Add pods to queues
        for i in range(6):
            pod = Pod(
                pod_id=f"pod-high-{i}",
                name=f"high-{i}",
                namespace="default",
                containers=[Container(
                    name="main",
                    image="app",
                    gpu_resources=GPUResources(count=1)
                )]
            )
            cluster.pods[pod.pod_id] = pod
            scheduler.enqueue(pod, "high")

        for i in range(3):
            pod = Pod(
                pod_id=f"pod-low-{i}",
                name=f"low-{i}",
                namespace="default",
                containers=[Container(
                    name="main",
                    image="app",
                    gpu_resources=GPUResources(count=1)
                )]
            )
            cluster.pods[pod.pod_id] = pod
            scheduler.enqueue(pod, "low")

        decisions = scheduler.schedule()

        # Should schedule proportionally to weights
        assert len(decisions) > 0


class TestPreemptionScheduler:
    """Tests for PreemptionScheduler."""

    def test_find_preemption_candidates(self):
        """Test finding preemption candidates."""
        cluster = Cluster("cluster-001")

        # Create node with running jobs
        node = create_node("host-001", num_gpus=4)

        # Add low priority running job
        low_job = create_training_job(
            "low-job",
            num_gpus=2,
            priority=PriorityClass.LOW,
            tenant_id="tenant-001"
        )
        low_job.preemptible = True
        low_job.pods[0].state = JobState.RUNNING
        node.gpus[0].allocated_jobs.append(low_job.job_id)

        cluster.add_node(node)
        cluster.jobs[low_job.job_id] = low_job

        scheduler = PreemptionScheduler(cluster)

        # High priority pod needing resources
        high_pod = Pod(
            pod_id="high-pod",
            name="high",
            namespace="default",
            containers=[],
            priority=PriorityClass.HIGH
        )

        candidates = scheduler.find_preemption_candidates(high_pod, node)

        assert len(candidates) == 1
        assert candidates[0].pod_id == low_job.pods[0].pod_id

    def test_schedule_with_preemption(self):
        """Test scheduling with preemption."""
        cluster = Cluster("cluster-001")

        # Create fully occupied node
        node = create_node("host-001", num_gpus=2)
        cluster.add_node(node)

        # Add low priority running job using all GPUs
        low_job = create_training_job(
            "low-job",
            num_gpus=2,
            priority=PriorityClass.LOW
        )
        low_job.preemptible = True
        low_job.pods[0].state = JobState.RUNNING
        low_job.pods[0].assigned_node = node.node_id

        for gpu in node.gpus:
            gpu.allocated_jobs.append(low_job.job_id)
            gpu.available_memory_gb = 0

        cluster.jobs[low_job.job_id] = low_job

        scheduler = PreemptionScheduler(cluster)

        # High priority pod needing resources
        container = Container(
            name="main",
            image="app",
            gpu_resources=GPUResources(count=1, memory_gb=40.0)
        )
        high_pod = Pod(
            pod_id="high-pod",
            name="high",
            namespace="default",
            containers=[container],
            priority=PriorityClass.HIGH
        )

        decision = scheduler.schedule_with_preemption(high_pod)

        assert decision.success is True
        assert "Preempting" in decision.reason