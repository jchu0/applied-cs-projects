"""Integration tests for multi-tenant GPU scheduler."""

import pytest
import time
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpusched.core.resources import (
    GPUType, JobState, PriorityClass, GPUResources,
    Cluster, Queue, Tenant, create_node, create_training_job
)
from gpusched.scheduler.scheduler import (
    GPUScheduler, QueueScheduler, PreemptionScheduler
)
from gpusched.allocator.allocator import AllocationManager
from gpusched.monitor.monitor import GPUMonitor


class TestEndToEndScheduling:
    """End-to-end scheduling integration tests."""

    def test_simple_job_scheduling(self):
        """Test scheduling a simple job from submission to completion."""
        # Setup cluster
        cluster = Cluster("test-cluster")

        # Add nodes
        node1 = create_node("host-001", num_gpus=4, gpu_type=GPUType.A100)
        node2 = create_node("host-002", num_gpus=4, gpu_type=GPUType.V100)
        cluster.add_node(node1)
        cluster.add_node(node2)

        # Setup scheduler and allocator
        scheduler = GPUScheduler(cluster)
        allocator = AllocationManager(cluster)

        # Create and submit job
        job = create_training_job(
            name="training-job",
            num_gpus=2,
            gpu_memory_gb=40.0,
            priority=PriorityClass.NORMAL
        )

        assert cluster.submit_job(job) is True
        assert len(cluster.pending_pods) == 1

        # Run scheduling cycle
        decisions = scheduler.run_scheduling_cycle()
        assert len(decisions) == 1
        assert decisions[0].success is True

        # Apply scheduling decision
        pod = cluster.pods[decisions[0].pod_id]
        pod.state = JobState.SCHEDULED
        pod.assigned_node = decisions[0].node_id
        pod.assigned_gpus = decisions[0].gpu_ids

        # Allocate resources
        allocations = allocator.allocate_pod(
            pod,
            decisions[0].node_id,
            decisions[0].gpu_ids
        )
        assert len(allocations) == 2

        # Verify allocation
        assert pod.pod_id in allocator.allocations

        # Start pod
        pod.state = JobState.RUNNING
        pod.started_at = time.time()

        # Complete pod
        pod.state = JobState.COMPLETED
        pod.completed_at = time.time()

        # Release resources
        allocator.release_pod(pod.pod_id)
        assert pod.pod_id not in allocator.allocations

    def test_multi_tenant_scheduling(self):
        """Test fair-share scheduling across multiple tenants."""
        cluster = Cluster("test-cluster")

        # Add nodes
        for i in range(3):
            node = create_node(f"host-{i:03d}", num_gpus=8)
            cluster.add_node(node)

        # Create tenants
        tenant1 = Tenant("tenant-001", "Team A", total_gpu_quota=12)
        tenant2 = Tenant("tenant-002", "Team B", total_gpu_quota=12)
        cluster.tenants["tenant-001"] = tenant1
        cluster.tenants["tenant-002"] = tenant2

        # Create queues
        queue1 = Queue("queue-a", "tenant-001", gpu_quota=12)
        queue2 = Queue("queue-b", "tenant-002", gpu_quota=12)
        cluster.queues["queue-a"] = queue1
        cluster.queues["queue-b"] = queue2

        scheduler = QueueScheduler(cluster)

        # Submit jobs from both tenants
        jobs = []
        for i in range(4):
            job = create_training_job(
                f"job-a-{i}",
                num_gpus=2,
                tenant_id="tenant-001"
            )
            job.queue_name = "queue-a"
            cluster.submit_job(job)
            jobs.append(job)
            for pod in job.pods:
                scheduler.enqueue(pod, "queue-a")

        for i in range(4):
            job = create_training_job(
                f"job-b-{i}",
                num_gpus=2,
                tenant_id="tenant-002"
            )
            job.queue_name = "queue-b"
            cluster.submit_job(job)
            jobs.append(job)
            for pod in job.pods:
                scheduler.enqueue(pod, "queue-b")

        # Run scheduling
        decisions = scheduler.schedule()

        # Should schedule pods from both queues fairly
        assert len(decisions) > 0

        # Track scheduled pods by tenant
        tenant_scheduled = {"tenant-001": 0, "tenant-002": 0}
        for decision in decisions:
            if decision.success:
                pod = cluster.pods[decision.pod_id]
                for job in jobs:
                    if any(p.pod_id == pod.pod_id for p in job.pods):
                        tenant_scheduled[job.tenant_id] += 1
                        break

        # Both tenants should get scheduled
        assert tenant_scheduled["tenant-001"] > 0
        assert tenant_scheduled["tenant-002"] > 0

    def test_gang_scheduling(self):
        """Test gang scheduling for distributed training."""
        cluster = Cluster("test-cluster")

        # Add large node for gang scheduling
        node = create_node("host-001", num_gpus=8)
        cluster.add_node(node)

        scheduler = GPUScheduler(cluster)

        # Create distributed training job requiring gang scheduling
        job = create_training_job(
            "distributed-training",
            num_gpus=2,
            parallelism=4  # 4 pods, 2 GPUs each = 8 GPUs total
        )
        job.gang_schedule = True

        cluster.submit_job(job)

        # Try scheduling
        decisions = scheduler.run_scheduling_cycle()

        # All pods should be scheduled together or none
        success_count = sum(1 for d in decisions if d.success)
        assert success_count == 0 or success_count == 4

        if success_count == 4:
            # All should be on the same node for this test
            node_ids = set(d.node_id for d in decisions)
            assert len(node_ids) == 1

    def test_preemption_scheduling(self):
        """Test preemption of low priority jobs."""
        cluster = Cluster("test-cluster")

        # Add node with limited resources
        node = create_node("host-001", num_gpus=4)
        cluster.add_node(node)

        scheduler = PreemptionScheduler(cluster)
        allocator = AllocationManager(cluster)

        # Submit and schedule low priority jobs
        low_jobs = []
        for i in range(2):
            job = create_training_job(
                f"low-pri-{i}",
                num_gpus=2,
                priority=PriorityClass.LOW
            )
            job.preemptible = True
            cluster.submit_job(job)
            low_jobs.append(job)

            # Manually schedule and allocate
            for pod in job.pods:
                pod.state = JobState.RUNNING
                pod.assigned_node = node.node_id
                gpu_ids = [node.gpus[i*2].gpu_id, node.gpus[i*2+1].gpu_id]
                pod.assigned_gpus = gpu_ids

                for gpu_id in gpu_ids:
                    for gpu in node.gpus:
                        if gpu.gpu_id == gpu_id:
                            gpu.allocated_jobs.append(job.job_id)
                            gpu.available_memory_gb = 0

        # Now try to schedule high priority job
        high_job = create_training_job(
            "high-pri",
            num_gpus=2,
            priority=PriorityClass.CRITICAL
        )
        cluster.submit_job(high_job)

        # Schedule with preemption
        decision = scheduler.schedule_with_preemption(high_job.pods[0])

        assert decision.success is True
        assert "Preempting" in decision.reason

    def test_resource_quota_enforcement(self):
        """Test that resource quotas are enforced."""
        cluster = Cluster("test-cluster")

        # Add nodes
        node = create_node("host-001", num_gpus=8)
        cluster.add_node(node)

        # Create queue with limited quota
        queue = Queue(
            "limited-queue",
            "tenant-001",
            gpu_quota=4,  # Only 4 GPUs allowed
            max_jobs=2    # Only 2 jobs allowed
        )
        cluster.queues["limited-queue"] = queue

        # Try to submit jobs exceeding quota
        jobs_submitted = []
        for i in range(3):
            job = create_training_job(
                f"job-{i}",
                num_gpus=2
            )
            job.queue_name = "limited-queue"

            if cluster.submit_job(job):
                jobs_submitted.append(job)

        # Should only accept 2 jobs due to max_jobs limit
        assert len(jobs_submitted) <= 2

        # Test GPU quota
        queue.max_jobs = 10  # Remove job limit
        queue.gpu_used = 3  # Already using 3 GPUs

        job = create_training_job("quota-test", num_gpus=2)
        job.queue_name = "limited-queue"

        # Should fail - would exceed GPU quota (3 + 2 > 4)
        assert cluster.submit_job(job) is False

    def test_node_affinity_scheduling(self):
        """Test node affinity and anti-affinity rules."""
        cluster = Cluster("test-cluster")

        # Add nodes with different labels
        node1 = create_node("host-001", num_gpus=4)
        node1.labels = {"zone": "us-west", "gpu": "a100"}

        node2 = create_node("host-002", num_gpus=4)
        node2.labels = {"zone": "us-east", "gpu": "v100"}

        cluster.add_node(node1)
        cluster.add_node(node2)

        scheduler = GPUScheduler(cluster)

        # Create job with node selector
        job = create_training_job("affinity-job", num_gpus=2)
        job.pods[0].node_selector = {"zone": "us-west"}

        cluster.submit_job(job)

        # Schedule
        decisions = scheduler.run_scheduling_cycle()

        assert len(decisions) == 1
        assert decisions[0].success is True
        assert decisions[0].node_id == node1.node_id  # Should select node1

    def test_monitoring_integration(self):
        """Test monitoring integration with scheduling."""
        cluster = Cluster("test-cluster")

        # Add nodes
        node = create_node("host-001", num_gpus=4)
        cluster.add_node(node)

        # Setup monitoring
        monitor = GPUMonitor(cluster)

        # Submit and schedule job
        job = create_training_job("monitored-job", num_gpus=2)
        cluster.submit_job(job)

        scheduler = GPUScheduler(cluster)
        decisions = scheduler.run_scheduling_cycle()

        # Update pod state
        if decisions[0].success:
            pod = cluster.pods[decisions[0].pod_id]
            pod.state = JobState.RUNNING
            pod.assigned_node = decisions[0].node_id

            # Simulate GPU utilization
            assigned_node = cluster.nodes[decisions[0].node_id]
            for gpu in assigned_node.gpus[:2]:
                gpu.utilization = 85.0
                gpu.temperature = 72.0
                gpu.allocated_jobs.append(job.job_id)

        # Collect metrics
        monitor.collector.collect_metrics()

        # Check metrics
        status = monitor.get_status()
        assert "cluster_health" in status
        assert "current_metrics" in status

        # Check for alerts
        metrics = monitor.collector.metrics_history[-1]
        alerts = monitor.alert_manager.check_alert_conditions(metrics)

        # May have high utilization alerts
        assert isinstance(alerts, list)

    def test_job_lifecycle(self):
        """Test complete job lifecycle from submission to completion."""
        cluster = Cluster("test-cluster")

        # Add node
        node = create_node("host-001", num_gpus=4)
        cluster.add_node(node)

        # Setup components
        scheduler = GPUScheduler(cluster)
        allocator = AllocationManager(cluster)
        monitor = GPUMonitor(cluster)

        # Submit job - create inside patch so created_at is correct
        with patch('time.time', return_value=1000.0):
            job = create_training_job(
                "lifecycle-job",
                num_gpus=2,
                parallelism=1
            )
            job.max_runtime_seconds = 3600
            cluster.submit_job(job)

        pod = job.pods[0]
        assert pod.state == JobState.PENDING

        # Schedule
        decision = scheduler.schedule_pod(pod)
        assert decision.success is True

        # Transition to scheduled
        pod.state = JobState.SCHEDULED
        pod.assigned_node = decision.node_id
        pod.assigned_gpus = decision.gpu_ids

        # Allocate resources
        allocations = allocator.allocate_pod(
            pod,
            decision.node_id,
            decision.gpu_ids
        )
        assert len(allocations) > 0

        # Start execution
        with patch('time.time', return_value=1100.0):
            pod.state = JobState.RUNNING
            pod.started_at = 1100.0

        # Monitor during execution
        monitor.collector.collect_metrics()
        gpu_stats = monitor.get_gpu_stats()
        assert len(gpu_stats) > 0

        # Complete job
        with patch('time.time', return_value=2000.0):
            pod.state = JobState.COMPLETED
            pod.completed_at = 2000.0

        # Verify completion
        assert job.state == JobState.COMPLETED
        assert pod.wait_time == 100.0  # 1100 - 1000

        # Cleanup
        allocator.release_pod(pod.pod_id)
        assert pod.pod_id not in allocator.allocations

    def test_failure_recovery(self):
        """Test handling of job and node failures."""
        cluster = Cluster("test-cluster")

        # Add nodes
        node1 = create_node("host-001", num_gpus=4)
        node2 = create_node("host-002", num_gpus=4)
        cluster.add_node(node1)
        cluster.add_node(node2)

        scheduler = GPUScheduler(cluster)

        # Submit job
        job = create_training_job("failure-test", num_gpus=2)
        cluster.submit_job(job)

        # Schedule on node1
        decision = scheduler.schedule_pod(job.pods[0])
        assert decision.node_id == node1.node_id

        job.pods[0].state = JobState.RUNNING
        job.pods[0].assigned_node = node1.node_id

        # Simulate node1 failure
        node1.conditions["Ready"] = False

        # Job should be reschedulable
        job.pods[0].state = JobState.FAILED

        # Retry scheduling
        job.pods[0].state = JobState.PENDING
        job.pods[0].assigned_node = None

        decision = scheduler.schedule_pod(job.pods[0])

        # Should schedule on node2 (node1 is not ready)
        assert decision.success is True
        assert decision.node_id == node2.node_id