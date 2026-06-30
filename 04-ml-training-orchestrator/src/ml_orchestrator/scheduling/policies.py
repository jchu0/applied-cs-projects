"""Scheduling policies for job scheduling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import structlog

from ml_orchestrator.core.models import (
    JobPriority,
    ResourceQuota,
    ResourceRequest,
    TrainingJob,
    WorkerInfo,
)


logger = structlog.get_logger(__name__)


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision."""

    job_id: str
    assigned_workers: list[str]
    should_preempt: list[str]  # Job IDs to preempt
    reason: str
    score: float = 0.0


class SchedulingPolicy(ABC):
    """Base class for scheduling policies."""

    @abstractmethod
    async def select_job(
        self,
        pending_jobs: list[TrainingJob],
        available_resources: dict[str, ResourceRequest],
        quotas: dict[str, ResourceQuota],
    ) -> Optional[SchedulingDecision]:
        """
        Select the next job to schedule.

        Args:
            pending_jobs: Jobs waiting to be scheduled
            available_resources: Available resources per worker
            quotas: User/team quotas

        Returns:
            SchedulingDecision if a job can be scheduled, None otherwise
        """
        pass

    def _find_workers_for_job(
        self,
        job: TrainingJob,
        available_resources: dict[str, ResourceRequest],
    ) -> list[str]:
        """Find workers that can run the job."""
        suitable_workers = []
        for worker_id, resources in available_resources.items():
            if job.resources.fits_in(resources):
                suitable_workers.append(worker_id)
        return suitable_workers

    def _check_quota(
        self,
        job: TrainingJob,
        quotas: dict[str, ResourceQuota],
    ) -> bool:
        """Check if job fits within user/team quota."""
        user_quota = quotas.get(job.user_id)
        if user_quota and not user_quota.can_allocate(job.resources):
            return False

        if job.team_id:
            team_quota = quotas.get(job.team_id)
            if team_quota and not team_quota.can_allocate(job.resources):
                return False

        return True


class FIFOPolicy(SchedulingPolicy):
    """First-In-First-Out scheduling policy."""

    async def select_job(
        self,
        pending_jobs: list[TrainingJob],
        available_resources: dict[str, ResourceRequest],
        quotas: dict[str, ResourceQuota],
    ) -> Optional[SchedulingDecision]:
        # Sort by queue time
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: j.queued_at or j.created_at,
        )

        for job in sorted_jobs:
            if not self._check_quota(job, quotas):
                continue

            workers = self._find_workers_for_job(job, available_resources)
            if workers:
                return SchedulingDecision(
                    job_id=job.id,
                    assigned_workers=[workers[0]],  # Single worker for now
                    should_preempt=[],
                    reason="FIFO: First queued job with available resources",
                )

        return None


class PriorityPolicy(SchedulingPolicy):
    """Priority-based scheduling with aging to prevent starvation."""

    def __init__(self, aging_factor: float = 0.5, max_age_hours: float = 24.0):
        self._aging_factor = aging_factor
        self._max_age_hours = max_age_hours

    def _calculate_score(self, job: TrainingJob) -> float:
        """Calculate job score based on priority and wait time."""
        base_score = job.priority.value

        # Add aging bonus
        queue_time = job.queued_at or job.created_at
        age_hours = min(
            (datetime.utcnow() - queue_time).total_seconds() / 3600,
            self._max_age_hours,
        )
        age_bonus = age_hours * self._aging_factor

        # User priority boost (from quota)
        priority_boost = 0

        return base_score + age_bonus + priority_boost

    async def select_job(
        self,
        pending_jobs: list[TrainingJob],
        available_resources: dict[str, ResourceRequest],
        quotas: dict[str, ResourceQuota],
    ) -> Optional[SchedulingDecision]:
        # Score and sort jobs
        scored_jobs = [
            (self._calculate_score(job), job) for job in pending_jobs
        ]
        scored_jobs.sort(key=lambda x: x[0], reverse=True)

        for score, job in scored_jobs:
            if not self._check_quota(job, quotas):
                continue

            workers = self._find_workers_for_job(job, available_resources)
            if workers:
                return SchedulingDecision(
                    job_id=job.id,
                    assigned_workers=[workers[0]],
                    should_preempt=[],
                    reason=f"Priority: score={score:.2f}",
                    score=score,
                )

        return None


class FairSharePolicy(SchedulingPolicy):
    """Fair share scheduling ensuring equitable resource distribution."""

    def __init__(self, share_weights: Optional[dict[str, float]] = None):
        self._share_weights = share_weights or {}
        self._usage_history: dict[str, float] = {}  # Track historical usage

    def _get_share(self, entity_id: str) -> float:
        """Get the fair share weight for an entity."""
        return self._share_weights.get(entity_id, 1.0)

    def _get_usage(self, entity_id: str) -> float:
        """Get current resource usage for an entity."""
        return self._usage_history.get(entity_id, 0.0)

    def _calculate_fair_score(self, job: TrainingJob) -> float:
        """Calculate fair share score (lower usage = higher score)."""
        entity_id = job.team_id or job.user_id
        share = self._get_share(entity_id)
        usage = self._get_usage(entity_id)

        # Score is share divided by usage (higher share, lower usage = higher score)
        if usage < 0.001:
            return share * 1000  # Boost for entities with no usage
        return share / usage

    async def select_job(
        self,
        pending_jobs: list[TrainingJob],
        available_resources: dict[str, ResourceRequest],
        quotas: dict[str, ResourceQuota],
    ) -> Optional[SchedulingDecision]:
        # Score jobs by fair share
        scored_jobs = [
            (self._calculate_fair_score(job), job) for job in pending_jobs
        ]
        scored_jobs.sort(key=lambda x: x[0], reverse=True)

        for score, job in scored_jobs:
            if not self._check_quota(job, quotas):
                continue

            workers = self._find_workers_for_job(job, available_resources)
            if workers:
                # Update usage tracking
                entity_id = job.team_id or job.user_id
                self._usage_history[entity_id] = self._get_usage(entity_id) + 1

                return SchedulingDecision(
                    job_id=job.id,
                    assigned_workers=[workers[0]],
                    should_preempt=[],
                    reason=f"FairShare: score={score:.2f}",
                    score=score,
                )

        return None

    def record_completion(self, job: TrainingJob) -> None:
        """Record job completion for usage tracking."""
        entity_id = job.team_id or job.user_id
        self._usage_history[entity_id] = max(0, self._get_usage(entity_id) - 0.5)


class GangSchedulingPolicy(SchedulingPolicy):
    """
    Gang scheduling for distributed training jobs.

    All workers for a job must be scheduled together or not at all.
    """

    async def select_job(
        self,
        pending_jobs: list[TrainingJob],
        available_resources: dict[str, ResourceRequest],
        quotas: dict[str, ResourceQuota],
    ) -> Optional[SchedulingDecision]:
        # Filter to distributed jobs first
        distributed_jobs = [
            j for j in pending_jobs if j.config.distributed.enabled
        ]
        single_jobs = [
            j for j in pending_jobs if not j.config.distributed.enabled
        ]

        # Sort by priority
        distributed_jobs.sort(key=lambda j: j.priority.value, reverse=True)

        for job in distributed_jobs:
            if not self._check_quota(job, quotas):
                continue

            world_size = job.config.distributed.world_size
            suitable_workers = self._find_workers_for_job(job, available_resources)

            if len(suitable_workers) >= world_size:
                # Can schedule all workers together
                assigned = suitable_workers[:world_size]
                return SchedulingDecision(
                    job_id=job.id,
                    assigned_workers=assigned,
                    should_preempt=[],
                    reason=f"Gang: {world_size} workers scheduled together",
                    score=job.priority.value,
                )

        # Fall back to single-worker jobs
        single_jobs.sort(key=lambda j: j.priority.value, reverse=True)
        for job in single_jobs:
            if not self._check_quota(job, quotas):
                continue

            workers = self._find_workers_for_job(job, available_resources)
            if workers:
                return SchedulingDecision(
                    job_id=job.id,
                    assigned_workers=[workers[0]],
                    should_preempt=[],
                    reason="Single worker job",
                )

        return None


class BackfillPolicy(SchedulingPolicy):
    """
    Backfill scheduling to fill resource gaps.

    Lower priority jobs can run if they don't delay higher priority jobs.
    """

    def __init__(self, max_backfill_duration_hours: float = 4.0):
        self._max_backfill_duration = max_backfill_duration_hours

    def _can_backfill(self, job: TrainingJob) -> bool:
        """Check if job is suitable for backfill."""
        if not job.config.timeout_hours:
            return False
        return job.config.timeout_hours <= self._max_backfill_duration

    async def select_job(
        self,
        pending_jobs: list[TrainingJob],
        available_resources: dict[str, ResourceRequest],
        quotas: dict[str, ResourceQuota],
    ) -> Optional[SchedulingDecision]:
        # First try high priority jobs
        high_priority = [
            j for j in pending_jobs
            if j.priority.value >= JobPriority.HIGH.value
        ]
        high_priority.sort(key=lambda j: j.priority.value, reverse=True)

        for job in high_priority:
            if not self._check_quota(job, quotas):
                continue

            workers = self._find_workers_for_job(job, available_resources)
            if workers:
                return SchedulingDecision(
                    job_id=job.id,
                    assigned_workers=[workers[0]],
                    should_preempt=[],
                    reason="High priority",
                )

        # Try backfill with lower priority jobs
        backfill_candidates = [
            j for j in pending_jobs
            if j.priority.value < JobPriority.HIGH.value and self._can_backfill(j)
        ]
        backfill_candidates.sort(
            key=lambda j: (j.resources.gpus, j.resources.cpus),
        )

        for job in backfill_candidates:
            if not self._check_quota(job, quotas):
                continue

            workers = self._find_workers_for_job(job, available_resources)
            if workers:
                return SchedulingDecision(
                    job_id=job.id,
                    assigned_workers=[workers[0]],
                    should_preempt=[],
                    reason=f"Backfill: max {job.config.timeout_hours}h",
                )

        return None


class PreemptivePolicy(SchedulingPolicy):
    """
    Preemptive scheduling allowing high priority jobs to preempt lower priority.
    """

    def __init__(
        self,
        priority_threshold: int = 25,
        min_run_time_minutes: int = 5,
    ):
        self._priority_threshold = priority_threshold
        self._min_run_time = timedelta(minutes=min_run_time_minutes)

    def _can_preempt(
        self,
        high_priority_job: TrainingJob,
        running_job: TrainingJob,
    ) -> bool:
        """Check if high priority job can preempt running job."""
        # Priority difference must exceed threshold
        priority_diff = high_priority_job.priority.value - running_job.priority.value
        if priority_diff < self._priority_threshold:
            return False

        # Running job must not be non-preemptible
        if not running_job.preemptible:
            return False

        # Running job must have run for minimum time
        if running_job.started_at:
            run_time = datetime.utcnow() - running_job.started_at
            if run_time < self._min_run_time:
                return False

        return True

    async def select_job(
        self,
        pending_jobs: list[TrainingJob],
        available_resources: dict[str, ResourceRequest],
        quotas: dict[str, ResourceQuota],
        running_jobs: Optional[list[TrainingJob]] = None,
    ) -> Optional[SchedulingDecision]:
        running_jobs = running_jobs or []

        # Sort pending by priority
        sorted_pending = sorted(
            pending_jobs,
            key=lambda j: j.priority.value,
            reverse=True,
        )

        for job in sorted_pending:
            if not self._check_quota(job, quotas):
                continue

            # First try without preemption
            workers = self._find_workers_for_job(job, available_resources)
            if workers:
                return SchedulingDecision(
                    job_id=job.id,
                    assigned_workers=[workers[0]],
                    should_preempt=[],
                    reason="Resources available",
                )

            # Check for preemption opportunities
            preemption_candidates = []
            for running in running_jobs:
                if self._can_preempt(job, running):
                    # Check if preempting would free enough resources
                    if job.resources.fits_in(running.resources):
                        preemption_candidates.append(running)

            # Sort by priority (preempt lowest first)
            preemption_candidates.sort(key=lambda j: j.priority.value)

            if preemption_candidates:
                to_preempt = preemption_candidates[0]
                return SchedulingDecision(
                    job_id=job.id,
                    assigned_workers=to_preempt.assigned_workers,
                    should_preempt=[to_preempt.id],
                    reason=f"Preempting job {to_preempt.id} (priority {to_preempt.priority.value})",
                )

        return None


class CompositePolicy(SchedulingPolicy):
    """
    Composite policy that combines multiple policies with weights.
    """

    def __init__(self, policies: list[tuple[SchedulingPolicy, float]]):
        """
        Initialize with weighted policies.

        Args:
            policies: List of (policy, weight) tuples
        """
        self._policies = policies
        total_weight = sum(w for _, w in policies)
        # Normalize weights
        self._normalized = [(p, w / total_weight) for p, w in policies]

    async def select_job(
        self,
        pending_jobs: list[TrainingJob],
        available_resources: dict[str, ResourceRequest],
        quotas: dict[str, ResourceQuota],
    ) -> Optional[SchedulingDecision]:
        decisions: list[tuple[SchedulingDecision, float]] = []

        for policy, weight in self._normalized:
            decision = await policy.select_job(
                pending_jobs, available_resources, quotas
            )
            if decision:
                decisions.append((decision, weight * decision.score))

        if not decisions:
            return None

        # Return decision with highest weighted score
        best = max(decisions, key=lambda x: x[1])
        return best[0]
