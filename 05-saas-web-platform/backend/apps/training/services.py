"""Training job management services."""

from typing import Optional, List, Dict, Any
from django.db import transaction
from django.db.models import Q, Count, Sum, F, Avg
from django.utils import timezone
import logging

from apps.tenants.models import Tenant
from apps.resources.models import ResourceAllocation
from apps.resources.services import ResourceManager
from .models import (
    TrainingJob, TrainingRun, Experiment,
    HyperparameterSweep, ModelArtifact, JobStatus
)

logger = logging.getLogger(__name__)


class TrainingJobManager:
    """Manages training job lifecycle."""

    def __init__(self):
        self.resource_manager = ResourceManager()

    @transaction.atomic
    def submit_job(self, job: TrainingJob) -> TrainingJob:
        """Submit a job for execution."""
        if job.status != JobStatus.PENDING:
            raise ValueError(f"Job is not pending: {job.status}")

        # Try to allocate resources
        try:
            allocation = self.resource_manager.allocate(
                tenant=job.tenant,
                created_by=job.created_by,
                requested_gpus=job.gpu_count * job.nodes_count,
                gpu_type=job.gpu_type or None,
                priority=job.priority,
                max_duration_hours=job.max_runtime_hours,
                preemptible=job.preemptible,
                job_id=str(job.id),
                job_type='training',
            )

            job.allocation = allocation
            job.status = JobStatus.INITIALIZING
            job.save()

            # Create first run
            TrainingRun.objects.create(
                job=job,
                run_number=1,
                status=JobStatus.INITIALIZING,
                started_at=timezone.now(),
            )

            logger.info(f"Submitted training job {job.id}: {job.name}")
            return job

        except ValueError as e:
            # Queue the job if resources unavailable
            job.status = JobStatus.QUEUED
            job.save()
            logger.info(f"Queued training job {job.id}: {e}")
            return job

    def start_job(self, job: TrainingJob, worker_hosts: List[str]) -> TrainingJob:
        """Mark job as running with assigned workers."""
        if job.status not in [JobStatus.INITIALIZING, JobStatus.QUEUED]:
            raise ValueError(f"Cannot start job in status: {job.status}")

        job.status = JobStatus.RUNNING
        job.worker_hosts = worker_hosts
        job.master_addr = worker_hosts[0] if worker_hosts else ''
        job.started_at = timezone.now()
        job.save()

        # Update current run
        run = job.runs.order_by('-run_number').first()
        if run:
            run.status = JobStatus.RUNNING
            run.started_at = timezone.now()
            run.save()

        logger.info(f"Started training job {job.id} on {len(worker_hosts)} workers")
        return job

    def update_progress(
        self,
        job: TrainingJob,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        throughput: Optional[float] = None,
        additional_metrics: Optional[Dict] = None,
    ) -> TrainingJob:
        """Update job progress and metrics."""
        if epoch is not None:
            job.current_epoch = epoch
        if step is not None:
            job.current_step = step
        if loss is not None:
            job.last_loss = loss
            if job.best_loss is None or loss < job.best_loss:
                job.best_loss = loss
        if learning_rate is not None:
            job.learning_rate = learning_rate
        if throughput is not None:
            job.throughput_samples_sec = throughput

        # Calculate progress
        if job.total_epochs and job.total_epochs > 0:
            job.progress_percent = (job.current_epoch / job.total_epochs) * 100
        elif job.total_steps and job.total_steps > 0:
            job.progress_percent = (job.current_step / job.total_steps) * 100

        # Estimate completion
        if job.started_at and job.progress_percent > 0:
            elapsed = (timezone.now() - job.started_at).total_seconds()
            estimated_total = elapsed / (job.progress_percent / 100)
            remaining = estimated_total - elapsed
            job.estimated_completion = timezone.now() + timezone.timedelta(seconds=remaining)

        # Merge additional metrics
        if additional_metrics:
            job.metrics = {**job.metrics, **additional_metrics}

        job.save()
        return job

    def save_checkpoint(
        self,
        job: TrainingJob,
        checkpoint_path: str,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
    ) -> ModelArtifact:
        """Save a checkpoint artifact."""
        job.latest_checkpoint = checkpoint_path
        job.status = JobStatus.CHECKPOINTING
        job.save()

        artifact = ModelArtifact.objects.create(
            job=job,
            name=f"checkpoint-epoch{job.current_epoch}-step{job.current_step}",
            artifact_type=ModelArtifact.ArtifactType.CHECKPOINT,
            file_path=checkpoint_path,
            step=job.current_step,
            epoch=job.current_epoch,
            metrics=metrics or {},
            is_best=is_best,
        )

        job.status = JobStatus.RUNNING
        job.save()

        logger.info(f"Saved checkpoint for job {job.id}: {checkpoint_path}")
        return artifact

    @transaction.atomic
    def complete_job(
        self,
        job: TrainingJob,
        success: bool = True,
        final_metrics: Optional[Dict] = None,
        error_message: str = "",
    ) -> TrainingJob:
        """Mark job as completed."""
        job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
        job.completed_at = timezone.now()
        job.progress_percent = 100 if success else job.progress_percent

        if final_metrics:
            job.metrics = {**job.metrics, **final_metrics}
        if error_message:
            job.error_message = error_message

        job.save()

        # Update current run
        run = job.runs.order_by('-run_number').first()
        if run:
            run.status = job.status
            run.completed_at = timezone.now()
            run.final_loss = job.best_loss
            run.final_metrics = final_metrics or {}
            run.error_message = error_message
            run.save()

        # Release resources
        if job.allocation:
            self.resource_manager.release(job.allocation)

        logger.info(f"Completed training job {job.id}: {job.status}")
        return job

    @transaction.atomic
    def cancel_job(self, job: TrainingJob) -> TrainingJob:
        """Cancel a running or queued job."""
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return job

        job.status = JobStatus.CANCELLED
        job.completed_at = timezone.now()
        job.save()

        if job.allocation:
            self.resource_manager.release(job.allocation)

        logger.info(f"Cancelled training job {job.id}")
        return job

    @transaction.atomic
    def retry_job(self, job: TrainingJob) -> TrainingJob:
        """Retry a failed job."""
        if job.status != JobStatus.FAILED:
            raise ValueError("Can only retry failed jobs")

        if job.retry_count >= job.max_retries:
            raise ValueError("Max retries exceeded")

        job.retry_count += 1
        job.status = JobStatus.PENDING
        job.error_message = ""
        job.save()

        # Create new run
        last_run = job.runs.order_by('-run_number').first()
        next_run_number = (last_run.run_number + 1) if last_run else 1

        TrainingRun.objects.create(
            job=job,
            run_number=next_run_number,
            status=JobStatus.PENDING,
        )

        return self.submit_job(job)

    def get_stats(self, tenant: Optional[Tenant] = None) -> Dict[str, Any]:
        """Get training job statistics."""
        queryset = TrainingJob.objects.all()
        if tenant:
            queryset = queryset.filter(tenant=tenant)

        stats = queryset.aggregate(
            total=Count('id'),
            running=Count('id', filter=Q(status=JobStatus.RUNNING)),
            completed=Count('id', filter=Q(status=JobStatus.COMPLETED)),
            failed=Count('id', filter=Q(status=JobStatus.FAILED)),
            queued=Count('id', filter=Q(status=JobStatus.QUEUED)),
        )

        # Calculate GPU hours
        completed_jobs = queryset.filter(
            status__in=[JobStatus.COMPLETED, JobStatus.FAILED],
            started_at__isnull=False,
            completed_at__isnull=False,
        )

        total_gpu_hours = 0
        for job in completed_jobs:
            total_gpu_hours += job.duration_hours * job.gpu_count * job.nodes_count

        completed_count = stats['completed'] + stats['failed']
        success_rate = (stats['completed'] / completed_count * 100) if completed_count > 0 else 0

        return {
            'total_jobs': stats['total'],
            'running_jobs': stats['running'],
            'completed_jobs': stats['completed'],
            'failed_jobs': stats['failed'],
            'queued_jobs': stats['queued'],
            'total_gpu_hours': round(total_gpu_hours, 2),
            'success_rate': round(success_rate, 2),
        }


class ExperimentManager:
    """Manages experiments and hyperparameter sweeps."""

    def create_experiment(
        self,
        tenant: Tenant,
        created_by,
        name: str,
        description: str = "",
        base_config: Optional[Dict] = None,
        hyperparameters: Optional[Dict] = None,
        best_metric_name: str = "loss",
        best_metric_minimize: bool = True,
        tags: Optional[Dict] = None,
    ) -> Experiment:
        """Create a new experiment."""
        return Experiment.objects.create(
            tenant=tenant,
            name=name,
            description=description,
            base_config=base_config or {},
            hyperparameters=hyperparameters or {},
            best_metric_name=best_metric_name,
            best_metric_minimize=best_metric_minimize,
            tags=tags or {},
            created_by=created_by,
        )

    def update_best_job(self, experiment: Experiment, job: TrainingJob) -> None:
        """Update best job if this one is better."""
        if not job.best_loss:
            return

        metric_value = job.best_loss

        should_update = False
        if experiment.best_metric_value is None:
            should_update = True
        elif experiment.best_metric_minimize:
            should_update = metric_value < experiment.best_metric_value
        else:
            should_update = metric_value > experiment.best_metric_value

        if should_update:
            experiment.best_job = job
            experiment.best_metric_value = metric_value
            experiment.save()

    def create_sweep(
        self,
        experiment: Experiment,
        name: str,
        strategy: str,
        parameter_space: Dict,
        objective_metric: str = "loss",
        objective_minimize: bool = True,
        max_trials: int = 10,
        max_parallel_trials: int = 2,
        early_stopping_patience: int = 5,
    ) -> HyperparameterSweep:
        """Create a hyperparameter sweep."""
        return HyperparameterSweep.objects.create(
            experiment=experiment,
            name=name,
            strategy=strategy,
            parameter_space=parameter_space,
            objective_metric=objective_metric,
            objective_minimize=objective_minimize,
            max_trials=max_trials,
            max_parallel_trials=max_parallel_trials,
            early_stopping_patience=early_stopping_patience,
        )

    def sample_hyperparameters(self, sweep: HyperparameterSweep) -> Dict[str, Any]:
        """Sample next hyperparameters for a trial."""
        import random

        params = {}
        for name, config in sweep.parameter_space.items():
            param_type = config.get('type')

            if param_type == 'choice':
                params[name] = random.choice(config['values'])
            elif param_type == 'uniform':
                params[name] = random.uniform(config['min'], config['max'])
            elif param_type == 'log_uniform':
                import math
                log_min = math.log(config['min'])
                log_max = math.log(config['max'])
                params[name] = math.exp(random.uniform(log_min, log_max))
            elif param_type == 'int_uniform':
                params[name] = random.randint(config['min'], config['max'])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return params

    def update_sweep_best(
        self,
        sweep: HyperparameterSweep,
        trial_params: Dict,
        trial_value: float,
    ) -> None:
        """Update sweep best if trial is better."""
        sweep.completed_trials += 1

        should_update = False
        if sweep.best_trial_value is None:
            should_update = True
        elif sweep.objective_minimize:
            should_update = trial_value < sweep.best_trial_value
        else:
            should_update = trial_value > sweep.best_trial_value

        if should_update:
            sweep.best_trial_params = trial_params
            sweep.best_trial_value = trial_value

        # Check if sweep is complete
        if sweep.completed_trials >= sweep.max_trials:
            sweep.is_active = False
            sweep.completed_at = timezone.now()

        sweep.save()
