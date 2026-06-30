"""Views for training app."""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Q

from .models import (
    TrainingJob, TrainingRun, Experiment,
    HyperparameterSweep, ModelArtifact, JobStatus
)
from .serializers import (
    TrainingJobSerializer, TrainingJobCreateSerializer,
    TrainingJobListSerializer, TrainingRunSerializer,
    ExperimentSerializer, HyperparameterSweepSerializer,
    ModelArtifactSerializer, JobMetricsSerializer,
)
from .services import TrainingJobManager, ExperimentManager


class TrainingJobViewSet(viewsets.ModelViewSet):
    """ViewSet for training jobs."""

    permission_classes = [IsAuthenticated]
    serializer_class = TrainingJobSerializer

    def get_queryset(self):
        queryset = TrainingJob.objects.all()
        tenant = getattr(self.request, 'tenant', None)

        if tenant:
            queryset = queryset.filter(tenant=tenant)

        # Filter by status
        job_status = self.request.query_params.get('status')
        if job_status:
            queryset = queryset.filter(status=job_status)

        # Filter by framework
        framework = self.request.query_params.get('framework')
        if framework:
            queryset = queryset.filter(framework=framework)

        return queryset.select_related('tenant', 'allocation', 'created_by')

    def get_serializer_class(self):
        if self.action == 'create':
            return TrainingJobCreateSerializer
        if self.action == 'list':
            return TrainingJobListSerializer
        return TrainingJobSerializer

    @action(detail=True, methods=['post'])
    def submit(self, request, pk=None):
        """Submit a pending job for execution."""
        job = self.get_object()
        manager = TrainingJobManager()

        try:
            job = manager.submit_job(job)
            return Response({
                'status': job.status,
                'allocation_id': str(job.allocation.id) if job.allocation else None,
            })
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel a job."""
        job = self.get_object()
        manager = TrainingJobManager()
        job = manager.cancel_job(job)

        return Response({'status': job.status})

    @action(detail=True, methods=['post'])
    def retry(self, request, pk=None):
        """Retry a failed job."""
        job = self.get_object()
        manager = TrainingJobManager()

        try:
            job = manager.retry_job(job)
            return Response({
                'status': job.status,
                'retry_count': job.retry_count,
            })
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    @action(detail=True, methods=['post'])
    def update_progress(self, request, pk=None):
        """Update job progress and metrics."""
        job = self.get_object()
        serializer = JobMetricsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        manager = TrainingJobManager()
        job = manager.update_progress(job, **serializer.validated_data)

        return Response({
            'progress_percent': job.progress_percent,
            'current_epoch': job.current_epoch,
            'current_step': job.current_step,
            'last_loss': job.last_loss,
            'best_loss': job.best_loss,
        })

    @action(detail=True, methods=['post'])
    def checkpoint(self, request, pk=None):
        """Save a checkpoint."""
        job = self.get_object()
        checkpoint_path = request.data.get('checkpoint_path')
        metrics = request.data.get('metrics', {})
        is_best = request.data.get('is_best', False)

        if not checkpoint_path:
            return Response(
                {'error': 'checkpoint_path required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        manager = TrainingJobManager()
        artifact = manager.save_checkpoint(
            job, checkpoint_path, metrics, is_best
        )

        return Response({
            'artifact_id': str(artifact.id),
            'latest_checkpoint': job.latest_checkpoint,
        })

    @action(detail=True, methods=['get'])
    def runs(self, request, pk=None):
        """Get run history for a job."""
        job = self.get_object()
        runs = job.runs.all()
        serializer = TrainingRunSerializer(runs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def artifacts(self, request, pk=None):
        """Get artifacts for a job."""
        job = self.get_object()
        artifacts = job.artifacts.all()
        serializer = ModelArtifactSerializer(artifacts, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def logs(self, request, pk=None):
        """Get job logs."""
        job = self.get_object()
        run = job.runs.order_by('-run_number').first()

        if not run:
            return Response({'logs': ''})

        return Response({
            'logs': run.logs,
            'logs_path': job.logs_path,
        })

    @action(detail=False, methods=['get'])
    def running(self, request):
        """Get currently running jobs."""
        queryset = self.get_queryset().filter(status=JobStatus.RUNNING)
        serializer = TrainingJobListSerializer(queryset, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def queued(self, request):
        """Get queued jobs."""
        queryset = self.get_queryset().filter(status=JobStatus.QUEUED)
        serializer = TrainingJobListSerializer(queryset, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get job statistics."""
        tenant = getattr(request, 'tenant', None)
        manager = TrainingJobManager()
        return Response(manager.get_stats(tenant))


class TrainingRunViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for training runs (read-only)."""

    permission_classes = [IsAuthenticated]
    serializer_class = TrainingRunSerializer

    def get_queryset(self):
        queryset = TrainingRun.objects.all()
        tenant = getattr(self.request, 'tenant', None)

        if tenant:
            queryset = queryset.filter(job__tenant=tenant)

        return queryset.select_related('job')


class ExperimentViewSet(viewsets.ModelViewSet):
    """ViewSet for experiments."""

    permission_classes = [IsAuthenticated]
    serializer_class = ExperimentSerializer

    def get_queryset(self):
        queryset = Experiment.objects.all()
        tenant = getattr(self.request, 'tenant', None)

        if tenant:
            queryset = queryset.filter(tenant=tenant)

        return queryset.select_related('tenant', 'best_job', 'created_by')

    def create(self, request, *args, **kwargs):
        """Create an experiment."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        tenant = getattr(request, 'tenant', None)
        if not tenant:
            return Response(
                {'error': 'No tenant context'},
                status=status.HTTP_400_BAD_REQUEST
            )

        manager = ExperimentManager()
        experiment = manager.create_experiment(
            tenant=tenant,
            created_by=request.user,
            **serializer.validated_data
        )

        return Response(
            self.get_serializer(experiment).data,
            status=status.HTTP_201_CREATED
        )

    @action(detail=True, methods=['get'])
    def jobs(self, request, pk=None):
        """Get jobs in this experiment."""
        experiment = self.get_object()
        jobs = experiment.tenant.training_jobs.filter(
            tags__contains={'experiment_id': str(experiment.id)}
        )
        serializer = TrainingJobListSerializer(jobs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def sweeps(self, request, pk=None):
        """Get hyperparameter sweeps for this experiment."""
        experiment = self.get_object()
        sweeps = experiment.sweeps.all()
        serializer = HyperparameterSweepSerializer(sweeps, many=True)
        return Response(serializer.data)


class HyperparameterSweepViewSet(viewsets.ModelViewSet):
    """ViewSet for hyperparameter sweeps."""

    permission_classes = [IsAuthenticated]
    serializer_class = HyperparameterSweepSerializer

    def get_queryset(self):
        queryset = HyperparameterSweep.objects.all()
        tenant = getattr(self.request, 'tenant', None)

        if tenant:
            queryset = queryset.filter(experiment__tenant=tenant)

        return queryset.select_related('experiment')

    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        """Start a sweep."""
        sweep = self.get_object()

        if not sweep.is_active:
            return Response(
                {'error': 'Sweep is not active'},
                status=status.HTTP_400_BAD_REQUEST
            )

        sweep.started_at = sweep.started_at or __import__('django.utils.timezone', fromlist=['now']).now()
        sweep.save()

        return Response({
            'status': 'started',
            'max_trials': sweep.max_trials,
        })

    @action(detail=True, methods=['post'])
    def stop(self, request, pk=None):
        """Stop a sweep."""
        sweep = self.get_object()
        sweep.is_active = False
        sweep.save()

        return Response({'status': 'stopped'})

    @action(detail=True, methods=['get'])
    def next_params(self, request, pk=None):
        """Get next hyperparameters to try."""
        sweep = self.get_object()

        if not sweep.is_active:
            return Response(
                {'error': 'Sweep is not active'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if sweep.completed_trials >= sweep.max_trials:
            return Response(
                {'error': 'Max trials reached'},
                status=status.HTTP_400_BAD_REQUEST
            )

        manager = ExperimentManager()
        params = manager.sample_hyperparameters(sweep)

        return Response({
            'params': params,
            'trial_number': sweep.completed_trials + 1,
        })

    @action(detail=True, methods=['post'])
    def report_trial(self, request, pk=None):
        """Report trial results."""
        sweep = self.get_object()
        params = request.data.get('params', {})
        value = request.data.get('value')

        if value is None:
            return Response(
                {'error': 'value required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        manager = ExperimentManager()
        manager.update_sweep_best(sweep, params, value)

        return Response({
            'completed_trials': sweep.completed_trials,
            'best_trial_value': sweep.best_trial_value,
            'best_trial_params': sweep.best_trial_params,
            'is_active': sweep.is_active,
        })


class ModelArtifactViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for model artifacts (read-only)."""

    permission_classes = [IsAuthenticated]
    serializer_class = ModelArtifactSerializer

    def get_queryset(self):
        queryset = ModelArtifact.objects.all()
        tenant = getattr(self.request, 'tenant', None)

        if tenant:
            queryset = queryset.filter(job__tenant=tenant)

        artifact_type = self.request.query_params.get('type')
        if artifact_type:
            queryset = queryset.filter(artifact_type=artifact_type)

        return queryset.select_related('job')

    @action(detail=False, methods=['get'])
    def best(self, request):
        """Get best model artifacts."""
        queryset = self.get_queryset().filter(is_best=True)
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
