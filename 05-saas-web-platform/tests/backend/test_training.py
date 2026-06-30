"""Tests for training app."""

import pytest
import importlib.util
from unittest.mock import Mock, patch, MagicMock

# Check if Django is available
DJANGO_AVAILABLE = importlib.util.find_spec("django") is not None

if DJANGO_AVAILABLE:
    from django.utils import timezone
    from rest_framework.test import APIClient
    from rest_framework import status
else:
    timezone = None
    APIClient = None
    status = None


pytestmark = pytest.mark.skipif(not DJANGO_AVAILABLE, reason="Django not installed")


class TestTrainingModels:
    """Tests for training models."""

    def test_training_job_creation(self, db, tenant_factory):
        """Test creating a training job."""
        from apps.training.models import (
            TrainingJob, TrainingFramework, DistributedStrategy, JobStatus
        )

        tenant = tenant_factory(slug='test-job')

        job = TrainingJob.objects.create(
            tenant=tenant,
            name='GPT-3 Fine-tune',
            framework=TrainingFramework.PYTORCH,
            strategy=DistributedStrategy.FSDP,
            entry_point='train.py',
            gpu_count=4,
            nodes_count=2,
            world_size=8,
            total_epochs=10,
        )

        assert job.id is not None
        assert job.name == 'GPT-3 Fine-tune'
        assert job.status == JobStatus.PENDING
        assert job.is_distributed is True
        assert job.world_size == 8

    def test_training_run_creation(self, db, tenant_factory):
        """Test creating a training run."""
        from apps.training.models import TrainingJob, TrainingRun, JobStatus

        tenant = tenant_factory(slug='test-run')
        job = TrainingJob.objects.create(
            tenant=tenant,
            name='Test Job',
            entry_point='train.py',
        )

        run = TrainingRun.objects.create(
            job=job,
            run_number=1,
            status=JobStatus.RUNNING,
            started_at=timezone.now(),
        )

        assert run.id is not None
        assert run.run_number == 1
        assert run.job == job

    def test_experiment_creation(self, db, tenant_factory):
        """Test creating an experiment."""
        from apps.training.models import Experiment

        tenant = tenant_factory(slug='test-exp')

        experiment = Experiment.objects.create(
            tenant=tenant,
            name='LLM Hyperparameter Search',
            description='Testing different learning rates',
            hyperparameters={
                'learning_rate': [1e-4, 1e-5, 1e-6],
                'batch_size': [16, 32, 64],
            },
        )

        assert experiment.id is not None
        assert experiment.name == 'LLM Hyperparameter Search'

    def test_hyperparameter_sweep_creation(self, db, tenant_factory):
        """Test creating a hyperparameter sweep."""
        from apps.training.models import Experiment, HyperparameterSweep

        tenant = tenant_factory(slug='test-sweep')
        experiment = Experiment.objects.create(
            tenant=tenant,
            name='Test Experiment',
        )

        sweep = HyperparameterSweep.objects.create(
            experiment=experiment,
            name='Learning Rate Sweep',
            strategy=HyperparameterSweep.SearchStrategy.BAYESIAN,
            parameter_space={
                'learning_rate': {
                    'type': 'log_uniform',
                    'min': 1e-6,
                    'max': 1e-3,
                },
                'batch_size': {
                    'type': 'choice',
                    'values': [16, 32, 64],
                },
            },
            max_trials=50,
        )

        assert sweep.id is not None
        assert sweep.strategy == HyperparameterSweep.SearchStrategy.BAYESIAN
        assert sweep.max_trials == 50

    def test_model_artifact_creation(self, db, tenant_factory):
        """Test creating a model artifact."""
        from apps.training.models import TrainingJob, ModelArtifact

        tenant = tenant_factory(slug='test-artifact')
        job = TrainingJob.objects.create(
            tenant=tenant,
            name='Test Job',
            entry_point='train.py',
        )

        artifact = ModelArtifact.objects.create(
            job=job,
            name='checkpoint-epoch5',
            artifact_type=ModelArtifact.ArtifactType.CHECKPOINT,
            file_path='/checkpoints/model-epoch5.pt',
            step=5000,
            epoch=5,
            metrics={'loss': 0.05, 'accuracy': 0.95},
        )

        assert artifact.id is not None
        assert artifact.file_path == '/checkpoints/model-epoch5.pt'


class TestTrainingJobManager:
    """Tests for TrainingJobManager service."""

    @pytest.fixture
    def setup_resources(self, db, tenant_factory, user_factory):
        """Set up resources for training tests."""
        from apps.resources.models import (
            ComputeNode, GPU, GPUType, ResourceQuota
        )

        tenant = tenant_factory(slug='training-test')
        user = user_factory(email='trainer@test.com')

        # Create quota
        ResourceQuota.objects.create(
            tenant=tenant,
            max_gpus=16,
            max_gpu_hours_monthly=1000,
            max_concurrent_jobs=10,
        )

        # Create compute node with GPUs
        node = ComputeNode.objects.create(
            name='training-node',
            hostname='training-node.local',
            gpu_type=GPUType.NVIDIA_A100_80GB,
            gpu_count=8,
            gpu_memory_gb=80,
            cpu_cores=128,
            ram_gb=1024,
            storage_tb=10.0,
            region='us-east-1',
            price_per_hour=3200,
        )

        for i in range(8):
            GPU.objects.create(
                node=node,
                device_index=i,
                gpu_type=GPUType.NVIDIA_A100_80GB,
                memory_gb=80,
            )

        return tenant, user, node

    def test_submit_job(self, setup_resources):
        """Test submitting a training job."""
        from apps.training.models import TrainingJob, JobStatus
        from apps.training.services import TrainingJobManager

        tenant, user, node = setup_resources

        job = TrainingJob.objects.create(
            tenant=tenant,
            name='Submit Test',
            entry_point='train.py',
            gpu_count=4,
            created_by=user,
        )

        manager = TrainingJobManager()
        job = manager.submit_job(job)

        assert job.status in [JobStatus.INITIALIZING, JobStatus.QUEUED]
        assert job.runs.count() >= 1

    def test_update_progress(self, setup_resources):
        """Test updating job progress."""
        from apps.training.models import TrainingJob, JobStatus
        from apps.training.services import TrainingJobManager

        tenant, user, node = setup_resources

        job = TrainingJob.objects.create(
            tenant=tenant,
            name='Progress Test',
            entry_point='train.py',
            status=JobStatus.RUNNING,
            total_epochs=10,
            started_at=timezone.now(),
            created_by=user,
        )

        manager = TrainingJobManager()
        job = manager.update_progress(
            job,
            epoch=5,
            step=2500,
            loss=0.05,
            learning_rate=1e-4,
            throughput=1000.0,
        )

        assert job.current_epoch == 5
        assert job.current_step == 2500
        assert job.last_loss == 0.05
        assert job.best_loss == 0.05
        assert job.progress_percent == 50.0

    def test_save_checkpoint(self, setup_resources):
        """Test saving a checkpoint."""
        from apps.training.models import TrainingJob, JobStatus, ModelArtifact
        from apps.training.services import TrainingJobManager

        tenant, user, node = setup_resources

        job = TrainingJob.objects.create(
            tenant=tenant,
            name='Checkpoint Test',
            entry_point='train.py',
            status=JobStatus.RUNNING,
            current_epoch=5,
            current_step=2500,
            created_by=user,
        )

        manager = TrainingJobManager()
        artifact = manager.save_checkpoint(
            job,
            checkpoint_path='/checkpoints/model-step2500.pt',
            metrics={'loss': 0.05},
            is_best=True,
        )

        assert artifact is not None
        assert artifact.artifact_type == ModelArtifact.ArtifactType.CHECKPOINT
        assert artifact.is_best is True
        assert job.latest_checkpoint == '/checkpoints/model-step2500.pt'

    def test_complete_job(self, setup_resources):
        """Test completing a job."""
        from apps.training.models import TrainingJob, TrainingRun, JobStatus
        from apps.training.services import TrainingJobManager

        tenant, user, node = setup_resources

        job = TrainingJob.objects.create(
            tenant=tenant,
            name='Complete Test',
            entry_point='train.py',
            status=JobStatus.RUNNING,
            started_at=timezone.now(),
            created_by=user,
        )

        TrainingRun.objects.create(
            job=job,
            run_number=1,
            status=JobStatus.RUNNING,
            started_at=timezone.now(),
        )

        manager = TrainingJobManager()
        job = manager.complete_job(
            job,
            success=True,
            final_metrics={'final_loss': 0.01},
        )

        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.progress_percent == 100

    def test_cancel_job(self, setup_resources):
        """Test cancelling a job."""
        from apps.training.models import TrainingJob, JobStatus
        from apps.training.services import TrainingJobManager

        tenant, user, node = setup_resources

        job = TrainingJob.objects.create(
            tenant=tenant,
            name='Cancel Test',
            entry_point='train.py',
            status=JobStatus.RUNNING,
            started_at=timezone.now(),
            created_by=user,
        )

        manager = TrainingJobManager()
        job = manager.cancel_job(job)

        assert job.status == JobStatus.CANCELLED

    def test_get_stats(self, setup_resources):
        """Test getting job statistics."""
        from apps.training.models import TrainingJob, JobStatus
        from apps.training.services import TrainingJobManager

        tenant, user, node = setup_resources

        # Create jobs with different statuses
        TrainingJob.objects.create(
            tenant=tenant,
            name='Running Job',
            entry_point='train.py',
            status=JobStatus.RUNNING,
            gpu_count=4,
            started_at=timezone.now(),
            created_by=user,
        )
        TrainingJob.objects.create(
            tenant=tenant,
            name='Completed Job',
            entry_point='train.py',
            status=JobStatus.COMPLETED,
            gpu_count=4,
            started_at=timezone.now() - timezone.timedelta(hours=2),
            completed_at=timezone.now(),
            created_by=user,
        )

        manager = TrainingJobManager()
        stats = manager.get_stats(tenant)

        assert stats['total_jobs'] == 2
        assert stats['running_jobs'] == 1
        assert stats['completed_jobs'] == 1


class TestExperimentManager:
    """Tests for ExperimentManager service."""

    def test_create_experiment(self, db, tenant_factory, user_factory):
        """Test creating an experiment."""
        from apps.training.services import ExperimentManager

        tenant = tenant_factory(slug='exp-create')
        user = user_factory(email='exp@test.com')

        manager = ExperimentManager()
        experiment = manager.create_experiment(
            tenant=tenant,
            created_by=user,
            name='Test Experiment',
            description='Testing experiment creation',
            hyperparameters={'lr': [1e-4, 1e-5]},
        )

        assert experiment.id is not None
        assert experiment.name == 'Test Experiment'

    def test_create_sweep(self, db, tenant_factory, user_factory):
        """Test creating a hyperparameter sweep."""
        from apps.training.models import Experiment
        from apps.training.services import ExperimentManager

        tenant = tenant_factory(slug='sweep-create')
        user = user_factory(email='sweep@test.com')

        experiment = Experiment.objects.create(
            tenant=tenant,
            name='Sweep Test Experiment',
            created_by=user,
        )

        manager = ExperimentManager()
        sweep = manager.create_sweep(
            experiment=experiment,
            name='LR Sweep',
            strategy='random',
            parameter_space={
                'learning_rate': {
                    'type': 'log_uniform',
                    'min': 1e-6,
                    'max': 1e-3,
                },
            },
            max_trials=20,
        )

        assert sweep.id is not None
        assert sweep.max_trials == 20

    def test_sample_hyperparameters(self, db, tenant_factory, user_factory):
        """Test sampling hyperparameters."""
        from apps.training.models import Experiment, HyperparameterSweep
        from apps.training.services import ExperimentManager

        tenant = tenant_factory(slug='sample-params')
        user = user_factory(email='sample@test.com')

        experiment = Experiment.objects.create(
            tenant=tenant,
            name='Sample Test',
            created_by=user,
        )

        sweep = HyperparameterSweep.objects.create(
            experiment=experiment,
            name='Sample Sweep',
            strategy='random',
            parameter_space={
                'learning_rate': {
                    'type': 'uniform',
                    'min': 0.0001,
                    'max': 0.01,
                },
                'batch_size': {
                    'type': 'choice',
                    'values': [16, 32, 64],
                },
                'hidden_dim': {
                    'type': 'int_uniform',
                    'min': 128,
                    'max': 1024,
                },
            },
        )

        manager = ExperimentManager()
        params = manager.sample_hyperparameters(sweep)

        assert 'learning_rate' in params
        assert 0.0001 <= params['learning_rate'] <= 0.01
        assert params['batch_size'] in [16, 32, 64]
        assert 128 <= params['hidden_dim'] <= 1024


class TestTrainingViewSets:
    """Tests for training API endpoints."""

    @pytest.fixture
    def authenticated_client(self, db):
        """Create an authenticated API client."""
        from apps.users.models import User

        client = APIClient()
        user = User.objects.create_user(
            email='test@example.com',
            password='TestPass123!',
        )
        client.force_authenticate(user=user)
        return client, user

    def test_list_jobs(self, authenticated_client, tenant_factory):
        """Test listing training jobs."""
        from apps.training.models import TrainingJob

        client, user = authenticated_client
        tenant = tenant_factory(slug='api-test')

        TrainingJob.objects.create(
            tenant=tenant,
            name='API Test Job',
            entry_point='train.py',
            created_by=user,
        )

        response = client.get('/api/v1/training/jobs/')

        assert response.status_code == status.HTTP_200_OK

    def test_get_job_stats(self, authenticated_client):
        """Test getting job statistics."""
        client, user = authenticated_client

        response = client.get('/api/v1/training/jobs/stats/')

        assert response.status_code == status.HTTP_200_OK
        assert 'total_jobs' in response.data

    def test_list_experiments(self, authenticated_client, tenant_factory):
        """Test listing experiments."""
        from apps.training.models import Experiment

        client, user = authenticated_client
        tenant = tenant_factory(slug='exp-api-test')

        Experiment.objects.create(
            tenant=tenant,
            name='API Test Experiment',
            created_by=user,
        )

        response = client.get('/api/v1/training/experiments/')

        assert response.status_code == status.HTTP_200_OK

    def test_list_sweeps(self, authenticated_client, tenant_factory):
        """Test listing hyperparameter sweeps."""
        from apps.training.models import Experiment, HyperparameterSweep

        client, user = authenticated_client
        tenant = tenant_factory(slug='sweep-api-test')

        experiment = Experiment.objects.create(
            tenant=tenant,
            name='Sweep API Test Experiment',
            created_by=user,
        )

        HyperparameterSweep.objects.create(
            experiment=experiment,
            name='API Test Sweep',
            strategy='random',
            parameter_space={'lr': {'type': 'uniform', 'min': 0.001, 'max': 0.1}},
        )

        response = client.get('/api/v1/training/sweeps/')

        assert response.status_code == status.HTTP_200_OK

    def test_list_artifacts(self, authenticated_client, tenant_factory):
        """Test listing model artifacts."""
        from apps.training.models import TrainingJob, ModelArtifact

        client, user = authenticated_client
        tenant = tenant_factory(slug='artifact-api-test')

        job = TrainingJob.objects.create(
            tenant=tenant,
            name='Artifact Test Job',
            entry_point='train.py',
            created_by=user,
        )

        ModelArtifact.objects.create(
            job=job,
            name='test-checkpoint',
            artifact_type=ModelArtifact.ArtifactType.CHECKPOINT,
            file_path='/checkpoints/test.pt',
        )

        response = client.get('/api/v1/training/artifacts/')

        assert response.status_code == status.HTTP_200_OK

    def test_get_best_artifacts(self, authenticated_client, tenant_factory):
        """Test getting best model artifacts."""
        from apps.training.models import TrainingJob, ModelArtifact

        client, user = authenticated_client
        tenant = tenant_factory(slug='best-artifact-test')

        job = TrainingJob.objects.create(
            tenant=tenant,
            name='Best Artifact Test Job',
            entry_point='train.py',
            created_by=user,
        )

        ModelArtifact.objects.create(
            job=job,
            name='best-checkpoint',
            artifact_type=ModelArtifact.ArtifactType.CHECKPOINT,
            file_path='/checkpoints/best.pt',
            is_best=True,
        )

        response = client.get('/api/v1/training/artifacts/best/')

        assert response.status_code == status.HTTP_200_OK


class TestTrainingSerializers:
    """Tests for training serializers."""

    def test_training_job_serializer(self, db, tenant_factory, user_factory):
        """Test TrainingJobSerializer."""
        from apps.training.models import TrainingJob
        from apps.training.serializers import TrainingJobSerializer

        tenant = tenant_factory(slug='serializer-test')
        user = user_factory(email='serializer@test.com')

        job = TrainingJob.objects.create(
            tenant=tenant,
            name='Serializer Test',
            entry_point='train.py',
            gpu_count=4,
            nodes_count=2,
            world_size=8,
            created_by=user,
        )

        serializer = TrainingJobSerializer(job)
        data = serializer.data

        assert data['name'] == 'Serializer Test'
        assert data['gpu_count'] == 4
        assert data['is_distributed'] is True
        assert 'id' in data

    def test_training_job_create_serializer_validation(self, db):
        """Test TrainingJobCreateSerializer validation."""
        from apps.training.serializers import TrainingJobCreateSerializer

        # Invalid - world_size exceeds nodes * gpus
        data = {
            'name': 'Invalid Job',
            'entry_point': 'train.py',
            'gpu_count': 2,
            'nodes_count': 1,
            'world_size': 8,  # Invalid: 8 > 1*2
        }

        serializer = TrainingJobCreateSerializer(data=data)
        assert not serializer.is_valid()

    def test_hyperparameter_sweep_serializer_validation(self, db):
        """Test HyperparameterSweepSerializer validation."""
        from apps.training.serializers import HyperparameterSweepSerializer

        # Invalid parameter_space (missing type)
        data = {
            'name': 'Invalid Sweep',
            'strategy': 'random',
            'parameter_space': {
                'learning_rate': {'min': 0.001, 'max': 0.1},  # Missing 'type'
            },
        }

        serializer = HyperparameterSweepSerializer(data=data)
        assert not serializer.is_valid()

    def test_job_metrics_serializer(self, db):
        """Test JobMetricsSerializer."""
        from apps.training.serializers import JobMetricsSerializer

        data = {
            'loss': 0.05,
            'learning_rate': 1e-4,
            'epoch': 5,
            'step': 2500,
            'throughput': 1000.0,
            'additional_metrics': {'accuracy': 0.95},
        }

        serializer = JobMetricsSerializer(data=data)
        assert serializer.is_valid()
        assert serializer.validated_data['loss'] == 0.05
