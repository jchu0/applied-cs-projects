"""Serializers for training app."""

from rest_framework import serializers
from .models import (
    TrainingJob, TrainingRun, Experiment,
    HyperparameterSweep, ModelArtifact
)


class TrainingRunSerializer(serializers.ModelSerializer):
    """Serializer for training runs."""

    class Meta:
        model = TrainingRun
        fields = [
            'id', 'job', 'run_number', 'status',
            'started_at', 'completed_at',
            'final_loss', 'final_metrics',
            'exit_code', 'error_message',
            'final_checkpoint', 'created_at'
        ]
        read_only_fields = ['id', 'job', 'run_number', 'created_at']


class ModelArtifactSerializer(serializers.ModelSerializer):
    """Serializer for model artifacts."""

    job_name = serializers.CharField(source='job.name', read_only=True)

    class Meta:
        model = ModelArtifact
        fields = [
            'id', 'job', 'job_name', 'name',
            'artifact_type', 'file_path', 'file_size_bytes', 'checksum',
            'step', 'epoch', 'metrics',
            'model_architecture', 'parameter_count', 'framework_version',
            'is_best', 'is_final', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class TrainingJobSerializer(serializers.ModelSerializer):
    """Serializer for training jobs."""

    tenant_name = serializers.CharField(source='tenant.name', read_only=True)
    duration_hours = serializers.FloatField(read_only=True)
    is_distributed = serializers.BooleanField(read_only=True)
    runs = TrainingRunSerializer(many=True, read_only=True)
    artifacts_count = serializers.SerializerMethodField()

    class Meta:
        model = TrainingJob
        fields = [
            'id', 'tenant', 'tenant_name', 'name', 'description',
            'framework', 'strategy',
            'code_repository', 'code_branch', 'code_commit',
            'docker_image', 'entry_point', 'arguments', 'environment',
            'gpu_count', 'gpu_type', 'nodes_count',
            'memory_gb', 'storage_gb',
            'world_size', 'gradient_accumulation_steps', 'mixed_precision',
            'distributed_config',
            'status', 'priority', 'preemptible',
            'allocation', 'worker_hosts', 'master_addr', 'master_port',
            'current_epoch', 'total_epochs', 'current_step', 'total_steps',
            'progress_percent',
            'metrics', 'last_loss', 'best_loss', 'learning_rate',
            'throughput_samples_sec',
            'checkpoint_enabled', 'checkpoint_interval_steps',
            'checkpoint_path', 'latest_checkpoint', 'resume_from_checkpoint',
            'submitted_at', 'started_at', 'completed_at',
            'estimated_completion', 'max_runtime_hours',
            'output_path', 'logs_path', 'tensorboard_path',
            'error_message', 'retry_count', 'max_retries',
            'duration_hours', 'is_distributed',
            'tags', 'runs', 'artifacts_count',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'tenant', 'allocation', 'worker_hosts',
            'master_addr', 'current_epoch', 'current_step', 'progress_percent',
            'metrics', 'last_loss', 'best_loss', 'learning_rate',
            'throughput_samples_sec', 'latest_checkpoint',
            'submitted_at', 'started_at', 'completed_at', 'estimated_completion',
            'error_message', 'retry_count',
            'created_at', 'updated_at'
        ]

    def get_artifacts_count(self, obj):
        return obj.artifacts.count()


class TrainingJobCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating training jobs."""

    class Meta:
        model = TrainingJob
        fields = [
            'name', 'description',
            'framework', 'strategy',
            'code_repository', 'code_branch', 'code_commit',
            'docker_image', 'entry_point', 'arguments', 'environment',
            'gpu_count', 'gpu_type', 'nodes_count',
            'memory_gb', 'storage_gb',
            'world_size', 'gradient_accumulation_steps', 'mixed_precision',
            'distributed_config',
            'priority', 'preemptible',
            'total_epochs', 'total_steps',
            'checkpoint_enabled', 'checkpoint_interval_steps',
            'checkpoint_path', 'resume_from_checkpoint',
            'max_runtime_hours',
            'output_path', 'logs_path', 'tensorboard_path',
            'max_retries', 'tags'
        ]

    def validate(self, data):
        # Validate distributed config
        nodes = data.get('nodes_count', 1)
        gpus = data.get('gpu_count', 1)
        world_size = data.get('world_size', 1)

        if world_size > nodes * gpus:
            raise serializers.ValidationError(
                "world_size cannot exceed nodes_count * gpu_count"
            )

        strategy = data.get('strategy')
        if strategy in ['model_parallel', 'pipeline_parallel', 'tensor_parallel']:
            if world_size < 2:
                raise serializers.ValidationError(
                    f"{strategy} requires world_size >= 2"
                )

        return data

    def create(self, validated_data):
        request = self.context.get('request')
        tenant = getattr(request, 'tenant', None)

        if not tenant:
            raise serializers.ValidationError("Tenant context required")

        return TrainingJob.objects.create(
            tenant=tenant,
            created_by=request.user,
            **validated_data
        )


class TrainingJobListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for job listing."""

    duration_hours = serializers.FloatField(read_only=True)

    class Meta:
        model = TrainingJob
        fields = [
            'id', 'name', 'framework', 'strategy', 'status',
            'gpu_count', 'nodes_count', 'world_size',
            'progress_percent', 'current_epoch', 'total_epochs',
            'last_loss', 'best_loss',
            'submitted_at', 'started_at', 'duration_hours'
        ]


class ExperimentSerializer(serializers.ModelSerializer):
    """Serializer for experiments."""

    tenant_name = serializers.CharField(source='tenant.name', read_only=True)
    jobs_count = serializers.SerializerMethodField()
    best_job_name = serializers.CharField(source='best_job.name', read_only=True)

    class Meta:
        model = Experiment
        fields = [
            'id', 'tenant', 'tenant_name', 'name', 'description',
            'base_config', 'hyperparameters',
            'is_active',
            'best_job', 'best_job_name',
            'best_metric_name', 'best_metric_value', 'best_metric_minimize',
            'jobs_count', 'tags',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'tenant', 'best_job',
            'best_metric_value', 'created_at', 'updated_at'
        ]

    def get_jobs_count(self, obj):
        return obj.tenant.training_jobs.count()


class HyperparameterSweepSerializer(serializers.ModelSerializer):
    """Serializer for hyperparameter sweeps."""

    experiment_name = serializers.CharField(source='experiment.name', read_only=True)

    class Meta:
        model = HyperparameterSweep
        fields = [
            'id', 'experiment', 'experiment_name', 'name',
            'strategy', 'parameter_space',
            'objective_metric', 'objective_minimize',
            'max_trials', 'max_parallel_trials', 'early_stopping_patience',
            'completed_trials', 'best_trial_params', 'best_trial_value',
            'is_active', 'started_at', 'completed_at',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'completed_trials', 'best_trial_params', 'best_trial_value',
            'started_at', 'completed_at', 'created_at', 'updated_at'
        ]

    def validate_parameter_space(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("parameter_space must be a dictionary")

        for param, config in value.items():
            if not isinstance(config, dict):
                raise serializers.ValidationError(
                    f"Invalid config for parameter '{param}'"
                )
            if 'type' not in config:
                raise serializers.ValidationError(
                    f"Parameter '{param}' must specify a type"
                )

        return value


class JobMetricsSerializer(serializers.Serializer):
    """Serializer for job metrics update."""

    loss = serializers.FloatField(required=False)
    learning_rate = serializers.FloatField(required=False)
    epoch = serializers.IntegerField(required=False)
    step = serializers.IntegerField(required=False)
    throughput = serializers.FloatField(required=False)
    additional_metrics = serializers.DictField(required=False)


class JobStatsSerializer(serializers.Serializer):
    """Serializer for job statistics."""

    total_jobs = serializers.IntegerField()
    running_jobs = serializers.IntegerField()
    completed_jobs = serializers.IntegerField()
    failed_jobs = serializers.IntegerField()
    queued_jobs = serializers.IntegerField()
    total_gpu_hours = serializers.FloatField()
    success_rate = serializers.FloatField()
