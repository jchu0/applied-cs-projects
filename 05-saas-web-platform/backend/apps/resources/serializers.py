"""Serializers for resource management."""

from rest_framework import serializers
from .models import (
    ComputeNode, GPU, ResourceAllocation,
    ResourceQuota, ResourceReservation
)


class GPUSerializer(serializers.ModelSerializer):
    """Serializer for GPU details."""

    node_name = serializers.CharField(source='node.name', read_only=True)

    class Meta:
        model = GPU
        fields = [
            'id', 'node', 'node_name', 'device_index',
            'gpu_type', 'memory_gb', 'cuda_version', 'driver_version',
            'status', 'temperature_celsius', 'memory_used_gb',
            'utilization_percent', 'power_watts', 'metrics_updated_at'
        ]
        read_only_fields = ['id', 'node', 'device_index']


class ComputeNodeSerializer(serializers.ModelSerializer):
    """Serializer for compute nodes."""

    gpus = GPUSerializer(many=True, read_only=True)
    available_gpus = serializers.SerializerMethodField()
    total_gpu_memory_gb = serializers.IntegerField(read_only=True)

    class Meta:
        model = ComputeNode
        fields = [
            'id', 'name', 'hostname', 'gpu_type', 'gpu_count',
            'gpu_memory_gb', 'cpu_cores', 'ram_gb', 'storage_tb',
            'ip_address', 'region', 'availability_zone',
            'status', 'health_status', 'last_health_check',
            'price_per_hour', 'tags',
            'created_at', 'updated_at',
            'gpus', 'available_gpus', 'total_gpu_memory_gb'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_available_gpus(self, obj):
        return obj.gpus.filter(status='available').count()


class ComputeNodeListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for node listing."""

    available_gpus = serializers.SerializerMethodField()

    class Meta:
        model = ComputeNode
        fields = [
            'id', 'name', 'hostname', 'gpu_type', 'gpu_count',
            'region', 'status', 'price_per_hour', 'available_gpus'
        ]

    def get_available_gpus(self, obj):
        return obj.gpus.filter(status='available').count()


class ResourceAllocationSerializer(serializers.ModelSerializer):
    """Serializer for resource allocations."""

    tenant_name = serializers.CharField(source='tenant.name', read_only=True)
    node_name = serializers.CharField(source='node.name', read_only=True)
    duration_hours = serializers.FloatField(read_only=True)
    allocated_gpus = GPUSerializer(many=True, read_only=True)

    class Meta:
        model = ResourceAllocation
        fields = [
            'id', 'tenant', 'tenant_name', 'node', 'node_name',
            'requested_gpus', 'requested_memory_gb', 'requested_cpu_cores',
            'status', 'priority',
            'requested_at', 'started_at', 'completed_at',
            'max_duration_hours', 'preemptible',
            'job_id', 'job_type',
            'estimated_cost_cents', 'actual_cost_cents',
            'duration_hours', 'allocated_gpus',
            'metadata', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'tenant', 'requested_at', 'started_at', 'completed_at',
            'actual_cost_cents', 'created_at', 'updated_at'
        ]


class ResourceAllocationCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating resource allocations."""

    gpu_type = serializers.ChoiceField(
        choices=[
            ('a100_40gb', 'NVIDIA A100 40GB'),
            ('a100_80gb', 'NVIDIA A100 80GB'),
            ('h100', 'NVIDIA H100'),
            ('l40s', 'NVIDIA L40S'),
            ('a10g', 'NVIDIA A10G'),
            ('t4', 'NVIDIA T4'),
        ],
        required=False
    )
    region = serializers.CharField(required=False, allow_blank=True)

    class Meta:
        model = ResourceAllocation
        fields = [
            'requested_gpus', 'requested_memory_gb', 'requested_cpu_cores',
            'priority', 'max_duration_hours', 'preemptible',
            'job_id', 'job_type', 'metadata',
            'gpu_type', 'region'
        ]

    def validate(self, data):
        requested_gpus = data.get('requested_gpus', 1)
        if requested_gpus < 1 or requested_gpus > 16:
            raise serializers.ValidationError(
                "requested_gpus must be between 1 and 16"
            )

        max_duration = data.get('max_duration_hours', 24)
        if max_duration < 1 or max_duration > 168:
            raise serializers.ValidationError(
                "max_duration_hours must be between 1 and 168 (one week)"
            )

        return data

    def create(self, validated_data):
        from .services import ResourceManager

        gpu_type = validated_data.pop('gpu_type', None)
        region = validated_data.pop('region', None)

        request = self.context.get('request')
        tenant = getattr(request, 'tenant', None)

        if not tenant:
            raise serializers.ValidationError("Tenant context required")

        manager = ResourceManager()
        return manager.allocate(
            tenant=tenant,
            created_by=request.user,
            gpu_type=gpu_type,
            region=region,
            **validated_data
        )


class ResourceQuotaSerializer(serializers.ModelSerializer):
    """Serializer for resource quotas."""

    tenant_name = serializers.CharField(source='tenant.name', read_only=True)
    gpu_utilization_percent = serializers.FloatField(read_only=True)
    hours_remaining = serializers.FloatField(read_only=True)

    class Meta:
        model = ResourceQuota
        fields = [
            'id', 'tenant', 'tenant_name',
            'max_gpus', 'max_gpu_hours_monthly',
            'max_concurrent_jobs', 'max_storage_gb',
            'current_gpus_allocated', 'gpu_hours_used_this_month',
            'current_jobs_running', 'storage_used_gb',
            'monthly_budget_cents', 'current_month_spent_cents',
            'gpu_utilization_percent', 'hours_remaining',
            'usage_reset_at', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'tenant', 'current_gpus_allocated',
            'gpu_hours_used_this_month', 'current_jobs_running',
            'storage_used_gb', 'current_month_spent_cents',
            'usage_reset_at', 'created_at', 'updated_at'
        ]


class ResourceReservationSerializer(serializers.ModelSerializer):
    """Serializer for resource reservations."""

    tenant_name = serializers.CharField(source='tenant.name', read_only=True)
    duration_hours = serializers.FloatField(read_only=True)

    class Meta:
        model = ResourceReservation
        fields = [
            'id', 'tenant', 'tenant_name',
            'gpu_type', 'gpu_count', 'region',
            'reservation_type', 'starts_at', 'ends_at',
            'recurring_schedule', 'is_active',
            'allocation', 'discount_percent', 'prepaid_amount_cents',
            'duration_hours', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'tenant', 'allocation',
            'created_at', 'updated_at'
        ]

    def validate(self, data):
        starts_at = data.get('starts_at')
        ends_at = data.get('ends_at')

        if starts_at and ends_at and starts_at >= ends_at:
            raise serializers.ValidationError(
                "ends_at must be after starts_at"
            )

        return data


class ResourceStatsSerializer(serializers.Serializer):
    """Serializer for resource statistics."""

    total_nodes = serializers.IntegerField()
    total_gpus = serializers.IntegerField()
    available_gpus = serializers.IntegerField()
    allocated_gpus = serializers.IntegerField()
    total_allocations = serializers.IntegerField()
    active_allocations = serializers.IntegerField()
    regions = serializers.ListField(child=serializers.DictField())


class GPUAvailabilitySerializer(serializers.Serializer):
    """Serializer for GPU availability query."""

    gpu_type = serializers.CharField(required=False)
    gpu_count = serializers.IntegerField(min_value=1, max_value=16, default=1)
    region = serializers.CharField(required=False, allow_blank=True)
