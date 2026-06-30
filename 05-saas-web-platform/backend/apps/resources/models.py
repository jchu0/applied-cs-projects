"""
Models for GPU and compute resource management.
"""
import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone
from apps.tenants.models import Tenant


class GPUType(models.TextChoices):
    """Supported GPU types."""
    NVIDIA_A100_40GB = 'a100_40gb', 'NVIDIA A100 40GB'
    NVIDIA_A100_80GB = 'a100_80gb', 'NVIDIA A100 80GB'
    NVIDIA_H100 = 'h100', 'NVIDIA H100'
    NVIDIA_L40S = 'l40s', 'NVIDIA L40S'
    NVIDIA_A10G = 'a10g', 'NVIDIA A10G'
    NVIDIA_T4 = 't4', 'NVIDIA T4'


class ResourceStatus(models.TextChoices):
    """Resource availability status."""
    AVAILABLE = 'available', 'Available'
    ALLOCATED = 'allocated', 'Allocated'
    MAINTENANCE = 'maintenance', 'Under Maintenance'
    OFFLINE = 'offline', 'Offline'
    RESERVED = 'reserved', 'Reserved'


class ComputeNode(models.Model):
    """Physical or virtual compute node with GPU resources."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    hostname = models.CharField(max_length=255, unique=True)

    # Hardware specifications
    gpu_type = models.CharField(max_length=50, choices=GPUType.choices)
    gpu_count = models.IntegerField(default=1)
    gpu_memory_gb = models.IntegerField()
    cpu_cores = models.IntegerField()
    ram_gb = models.IntegerField()
    storage_tb = models.DecimalField(max_digits=5, decimal_places=2)

    # Network
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    region = models.CharField(max_length=50)
    availability_zone = models.CharField(max_length=50, blank=True)

    # Status
    status = models.CharField(
        max_length=20,
        choices=ResourceStatus.choices,
        default=ResourceStatus.AVAILABLE
    )
    health_status = models.JSONField(default=dict)
    last_health_check = models.DateTimeField(null=True, blank=True)

    # Pricing (per hour in cents)
    price_per_hour = models.IntegerField(help_text="Price per hour in cents")

    # Metadata
    tags = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'compute_nodes'
        ordering = ['region', 'name']
        indexes = [
            models.Index(fields=['gpu_type', 'status']),
            models.Index(fields=['region', 'status']),
        ]

    def __str__(self):
        return f"{self.name} ({self.gpu_count}x {self.get_gpu_type_display()})"

    @property
    def is_available(self):
        return self.status == ResourceStatus.AVAILABLE

    @property
    def total_gpu_memory_gb(self):
        return self.gpu_count * self.gpu_memory_gb


class GPU(models.Model):
    """Individual GPU device within a compute node."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    node = models.ForeignKey(
        ComputeNode,
        on_delete=models.CASCADE,
        related_name='gpus'
    )
    device_index = models.IntegerField()

    # GPU details
    gpu_type = models.CharField(max_length=50, choices=GPUType.choices)
    memory_gb = models.IntegerField()
    cuda_version = models.CharField(max_length=20, blank=True)
    driver_version = models.CharField(max_length=50, blank=True)

    # Status
    status = models.CharField(
        max_length=20,
        choices=ResourceStatus.choices,
        default=ResourceStatus.AVAILABLE
    )

    # Current metrics
    temperature_celsius = models.IntegerField(null=True, blank=True)
    memory_used_gb = models.FloatField(null=True, blank=True)
    utilization_percent = models.IntegerField(null=True, blank=True)
    power_watts = models.IntegerField(null=True, blank=True)
    metrics_updated_at = models.DateTimeField(null=True, blank=True)

    # Allocation
    current_allocation = models.ForeignKey(
        'ResourceAllocation',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='allocated_gpus'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'gpus'
        unique_together = ['node', 'device_index']
        ordering = ['node', 'device_index']

    def __str__(self):
        return f"{self.node.name} GPU:{self.device_index} ({self.get_gpu_type_display()})"


class ResourceAllocation(models.Model):
    """GPU/compute resource allocation to a tenant/job."""

    class AllocationStatus(models.TextChoices):
        PENDING = 'pending', 'Pending'
        ACTIVE = 'active', 'Active'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
        CANCELLED = 'cancelled', 'Cancelled'
        PREEMPTED = 'preempted', 'Preempted'

    class Priority(models.IntegerChoices):
        LOW = 1, 'Low'
        NORMAL = 5, 'Normal'
        HIGH = 10, 'High'
        CRITICAL = 20, 'Critical'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        Tenant,
        on_delete=models.CASCADE,
        related_name='resource_allocations'
    )
    node = models.ForeignKey(
        ComputeNode,
        on_delete=models.CASCADE,
        related_name='allocations'
    )

    # Allocation request
    requested_gpus = models.IntegerField(default=1)
    requested_memory_gb = models.IntegerField(null=True, blank=True)
    requested_cpu_cores = models.IntegerField(null=True, blank=True)

    # Status
    status = models.CharField(
        max_length=20,
        choices=AllocationStatus.choices,
        default=AllocationStatus.PENDING
    )
    priority = models.IntegerField(
        choices=Priority.choices,
        default=Priority.NORMAL
    )

    # Scheduling
    requested_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    max_duration_hours = models.IntegerField(default=24)
    preemptible = models.BooleanField(default=False)

    # Job reference
    job_id = models.CharField(max_length=255, blank=True)
    job_type = models.CharField(max_length=50, blank=True)

    # Cost tracking
    estimated_cost_cents = models.IntegerField(null=True, blank=True)
    actual_cost_cents = models.IntegerField(null=True, blank=True)

    # Metadata
    metadata = models.JSONField(default=dict)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='resource_allocations'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'resource_allocations'
        ordering = ['-priority', 'requested_at']
        indexes = [
            models.Index(fields=['tenant', 'status']),
            models.Index(fields=['node', 'status']),
            models.Index(fields=['status', 'priority']),
        ]

    def __str__(self):
        return f"Allocation {self.id} - {self.tenant.name} ({self.requested_gpus} GPUs)"

    @property
    def duration_hours(self):
        if not self.started_at:
            return 0
        end = self.completed_at or timezone.now()
        return (end - self.started_at).total_seconds() / 3600

    def calculate_cost(self):
        """Calculate actual cost based on usage."""
        hours = self.duration_hours
        hourly_rate = self.node.price_per_hour * self.requested_gpus
        return int(hours * hourly_rate)


class ResourceQuota(models.Model):
    """Resource usage quotas for tenants."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.OneToOneField(
        Tenant,
        on_delete=models.CASCADE,
        related_name='resource_quota'
    )

    # GPU quotas
    max_gpus = models.IntegerField(default=4)
    max_gpu_hours_monthly = models.IntegerField(default=100)

    # Compute quotas
    max_concurrent_jobs = models.IntegerField(default=5)
    max_storage_gb = models.IntegerField(default=100)

    # Usage tracking
    current_gpus_allocated = models.IntegerField(default=0)
    gpu_hours_used_this_month = models.FloatField(default=0)
    current_jobs_running = models.IntegerField(default=0)
    storage_used_gb = models.IntegerField(default=0)

    # Budget
    monthly_budget_cents = models.IntegerField(null=True, blank=True)
    current_month_spent_cents = models.IntegerField(default=0)

    # Timestamps
    usage_reset_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'resource_quotas'

    def __str__(self):
        return f"Quota for {self.tenant.name}"

    @property
    def gpu_utilization_percent(self):
        if self.max_gpus == 0:
            return 0
        return (self.current_gpus_allocated / self.max_gpus) * 100

    @property
    def hours_remaining(self):
        return max(0, self.max_gpu_hours_monthly - self.gpu_hours_used_this_month)

    def can_allocate_gpus(self, count: int) -> bool:
        return (self.current_gpus_allocated + count) <= self.max_gpus


class ResourceReservation(models.Model):
    """Future resource reservation."""

    class ReservationType(models.TextChoices):
        SCHEDULED = 'scheduled', 'Scheduled'
        RECURRING = 'recurring', 'Recurring'
        ON_DEMAND = 'on_demand', 'On-Demand'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        Tenant,
        on_delete=models.CASCADE,
        related_name='resource_reservations'
    )

    # Resource requirements
    gpu_type = models.CharField(max_length=50, choices=GPUType.choices)
    gpu_count = models.IntegerField(default=1)
    region = models.CharField(max_length=50, blank=True)

    # Scheduling
    reservation_type = models.CharField(
        max_length=20,
        choices=ReservationType.choices,
        default=ReservationType.SCHEDULED
    )
    starts_at = models.DateTimeField()
    ends_at = models.DateTimeField()
    recurring_schedule = models.JSONField(null=True, blank=True)

    # Status
    is_active = models.BooleanField(default=True)
    allocation = models.OneToOneField(
        ResourceAllocation,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reservation'
    )

    # Pricing
    discount_percent = models.IntegerField(default=0)
    prepaid_amount_cents = models.IntegerField(default=0)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'resource_reservations'
        ordering = ['starts_at']

    def __str__(self):
        return f"Reservation {self.tenant.name} - {self.gpu_count}x {self.gpu_type}"

    @property
    def duration_hours(self):
        return (self.ends_at - self.starts_at).total_seconds() / 3600
