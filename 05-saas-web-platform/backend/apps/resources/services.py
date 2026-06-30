"""Resource management services."""

from typing import Optional, List, Dict, Any
from django.db import transaction
from django.db.models import Q, Count, Sum, F
from django.utils import timezone
import logging

from apps.tenants.models import Tenant
from .models import (
    ComputeNode, GPU, ResourceAllocation, ResourceQuota,
    ResourceReservation, ResourceStatus, GPUType
)

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages GPU and compute resource allocation."""

    def find_available_nodes(
        self,
        gpu_type: Optional[str] = None,
        gpu_count: int = 1,
        region: Optional[str] = None,
    ) -> List[ComputeNode]:
        """Find nodes with available GPUs matching requirements."""
        queryset = ComputeNode.objects.filter(
            status=ResourceStatus.AVAILABLE
        ).annotate(
            available_gpu_count=Count(
                'gpus',
                filter=Q(gpus__status=ResourceStatus.AVAILABLE)
            )
        ).filter(
            available_gpu_count__gte=gpu_count
        )

        if gpu_type:
            queryset = queryset.filter(gpu_type=gpu_type)

        if region:
            queryset = queryset.filter(region=region)

        return list(queryset.order_by('price_per_hour'))

    def check_quota(self, tenant: Tenant, gpu_count: int) -> tuple[bool, str]:
        """Check if tenant has sufficient quota for allocation."""
        try:
            quota = tenant.resource_quota
        except ResourceQuota.DoesNotExist:
            quota = ResourceQuota.objects.create(tenant=tenant)

        if not quota.can_allocate_gpus(gpu_count):
            return False, f"GPU quota exceeded. Max: {quota.max_gpus}, Current: {quota.current_gpus_allocated}"

        if quota.hours_remaining <= 0:
            return False, "Monthly GPU hours quota exhausted"

        if quota.current_jobs_running >= quota.max_concurrent_jobs:
            return False, f"Maximum concurrent jobs ({quota.max_concurrent_jobs}) reached"

        if quota.monthly_budget_cents and quota.current_month_spent_cents >= quota.monthly_budget_cents:
            return False, "Monthly budget exhausted"

        return True, ""

    @transaction.atomic
    def allocate(
        self,
        tenant: Tenant,
        created_by,
        requested_gpus: int = 1,
        gpu_type: Optional[str] = None,
        region: Optional[str] = None,
        priority: int = 5,
        max_duration_hours: int = 24,
        preemptible: bool = False,
        job_id: str = "",
        job_type: str = "",
        requested_memory_gb: Optional[int] = None,
        requested_cpu_cores: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResourceAllocation:
        """Allocate GPU resources to a tenant."""

        # Check quota
        can_allocate, reason = self.check_quota(tenant, requested_gpus)
        if not can_allocate:
            raise ValueError(reason)

        # Find available node
        nodes = self.find_available_nodes(
            gpu_type=gpu_type,
            gpu_count=requested_gpus,
            region=region,
        )

        if not nodes:
            raise ValueError(
                f"No available nodes with {requested_gpus} GPUs"
                f"{' of type ' + gpu_type if gpu_type else ''}"
                f"{' in region ' + region if region else ''}"
            )

        node = nodes[0]

        # Calculate estimated cost
        estimated_cost = node.price_per_hour * requested_gpus * max_duration_hours

        # Create allocation
        allocation = ResourceAllocation.objects.create(
            tenant=tenant,
            node=node,
            requested_gpus=requested_gpus,
            requested_memory_gb=requested_memory_gb,
            requested_cpu_cores=requested_cpu_cores,
            priority=priority,
            max_duration_hours=max_duration_hours,
            preemptible=preemptible,
            job_id=job_id,
            job_type=job_type,
            estimated_cost_cents=estimated_cost,
            metadata=metadata or {},
            created_by=created_by,
            status=ResourceAllocation.AllocationStatus.PENDING,
        )

        # Allocate GPUs
        available_gpus = node.gpus.filter(
            status=ResourceStatus.AVAILABLE
        ).order_by('device_index')[:requested_gpus]

        for gpu in available_gpus:
            gpu.status = ResourceStatus.ALLOCATED
            gpu.current_allocation = allocation
            gpu.save()

        # Update quota
        quota = tenant.resource_quota
        quota.current_gpus_allocated += requested_gpus
        quota.current_jobs_running += 1
        quota.save()

        # Mark as active
        allocation.status = ResourceAllocation.AllocationStatus.ACTIVE
        allocation.started_at = timezone.now()
        allocation.save()

        logger.info(
            f"Allocated {requested_gpus} GPUs on {node.name} to {tenant.name} "
            f"(allocation_id={allocation.id})"
        )

        return allocation

    @transaction.atomic
    def release(self, allocation: ResourceAllocation) -> None:
        """Release allocated resources."""
        if allocation.status not in [
            ResourceAllocation.AllocationStatus.ACTIVE,
            ResourceAllocation.AllocationStatus.PREEMPTED,
        ]:
            return

        # Release GPUs
        allocation.allocated_gpus.update(
            status=ResourceStatus.AVAILABLE,
            current_allocation=None,
        )

        # Calculate actual cost
        allocation.actual_cost_cents = allocation.calculate_cost()
        allocation.completed_at = timezone.now()
        allocation.status = ResourceAllocation.AllocationStatus.COMPLETED
        allocation.save()

        # Update quota
        tenant = allocation.tenant
        try:
            quota = tenant.resource_quota
            quota.current_gpus_allocated = max(0, quota.current_gpus_allocated - allocation.requested_gpus)
            quota.current_jobs_running = max(0, quota.current_jobs_running - 1)
            quota.gpu_hours_used_this_month += allocation.duration_hours
            quota.current_month_spent_cents += allocation.actual_cost_cents
            quota.save()
        except ResourceQuota.DoesNotExist:
            pass

        logger.info(
            f"Released allocation {allocation.id} - "
            f"Duration: {allocation.duration_hours:.2f}h, Cost: ${allocation.actual_cost_cents/100:.2f}"
        )

    @transaction.atomic
    def preempt(self, allocation: ResourceAllocation, reason: str = "") -> None:
        """Preempt an allocation (for higher priority jobs)."""
        if not allocation.preemptible:
            raise ValueError("Allocation is not preemptible")

        if allocation.status != ResourceAllocation.AllocationStatus.ACTIVE:
            return

        allocation.status = ResourceAllocation.AllocationStatus.PREEMPTED
        allocation.metadata['preempt_reason'] = reason
        allocation.save()

        self.release(allocation)

        logger.info(f"Preempted allocation {allocation.id}: {reason}")

    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        nodes = ComputeNode.objects.all()
        gpus = GPU.objects.all()

        total_nodes = nodes.count()
        total_gpus = gpus.count()
        available_gpus = gpus.filter(status=ResourceStatus.AVAILABLE).count()
        allocated_gpus = gpus.filter(status=ResourceStatus.ALLOCATED).count()

        allocations = ResourceAllocation.objects.all()
        total_allocations = allocations.count()
        active_allocations = allocations.filter(
            status=ResourceAllocation.AllocationStatus.ACTIVE
        ).count()

        # Region breakdown
        regions = list(nodes.values('region').annotate(
            node_count=Count('id'),
            gpu_count=Sum('gpu_count'),
        ))

        return {
            'total_nodes': total_nodes,
            'total_gpus': total_gpus,
            'available_gpus': available_gpus,
            'allocated_gpus': allocated_gpus,
            'total_allocations': total_allocations,
            'active_allocations': active_allocations,
            'regions': regions,
        }

    def get_availability(
        self,
        gpu_type: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get GPU availability summary."""
        queryset = GPU.objects.filter(status=ResourceStatus.AVAILABLE)

        if gpu_type:
            queryset = queryset.filter(gpu_type=gpu_type)
        if region:
            queryset = queryset.filter(node__region=region)

        by_type = list(queryset.values('gpu_type').annotate(
            available=Count('id')
        ))

        by_region = list(queryset.values('node__region').annotate(
            available=Count('id')
        ))

        return {
            'total_available': queryset.count(),
            'by_type': by_type,
            'by_region': by_region,
        }


class QuotaManager:
    """Manages tenant resource quotas."""

    def get_or_create_quota(self, tenant: Tenant) -> ResourceQuota:
        """Get or create quota for tenant."""
        quota, _ = ResourceQuota.objects.get_or_create(
            tenant=tenant,
            defaults={
                'max_gpus': 4,
                'max_gpu_hours_monthly': 100,
                'max_concurrent_jobs': 5,
                'max_storage_gb': 100,
            }
        )
        return quota

    def update_quota(
        self,
        tenant: Tenant,
        max_gpus: Optional[int] = None,
        max_gpu_hours_monthly: Optional[int] = None,
        max_concurrent_jobs: Optional[int] = None,
        max_storage_gb: Optional[int] = None,
        monthly_budget_cents: Optional[int] = None,
    ) -> ResourceQuota:
        """Update tenant quota limits."""
        quota = self.get_or_create_quota(tenant)

        if max_gpus is not None:
            quota.max_gpus = max_gpus
        if max_gpu_hours_monthly is not None:
            quota.max_gpu_hours_monthly = max_gpu_hours_monthly
        if max_concurrent_jobs is not None:
            quota.max_concurrent_jobs = max_concurrent_jobs
        if max_storage_gb is not None:
            quota.max_storage_gb = max_storage_gb
        if monthly_budget_cents is not None:
            quota.monthly_budget_cents = monthly_budget_cents

        quota.save()
        return quota

    def reset_monthly_usage(self, tenant: Tenant) -> None:
        """Reset monthly usage counters."""
        try:
            quota = tenant.resource_quota
            quota.gpu_hours_used_this_month = 0
            quota.current_month_spent_cents = 0
            quota.usage_reset_at = timezone.now()
            quota.save()
        except ResourceQuota.DoesNotExist:
            pass


class ReservationManager:
    """Manages resource reservations."""

    def create_reservation(
        self,
        tenant: Tenant,
        created_by,
        gpu_type: str,
        gpu_count: int,
        starts_at,
        ends_at,
        region: str = "",
        reservation_type: str = "scheduled",
        recurring_schedule: Optional[Dict] = None,
        discount_percent: int = 0,
    ) -> ResourceReservation:
        """Create a resource reservation."""
        # Check availability for time slot
        overlapping = ResourceReservation.objects.filter(
            gpu_type=gpu_type,
            is_active=True,
            starts_at__lt=ends_at,
            ends_at__gt=starts_at,
        )

        if region:
            overlapping = overlapping.filter(region=region)

        total_reserved = overlapping.aggregate(
            total=Sum('gpu_count')
        )['total'] or 0

        # Get total available GPUs of this type
        total_available = GPU.objects.filter(gpu_type=gpu_type)
        if region:
            total_available = total_available.filter(node__region=region)
        total_available = total_available.count()

        if total_reserved + gpu_count > total_available:
            raise ValueError(
                f"Insufficient GPUs available for reservation. "
                f"Requested: {gpu_count}, Available: {total_available - total_reserved}"
            )

        reservation = ResourceReservation.objects.create(
            tenant=tenant,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            region=region,
            reservation_type=reservation_type,
            starts_at=starts_at,
            ends_at=ends_at,
            recurring_schedule=recurring_schedule,
            discount_percent=discount_percent,
            created_by=created_by,
        )

        logger.info(
            f"Created reservation {reservation.id} for {tenant.name}: "
            f"{gpu_count}x {gpu_type} from {starts_at} to {ends_at}"
        )

        return reservation

    def cancel_reservation(self, reservation: ResourceReservation) -> None:
        """Cancel a reservation."""
        reservation.is_active = False
        reservation.save()

        if reservation.allocation:
            ResourceManager().release(reservation.allocation)

        logger.info(f"Cancelled reservation {reservation.id}")

    def activate_reservation(self, reservation: ResourceReservation) -> Optional[ResourceAllocation]:
        """Activate a reservation by creating an allocation."""
        if not reservation.is_active:
            return None

        if reservation.allocation:
            return reservation.allocation

        manager = ResourceManager()
        try:
            allocation = manager.allocate(
                tenant=reservation.tenant,
                created_by=reservation.created_by,
                requested_gpus=reservation.gpu_count,
                gpu_type=reservation.gpu_type,
                region=reservation.region or None,
                max_duration_hours=int(reservation.duration_hours),
            )

            reservation.allocation = allocation
            reservation.save()

            return allocation

        except ValueError as e:
            logger.error(f"Failed to activate reservation {reservation.id}: {e}")
            return None

    def get_upcoming_reservations(
        self,
        tenant: Optional[Tenant] = None,
        hours_ahead: int = 24,
    ) -> List[ResourceReservation]:
        """Get reservations starting soon."""
        now = timezone.now()
        cutoff = now + timezone.timedelta(hours=hours_ahead)

        queryset = ResourceReservation.objects.filter(
            is_active=True,
            allocation__isnull=True,
            starts_at__gte=now,
            starts_at__lte=cutoff,
        )

        if tenant:
            queryset = queryset.filter(tenant=tenant)

        return list(queryset.order_by('starts_at'))
