"""Views for resource management."""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Count, Q

from .models import (
    ComputeNode, GPU, ResourceAllocation,
    ResourceQuota, ResourceReservation, ResourceStatus
)
from .serializers import (
    ComputeNodeSerializer, ComputeNodeListSerializer,
    GPUSerializer, ResourceAllocationSerializer,
    ResourceAllocationCreateSerializer, ResourceQuotaSerializer,
    ResourceReservationSerializer, GPUAvailabilitySerializer,
)
from .services import ResourceManager, QuotaManager, ReservationManager


class ComputeNodeViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for compute nodes (read-only for non-admins)."""

    permission_classes = [IsAuthenticated]
    serializer_class = ComputeNodeSerializer

    def get_queryset(self):
        queryset = ComputeNode.objects.all()

        # Filter by status
        node_status = self.request.query_params.get('status')
        if node_status:
            queryset = queryset.filter(status=node_status)

        # Filter by GPU type
        gpu_type = self.request.query_params.get('gpu_type')
        if gpu_type:
            queryset = queryset.filter(gpu_type=gpu_type)

        # Filter by region
        region = self.request.query_params.get('region')
        if region:
            queryset = queryset.filter(region=region)

        return queryset.prefetch_related('gpus')

    def get_serializer_class(self):
        if self.action == 'list':
            return ComputeNodeListSerializer
        return ComputeNodeSerializer

    @action(detail=False, methods=['get'])
    def regions(self, request):
        """Get available regions."""
        regions = ComputeNode.objects.values('region').annotate(
            node_count=Count('id'),
        ).order_by('region')

        return Response(list(regions))

    @action(detail=False, methods=['get'])
    def gpu_types(self, request):
        """Get available GPU types."""
        types = ComputeNode.objects.values('gpu_type').annotate(
            count=Count('id'),
        ).order_by('gpu_type')

        return Response(list(types))

    @action(detail=True, methods=['get'])
    def gpus(self, request, pk=None):
        """Get GPUs for a specific node."""
        node = self.get_object()
        gpus = node.gpus.all()
        serializer = GPUSerializer(gpus, many=True)
        return Response(serializer.data)


class GPUViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for GPUs (read-only)."""

    permission_classes = [IsAuthenticated]
    serializer_class = GPUSerializer

    def get_queryset(self):
        queryset = GPU.objects.all()

        # Filter by status
        gpu_status = self.request.query_params.get('status')
        if gpu_status:
            queryset = queryset.filter(status=gpu_status)

        # Filter by type
        gpu_type = self.request.query_params.get('gpu_type')
        if gpu_type:
            queryset = queryset.filter(gpu_type=gpu_type)

        # Filter by region
        region = self.request.query_params.get('region')
        if region:
            queryset = queryset.filter(node__region=region)

        return queryset.select_related('node')

    @action(detail=False, methods=['get'])
    def available(self, request):
        """Get available GPUs."""
        queryset = self.get_queryset().filter(status=ResourceStatus.AVAILABLE)
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def availability(self, request):
        """Check GPU availability."""
        serializer = GPUAvailabilitySerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)

        manager = ResourceManager()
        availability = manager.get_availability(
            gpu_type=serializer.validated_data.get('gpu_type'),
            region=serializer.validated_data.get('region'),
        )

        return Response(availability)


class ResourceAllocationViewSet(viewsets.ModelViewSet):
    """ViewSet for resource allocations."""

    permission_classes = [IsAuthenticated]
    serializer_class = ResourceAllocationSerializer

    def get_queryset(self):
        queryset = ResourceAllocation.objects.all()
        tenant = getattr(self.request, 'tenant', None)

        if tenant:
            queryset = queryset.filter(tenant=tenant)

        # Filter by status
        alloc_status = self.request.query_params.get('status')
        if alloc_status:
            queryset = queryset.filter(status=alloc_status)

        return queryset.select_related('tenant', 'node', 'created_by')

    def get_serializer_class(self):
        if self.action == 'create':
            return ResourceAllocationCreateSerializer
        return ResourceAllocationSerializer

    @action(detail=True, methods=['post'])
    def release(self, request, pk=None):
        """Release an allocation."""
        allocation = self.get_object()

        if allocation.status != ResourceAllocation.AllocationStatus.ACTIVE:
            return Response(
                {'error': 'Allocation is not active'},
                status=status.HTTP_400_BAD_REQUEST
            )

        manager = ResourceManager()
        manager.release(allocation)

        return Response({
            'status': 'released',
            'actual_cost_cents': allocation.actual_cost_cents,
            'duration_hours': allocation.duration_hours,
        })

    @action(detail=True, methods=['post'])
    def extend(self, request, pk=None):
        """Extend allocation duration."""
        allocation = self.get_object()
        hours = request.data.get('hours', 1)

        if allocation.status != ResourceAllocation.AllocationStatus.ACTIVE:
            return Response(
                {'error': 'Allocation is not active'},
                status=status.HTTP_400_BAD_REQUEST
            )

        allocation.max_duration_hours += hours
        allocation.save()

        return Response({
            'max_duration_hours': allocation.max_duration_hours,
            'estimated_cost_cents': allocation.estimated_cost_cents,
        })

    @action(detail=False, methods=['get'])
    def active(self, request):
        """Get active allocations."""
        queryset = self.get_queryset().filter(
            status=ResourceAllocation.AllocationStatus.ACTIVE
        )
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get resource statistics."""
        manager = ResourceManager()
        return Response(manager.get_stats())


class ResourceQuotaViewSet(viewsets.ModelViewSet):
    """ViewSet for resource quotas."""

    permission_classes = [IsAuthenticated]
    serializer_class = ResourceQuotaSerializer

    def get_queryset(self):
        queryset = ResourceQuota.objects.all()
        tenant = getattr(self.request, 'tenant', None)

        if tenant:
            queryset = queryset.filter(tenant=tenant)

        return queryset.select_related('tenant')

    @action(detail=False, methods=['get'])
    def current(self, request):
        """Get quota for current tenant."""
        tenant = getattr(request, 'tenant', None)
        if not tenant:
            return Response(
                {'error': 'No tenant context'},
                status=status.HTTP_400_BAD_REQUEST
            )

        manager = QuotaManager()
        quota = manager.get_or_create_quota(tenant)
        serializer = self.get_serializer(quota)

        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def reset_usage(self, request, pk=None):
        """Reset monthly usage for a quota."""
        quota = self.get_object()
        manager = QuotaManager()
        manager.reset_monthly_usage(quota.tenant)

        quota.refresh_from_db()
        serializer = self.get_serializer(quota)

        return Response(serializer.data)


class ResourceReservationViewSet(viewsets.ModelViewSet):
    """ViewSet for resource reservations."""

    permission_classes = [IsAuthenticated]
    serializer_class = ResourceReservationSerializer

    def get_queryset(self):
        queryset = ResourceReservation.objects.all()
        tenant = getattr(self.request, 'tenant', None)

        if tenant:
            queryset = queryset.filter(tenant=tenant)

        return queryset.select_related('tenant', 'allocation', 'created_by')

    def create(self, request, *args, **kwargs):
        """Create a reservation."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        tenant = getattr(request, 'tenant', None)
        if not tenant:
            return Response(
                {'error': 'No tenant context'},
                status=status.HTTP_400_BAD_REQUEST
            )

        manager = ReservationManager()
        try:
            reservation = manager.create_reservation(
                tenant=tenant,
                created_by=request.user,
                **serializer.validated_data
            )
            return Response(
                self.get_serializer(reservation).data,
                status=status.HTTP_201_CREATED
            )
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Cancel a reservation."""
        reservation = self.get_object()
        manager = ReservationManager()
        manager.cancel_reservation(reservation)

        return Response({'status': 'cancelled'})

    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """Activate a reservation (create allocation)."""
        reservation = self.get_object()
        manager = ReservationManager()

        allocation = manager.activate_reservation(reservation)
        if allocation:
            return Response({
                'status': 'activated',
                'allocation_id': str(allocation.id),
            })
        return Response(
            {'error': 'Failed to activate reservation'},
            status=status.HTTP_400_BAD_REQUEST
        )

    @action(detail=False, methods=['get'])
    def upcoming(self, request):
        """Get upcoming reservations."""
        tenant = getattr(request, 'tenant', None)
        hours = int(request.query_params.get('hours', 24))

        manager = ReservationManager()
        reservations = manager.get_upcoming_reservations(
            tenant=tenant,
            hours_ahead=hours,
        )

        serializer = self.get_serializer(reservations, many=True)
        return Response(serializer.data)
