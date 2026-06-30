"""Tests for resources app."""

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


class TestResourceModels:
    """Tests for resource models."""

    def test_compute_node_creation(self, db):
        """Test creating a compute node."""
        from apps.resources.models import ComputeNode, GPUType, ResourceStatus

        node = ComputeNode.objects.create(
            name='gpu-node-1',
            hostname='gpu-node-1.cluster.local',
            gpu_type=GPUType.NVIDIA_A100_80GB,
            gpu_count=8,
            gpu_memory_gb=80,
            cpu_cores=128,
            ram_gb=1024,
            storage_tb=10.0,
            region='us-east-1',
            price_per_hour=3200,
        )

        assert node.id is not None
        assert node.name == 'gpu-node-1'
        assert node.gpu_type == GPUType.NVIDIA_A100_80GB
        assert node.total_gpu_memory_gb == 640  # 8 * 80
        assert node.is_available is True

    def test_gpu_creation(self, db):
        """Test creating a GPU."""
        from apps.resources.models import ComputeNode, GPU, GPUType, ResourceStatus

        node = ComputeNode.objects.create(
            name='gpu-node-1',
            hostname='gpu-node-1.cluster.local',
            gpu_type=GPUType.NVIDIA_H100,
            gpu_count=8,
            gpu_memory_gb=80,
            cpu_cores=128,
            ram_gb=1024,
            storage_tb=10.0,
            region='us-east-1',
            price_per_hour=4000,
        )

        gpu = GPU.objects.create(
            node=node,
            device_index=0,
            gpu_type=GPUType.NVIDIA_H100,
            memory_gb=80,
            cuda_version='12.2',
            driver_version='535.104.05',
        )

        assert gpu.id is not None
        assert gpu.node == node
        assert gpu.device_index == 0
        assert gpu.status == ResourceStatus.AVAILABLE

    def test_resource_allocation_creation(self, db, tenant_factory):
        """Test creating a resource allocation."""
        from apps.resources.models import (
            ComputeNode, GPU, ResourceAllocation, GPUType
        )

        tenant = tenant_factory(slug='test-alloc')
        node = ComputeNode.objects.create(
            name='gpu-node-1',
            hostname='gpu-node-1.cluster.local',
            gpu_type=GPUType.NVIDIA_A100_40GB,
            gpu_count=8,
            gpu_memory_gb=40,
            cpu_cores=64,
            ram_gb=512,
            storage_tb=5.0,
            region='us-west-2',
            price_per_hour=2400,
        )

        allocation = ResourceAllocation.objects.create(
            tenant=tenant,
            node=node,
            requested_gpus=4,
            max_duration_hours=24,
            priority=5,
        )

        assert allocation.id is not None
        assert allocation.tenant == tenant
        assert allocation.node == node
        assert allocation.requested_gpus == 4

    def test_resource_quota_creation(self, db, tenant_factory):
        """Test creating a resource quota."""
        from apps.resources.models import ResourceQuota

        tenant = tenant_factory(slug='test-quota')

        quota = ResourceQuota.objects.create(
            tenant=tenant,
            max_gpus=8,
            max_gpu_hours_monthly=500,
            max_concurrent_jobs=10,
            max_storage_gb=500,
        )

        assert quota.id is not None
        assert quota.max_gpus == 8
        assert quota.gpu_utilization_percent == 0
        assert quota.can_allocate_gpus(4) is True
        assert quota.can_allocate_gpus(10) is False

    def test_resource_reservation_creation(self, db, tenant_factory):
        """Test creating a resource reservation."""
        from apps.resources.models import ResourceReservation, GPUType

        tenant = tenant_factory(slug='test-reservation')
        now = timezone.now()

        reservation = ResourceReservation.objects.create(
            tenant=tenant,
            gpu_type=GPUType.NVIDIA_A100_80GB,
            gpu_count=4,
            region='us-east-1',
            starts_at=now + timezone.timedelta(hours=1),
            ends_at=now + timezone.timedelta(hours=5),
        )

        assert reservation.id is not None
        assert reservation.gpu_count == 4
        assert reservation.duration_hours == 4


class TestResourceManager:
    """Tests for ResourceManager service."""

    @pytest.fixture
    def setup_nodes(self, db):
        """Create test compute nodes and GPUs."""
        from apps.resources.models import ComputeNode, GPU, GPUType

        node1 = ComputeNode.objects.create(
            name='node-1',
            hostname='node-1.local',
            gpu_type=GPUType.NVIDIA_A100_80GB,
            gpu_count=4,
            gpu_memory_gb=80,
            cpu_cores=64,
            ram_gb=512,
            storage_tb=5.0,
            region='us-east-1',
            price_per_hour=3200,
        )

        for i in range(4):
            GPU.objects.create(
                node=node1,
                device_index=i,
                gpu_type=GPUType.NVIDIA_A100_80GB,
                memory_gb=80,
            )

        node2 = ComputeNode.objects.create(
            name='node-2',
            hostname='node-2.local',
            gpu_type=GPUType.NVIDIA_H100,
            gpu_count=8,
            gpu_memory_gb=80,
            cpu_cores=128,
            ram_gb=1024,
            storage_tb=10.0,
            region='us-west-2',
            price_per_hour=4000,
        )

        for i in range(8):
            GPU.objects.create(
                node=node2,
                device_index=i,
                gpu_type=GPUType.NVIDIA_H100,
                memory_gb=80,
            )

        return node1, node2

    def test_find_available_nodes(self, setup_nodes):
        """Test finding available nodes."""
        from apps.resources.services import ResourceManager

        manager = ResourceManager()
        nodes = manager.find_available_nodes(gpu_count=4)

        assert len(nodes) >= 1

    def test_find_available_nodes_by_type(self, setup_nodes):
        """Test finding available nodes by GPU type."""
        from apps.resources.services import ResourceManager
        from apps.resources.models import GPUType

        manager = ResourceManager()
        nodes = manager.find_available_nodes(
            gpu_type=GPUType.NVIDIA_H100,
            gpu_count=4,
        )

        assert len(nodes) == 1
        assert nodes[0].gpu_type == GPUType.NVIDIA_H100

    def test_find_available_nodes_by_region(self, setup_nodes):
        """Test finding available nodes by region."""
        from apps.resources.services import ResourceManager

        manager = ResourceManager()
        nodes = manager.find_available_nodes(
            gpu_count=2,
            region='us-east-1',
        )

        assert len(nodes) == 1
        assert nodes[0].region == 'us-east-1'

    def test_check_quota(self, db, tenant_factory):
        """Test quota checking."""
        from apps.resources.services import ResourceManager
        from apps.resources.models import ResourceQuota

        tenant = tenant_factory(slug='test-check-quota')
        ResourceQuota.objects.create(
            tenant=tenant,
            max_gpus=8,
            max_gpu_hours_monthly=100,
        )

        manager = ResourceManager()

        # Should pass
        can_allocate, reason = manager.check_quota(tenant, 4)
        assert can_allocate is True

        # Should fail - exceeds max
        can_allocate, reason = manager.check_quota(tenant, 10)
        assert can_allocate is False
        assert 'quota exceeded' in reason.lower()

    def test_allocate_gpus(self, setup_nodes, tenant_factory, user_factory):
        """Test GPU allocation."""
        from apps.resources.services import ResourceManager
        from apps.resources.models import (
            ResourceQuota, ResourceAllocation, ResourceStatus
        )

        tenant = tenant_factory(slug='test-allocate')
        user = user_factory(email='allocate@test.com')
        ResourceQuota.objects.create(
            tenant=tenant,
            max_gpus=8,
            max_gpu_hours_monthly=100,
        )

        manager = ResourceManager()
        allocation = manager.allocate(
            tenant=tenant,
            created_by=user,
            requested_gpus=2,
        )

        assert allocation is not None
        assert allocation.status == ResourceAllocation.AllocationStatus.ACTIVE
        assert allocation.requested_gpus == 2
        assert allocation.allocated_gpus.count() == 2

        # Check GPUs are marked as allocated
        for gpu in allocation.allocated_gpus.all():
            assert gpu.status == ResourceStatus.ALLOCATED

        # Check quota was updated
        quota = tenant.resource_quota
        assert quota.current_gpus_allocated == 2
        assert quota.current_jobs_running == 1

    def test_release_allocation(self, setup_nodes, tenant_factory, user_factory):
        """Test releasing an allocation."""
        from apps.resources.services import ResourceManager
        from apps.resources.models import (
            ResourceQuota, ResourceAllocation, ResourceStatus
        )

        tenant = tenant_factory(slug='test-release')
        user = user_factory(email='release@test.com')
        ResourceQuota.objects.create(
            tenant=tenant,
            max_gpus=8,
            max_gpu_hours_monthly=100,
        )

        manager = ResourceManager()
        allocation = manager.allocate(
            tenant=tenant,
            created_by=user,
            requested_gpus=2,
        )

        # Release
        manager.release(allocation)

        allocation.refresh_from_db()
        assert allocation.status == ResourceAllocation.AllocationStatus.COMPLETED
        assert allocation.completed_at is not None
        assert allocation.actual_cost_cents is not None

        # Check GPUs are released
        for gpu in allocation.node.gpus.filter(device_index__lt=2):
            gpu.refresh_from_db()
            assert gpu.status == ResourceStatus.AVAILABLE

        # Check quota was updated
        quota = tenant.resource_quota
        assert quota.current_gpus_allocated == 0

    def test_get_stats(self, setup_nodes):
        """Test getting resource statistics."""
        from apps.resources.services import ResourceManager

        manager = ResourceManager()
        stats = manager.get_stats()

        assert stats['total_nodes'] == 2
        assert stats['total_gpus'] == 12  # 4 + 8
        assert stats['available_gpus'] == 12


class TestResourceViewSets:
    """Tests for resource API endpoints."""

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

    @pytest.fixture
    def setup_nodes(self, db):
        """Create test compute nodes and GPUs."""
        from apps.resources.models import ComputeNode, GPU, GPUType

        node = ComputeNode.objects.create(
            name='test-node',
            hostname='test-node.local',
            gpu_type=GPUType.NVIDIA_A100_40GB,
            gpu_count=4,
            gpu_memory_gb=40,
            cpu_cores=64,
            ram_gb=512,
            storage_tb=5.0,
            region='us-east-1',
            price_per_hour=2400,
        )

        for i in range(4):
            GPU.objects.create(
                node=node,
                device_index=i,
                gpu_type=GPUType.NVIDIA_A100_40GB,
                memory_gb=40,
            )

        return node

    def test_list_nodes(self, authenticated_client, setup_nodes):
        """Test listing compute nodes."""
        client, user = authenticated_client

        response = client.get('/api/v1/resources/nodes/')

        assert response.status_code == status.HTTP_200_OK

    def test_get_node(self, authenticated_client, setup_nodes):
        """Test getting a specific node."""
        client, user = authenticated_client

        response = client.get(f'/api/v1/resources/nodes/{setup_nodes.id}/')

        assert response.status_code == status.HTTP_200_OK
        assert response.data['name'] == 'test-node'

    def test_list_gpus(self, authenticated_client, setup_nodes):
        """Test listing GPUs."""
        client, user = authenticated_client

        response = client.get('/api/v1/resources/gpus/')

        assert response.status_code == status.HTTP_200_OK

    def test_get_available_gpus(self, authenticated_client, setup_nodes):
        """Test getting available GPUs."""
        client, user = authenticated_client

        response = client.get('/api/v1/resources/gpus/available/')

        assert response.status_code == status.HTTP_200_OK

    def test_get_gpu_availability(self, authenticated_client, setup_nodes):
        """Test checking GPU availability."""
        client, user = authenticated_client

        response = client.get('/api/v1/resources/gpus/availability/')

        assert response.status_code == status.HTTP_200_OK
        assert 'total_available' in response.data

    def test_get_regions(self, authenticated_client, setup_nodes):
        """Test getting available regions."""
        client, user = authenticated_client

        response = client.get('/api/v1/resources/nodes/regions/')

        assert response.status_code == status.HTTP_200_OK

    def test_get_gpu_types(self, authenticated_client, setup_nodes):
        """Test getting available GPU types."""
        client, user = authenticated_client

        response = client.get('/api/v1/resources/nodes/gpu_types/')

        assert response.status_code == status.HTTP_200_OK

    def test_list_allocations(self, authenticated_client):
        """Test listing allocations."""
        client, user = authenticated_client

        response = client.get('/api/v1/resources/allocations/')

        assert response.status_code == status.HTTP_200_OK

    def test_get_allocation_stats(self, authenticated_client):
        """Test getting allocation statistics."""
        client, user = authenticated_client

        response = client.get('/api/v1/resources/allocations/stats/')

        assert response.status_code == status.HTTP_200_OK
        assert 'total_nodes' in response.data


class TestResourceSerializers:
    """Tests for resource serializers."""

    def test_compute_node_serializer(self, db):
        """Test ComputeNodeSerializer."""
        from apps.resources.models import ComputeNode, GPUType
        from apps.resources.serializers import ComputeNodeSerializer

        node = ComputeNode.objects.create(
            name='serializer-test',
            hostname='serializer-test.local',
            gpu_type=GPUType.NVIDIA_A100_40GB,
            gpu_count=4,
            gpu_memory_gb=40,
            cpu_cores=64,
            ram_gb=512,
            storage_tb=5.0,
            region='us-east-1',
            price_per_hour=2400,
        )

        serializer = ComputeNodeSerializer(node)
        data = serializer.data

        assert data['name'] == 'serializer-test'
        assert data['gpu_count'] == 4
        assert 'id' in data

    def test_resource_allocation_create_serializer_validation(self, db):
        """Test ResourceAllocationCreateSerializer validation."""
        from apps.resources.serializers import ResourceAllocationCreateSerializer

        # Invalid gpu count
        data = {
            'requested_gpus': 100,  # Too many
        }

        serializer = ResourceAllocationCreateSerializer(data=data)
        assert not serializer.is_valid()

    def test_resource_quota_serializer(self, db, tenant_factory):
        """Test ResourceQuotaSerializer."""
        from apps.resources.models import ResourceQuota
        from apps.resources.serializers import ResourceQuotaSerializer

        tenant = tenant_factory(slug='quota-serializer')
        quota = ResourceQuota.objects.create(
            tenant=tenant,
            max_gpus=8,
            max_gpu_hours_monthly=100,
            current_gpus_allocated=2,
        )

        serializer = ResourceQuotaSerializer(quota)
        data = serializer.data

        assert data['max_gpus'] == 8
        assert 'gpu_utilization_percent' in data
        assert 'hours_remaining' in data
