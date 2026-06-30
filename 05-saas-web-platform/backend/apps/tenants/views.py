"""
Tenant views for workspace management.
"""
import secrets
from datetime import timedelta
from rest_framework import status, views
from rest_framework.response import Response
from django.utils import timezone
from django.shortcuts import get_object_or_404

from .models import Tenant, TenantMembership, Invitation
from .serializers import (
    TenantSerializer,
    CreateTenantSerializer,
    TenantMembershipSerializer,
    InvitationSerializer,
    CreateInvitationSerializer,
)


class TenantListView(views.APIView):
    """List user's tenants or create a new tenant."""

    def get(self, request):
        memberships = request.user.tenant_memberships.select_related('tenant')
        tenants = [m.tenant for m in memberships]
        return Response(TenantSerializer(tenants, many=True).data)

    def post(self, request):
        serializer = CreateTenantSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        tenant = Tenant.objects.create(
            name=serializer.validated_data['name'],
            slug=serializer.validated_data['slug'],
        )

        # Make creator the owner
        TenantMembership.objects.create(
            tenant=tenant,
            user=request.user,
            role=TenantMembership.Role.OWNER,
            accepted_at=timezone.now(),
        )

        return Response(TenantSerializer(tenant).data, status=status.HTTP_201_CREATED)


class TenantDetailView(views.APIView):
    """Get, update, or delete a tenant."""

    def get_tenant(self, request, tenant_id):
        tenant = get_object_or_404(Tenant, id=tenant_id)
        membership = get_object_or_404(TenantMembership, tenant=tenant, user=request.user)
        return tenant, membership

    def get(self, request, tenant_id):
        tenant, _ = self.get_tenant(request, tenant_id)
        return Response(TenantSerializer(tenant).data)

    def patch(self, request, tenant_id):
        tenant, membership = self.get_tenant(request, tenant_id)

        if membership.role not in [TenantMembership.Role.OWNER, TenantMembership.Role.ADMIN]:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        serializer = TenantSerializer(tenant, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    def delete(self, request, tenant_id):
        tenant, membership = self.get_tenant(request, tenant_id)

        if membership.role != TenantMembership.Role.OWNER:
            return Response({'error': 'Only owner can delete tenant'}, status=status.HTTP_403_FORBIDDEN)

        tenant.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class TenantMemberListView(views.APIView):
    """List or invite members to a tenant."""

    def get_tenant(self, request, tenant_id):
        tenant = get_object_or_404(Tenant, id=tenant_id)
        membership = get_object_or_404(TenantMembership, tenant=tenant, user=request.user)
        return tenant, membership

    def get(self, request, tenant_id):
        tenant, _ = self.get_tenant(request, tenant_id)
        memberships = tenant.memberships.select_related('user')
        return Response(TenantMembershipSerializer(memberships, many=True).data)

    def post(self, request, tenant_id):
        tenant, membership = self.get_tenant(request, tenant_id)

        if membership.role not in [TenantMembership.Role.OWNER, TenantMembership.Role.ADMIN]:
            return Response({'error': 'Permission denied'}, status=status.HTTP_403_FORBIDDEN)

        serializer = CreateInvitationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        invitation = Invitation.objects.create(
            tenant=tenant,
            email=serializer.validated_data['email'],
            role=serializer.validated_data['role'],
            token=secrets.token_urlsafe(32),
            invited_by=request.user,
            expires_at=timezone.now() + timedelta(days=7),
        )

        # TODO: Send invitation email

        return Response(InvitationSerializer(invitation).data, status=status.HTTP_201_CREATED)


class AcceptInvitationView(views.APIView):
    """Accept a tenant invitation."""

    def post(self, request, token):
        invitation = get_object_or_404(Invitation, token=token)

        if invitation.accepted_at:
            return Response({'error': 'Invitation already accepted'}, status=status.HTTP_400_BAD_REQUEST)

        if invitation.expires_at < timezone.now():
            return Response({'error': 'Invitation has expired'}, status=status.HTTP_400_BAD_REQUEST)

        # Create membership
        membership, created = TenantMembership.objects.get_or_create(
            tenant=invitation.tenant,
            user=request.user,
            defaults={
                'role': invitation.role,
                'invited_by': invitation.invited_by,
                'invited_at': invitation.created_at,
                'accepted_at': timezone.now(),
            }
        )

        if not created:
            return Response({'error': 'Already a member'}, status=status.HTTP_400_BAD_REQUEST)

        invitation.accepted_at = timezone.now()
        invitation.save()

        return Response(TenantSerializer(invitation.tenant).data)
