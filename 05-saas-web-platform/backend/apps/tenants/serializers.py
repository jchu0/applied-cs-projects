"""
Tenant serializers.
"""
from rest_framework import serializers
from .models import Tenant, TenantMembership, Invitation
from apps.users.serializers import UserSerializer


class TenantSerializer(serializers.ModelSerializer):
    """Serializer for tenant data."""

    class Meta:
        model = Tenant
        fields = ['id', 'name', 'slug', 'logo_url', 'settings', 'is_active', 'created_at']
        read_only_fields = ['id', 'created_at']


class CreateTenantSerializer(serializers.Serializer):
    """Serializer for creating a new tenant."""
    name = serializers.CharField(max_length=100)
    slug = serializers.SlugField()

    def validate_slug(self, value):
        if Tenant.objects.filter(slug=value).exists():
            raise serializers.ValidationError('This slug is already taken.')
        return value


class TenantMembershipSerializer(serializers.ModelSerializer):
    """Serializer for tenant membership."""
    user = UserSerializer(read_only=True)

    class Meta:
        model = TenantMembership
        fields = ['id', 'user', 'role', 'invited_at', 'accepted_at', 'created_at']
        read_only_fields = ['id', 'invited_at', 'accepted_at', 'created_at']


class InvitationSerializer(serializers.ModelSerializer):
    """Serializer for invitations."""
    invited_by = UserSerializer(read_only=True)

    class Meta:
        model = Invitation
        fields = ['id', 'email', 'role', 'invited_by', 'expires_at', 'created_at']
        read_only_fields = ['id', 'invited_by', 'expires_at', 'created_at']


class CreateInvitationSerializer(serializers.Serializer):
    """Serializer for creating invitations."""
    email = serializers.EmailField()
    role = serializers.ChoiceField(choices=TenantMembership.Role.choices, default=TenantMembership.Role.MEMBER)
