"""
User serializers for registration, login, and profile management.
"""
from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from .models import User, APIKey


class UserSerializer(serializers.ModelSerializer):
    """Serializer for user profile data."""
    full_name = serializers.ReadOnlyField()

    class Meta:
        model = User
        fields = [
            'id', 'email', 'email_verified', 'first_name', 'last_name',
            'full_name', 'avatar_url', 'created_at', 'last_login_at',
        ]
        read_only_fields = ['id', 'email', 'email_verified', 'created_at', 'last_login_at']


class RegisterSerializer(serializers.Serializer):
    """Serializer for user registration."""
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, min_length=8)
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)

    def validate_email(self, value):
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError('A user with this email already exists.')
        return value.lower()

    def validate_password(self, value):
        validate_password(value)
        return value


class LoginSerializer(serializers.Serializer):
    """Serializer for user login."""
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)


class ChangePasswordSerializer(serializers.Serializer):
    """Serializer for password change."""
    current_password = serializers.CharField(write_only=True)
    new_password = serializers.CharField(write_only=True, min_length=8)

    def validate_new_password(self, value):
        validate_password(value)
        return value


class APIKeySerializer(serializers.ModelSerializer):
    """Serializer for API keys."""

    class Meta:
        model = APIKey
        fields = ['id', 'name', 'prefix', 'scopes', 'last_used_at', 'expires_at', 'created_at']
        read_only_fields = ['id', 'prefix', 'last_used_at', 'created_at']


class CreateAPIKeySerializer(serializers.Serializer):
    """Serializer for creating API keys."""
    name = serializers.CharField(max_length=100)
    scopes = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        default=list
    )
    expires_in_days = serializers.IntegerField(required=False, min_value=1, max_value=365)
