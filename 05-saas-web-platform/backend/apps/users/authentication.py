"""
JWT authentication for the SaaS platform.
"""
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from django.conf import settings
from django.utils import timezone
from rest_framework import authentication, exceptions

from .models import User, APIKey


def create_jwt_token(user, expires_in_hours=24):
    """Create a JWT token for a user."""
    payload = {
        'user_id': str(user.id),
        'email': user.email,
        'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
        'iat': datetime.utcnow(),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm='HS256')


def decode_jwt_token(token):
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        raise exceptions.AuthenticationFailed('Token has expired')
    except jwt.InvalidTokenError:
        raise exceptions.AuthenticationFailed('Invalid token')


class JWTAuthentication(authentication.BaseAuthentication):
    """JWT token authentication."""

    def authenticate(self, request):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return None

        try:
            prefix, token = auth_header.split(' ')
            if prefix.lower() != 'bearer':
                return None
        except ValueError:
            return None

        payload = decode_jwt_token(token)

        try:
            user = User.objects.get(id=payload['user_id'])
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed('User not found')

        if not user.is_active:
            raise exceptions.AuthenticationFailed('User account is disabled')

        return (user, token)


class APIKeyAuthentication(authentication.BaseAuthentication):
    """API key authentication."""

    def authenticate(self, request):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return None

        # Extract prefix (first 8 characters)
        if len(api_key) < 8:
            raise exceptions.AuthenticationFailed('Invalid API key format')

        prefix = api_key[:8]

        # Find potential keys by prefix
        try:
            api_key_obj = APIKey.objects.select_related('user').get(prefix=prefix)
        except APIKey.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid API key')

        # Verify full key hash
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash != api_key_obj.key_hash:
            raise exceptions.AuthenticationFailed('Invalid API key')

        # Check expiration
        if api_key_obj.expires_at and api_key_obj.expires_at < timezone.now():
            raise exceptions.AuthenticationFailed('API key has expired')

        # Update last used
        api_key_obj.last_used_at = timezone.now()
        api_key_obj.save(update_fields=['last_used_at'])

        user = api_key_obj.user
        if not user.is_active:
            raise exceptions.AuthenticationFailed('User account is disabled')

        return (user, api_key_obj)


def generate_api_key():
    """Generate a new API key."""
    key = secrets.token_urlsafe(32)
    prefix = key[:8]
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, prefix, key_hash
