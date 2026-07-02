"""
Test suite for authentication and authorization functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

pytest.importorskip("django")

from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
import jwt

from apps.users.models import User


class AuthenticationTestCase(TestCase):
    """Test authentication endpoints and JWT token management."""

    def setUp(self):
        """Set up test client and sample users."""
        self.client = APIClient()
        self.test_user = User.objects.create_user(
            email='test@example.com',
            password='TestPass123!'
        )
        self.admin_user = User.objects.create_user(
            email='admin@example.com',
            password='AdminPass123!',
            is_staff=True,
            is_superuser=True
        )

    def test_user_registration(self):
        """Test user registration endpoint."""
        url = reverse('auth:register')
        data = {
            'email': 'newuser@example.com',
            'password': 'NewPass123!',
            'first_name': 'New',
            'last_name': 'User'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('access_token', response.data)
        self.assertIn('refresh_token', response.data)

        # Verify user was created
        user = User.objects.get(email='newuser@example.com')
        self.assertEqual(user.email, 'newuser@example.com')

    def test_user_registration_duplicate_email(self):
        """Test registration with duplicate email."""
        url = reverse('auth:register')
        data = {
            'email': 'test@example.com',  # Already exists
            'password': 'NewPass123!'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_user_login(self):
        """Test user login endpoint."""
        url = reverse('auth:login')
        data = {
            'email': 'test@example.com',
            'password': 'TestPass123!'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access_token', response.data)
        self.assertIn('refresh_token', response.data)
        self.assertIn('user', response.data)

    def test_invalid_login(self):
        """Test login with invalid credentials."""
        url = reverse('auth:login')
        data = {
            'email': 'test@example.com',
            'password': 'WrongPassword'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_token_refresh(self):
        """Test JWT token refresh functionality."""
        # First login to get tokens
        login_url = reverse('auth:login')
        login_data = {
            'email': 'test@example.com',
            'password': 'TestPass123!'
        }
        login_response = self.client.post(login_url, login_data, format='json')
        refresh_token = login_response.data['refresh_token']

        # Use refresh token to get new access token
        refresh_url = reverse('auth:token-refresh')
        refresh_data = {'refresh_token': refresh_token}
        response = self.client.post(refresh_url, refresh_data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access_token', response.data)

    def test_logout(self):
        """Test user logout functionality."""
        # Login first
        self.client.force_authenticate(user=self.test_user)

        # Logout
        url = reverse('auth:logout')
        response = self.client.post(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_password_reset_request(self):
        """Test password reset request."""
        url = reverse('auth:password-reset')
        data = {'email': 'test@example.com'}
        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('message', response.data)

    def test_password_reset_nonexistent_email(self):
        """Test password reset for nonexistent email (should not reveal if user exists)."""
        url = reverse('auth:password-reset')
        data = {'email': 'nonexistent@example.com'}
        response = self.client.post(url, data, format='json')

        # Should still return OK (don't reveal if user exists)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_password_change(self):
        """Test password change for authenticated user."""
        self.client.force_authenticate(user=self.test_user)

        url = reverse('auth:password-change')
        data = {
            'current_password': 'TestPass123!',
            'new_password': 'NewTestPass123!'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Verify password was changed
        self.test_user.refresh_from_db()
        self.assertTrue(self.test_user.check_password('NewTestPass123!'))

    def test_password_change_wrong_current(self):
        """Test password change with wrong current password."""
        self.client.force_authenticate(user=self.test_user)

        url = reverse('auth:password-change')
        data = {
            'current_password': 'WrongPassword',
            'new_password': 'NewTestPass123!'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class ProfileTestCase(TestCase):
    """Test user profile management."""

    def setUp(self):
        """Set up test users."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='profile@example.com',
            password='ProfilePass123!',
            first_name='Test',
            last_name='User'
        )

    def test_get_profile(self):
        """Test getting user profile."""
        self.client.force_authenticate(user=self.user)

        url = reverse('auth:profile')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['email'], 'profile@example.com')

    def test_update_profile(self):
        """Test updating user profile."""
        self.client.force_authenticate(user=self.user)

        url = reverse('auth:profile')
        data = {
            'first_name': 'Updated',
            'last_name': 'Name'
        }
        response = self.client.patch(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.user.refresh_from_db()
        self.assertEqual(self.user.first_name, 'Updated')
        self.assertEqual(self.user.last_name, 'Name')

    def test_profile_unauthenticated(self):
        """Test profile access without authentication."""
        url = reverse('auth:profile')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class PermissionTestCase(TestCase):
    """Test permission and role-based access control."""

    def setUp(self):
        """Set up test users with different roles."""
        self.client = APIClient()
        self.regular_user = User.objects.create_user(
            email='regular@example.com',
            password='RegularPass123!'
        )
        self.admin_user = User.objects.create_user(
            email='admin@example.com',
            password='AdminPass123!',
            is_staff=True
        )

    def test_regular_user_access(self):
        """Test regular user access to basic endpoints."""
        self.client.force_authenticate(user=self.regular_user)

        # Should access own profile
        url = reverse('auth:profile')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Should not access admin endpoints
        admin_url = reverse('admin:dashboard')
        response = self.client.get(admin_url)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_admin_access(self):
        """Test admin access to protected endpoints."""
        self.client.force_authenticate(user=self.admin_user)

        # Should access admin endpoints
        url = reverse('admin:dashboard')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_unauthenticated_access(self):
        """Test unauthenticated access to protected endpoints."""
        # Should not access protected endpoints
        url = reverse('auth:profile')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class CurrentUserTestCase(TestCase):
    """Test current user endpoint."""

    def setUp(self):
        """Set up test environment."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='current@example.com',
            password='CurrentPass123!',
            first_name='Current',
            last_name='User'
        )

    def test_get_current_user(self):
        """Test getting current user info."""
        self.client.force_authenticate(user=self.user)

        url = reverse('auth:current-user')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['email'], 'current@example.com')
        self.assertEqual(response.data['first_name'], 'Current')

    def test_update_current_user(self):
        """Test updating current user info."""
        self.client.force_authenticate(user=self.user)

        url = reverse('auth:current-user')
        data = {'first_name': 'NewFirst'}
        response = self.client.patch(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['first_name'], 'NewFirst')


class SessionManagementTestCase(TestCase):
    """Test session management and security."""

    def setUp(self):
        """Set up test environment."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            email='session@example.com',
            password='SessionPass123!'
        )

    def test_concurrent_sessions(self):
        """Test handling of concurrent sessions."""
        # Create multiple sessions
        client1 = APIClient()
        client2 = APIClient()

        url = reverse('auth:login')
        data = {
            'email': 'session@example.com',
            'password': 'SessionPass123!'
        }

        response1 = client1.post(url, data, format='json')
        response2 = client2.post(url, data, format='json')

        self.assertEqual(response1.status_code, status.HTTP_200_OK)
        self.assertEqual(response2.status_code, status.HTTP_200_OK)

        # Both sessions should have different tokens
        self.assertNotEqual(
            response1.data['access_token'],
            response2.data['access_token']
        )


class JWTSecretConsolidationTestCase(TestCase):
    """Regression tests: JWTs are signed/verified with JWT_SECRET_KEY.

    Phase 0 signed tokens with ``settings.SECRET_KEY`` while a separate
    ``JWT_SECRET_KEY`` existed. Token signing must consistently use
    ``JWT_SECRET_KEY`` so the two secrets can be rotated independently.
    """

    def setUp(self):
        self.user = User.objects.create_user(
            email='jwt@example.com',
            password='JwtPass123!',
        )

    def test_token_is_signed_with_jwt_secret_key(self):
        from django.conf import settings
        from apps.users.authentication import create_jwt_token

        token = create_jwt_token(self.user, expires_in_hours=1)

        # Decodable with JWT_SECRET_KEY...
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        self.assertEqual(payload['user_id'], str(self.user.id))

        # ...and NOT with Django's SECRET_KEY (the two secrets differ in dev).
        self.assertNotEqual(settings.SECRET_KEY, settings.JWT_SECRET_KEY)
        with self.assertRaises(jwt.InvalidSignatureError):
            jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
            )

    def test_decode_rejects_token_signed_with_django_secret_key(self):
        """A token minted with the wrong secret must not authenticate."""
        from django.conf import settings
        from rest_framework import exceptions
        from apps.users.authentication import decode_jwt_token

        forged = jwt.encode(
            {
                'user_id': str(self.user.id),
                'email': self.user.email,
                'exp': datetime.utcnow() + timedelta(hours=1),
                'iat': datetime.utcnow(),
            },
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )
        with self.assertRaises(exceptions.AuthenticationFailed):
            decode_jwt_token(forged)

    def test_authenticated_request_uses_jwt_secret_key(self):
        """End-to-end: a Bearer token from login authenticates a protected view."""
        from apps.users.authentication import create_jwt_token

        token = create_jwt_token(self.user, expires_in_hours=1)
        client = APIClient()
        client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        response = client.get(reverse('auth:current-user'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['email'], 'jwt@example.com')

    def test_refresh_rejects_token_with_bad_signature(self):
        """TokenRefreshView must reject refresh tokens signed with SECRET_KEY."""
        from django.conf import settings

        forged = jwt.encode(
            {
                'user_id': str(self.user.id),
                'email': self.user.email,
                'exp': datetime.utcnow() + timedelta(days=7),
                'iat': datetime.utcnow(),
            },
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )
        client = APIClient()
        response = client.post(
            reverse('auth:token-refresh'),
            {'refresh_token': forged},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class AuthThrottlingTestCase(TestCase):
    """Regression tests: sensitive auth endpoints are rate limited."""

    def setUp(self):
        from django.core.cache import cache
        # Throttle history lives in the cache; start each test clean so the
        # counts are deterministic.
        cache.clear()
        self.addCleanup(cache.clear)
        self.client = APIClient()

    def test_login_endpoint_throttles_after_limit(self):
        from unittest.mock import patch
        from rest_framework.throttling import ScopedRateThrottle

        # SimpleRateThrottle reads THROTTLE_RATES (a class attribute captured at
        # import time), so patch it directly to make the limit deterministic
        # regardless of DEFAULT_THROTTLE_RATES / test ordering.
        rates = dict(ScopedRateThrottle.THROTTLE_RATES, auth_login='3/min')
        with patch.object(ScopedRateThrottle, 'THROTTLE_RATES', rates):
            url = reverse('auth:login')
            data = {'email': 'nobody@example.com', 'password': 'whatever'}
            statuses = [
                self.client.post(url, data, format='json').status_code
                for _ in range(5)
            ]
        # First 3 allowed (401 invalid creds), then throttled (429).
        self.assertIn(status.HTTP_429_TOO_MANY_REQUESTS, statuses, statuses)
        self.assertEqual(statuses[:3], [status.HTTP_401_UNAUTHORIZED] * 3, statuses)
        self.assertEqual(statuses[3:], [status.HTTP_429_TOO_MANY_REQUESTS] * 2, statuses)

    def test_login_view_declares_scoped_throttle(self):
        from rest_framework.throttling import ScopedRateThrottle
        from apps.users.views import LoginView, RegisterView, PasswordResetView

        for view in (LoginView, RegisterView, PasswordResetView):
            self.assertIn(ScopedRateThrottle, view.throttle_classes)
            self.assertTrue(view.throttle_scope)


if __name__ == '__main__':
    pytest.main([__file__])
