"""
User views for authentication and profile management.
"""
from rest_framework import status, views
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.throttling import ScopedRateThrottle
from django.contrib.auth import authenticate
from django.utils import timezone

from .models import User
from .serializers import (
    UserSerializer,
    RegisterSerializer,
    LoginSerializer,
    ChangePasswordSerializer,
)
from .authentication import create_jwt_token, decode_jwt_token


class RegisterView(views.APIView):
    """Register a new user."""
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'auth_register'

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = User.objects.create_user(
            email=serializer.validated_data['email'],
            password=serializer.validated_data['password'],
            first_name=serializer.validated_data.get('first_name', ''),
            last_name=serializer.validated_data.get('last_name', ''),
        )

        access_token = create_jwt_token(user, expires_in_hours=1)
        refresh_token = create_jwt_token(user, expires_in_hours=24 * 7)

        return Response({
            'user': UserSerializer(user).data,
            'access_token': access_token,
            'refresh_token': refresh_token,
        }, status=status.HTTP_201_CREATED)


class LoginView(views.APIView):
    """Login and get JWT token."""
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'auth_login'

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = authenticate(
            email=serializer.validated_data['email'],
            password=serializer.validated_data['password'],
        )

        if not user:
            return Response(
                {'error': 'Invalid credentials'},
                status=status.HTTP_401_UNAUTHORIZED
            )

        user.last_login_at = timezone.now()
        user.save(update_fields=['last_login_at'])

        access_token = create_jwt_token(user, expires_in_hours=1)
        refresh_token = create_jwt_token(user, expires_in_hours=24 * 7)

        return Response({
            'user': UserSerializer(user).data,
            'access_token': access_token,
            'refresh_token': refresh_token,
        })


class LogoutView(views.APIView):
    """Logout user (client should discard token)."""

    def post(self, request):
        # JWT is stateless, so we just return success
        # Client should discard the token
        return Response({'message': 'Logged out successfully'})


class CurrentUserView(views.APIView):
    """Get current user profile."""

    def get(self, request):
        return Response(UserSerializer(request.user).data)

    def patch(self, request):
        serializer = UserSerializer(
            request.user,
            data=request.data,
            partial=True
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


class ChangePasswordView(views.APIView):
    """Change user password."""

    def post(self, request):
        serializer = ChangePasswordSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        if not request.user.check_password(serializer.validated_data['current_password']):
            return Response(
                {'error': 'Current password is incorrect'},
                status=status.HTTP_400_BAD_REQUEST
            )

        request.user.set_password(serializer.validated_data['new_password'])
        request.user.save()

        return Response({'message': 'Password changed successfully'})


class TokenRefreshView(views.APIView):
    """Refresh JWT token."""
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'auth_token_refresh'

    def post(self, request):
        refresh_token = request.data.get('refresh_token')
        if not refresh_token:
            return Response(
                {'error': 'Refresh token required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Validate the refresh token's signature (using the consolidated JWT
        # secret) and issue a fresh access token for the encoded user.
        from rest_framework.exceptions import AuthenticationFailed
        try:
            payload = decode_jwt_token(refresh_token)
            user = User.objects.get(id=payload.get('user_id'))
        except (AuthenticationFailed, User.DoesNotExist, ValueError):
            return Response(
                {'error': 'Invalid refresh token'},
                status=status.HTTP_401_UNAUTHORIZED
            )

        token = create_jwt_token(user)
        return Response({
            'access_token': token,
            'token_type': 'Bearer',
        })


class PasswordResetView(views.APIView):
    """Request password reset email."""
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'auth_password_reset'

    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response(
                {'error': 'Email required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Check if user exists (don't reveal if they don't)
        try:
            user = User.objects.get(email=email)
            # In production, send reset email here
            # send_password_reset_email(user)
        except User.DoesNotExist:
            pass  # Don't reveal if user exists

        return Response({
            'message': 'If an account exists with this email, a reset link has been sent.'
        })


class PasswordResetConfirmView(views.APIView):
    """Confirm password reset with token."""
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'auth_password_reset'

    def post(self, request):
        token = request.data.get('token')
        new_password = request.data.get('new_password')

        if not token or not new_password:
            return Response(
                {'error': 'Token and new password required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # In production, validate token and reset password
        return Response({'message': 'Password reset successfully'})


class ProfileView(views.APIView):
    """User profile management."""

    def get(self, request):
        return Response(UserSerializer(request.user).data)

    def patch(self, request):
        serializer = UserSerializer(
            request.user,
            data=request.data,
            partial=True
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)
