"""
Custom exceptions and error handling.
"""
from rest_framework import status
from rest_framework.exceptions import APIException
from rest_framework.views import exception_handler


class BaseAPIException(APIException):
    """Base exception for API errors."""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'An error occurred.'
    default_code = 'error'


class PermissionDenied(BaseAPIException):
    """Permission denied error."""
    status_code = status.HTTP_403_FORBIDDEN
    default_detail = 'You do not have permission to perform this action.'
    default_code = 'permission_denied'


class ResourceNotFound(BaseAPIException):
    """Resource not found error."""
    status_code = status.HTTP_404_NOT_FOUND
    default_detail = 'The requested resource was not found.'
    default_code = 'not_found'


class ValidationError(BaseAPIException):
    """Validation error."""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = 'Invalid input.'
    default_code = 'validation_error'


class ConflictError(BaseAPIException):
    """Conflict error."""
    status_code = status.HTTP_409_CONFLICT
    default_detail = 'A conflict occurred with the current state.'
    default_code = 'conflict'


class RateLimitExceeded(BaseAPIException):
    """Rate limit exceeded error."""
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    default_detail = 'Rate limit exceeded. Please try again later.'
    default_code = 'rate_limit_exceeded'


class PaymentRequired(BaseAPIException):
    """Payment required error."""
    status_code = status.HTTP_402_PAYMENT_REQUIRED
    default_detail = 'Payment is required to access this resource.'
    default_code = 'payment_required'


class SubscriptionRequired(BaseAPIException):
    """Active subscription required."""
    status_code = status.HTTP_402_PAYMENT_REQUIRED
    default_detail = 'An active subscription is required.'
    default_code = 'subscription_required'


class QuotaExceeded(BaseAPIException):
    """Quota exceeded error."""
    status_code = status.HTTP_402_PAYMENT_REQUIRED
    default_detail = 'Your plan quota has been exceeded.'
    default_code = 'quota_exceeded'


class ServiceUnavailable(BaseAPIException):
    """Service unavailable error."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = 'Service temporarily unavailable.'
    default_code = 'service_unavailable'


def custom_exception_handler(exc, context):
    """Custom exception handler with additional formatting."""
    response = exception_handler(exc, context)

    if response is not None:
        # Ensure consistent error format
        if isinstance(response.data, dict):
            error_data = {
                'error': {
                    'code': getattr(exc, 'default_code', 'error'),
                    'message': str(exc.detail) if hasattr(exc, 'detail') else str(exc),
                    'status': response.status_code,
                }
            }

            # Add field errors for validation
            if hasattr(exc, 'detail') and isinstance(exc.detail, dict):
                error_data['error']['fields'] = exc.detail

            response.data = error_data

        # Log server errors
        if response.status_code >= 500:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Server error: {exc}",
                exc_info=True,
                extra={
                    'request': context.get('request'),
                }
            )

    return response
