"""
Email notification service.
"""
from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string


class EmailService:
    """Service for sending email notifications."""

    @staticmethod
    def send_welcome_email(user):
        """Send welcome email to new user."""
        subject = 'Welcome to SaaS Platform'
        context = {
            'user': user,
            'login_url': f"{settings.FRONTEND_URL}/login",
        }
        return EmailService._send_email(
            user.email,
            subject,
            'emails/welcome.html',
            context
        )

    @staticmethod
    def send_invitation_email(invitation):
        """Send team invitation email."""
        subject = f"You've been invited to join {invitation.tenant.name}"
        context = {
            'invitation': invitation,
            'accept_url': f"{settings.FRONTEND_URL}/invitations/{invitation.token}",
            'tenant': invitation.tenant,
            'invited_by': invitation.invited_by,
        }
        return EmailService._send_email(
            invitation.email,
            subject,
            'emails/invitation.html',
            context
        )

    @staticmethod
    def send_password_reset_email(user, reset_token):
        """Send password reset email."""
        subject = 'Reset your password'
        context = {
            'user': user,
            'reset_url': f"{settings.FRONTEND_URL}/reset-password/{reset_token}",
        }
        return EmailService._send_email(
            user.email,
            subject,
            'emails/password_reset.html',
            context
        )

    @staticmethod
    def send_email_verification(user, verification_token):
        """Send email verification."""
        subject = 'Verify your email address'
        context = {
            'user': user,
            'verify_url': f"{settings.FRONTEND_URL}/verify-email/{verification_token}",
        }
        return EmailService._send_email(
            user.email,
            subject,
            'emails/verify_email.html',
            context
        )

    @staticmethod
    def send_payment_failed_email(tenant, invoice):
        """Send payment failed notification."""
        # Get tenant owner
        owner = tenant.memberships.filter(role='owner').first()
        if not owner:
            return False

        subject = 'Payment failed - Action required'
        context = {
            'tenant': tenant,
            'invoice': invoice,
            'billing_url': f"{settings.FRONTEND_URL}/dashboard/billing",
        }
        return EmailService._send_email(
            owner.user.email,
            subject,
            'emails/payment_failed.html',
            context
        )

    @staticmethod
    def send_subscription_canceled_email(tenant, subscription):
        """Send subscription cancellation notification."""
        owner = tenant.memberships.filter(role='owner').first()
        if not owner:
            return False

        subject = 'Subscription canceled'
        context = {
            'tenant': tenant,
            'subscription': subscription,
            'resubscribe_url': f"{settings.FRONTEND_URL}/dashboard/billing",
        }
        return EmailService._send_email(
            owner.user.email,
            subject,
            'emails/subscription_canceled.html',
            context
        )

    @staticmethod
    def send_trial_ending_email(tenant, days_remaining):
        """Send trial ending reminder."""
        owner = tenant.memberships.filter(role='owner').first()
        if not owner:
            return False

        subject = f'Your trial ends in {days_remaining} days'
        context = {
            'tenant': tenant,
            'days_remaining': days_remaining,
            'billing_url': f"{settings.FRONTEND_URL}/dashboard/billing",
        }
        return EmailService._send_email(
            owner.user.email,
            subject,
            'emails/trial_ending.html',
            context
        )

    @staticmethod
    def _send_email(to_email, subject, template, context):
        """Send an email using template."""
        try:
            # Render HTML content
            html_content = render_to_string(template, context)

            # Create plain text version (simple strip)
            text_content = html_content.replace('<br>', '\n').replace('</p>', '\n')
            import re
            text_content = re.sub('<[^<]+?>', '', text_content)

            send_mail(
                subject=subject,
                message=text_content,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[to_email],
                html_message=html_content,
                fail_silently=False,
            )
            return True
        except Exception as e:
            # Log error but don't raise
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False


# Email templates directory structure expected:
# templates/
#   emails/
#     welcome.html
#     invitation.html
#     password_reset.html
#     verify_email.html
#     payment_failed.html
#     subscription_canceled.html
#     trial_ending.html
