import logging
from typing import Dict, Optional

from jinja2 import Environment, BaseLoader, TemplateNotFound, TemplateSyntaxError

from models import Template, Channel, EmailContent, SMSContent

logger = logging.getLogger(__name__)


class TemplateEngine:
    """Jinja2-based template engine for notifications."""

    def __init__(self):
        self.env = Environment(
            loader=BaseLoader(),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        # In-memory template cache (in production, use Redis)
        self.templates: Dict[str, Template] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load default notification templates."""
        # Welcome email template
        self.templates["welcome_email"] = Template(
            id="welcome_email",
            tenant_id="default",
            name="Welcome Email",
            channel=Channel.EMAIL,
            subject="Welcome to {{ company_name }}!",
            body="""
            <html>
            <body>
                <h1>Welcome, {{ first_name }}!</h1>
                <p>Thank you for signing up for {{ company_name }}.</p>
                <p>Your account has been created successfully.</p>
                {% if verification_link %}
                <p>Please verify your email by clicking the link below:</p>
                <a href="{{ verification_link }}">Verify Email</a>
                {% endif %}
                <p>Best regards,<br>The {{ company_name }} Team</p>
            </body>
            </html>
            """,
            default_variables={"company_name": "Microservices Platform"}
        )

        # Password reset template
        self.templates["password_reset"] = Template(
            id="password_reset",
            tenant_id="default",
            name="Password Reset",
            channel=Channel.EMAIL,
            subject="Reset Your Password",
            body="""
            <html>
            <body>
                <h1>Password Reset Request</h1>
                <p>Hi {{ first_name }},</p>
                <p>We received a request to reset your password.</p>
                <p>Click the link below to reset your password:</p>
                <a href="{{ reset_link }}">Reset Password</a>
                <p>This link will expire in {{ expiry_hours }} hours.</p>
                <p>If you didn't request this, please ignore this email.</p>
            </body>
            </html>
            """,
            default_variables={"expiry_hours": "24"}
        )

        # Subscription confirmation
        self.templates["subscription_created"] = Template(
            id="subscription_created",
            tenant_id="default",
            name="Subscription Created",
            channel=Channel.EMAIL,
            subject="Your {{ plan_name }} subscription is active",
            body="""
            <html>
            <body>
                <h1>Subscription Confirmed</h1>
                <p>Hi {{ first_name }},</p>
                <p>Your {{ plan_name }} subscription is now active.</p>
                <p><strong>Details:</strong></p>
                <ul>
                    <li>Plan: {{ plan_name }}</li>
                    <li>Amount: ${{ amount }}</li>
                    <li>Next billing date: {{ next_billing_date }}</li>
                </ul>
                <p>Thank you for your business!</p>
            </body>
            </html>
            """
        )

        # Invoice notification
        self.templates["invoice_paid"] = Template(
            id="invoice_paid",
            tenant_id="default",
            name="Invoice Paid",
            channel=Channel.EMAIL,
            subject="Payment Received - Invoice #{{ invoice_number }}",
            body="""
            <html>
            <body>
                <h1>Payment Received</h1>
                <p>Hi {{ first_name }},</p>
                <p>We've received your payment of ${{ amount }} for invoice #{{ invoice_number }}.</p>
                {% if invoice_url %}
                <p><a href="{{ invoice_url }}">View Invoice</a></p>
                {% endif %}
                <p>Thank you!</p>
            </body>
            </html>
            """
        )

        # SMS verification code
        self.templates["sms_verification"] = Template(
            id="sms_verification",
            tenant_id="default",
            name="SMS Verification",
            channel=Channel.SMS,
            subject="",
            body="Your verification code is: {{ code }}. Valid for {{ expiry_minutes }} minutes."
        )

    def get_template(self, template_id: str) -> Optional[Template]:
        """Get a template by ID."""
        return self.templates.get(template_id)

    def add_template(self, template: Template):
        """Add a template to the cache."""
        self.templates[template.id] = template

    def render(self, template_id: str, variables: Dict[str, str]) -> Optional[str]:
        """Render a template with the given variables."""
        template = self.get_template(template_id)
        if not template:
            logger.error(f"Template not found: {template_id}")
            return None

        try:
            # Merge default variables with provided variables
            merged_vars = {**template.default_variables, **variables}

            # Render the template
            jinja_template = self.env.from_string(template.body)
            return jinja_template.render(**merged_vars)

        except TemplateSyntaxError as e:
            logger.error(f"Template syntax error in {template_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to render template {template_id}: {e}")
            return None

    def render_subject(self, template_id: str, variables: Dict[str, str]) -> Optional[str]:
        """Render a template subject with the given variables."""
        template = self.get_template(template_id)
        if not template:
            return None

        try:
            merged_vars = {**template.default_variables, **variables}
            jinja_template = self.env.from_string(template.subject)
            return jinja_template.render(**merged_vars)
        except Exception as e:
            logger.error(f"Failed to render subject for {template_id}: {e}")
            return None

    def render_email(self, template_id: str, variables: Dict[str, str], recipient: str) -> Optional[EmailContent]:
        """Render a complete email from a template."""
        template = self.get_template(template_id)
        if not template or template.channel != Channel.EMAIL:
            return None

        subject = self.render_subject(template_id, variables)
        body = self.render(template_id, variables)

        if not subject or not body:
            return None

        return EmailContent(
            to=recipient,
            subject=subject,
            html_body=body,
            text_body=self._html_to_text(body)
        )

    def render_sms(self, template_id: str, variables: Dict[str, str], recipient: str) -> Optional[SMSContent]:
        """Render SMS content from a template."""
        template = self.get_template(template_id)
        if not template or template.channel != Channel.SMS:
            return None

        body = self.render(template_id, variables)
        if not body:
            return None

        return SMSContent(
            to=recipient,
            body=body
        )

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Simple HTML to text conversion."""
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
