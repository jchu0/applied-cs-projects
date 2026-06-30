import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Service configuration
    service_name: str = "notification-service"
    grpc_port: int = 9090
    http_port: int = 8080

    # External services
    nats_url: str = "nats://localhost:4222"
    redis_url: str = "redis://localhost:6379"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"

    # Email providers
    sendgrid_api_key: str = ""
    sendgrid_from_email: str = "noreply@example.com"
    sendgrid_from_name: str = "Microservices Platform"

    # SMS provider (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # Push notifications (Firebase)
    firebase_project_id: str = ""
    firebase_credentials_path: str = ""

    # Rate limiting
    rate_limit_per_minute: int = 100

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 60

    class Config:
        env_file = ".env"


settings = Settings()
