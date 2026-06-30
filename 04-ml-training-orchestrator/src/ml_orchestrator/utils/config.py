"""Configuration management."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_json: bool = Field(default=False, description="JSON log format")

    # Scheduler
    scheduling_interval: float = Field(
        default=1.0, description="Scheduling interval in seconds"
    )
    scheduling_policy: str = Field(default="priority", description="Scheduling policy")

    # Workers
    heartbeat_timeout: int = Field(
        default=60, description="Worker heartbeat timeout in seconds"
    )
    health_check_interval: float = Field(
        default=10.0, description="Health check interval in seconds"
    )

    # Checkpoints
    checkpoint_path: str = Field(default="/checkpoints", description="Checkpoint storage path")
    checkpoint_keep_last_n: int = Field(
        default=3, description="Number of checkpoints to keep"
    )

    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./orchestrator.db",
        description="Database URL",
    )

    # Redis (optional)
    redis_url: Optional[str] = Field(default=None, description="Redis URL")

    # S3 (optional)
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for checkpoints")
    s3_region: str = Field(default="us-east-1", description="S3 region")
    s3_endpoint_url: Optional[str] = Field(default=None, description="S3 endpoint URL")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
