"""Configuration management for the job queue system."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Redis (for future persistence)
    redis_url: str = "redis://localhost:6379/0"

    # Worker
    worker_queues: list[str] = ["default"]
    worker_concurrency: int = 4
    worker_poll_interval: float = 1.0
    worker_heartbeat_interval: float = 5.0

    # Task defaults
    default_timeout_ms: int = 30000
    default_max_retries: int = 3
    visibility_timeout_ms: int = 30000

    # Result store
    result_ttl_seconds: int = 3600

    class Config:
        env_prefix = "JOBQUEUE_"
        env_file = ".env"


settings = Settings()
