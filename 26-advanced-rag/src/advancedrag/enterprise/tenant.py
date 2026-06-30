"""Multi-tenant configuration management for RAG system."""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class TenantConfig:
    """Per-tenant RAG configuration."""

    tenant_id: str

    # Retrieval settings
    retrieval_top_k: int = 100
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    fusion_strategy: str = "rrf"  # rrf, linear

    # Reranking settings
    reranker_type: str = "cross-encoder"  # cross-encoder, slm, mock
    reranking_top_k: int = 10
    use_diversity_filter: bool = True
    diversity_lambda: float = 0.7

    # Generation settings
    llm_model: str = "gpt-4"
    max_context_tokens: int = 4000
    include_citations: bool = True

    # Quality settings
    hallucination_check: bool = True
    min_confidence_threshold: float = 0.3

    # Rate limiting
    requests_per_minute: int = 100
    max_concurrent: int = 10

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TenantConfig":
        """Create from dictionary."""
        return cls(**data)


class TenantConfigManager:
    """Manages per-tenant configurations with caching."""

    def __init__(self, config_store=None):
        """Initialize manager.

        Args:
            config_store: Storage backend (dict-like with async get/set)
        """
        self.store = config_store or InMemoryConfigStore()
        self.cache: dict[str, TenantConfig] = {}

    async def get_config(self, tenant_id: str) -> TenantConfig:
        """Get configuration for tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant configuration (defaults if not found)
        """
        # Check cache first
        if tenant_id in self.cache:
            return self.cache[tenant_id]

        # Load from store
        config_data = await self.store.get(tenant_id)
        if not config_data:
            # Return defaults
            config = TenantConfig(tenant_id=tenant_id)
        else:
            config = TenantConfig.from_dict(config_data)

        # Cache it
        self.cache[tenant_id] = config
        return config

    async def update_config(self, tenant_id: str, updates: dict) -> TenantConfig:
        """Update tenant configuration.

        Args:
            tenant_id: Tenant identifier
            updates: Configuration updates

        Returns:
            Updated configuration
        """
        # Get current config
        current = await self.get_config(tenant_id)

        # Apply updates
        for key, value in updates.items():
            if hasattr(current, key):
                setattr(current, key, value)

        # Save to store
        await self.store.set(tenant_id, current.to_dict())

        # Update cache
        self.cache[tenant_id] = current

        return current

    async def delete_config(self, tenant_id: str):
        """Delete tenant configuration.

        Args:
            tenant_id: Tenant identifier
        """
        await self.store.delete(tenant_id)
        if tenant_id in self.cache:
            del self.cache[tenant_id]

    async def list_tenants(self) -> list[str]:
        """List all tenant IDs.

        Returns:
            List of tenant identifiers
        """
        return await self.store.list_keys()

    def invalidate_cache(self, tenant_id: Optional[str] = None):
        """Invalidate cache.

        Args:
            tenant_id: Specific tenant or None for all
        """
        if tenant_id:
            self.cache.pop(tenant_id, None)
        else:
            self.cache.clear()


class InMemoryConfigStore:
    """Simple in-memory configuration store."""

    def __init__(self):
        self._data: dict[str, dict] = {}

    async def get(self, key: str) -> Optional[dict]:
        """Get configuration by key."""
        return self._data.get(key)

    async def set(self, key: str, value: dict):
        """Set configuration."""
        self._data[key] = value

    async def delete(self, key: str):
        """Delete configuration."""
        self._data.pop(key, None)

    async def list_keys(self) -> list[str]:
        """List all keys."""
        return list(self._data.keys())


class RedisConfigStore:
    """Redis-backed configuration store."""

    def __init__(self, redis_client, prefix: str = "tenant:"):
        self.redis = redis_client
        self.prefix = prefix

    async def get(self, key: str) -> Optional[dict]:
        """Get configuration by key."""
        data = await self.redis.get(f"{self.prefix}{key}")
        if data:
            return json.loads(data)
        return None

    async def set(self, key: str, value: dict):
        """Set configuration."""
        await self.redis.set(
            f"{self.prefix}{key}",
            json.dumps(value)
        )

    async def delete(self, key: str):
        """Delete configuration."""
        await self.redis.delete(f"{self.prefix}{key}")

    async def list_keys(self) -> list[str]:
        """List all keys."""
        keys = await self.redis.keys(f"{self.prefix}*")
        return [k.replace(self.prefix, "") for k in keys]
