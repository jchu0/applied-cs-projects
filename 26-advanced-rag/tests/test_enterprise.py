"""Tests for enterprise features (tenant config, registries)."""

import pytest
from unittest.mock import AsyncMock

from advancedrag.enterprise.tenant import (
    TenantConfig,
    TenantConfigManager,
    InMemoryConfigStore,
)
from advancedrag.enterprise.registry import RerankerRegistry
from advancedrag.reranking.reranker import BaseReranker, MockReranker


class TestTenantConfig:
    """Test TenantConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = TenantConfig(tenant_id="t1")
        assert config.tenant_id == "t1"
        assert config.retrieval_top_k == 100
        assert config.bm25_weight == 0.4
        assert config.vector_weight == 0.6
        assert config.reranker_type == "cross-encoder"
        assert config.llm_model == "gpt-4"
        assert config.max_context_tokens == 4000
        assert config.requests_per_minute == 100

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        config = TenantConfig(tenant_id="t1", retrieval_top_k=50, bm25_weight=0.3)
        d = config.to_dict()
        restored = TenantConfig.from_dict(d)
        assert restored.tenant_id == "t1"
        assert restored.retrieval_top_k == 50
        assert restored.bm25_weight == 0.3

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = TenantConfig(
            tenant_id="enterprise",
            retrieval_top_k=200,
            reranker_type="slm",
            llm_model="gpt-3.5-turbo",
            max_concurrent=20,
        )
        assert config.retrieval_top_k == 200
        assert config.reranker_type == "slm"
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.max_concurrent == 20


class TestTenantConfigManager:
    """Test TenantConfigManager."""

    @pytest.fixture
    def manager(self):
        return TenantConfigManager()

    @pytest.mark.asyncio
    async def test_get_default_config(self, manager):
        """Getting a non-existent tenant should return defaults."""
        config = await manager.get_config("new_tenant")
        assert config.tenant_id == "new_tenant"
        assert config.retrieval_top_k == 100  # default

    @pytest.mark.asyncio
    async def test_update_config(self, manager):
        """Updating config should persist changes."""
        await manager.get_config("t1")
        updated = await manager.update_config("t1", {"retrieval_top_k": 50, "bm25_weight": 0.5})
        assert updated.retrieval_top_k == 50
        assert updated.bm25_weight == 0.5
        # Verify persistence
        fetched = await manager.get_config("t1")
        assert fetched.retrieval_top_k == 50

    @pytest.mark.asyncio
    async def test_delete_config(self, manager):
        """Deleting config should remove it."""
        await manager.get_config("t1")
        await manager.delete_config("t1")
        # Should get fresh defaults
        config = await manager.get_config("t1")
        assert config.retrieval_top_k == 100  # back to default

    @pytest.mark.asyncio
    async def test_list_tenants(self, manager):
        """Should list all tenant IDs with stored configs."""
        await manager.update_config("t1", {"retrieval_top_k": 50})
        await manager.update_config("t2", {"retrieval_top_k": 75})
        tenants = await manager.list_tenants()
        assert "t1" in tenants
        assert "t2" in tenants


class TestRerankerRegistry:
    """Test RerankerRegistry."""

    @pytest.fixture
    def registry(self):
        return RerankerRegistry()

    def test_register_and_get(self, registry):
        """Register and retrieve a reranker."""
        registry.register("mock", MockReranker)
        reranker = registry.get("mock")
        assert isinstance(reranker, MockReranker)

    def test_get_cached_returns_same_instance(self, registry):
        """Cached get should return the same instance."""
        registry.register("mock", MockReranker)
        r1 = registry.get_cached("mock")
        r2 = registry.get_cached("mock")
        assert r1 is r2

    def test_unregister(self, registry):
        """Unregistering should remove the reranker."""
        registry.register("mock", MockReranker)
        registry.unregister("mock")
        with pytest.raises(ValueError, match="Unknown reranker"):
            registry.get("mock")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
