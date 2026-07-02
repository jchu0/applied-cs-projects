"""Regression tests: feature view names must not allow SQL injection."""

import numpy as np
import pytest

from feature_platform.core.models import DataType, Entity, Feature, FeatureView
from feature_platform.store.offline import FeatureData, validate_feature_view_name
from feature_platform.store.registry import FeatureRegistry

HOSTILE_VIEW_NAME = "x; DROP TABLE features_users;--"


class TestValidateFeatureViewName:
    def test_accepts_safe_names(self):
        for name in ("users", "user_features_v2", "F1", "_hidden"):
            assert validate_feature_view_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            HOSTILE_VIEW_NAME,
            "users; --",
            "users'",
            'users"',
            "users.other",
            "users features",
            "",
            None,
        ],
    )
    def test_rejects_hostile_names(self, name):
        with pytest.raises(ValueError, match="Invalid feature view name"):
            validate_feature_view_name(name)


class TestRegistryRejectsHostileNames:
    def test_register_feature_view_rejects_hostile_name(self, tmp_path):
        registry = FeatureRegistry(path=str(tmp_path / "registry"))
        view = FeatureView(
            name=HOSTILE_VIEW_NAME,
            entities=[Entity(name="user", join_keys=["user_id"])],
            schema=[Feature(name="age", dtype=DataType.INT64)],
        )

        with pytest.raises(ValueError, match="Invalid feature view name"):
            registry.register_feature_view(view)

        # Nothing was registered
        assert registry.get_feature_view(HOSTILE_VIEW_NAME) is None

    def test_register_feature_view_accepts_safe_name(self, tmp_path):
        registry = FeatureRegistry(path=str(tmp_path / "registry"))
        view = FeatureView(
            name="user_features",
            entities=[Entity(name="user", join_keys=["user_id"])],
            schema=[Feature(name="age", dtype=DataType.INT64)],
        )

        registry.register_feature_view(view)
        assert registry.get_feature_view("user_features") is not None


class TestDuckDBStoreRejectsHostileNames:
    @pytest.fixture
    def store(self, tmp_path):
        duckdb = pytest.importorskip("duckdb")
        from feature_platform.store.offline import DuckDBOfflineStore

        store = DuckDBOfflineStore(path=str(tmp_path / "store.duckdb"))
        yield store
        store.close()

    @pytest.fixture
    def sample_data(self):
        # object dtype so rows are inserted as plain Python scalars
        # (older duckdb versions cannot bind numpy scalar types)
        return FeatureData(
            entity_ids={"user_id": ["u1", "u2"]},
            features={"age": np.array([30, 40], dtype=object)},
        )

    def test_write_rejects_hostile_view_name(self, store, sample_data):
        with pytest.raises(ValueError, match="Invalid feature view name"):
            store.write_features(HOSTILE_VIEW_NAME, sample_data)

    def test_read_rejects_hostile_view_name(self, store):
        with pytest.raises(ValueError, match="Invalid feature view name"):
            store.read_features(HOSTILE_VIEW_NAME)

    def test_delete_rejects_hostile_view_name(self, store):
        with pytest.raises(ValueError, match="Invalid feature view name"):
            store.delete_features(HOSTILE_VIEW_NAME)

    def test_hostile_view_name_cannot_drop_tables(self, store, sample_data):
        # Set up a legitimate table, then confirm a hostile view name cannot
        # reach the SQL layer to drop it.
        store.write_features("users", sample_data)

        with pytest.raises(ValueError):
            store.write_features("users; DROP TABLE features_users;--", sample_data)

        result = store.read_features("users")
        assert len(result) == 2

    def test_hostile_column_names_rejected(self, store):
        data = FeatureData(
            entity_ids={"user_id": ["u1"]},
            features={"age BIGINT); DROP TABLE x;--": np.array([1])},
        )
        with pytest.raises(ValueError, match="Invalid column name"):
            store.write_features("users2", data)

    def test_round_trip_with_safe_names(self, store, sample_data):
        store.write_features("user_features", sample_data)
        result = store.read_features("user_features")
        assert len(result) == 2
        assert set(result.features) == {"age"}
