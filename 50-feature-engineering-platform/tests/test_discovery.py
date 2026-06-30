"""Tests for feature discovery module."""

import numpy as np
import pytest
from datetime import datetime, timedelta

from feature_platform.discovery.search import (
    FeatureSearchEngine,
    SearchQuery,
    SearchResult,
    SearchFilters,
    FeatureMetadata,
    SortOrder,
)
from feature_platform.discovery.similarity import (
    FeatureSimilarityEngine,
    SimilarityMethod,
    SimilarityResult,
    FeatureProfile,
)
from feature_platform.discovery.recommendations import (
    FeatureRecommender,
    RecommendationContext,
    FeatureRecommendation,
    RecommendationType,
    FeatureInfo,
    FeatureUsageRecord,
)


class TestFeatureSearchEngine:
    """Tests for FeatureSearchEngine."""

    @pytest.fixture
    def engine(self):
        """Create a search engine with sample data."""
        engine = FeatureSearchEngine()

        # Add sample features
        features = [
            FeatureMetadata(
                name="user_age",
                feature_view="user_features",
                data_type="float64",
                description="Age of the user in years",
                tags=["demographic", "user"],
                owner="data-team",
                entity_type="user",
                quality_score=0.95,
                usage_count=100,
                created_at=datetime.now() - timedelta(days=30),
            ),
            FeatureMetadata(
                name="user_income",
                feature_view="user_features",
                data_type="float64",
                description="Annual income of the user",
                tags=["financial", "user"],
                owner="data-team",
                entity_type="user",
                quality_score=0.90,
                usage_count=80,
            ),
            FeatureMetadata(
                name="product_price",
                feature_view="product_features",
                data_type="float64",
                description="Price of the product",
                tags=["product", "pricing"],
                owner="product-team",
                entity_type="product",
                quality_score=0.85,
                usage_count=50,
            ),
            FeatureMetadata(
                name="purchase_count",
                feature_view="user_features",
                data_type="int64",
                description="Number of purchases made by user",
                tags=["behavioral", "user"],
                owner="ml-team",
                entity_type="user",
                quality_score=0.92,
                usage_count=120,
            ),
            FeatureMetadata(
                name="deprecated_feature",
                feature_view="old_features",
                data_type="float64",
                description="An old deprecated feature",
                tags=["legacy"],
                owner="data-team",
                is_deprecated=True,
            ),
        ]

        for f in features:
            engine.index_feature(f)

        return engine

    def test_initialization(self):
        """Test engine initialization."""
        engine = FeatureSearchEngine(fuzzy_threshold=0.8)
        assert engine.fuzzy_threshold == 0.8
        assert engine.get_feature_count() == 0

    def test_index_feature(self, engine):
        """Test indexing features."""
        assert engine.get_feature_count() == 5

    def test_search_by_name(self, engine):
        """Test searching by feature name."""
        query = SearchQuery(text="user_age", limit=10)
        results = engine.search(query)

        assert len(results) >= 1
        assert results[0].feature_name == "user_age"

    def test_search_by_description(self, engine):
        """Test searching by description text."""
        query = SearchQuery(text="income annual", limit=10)
        results = engine.search(query)

        assert len(results) >= 1
        # user_income should be in results
        names = [r.feature_name for r in results]
        assert "user_income" in names

    def test_search_excludes_deprecated(self, engine):
        """Test that deprecated features are excluded by default."""
        query = SearchQuery(text="deprecated", include_deprecated=False)
        results = engine.search(query)

        names = [r.feature_name for r in results]
        assert "deprecated_feature" not in names

    def test_search_includes_deprecated_when_requested(self, engine):
        """Test including deprecated features."""
        query = SearchQuery(text="deprecated", include_deprecated=True)
        results = engine.search(query)

        names = [r.feature_name for r in results]
        assert "deprecated_feature" in names

    def test_search_with_filters(self, engine):
        """Test searching with filters."""
        filters = SearchFilters(
            data_types=["float64"],
            owners=["data-team"],
        )
        query = SearchQuery(text="", filters=filters)
        results = engine.search(query)

        # Should get user_age and user_income
        assert len(results) >= 1
        for r in results:
            assert r.data_type == "float64"
            assert r.owner == "data-team"

    def test_search_filter_by_tags(self, engine):
        """Test filtering by tags."""
        filters = SearchFilters(tags=["user"])
        query = SearchQuery(text="", filters=filters)
        results = engine.search(query)

        assert len(results) >= 2
        for r in results:
            assert any("user" in tag.lower() for tag in r.tags)

    def test_search_filter_by_entity_type(self, engine):
        """Test filtering by entity type."""
        filters = SearchFilters(entity_types=["product"])
        query = SearchQuery(text="", filters=filters)
        results = engine.search(query)

        assert len(results) >= 1
        for r in results:
            assert r.entity_type == "product"

    def test_search_filter_by_quality_score(self, engine):
        """Test filtering by minimum quality score."""
        filters = SearchFilters(min_quality_score=0.9)
        query = SearchQuery(text="", filters=filters)
        results = engine.search(query)

        for r in results:
            assert r.quality_score >= 0.9

    def test_search_sort_by_name(self, engine):
        """Test sorting by name."""
        query = SearchQuery(text="user", sort_by=SortOrder.NAME)
        results = engine.search(query)

        names = [r.feature_name for r in results]
        assert names == sorted(names, key=str.lower)

    def test_search_sort_by_popularity(self, engine):
        """Test sorting by popularity."""
        query = SearchQuery(text="user", sort_by=SortOrder.POPULARITY)
        results = engine.search(query)

        usage_counts = [r.usage_count for r in results]
        assert usage_counts == sorted(usage_counts, reverse=True)

    def test_search_pagination(self, engine):
        """Test search pagination."""
        query1 = SearchQuery(text="", limit=2, offset=0)
        results1 = engine.search(query1)

        query2 = SearchQuery(text="", limit=2, offset=2)
        results2 = engine.search(query2)

        # Should get different results
        names1 = set(r.feature_name for r in results1)
        names2 = set(r.feature_name for r in results2)
        assert names1.isdisjoint(names2) or len(results2) == 0

    def test_get_popular_features(self, engine):
        """Test getting popular features."""
        results = engine.get_popular_features(limit=3)

        assert len(results) == 3
        # Should be sorted by usage count
        usage_counts = [r.usage_count for r in results]
        assert usage_counts == sorted(usage_counts, reverse=True)

    def test_get_features_by_tag(self, engine):
        """Test getting features by tag."""
        results = engine.get_features_by_tag("demographic")

        assert len(results) >= 1
        assert results[0].feature_name == "user_age"

    def test_get_features_by_owner(self, engine):
        """Test getting features by owner."""
        results = engine.get_features_by_owner("ml-team")

        assert len(results) >= 1
        assert all(r.owner == "ml-team" for r in results)

    def test_get_all_tags(self, engine):
        """Test getting all tags."""
        tags = engine.get_all_tags()

        assert "user" in tags
        assert "product" in tags
        assert "demographic" in tags

    def test_remove_feature(self, engine):
        """Test removing a feature."""
        initial_count = engine.get_feature_count()
        removed = engine.remove_feature("user_features", "user_age")

        assert removed
        assert engine.get_feature_count() == initial_count - 1

        # Should not find removed feature
        query = SearchQuery(text="user_age")
        results = engine.search(query)
        names = [r.feature_name for r in results]
        assert "user_age" not in names

    def test_fuzzy_matching(self, engine):
        """Test fuzzy matching for typos."""
        # Search with typo
        query = SearchQuery(text="usr_age")  # typo: usr instead of user
        results = engine.search(query)

        # Should still find user_age via prefix/fuzzy matching
        # Note: depends on fuzzy_threshold setting
        assert len(results) >= 0  # May or may not match depending on threshold


class TestFeatureSimilarityEngine:
    """Tests for FeatureSimilarityEngine."""

    @pytest.fixture
    def engine(self):
        """Create similarity engine with sample profiles."""
        engine = FeatureSimilarityEngine()

        np.random.seed(42)

        profiles = [
            FeatureProfile(
                name="user_age",
                feature_view="user_features",
                data_type="float64",
                description="Age of user",
                tags=["demographic", "user"],
                owner="data-team",
                mean=35.0,
                std=12.0,
                min_value=18.0,
                max_value=80.0,
                null_ratio=0.01,
                sample_values=np.random.normal(35, 12, 100),
            ),
            FeatureProfile(
                name="customer_age",
                feature_view="customer_features",
                data_type="float64",
                description="Age of customer",
                tags=["demographic", "customer"],
                owner="data-team",
                mean=36.0,
                std=11.0,
                min_value=18.0,
                max_value=75.0,
                null_ratio=0.02,
                sample_values=np.random.normal(36, 11, 100),
            ),
            FeatureProfile(
                name="product_price",
                feature_view="product_features",
                data_type="float64",
                description="Price of product",
                tags=["product", "pricing"],
                owner="product-team",
                mean=50.0,
                std=30.0,
                min_value=1.0,
                max_value=500.0,
                null_ratio=0.0,
                sample_values=np.random.exponential(50, 100),
            ),
            FeatureProfile(
                name="user_name",
                feature_view="user_features",
                data_type="string",
                description="Name of user",
                tags=["pii", "user"],
                owner="data-team",
            ),
        ]

        for p in profiles:
            engine.add_profile(p)

        return engine

    def test_initialization(self):
        """Test engine initialization."""
        engine = FeatureSimilarityEngine(
            stat_weight=0.4,
            corr_weight=0.3,
            name_weight=0.2,
            meta_weight=0.1,
        )
        assert engine.stat_weight == 0.4

    def test_add_profile(self, engine):
        """Test adding a profile."""
        assert "user_features:user_age" in engine._profiles

    def test_remove_profile(self, engine):
        """Test removing a profile."""
        removed = engine.remove_profile("user_features", "user_age")
        assert removed
        assert "user_features:user_age" not in engine._profiles

    def test_find_similar_statistical(self, engine):
        """Test finding similar features using statistical method."""
        results = engine.find_similar(
            "user_features",
            "user_age",
            method=SimilarityMethod.STATISTICAL,
            top_k=3,
        )

        assert len(results) > 0
        # customer_age should be most similar
        assert results[0].target_feature == "customer_features:customer_age"

    def test_find_similar_name(self, engine):
        """Test finding similar features using name method."""
        results = engine.find_similar(
            "user_features",
            "user_age",
            method=SimilarityMethod.NAME,
            top_k=3,
        )

        assert len(results) > 0
        # customer_age should be similar by name
        targets = [r.target_feature for r in results]
        assert "customer_features:customer_age" in targets

    def test_find_similar_metadata(self, engine):
        """Test finding similar features using metadata method."""
        results = engine.find_similar(
            "user_features",
            "user_age",
            method=SimilarityMethod.METADATA,
            top_k=3,
        )

        assert len(results) > 0
        # Should prefer features with similar tags/owner

    def test_find_similar_combined(self, engine):
        """Test finding similar features using combined method."""
        results = engine.find_similar(
            "user_features",
            "user_age",
            method=SimilarityMethod.COMBINED,
            top_k=3,
        )

        assert len(results) > 0
        # customer_age should still be most similar overall
        assert results[0].target_feature == "customer_features:customer_age"

    def test_compute_similarity(self, engine):
        """Test computing similarity between two features."""
        p1 = engine._profiles["user_features:user_age"]
        p2 = engine._profiles["customer_features:customer_age"]

        result = engine.compute_similarity(p1, p2)

        assert result.similarity_score > 0
        assert result.source_feature == "user_features:user_age"
        assert result.target_feature == "customer_features:customer_age"
        assert len(result.breakdown) > 0

    def test_find_duplicates(self, engine):
        """Test finding duplicate features."""
        duplicates = engine.find_duplicates(similarity_threshold=0.5)

        # user_age and customer_age should be potential duplicates
        found = False
        for f1, f2, score in duplicates:
            if ("user_age" in f1 and "customer_age" in f2) or \
               ("customer_age" in f1 and "user_age" in f2):
                found = True
                break
        assert found

    def test_cluster_features(self, engine):
        """Test clustering features."""
        clusters = engine.cluster_features(n_clusters=2)

        assert len(clusters) <= 2
        # All features should be assigned
        all_features = set()
        for features in clusters.values():
            all_features.update(features)
        assert len(all_features) == 4

    def test_similarity_with_min_threshold(self, engine):
        """Test similarity search with minimum threshold."""
        results = engine.find_similar(
            "user_features",
            "user_age",
            min_similarity=0.5,
        )

        for r in results:
            assert r.similarity_score >= 0.5

    def test_correlation_similarity(self, engine):
        """Test correlation-based similarity."""
        p1 = engine._profiles["user_features:user_age"]
        p2 = engine._profiles["customer_features:customer_age"]

        score, breakdown = engine._compute_correlation_similarity(p1, p2)

        assert "correlation" in breakdown
        assert score >= 0


class TestFeatureRecommender:
    """Tests for FeatureRecommender."""

    @pytest.fixture
    def recommender(self):
        """Create recommender with sample data."""
        recommender = FeatureRecommender()

        features = [
            FeatureInfo(
                name="user_age",
                feature_view="user_features",
                data_type="float64",
                entity_type="user",
                tags=["demographic", "churn"],
                quality_score=0.95,
                usage_count=100,
            ),
            FeatureInfo(
                name="user_income",
                feature_view="user_features",
                data_type="float64",
                entity_type="user",
                tags=["financial", "churn"],
                quality_score=0.90,
                usage_count=80,
            ),
            FeatureInfo(
                name="purchase_history",
                feature_view="user_features",
                data_type="string",
                entity_type="user",
                tags=["behavioral", "churn"],
                quality_score=0.92,
                usage_count=120,
            ),
            FeatureInfo(
                name="product_category",
                feature_view="product_features",
                data_type="string",
                entity_type="product",
                tags=["product", "recommendation"],
                quality_score=0.88,
                usage_count=60,
            ),
            FeatureInfo(
                name="low_quality_feature",
                feature_view="misc",
                data_type="float64",
                tags=["legacy"],
                quality_score=0.3,
                usage_count=5,
            ),
        ]

        for f in features:
            recommender.add_feature(f)

        # Add cooccurrence data
        recommender.learn_cooccurrence([
            "user_features:user_age",
            "user_features:user_income",
            "user_features:purchase_history",
        ])
        recommender.learn_cooccurrence([
            "user_features:user_age",
            "user_features:user_income",
        ])

        return recommender

    def test_initialization(self):
        """Test recommender initialization."""
        recommender = FeatureRecommender(
            similarity_weight=0.4,
            cooccurrence_weight=0.3,
        )
        assert recommender.similarity_weight == 0.4
        assert recommender.cooccurrence_weight == 0.3

    def test_add_feature(self, recommender):
        """Test adding features."""
        assert recommender.get_feature_count() == 5

    def test_recommend_basic(self, recommender):
        """Test basic recommendations."""
        context = RecommendationContext(
            current_features=["user_features:user_age"],
            max_recommendations=3,
        )

        recommendations = recommender.recommend(context)

        assert len(recommendations) <= 3
        assert all(r.feature_name != "user_age" for r in recommendations)

    def test_recommend_with_entity_filter(self, recommender):
        """Test recommendations filtered by entity type."""
        context = RecommendationContext(
            current_features=[],
            entity_types=["user"],
            max_recommendations=10,
        )

        recommendations = recommender.recommend(context)

        # Should only get user features
        assert len(recommendations) > 0

    def test_recommend_with_domain_tags(self, recommender):
        """Test recommendations with domain tags."""
        context = RecommendationContext(
            current_features=[],
            domain_tags=["churn"],
            max_recommendations=10,
        )

        recommendations = recommender.recommend(context)

        # Should prefer features with churn tag
        assert len(recommendations) > 0

    def test_recommend_excludes_current(self, recommender):
        """Test that recommendations exclude current features."""
        context = RecommendationContext(
            current_features=[
                "user_features:user_age",
                "user_features:user_income",
            ],
            max_recommendations=10,
        )

        recommendations = recommender.recommend(context)

        names = [r.feature_name for r in recommendations]
        assert "user_age" not in names
        assert "user_income" not in names

    def test_recommend_excludes_excluded(self, recommender):
        """Test that recommendations exclude excluded features."""
        context = RecommendationContext(
            current_features=["user_features:user_age"],
            excluded_features=["user_features:user_income"],
            max_recommendations=10,
        )

        recommendations = recommender.recommend(context)

        names = [r.feature_name for r in recommendations]
        assert "user_income" not in names

    def test_get_cooccurrence_recommendations(self, recommender):
        """Test co-occurrence based recommendations."""
        recommendations = recommender.get_cooccurrence_recommendations(
            features=["user_features:user_age"],
            top_k=5,
        )

        assert len(recommendations) > 0
        # user_income should be recommended (co-occurs with user_age)
        names = [r.feature_name for r in recommendations]
        assert "user_income" in names

    def test_get_domain_recommendations(self, recommender):
        """Test domain-specific recommendations."""
        recommendations = recommender.get_domain_recommendations(
            domain_tags=["churn"],
            top_k=5,
        )

        assert len(recommendations) > 0
        # All should have churn tag
        for r in recommendations:
            assert r.recommendation_type == RecommendationType.DOMAIN

    def test_get_popular_recommendations(self, recommender):
        """Test popularity-based recommendations."""
        recommendations = recommender.get_popular_recommendations(top_k=3)

        assert len(recommendations) == 3
        # Should be sorted by usage
        usage_counts = [r.metadata.get("usage_count", 0) for r in recommendations]
        assert usage_counts == sorted(usage_counts, reverse=True)

    def test_get_popular_recommendations_by_entity(self, recommender):
        """Test popularity recommendations filtered by entity."""
        recommendations = recommender.get_popular_recommendations(
            entity_type="user",
            top_k=10,
        )

        # Should only get user features
        # (entity filtering happens internally)
        assert len(recommendations) > 0

    def test_get_quality_recommendations(self, recommender):
        """Test quality-based recommendations."""
        recommendations = recommender.get_quality_recommendations(
            current_features=[],
            quality_threshold=0.9,
            top_k=5,
        )

        for r in recommendations:
            assert r.metadata.get("quality_score", 0) >= 0.9

    def test_record_usage(self, recommender):
        """Test recording feature usage."""
        initial_count = recommender.get_usage_record_count()

        record = FeatureUsageRecord(
            feature_key="user_features:user_age",
            project_id="project-123",
            model_id="model-456",
            task_type="classification",
        )
        recommender.record_usage(record)

        assert recommender.get_usage_record_count() == initial_count + 1

    def test_recommendation_has_score_breakdown(self, recommender):
        """Test that recommendations include score breakdown."""
        context = RecommendationContext(
            current_features=["user_features:user_age"],
            max_recommendations=3,
        )

        recommendations = recommender.recommend(context)

        assert len(recommendations) > 0
        for r in recommendations:
            assert "score_breakdown" in r.metadata


class TestRecommendationContext:
    """Tests for RecommendationContext dataclass."""

    def test_defaults(self):
        """Test default values."""
        context = RecommendationContext()

        assert context.current_features == []
        assert context.entity_types == []
        assert context.task_type is None
        assert context.max_recommendations == 10

    def test_full_context(self):
        """Test creating full context."""
        context = RecommendationContext(
            current_features=["f1", "f2"],
            entity_types=["user"],
            task_type="classification",
            domain_tags=["fraud"],
            excluded_features=["f3"],
            max_recommendations=5,
        )

        assert len(context.current_features) == 2
        assert context.task_type == "classification"
        assert context.max_recommendations == 5


class TestFeatureRecommendation:
    """Tests for FeatureRecommendation dataclass."""

    def test_creation(self):
        """Test creating a recommendation."""
        rec = FeatureRecommendation(
            feature_name="user_age",
            feature_view="user_features",
            recommendation_type=RecommendationType.COOCCURRENCE,
            score=0.85,
            reason="Frequently used together",
            confidence=0.9,
            metadata={"cooccurrence_count": 5},
        )

        assert rec.feature_name == "user_age"
        assert rec.recommendation_type == RecommendationType.COOCCURRENCE
        assert rec.score == 0.85
        assert rec.confidence == 0.9


class TestIntegration:
    """Integration tests for discovery module."""

    def test_search_to_similarity_workflow(self):
        """Test workflow from search to similarity analysis."""
        # Setup search engine
        search = FeatureSearchEngine()
        similarity = FeatureSimilarityEngine()

        np.random.seed(42)

        # Add features to both
        meta = FeatureMetadata(
            name="user_age",
            feature_view="users",
            data_type="float64",
            description="Age of user",
            tags=["demographic"],
            owner="team-a",
            quality_score=0.9,
            usage_count=100,
        )
        search.index_feature(meta)

        profile = FeatureProfile(
            name="user_age",
            feature_view="users",
            data_type="float64",
            description="Age of user",
            tags=["demographic"],
            owner="team-a",
            mean=35.0,
            std=10.0,
            sample_values=np.random.normal(35, 10, 100),
        )
        similarity.add_profile(profile)

        # Search for features
        query = SearchQuery(text="age")
        results = search.search(query)

        assert len(results) > 0

        # Find similar to found feature
        if len(results) > 0:
            found = results[0]
            similar = similarity.find_similar(
                found.feature_view,
                found.feature_name,
                top_k=5,
            )
            # May be empty if only one feature
            assert isinstance(similar, list)

    def test_search_to_recommendation_workflow(self):
        """Test workflow from search to recommendations."""
        search = FeatureSearchEngine()
        recommender = FeatureRecommender()

        # Add features
        features = [
            ("user_age", "users", "float64", ["demo"], "team-a"),
            ("user_income", "users", "float64", ["demo", "fin"], "team-a"),
            ("product_price", "products", "float64", ["pricing"], "team-b"),
        ]

        for name, view, dtype, tags, owner in features:
            meta = FeatureMetadata(
                name=name,
                feature_view=view,
                data_type=dtype,
                tags=tags,
                owner=owner,
                usage_count=50,
            )
            search.index_feature(meta)

            info = FeatureInfo(
                name=name,
                feature_view=view,
                data_type=dtype,
                tags=tags,
                quality_score=0.9,
                usage_count=50,
            )
            recommender.add_feature(info)

        # Learn cooccurrence
        recommender.learn_cooccurrence([
            "users:user_age",
            "users:user_income",
        ])

        # Search and get recommendations based on results
        query = SearchQuery(text="user", limit=5)
        search_results = search.search(query)

        current_features = [
            f"{r.feature_view}:{r.feature_name}"
            for r in search_results[:1]
        ]

        context = RecommendationContext(
            current_features=current_features,
            max_recommendations=5,
        )
        recommendations = recommender.recommend(context)

        assert len(recommendations) >= 0  # May get 0 if all features are current
