"""Feature discovery and search module."""

from feature_platform.discovery.search import (
    FeatureSearchEngine,
    SearchQuery,
    SearchResult,
    SearchFilters,
)
from feature_platform.discovery.similarity import (
    FeatureSimilarityEngine,
    SimilarityMethod,
    SimilarityResult,
)
from feature_platform.discovery.recommendations import (
    FeatureRecommender,
    RecommendationContext,
    FeatureRecommendation,
)

__all__ = [
    # Search
    "FeatureSearchEngine",
    "SearchQuery",
    "SearchResult",
    "SearchFilters",
    # Similarity
    "FeatureSimilarityEngine",
    "SimilarityMethod",
    "SimilarityResult",
    # Recommendations
    "FeatureRecommender",
    "RecommendationContext",
    "FeatureRecommendation",
]
