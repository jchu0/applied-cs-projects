"""Feature recommendations for ML projects."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict


class RecommendationType(Enum):
    """Types of feature recommendations."""

    SIMILAR = "similar"  # Features similar to currently used ones
    POPULAR = "popular"  # Popular features in the org
    COOCCURRENCE = "cooccurrence"  # Features often used together
    DOMAIN = "domain"  # Domain-specific recommendations
    QUALITY = "quality"  # High-quality alternatives


@dataclass
class RecommendationContext:
    """Context for generating feature recommendations."""

    # Current features in use
    current_features: List[str] = field(default_factory=list)

    # Target entity types (e.g., "user", "product")
    entity_types: List[str] = field(default_factory=list)

    # ML task type (e.g., "classification", "regression", "ranking")
    task_type: Optional[str] = None

    # Domain tags (e.g., "fraud", "churn", "recommendation")
    domain_tags: List[str] = field(default_factory=list)

    # Excluded features (already considered/rejected)
    excluded_features: List[str] = field(default_factory=list)

    # Maximum recommendations to return
    max_recommendations: int = 10


@dataclass
class FeatureRecommendation:
    """A feature recommendation."""

    feature_name: str
    feature_view: str
    recommendation_type: RecommendationType
    score: float
    reason: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureUsageRecord:
    """Record of feature usage in a project/model."""

    feature_key: str
    project_id: str
    model_id: Optional[str] = None
    task_type: Optional[str] = None
    entity_types: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    used_at: Optional[datetime] = None
    performance_impact: Optional[float] = None


@dataclass
class FeatureInfo:
    """Information about a feature for recommendations."""

    name: str
    feature_view: str
    data_type: str
    entity_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    quality_score: Optional[float] = None
    usage_count: int = 0
    description: Optional[str] = None


class FeatureRecommender:
    """
    Recommender system for discovering useful features.

    Provides recommendations based on:
    - Feature similarity to currently used features
    - Co-occurrence patterns (features used together)
    - Popularity and quality metrics
    - Domain and task-specific matching
    """

    def __init__(
        self,
        similarity_weight: float = 0.3,
        cooccurrence_weight: float = 0.3,
        popularity_weight: float = 0.2,
        quality_weight: float = 0.2,
    ):
        self.similarity_weight = similarity_weight
        self.cooccurrence_weight = cooccurrence_weight
        self.popularity_weight = popularity_weight
        self.quality_weight = quality_weight

        self._features: Dict[str, FeatureInfo] = {}
        self._usage_records: List[FeatureUsageRecord] = []
        self._cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._tag_features: Dict[str, Set[str]] = defaultdict(set)
        self._entity_features: Dict[str, Set[str]] = defaultdict(set)
        self._task_features: Dict[str, Set[str]] = defaultdict(set)

    def add_feature(self, info: FeatureInfo) -> None:
        """Add a feature to the recommender."""
        key = f"{info.feature_view}:{info.name}"
        self._features[key] = info

        # Index by tags
        for tag in info.tags:
            self._tag_features[tag.lower()].add(key)

        # Index by entity type
        if info.entity_type:
            self._entity_features[info.entity_type.lower()].add(key)

    def record_usage(self, record: FeatureUsageRecord) -> None:
        """Record feature usage for learning co-occurrence patterns."""
        self._usage_records.append(record)

        # Update task index
        if record.task_type:
            self._task_features[record.task_type.lower()].add(record.feature_key)

    def learn_cooccurrence(self, project_features: List[str]) -> None:
        """
        Learn co-occurrence from a list of features used together.

        Call this with features from completed projects to build
        co-occurrence statistics.
        """
        for i, f1 in enumerate(project_features):
            for f2 in project_features[i + 1:]:
                self._cooccurrence[f1][f2] += 1
                self._cooccurrence[f2][f1] += 1

    def recommend(
        self,
        context: RecommendationContext,
    ) -> List[FeatureRecommendation]:
        """
        Generate feature recommendations based on context.

        Args:
            context: RecommendationContext with current features and constraints

        Returns:
            List of FeatureRecommendation sorted by relevance
        """
        candidates = self._get_candidates(context)
        scored = self._score_candidates(candidates, context)

        # Sort by score and limit
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:context.max_recommendations]

    def get_cooccurrence_recommendations(
        self,
        features: List[str],
        top_k: int = 10,
    ) -> List[FeatureRecommendation]:
        """Get recommendations based on co-occurrence with given features."""
        cooccurrence_scores: Dict[str, int] = defaultdict(int)

        for feature in features:
            if feature in self._cooccurrence:
                for other_feature, count in self._cooccurrence[feature].items():
                    if other_feature not in features:
                        cooccurrence_scores[other_feature] += count

        # Sort by score
        sorted_features = sorted(
            cooccurrence_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        recommendations = []
        for feature_key, count in sorted_features[:top_k]:
            if feature_key not in self._features:
                continue

            info = self._features[feature_key]
            recommendations.append(
                FeatureRecommendation(
                    feature_name=info.name,
                    feature_view=info.feature_view,
                    recommendation_type=RecommendationType.COOCCURRENCE,
                    score=count / max(1, len(features)),
                    reason=f"Frequently used with {count} of your current features",
                    confidence=min(count / 10, 1.0),
                    metadata={"cooccurrence_count": count},
                )
            )

        return recommendations

    def get_domain_recommendations(
        self,
        domain_tags: List[str],
        excluded: Optional[Set[str]] = None,
        top_k: int = 10,
    ) -> List[FeatureRecommendation]:
        """Get recommendations for a specific domain."""
        excluded = excluded or set()
        candidates: Dict[str, int] = defaultdict(int)

        for tag in domain_tags:
            tag_lower = tag.lower()
            if tag_lower in self._tag_features:
                for feature_key in self._tag_features[tag_lower]:
                    if feature_key not in excluded:
                        candidates[feature_key] += 1

        # Score by tag match count and quality
        recommendations = []
        for feature_key, tag_matches in candidates.items():
            if feature_key not in self._features:
                continue

            info = self._features[feature_key]
            quality_boost = info.quality_score or 0.5
            score = (tag_matches / len(domain_tags)) * quality_boost

            recommendations.append(
                FeatureRecommendation(
                    feature_name=info.name,
                    feature_view=info.feature_view,
                    recommendation_type=RecommendationType.DOMAIN,
                    score=score,
                    reason=f"Matches {tag_matches} domain tags",
                    confidence=quality_boost,
                    metadata={"matching_tags": tag_matches},
                )
            )

        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:top_k]

    def get_popular_recommendations(
        self,
        entity_type: Optional[str] = None,
        excluded: Optional[Set[str]] = None,
        top_k: int = 10,
    ) -> List[FeatureRecommendation]:
        """Get most popular features, optionally filtered by entity type."""
        excluded = excluded or set()
        candidates = []

        for key, info in self._features.items():
            if key in excluded:
                continue

            if entity_type and info.entity_type:
                if info.entity_type.lower() != entity_type.lower():
                    continue

            candidates.append((key, info))

        # Sort by usage count
        candidates.sort(key=lambda x: x[1].usage_count, reverse=True)

        recommendations = []
        for key, info in candidates[:top_k]:
            recommendations.append(
                FeatureRecommendation(
                    feature_name=info.name,
                    feature_view=info.feature_view,
                    recommendation_type=RecommendationType.POPULAR,
                    score=info.usage_count / max(1, candidates[0][1].usage_count),
                    reason=f"Popular feature with {info.usage_count} uses",
                    confidence=min(info.usage_count / 100, 1.0),
                    metadata={"usage_count": info.usage_count},
                )
            )

        return recommendations

    def get_quality_recommendations(
        self,
        current_features: List[str],
        quality_threshold: float = 0.8,
        top_k: int = 10,
    ) -> List[FeatureRecommendation]:
        """Get high-quality features that could replace or complement current features."""
        recommendations = []

        for key, info in self._features.items():
            if key in current_features:
                continue

            if info.quality_score is None or info.quality_score < quality_threshold:
                continue

            recommendations.append(
                FeatureRecommendation(
                    feature_name=info.name,
                    feature_view=info.feature_view,
                    recommendation_type=RecommendationType.QUALITY,
                    score=info.quality_score,
                    reason=f"High quality feature (score: {info.quality_score:.2f})",
                    confidence=info.quality_score,
                    metadata={"quality_score": info.quality_score},
                )
            )

        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:top_k]

    def _get_candidates(self, context: RecommendationContext) -> Set[str]:
        """Get candidate features based on context filters."""
        excluded = set(context.current_features + context.excluded_features)
        candidates = set(self._features.keys()) - excluded

        # Filter by entity type if specified
        if context.entity_types:
            entity_candidates: Set[str] = set()
            for entity in context.entity_types:
                entity_candidates.update(
                    self._entity_features.get(entity.lower(), set())
                )
            candidates &= entity_candidates

        return candidates

    def _score_candidates(
        self,
        candidates: Set[str],
        context: RecommendationContext,
    ) -> List[FeatureRecommendation]:
        """Score candidate features."""
        recommendations = []
        excluded = set(context.current_features + context.excluded_features)

        for key in candidates:
            if key not in self._features:
                continue

            info = self._features[key]
            scores = {}
            reasons = []

            # Co-occurrence score
            cooccurrence_score = 0.0
            if context.current_features:
                cooccurrence_count = sum(
                    self._cooccurrence.get(f, {}).get(key, 0)
                    for f in context.current_features
                )
                if cooccurrence_count > 0:
                    cooccurrence_score = min(cooccurrence_count / 10, 1.0)
                    reasons.append(
                        f"Used with {cooccurrence_count} current features"
                    )
            scores["cooccurrence"] = cooccurrence_score

            # Domain match score
            domain_score = 0.0
            if context.domain_tags:
                matching_tags = sum(
                    1 for tag in context.domain_tags
                    if tag.lower() in [t.lower() for t in info.tags]
                )
                if matching_tags > 0:
                    domain_score = matching_tags / len(context.domain_tags)
                    reasons.append(f"Matches {matching_tags} domain tags")
            scores["domain"] = domain_score

            # Popularity score
            max_usage = max(
                (f.usage_count for f in self._features.values()),
                default=1
            )
            popularity_score = info.usage_count / max(max_usage, 1)
            scores["popularity"] = popularity_score

            # Quality score
            quality_score = info.quality_score or 0.5
            scores["quality"] = quality_score

            # Combined score
            total_score = (
                self.cooccurrence_weight * cooccurrence_score +
                self.similarity_weight * domain_score +  # Using domain as proxy for similarity
                self.popularity_weight * popularity_score +
                self.quality_weight * quality_score
            )

            # Determine recommendation type
            if cooccurrence_score > 0.3:
                rec_type = RecommendationType.COOCCURRENCE
            elif domain_score > 0.3:
                rec_type = RecommendationType.DOMAIN
            elif quality_score > 0.8:
                rec_type = RecommendationType.QUALITY
            else:
                rec_type = RecommendationType.POPULAR

            reason = "; ".join(reasons) if reasons else "Potentially useful feature"

            recommendations.append(
                FeatureRecommendation(
                    feature_name=info.name,
                    feature_view=info.feature_view,
                    recommendation_type=rec_type,
                    score=total_score,
                    reason=reason,
                    confidence=quality_score,
                    metadata={"score_breakdown": scores},
                )
            )

        return recommendations

    def get_feature_count(self) -> int:
        """Get total number of features in the recommender."""
        return len(self._features)

    def get_usage_record_count(self) -> int:
        """Get total number of usage records."""
        return len(self._usage_records)
