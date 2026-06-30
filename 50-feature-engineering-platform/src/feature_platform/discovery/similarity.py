"""Feature similarity analysis for discovering related features."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class SimilarityMethod(Enum):
    """Methods for computing feature similarity."""

    STATISTICAL = "statistical"  # Statistical profile similarity
    CORRELATION = "correlation"  # Correlation-based similarity
    NAME = "name"  # Name/description text similarity
    METADATA = "metadata"  # Metadata similarity (tags, owner, etc.)
    COMBINED = "combined"  # Weighted combination


@dataclass
class SimilarityResult:
    """Result of feature similarity computation."""

    source_feature: str
    target_feature: str
    similarity_score: float
    method: SimilarityMethod
    breakdown: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureProfile:
    """Statistical profile of a feature for similarity comparison."""

    name: str
    feature_view: str
    data_type: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None

    # Statistical profile
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    median: Optional[float] = None
    null_ratio: Optional[float] = None
    unique_ratio: Optional[float] = None
    distribution_type: Optional[str] = None

    # Sample values for correlation
    sample_values: Optional[np.ndarray] = None


class FeatureSimilarityEngine:
    """
    Engine for computing and finding similar features.

    Supports multiple similarity methods:
    - Statistical: Compare statistical profiles (mean, std, distribution)
    - Correlation: Compute correlation between feature values
    - Name: Text similarity on names and descriptions
    - Metadata: Similarity based on tags, owners, entity types
    - Combined: Weighted combination of all methods
    """

    def __init__(
        self,
        stat_weight: float = 0.3,
        corr_weight: float = 0.3,
        name_weight: float = 0.2,
        meta_weight: float = 0.2,
    ):
        self.stat_weight = stat_weight
        self.corr_weight = corr_weight
        self.name_weight = name_weight
        self.meta_weight = meta_weight

        self._profiles: Dict[str, FeatureProfile] = {}

    def add_profile(self, profile: FeatureProfile) -> None:
        """Add a feature profile for similarity comparison."""
        key = f"{profile.feature_view}:{profile.name}"
        self._profiles[key] = profile

    def remove_profile(self, feature_view: str, name: str) -> bool:
        """Remove a feature profile."""
        key = f"{feature_view}:{name}"
        if key in self._profiles:
            del self._profiles[key]
            return True
        return False

    def find_similar(
        self,
        feature_view: str,
        name: str,
        method: SimilarityMethod = SimilarityMethod.COMBINED,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SimilarityResult]:
        """
        Find features similar to the given feature.

        Args:
            feature_view: Feature view name
            name: Feature name
            method: Similarity computation method
            top_k: Number of similar features to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of SimilarityResult sorted by similarity score
        """
        key = f"{feature_view}:{name}"
        if key not in self._profiles:
            return []

        source_profile = self._profiles[key]
        results = []

        for target_key, target_profile in self._profiles.items():
            if target_key == key:
                continue

            similarity = self.compute_similarity(
                source_profile, target_profile, method
            )

            if similarity.similarity_score >= min_similarity:
                results.append(similarity)

        # Sort by similarity score (descending)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]

    def compute_similarity(
        self,
        source: FeatureProfile,
        target: FeatureProfile,
        method: SimilarityMethod = SimilarityMethod.COMBINED,
    ) -> SimilarityResult:
        """Compute similarity between two feature profiles."""
        source_key = f"{source.feature_view}:{source.name}"
        target_key = f"{target.feature_view}:{target.name}"

        if method == SimilarityMethod.STATISTICAL:
            score, breakdown = self._compute_statistical_similarity(source, target)
        elif method == SimilarityMethod.CORRELATION:
            score, breakdown = self._compute_correlation_similarity(source, target)
        elif method == SimilarityMethod.NAME:
            score, breakdown = self._compute_name_similarity(source, target)
        elif method == SimilarityMethod.METADATA:
            score, breakdown = self._compute_metadata_similarity(source, target)
        elif method == SimilarityMethod.COMBINED:
            score, breakdown = self._compute_combined_similarity(source, target)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

        return SimilarityResult(
            source_feature=source_key,
            target_feature=target_key,
            similarity_score=score,
            method=method,
            breakdown=breakdown,
        )

    def find_duplicates(
        self,
        similarity_threshold: float = 0.95,
    ) -> List[Tuple[str, str, float]]:
        """
        Find potentially duplicate features.

        Returns list of (feature1, feature2, similarity) tuples.
        """
        duplicates = []
        keys = list(self._profiles.keys())

        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1:]:
                profile1 = self._profiles[key1]
                profile2 = self._profiles[key2]

                result = self.compute_similarity(
                    profile1, profile2, SimilarityMethod.COMBINED
                )

                if result.similarity_score >= similarity_threshold:
                    duplicates.append((key1, key2, result.similarity_score))

        # Sort by similarity (highest first)
        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates

    def cluster_features(
        self,
        n_clusters: int = 5,
        method: SimilarityMethod = SimilarityMethod.COMBINED,
    ) -> Dict[int, List[str]]:
        """
        Cluster features by similarity.

        Uses a simple greedy clustering approach.
        Returns mapping of cluster_id -> list of feature keys.
        """
        if not self._profiles:
            return {}

        keys = list(self._profiles.keys())
        n = len(keys)

        if n <= n_clusters:
            return {i: [k] for i, k in enumerate(keys)}

        # Build similarity matrix
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                result = self.compute_similarity(
                    self._profiles[keys[i]],
                    self._profiles[keys[j]],
                    method,
                )
                sim_matrix[i, j] = result.similarity_score
                sim_matrix[j, i] = result.similarity_score

        # Greedy clustering
        assigned = [-1] * n
        clusters: Dict[int, List[str]] = defaultdict(list)

        # Select initial centroids (most dissimilar)
        centroids = [0]
        for _ in range(1, n_clusters):
            min_sim = float('inf')
            best_idx = -1
            for i in range(n):
                if i in centroids:
                    continue
                max_sim_to_centroids = max(sim_matrix[i, c] for c in centroids)
                if max_sim_to_centroids < min_sim:
                    min_sim = max_sim_to_centroids
                    best_idx = i
            if best_idx >= 0:
                centroids.append(best_idx)

        # Assign to clusters
        for i in range(n):
            best_cluster = 0
            best_sim = sim_matrix[i, centroids[0]]
            for c_idx, centroid in enumerate(centroids):
                if sim_matrix[i, centroid] > best_sim:
                    best_sim = sim_matrix[i, centroid]
                    best_cluster = c_idx
            assigned[i] = best_cluster
            clusters[best_cluster].append(keys[i])

        return dict(clusters)

    def _compute_statistical_similarity(
        self,
        source: FeatureProfile,
        target: FeatureProfile,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute similarity based on statistical profiles."""
        breakdown = {}

        # Data type match (binary)
        type_sim = 1.0 if source.data_type == target.data_type else 0.0
        breakdown["type_match"] = type_sim

        # Skip numeric comparisons for non-numeric types
        if source.data_type not in ("float64", "float32", "int64", "int32", "numeric"):
            return type_sim, breakdown

        # Mean similarity (normalized difference)
        mean_sim = 0.0
        if source.mean is not None and target.mean is not None:
            max_mean = max(abs(source.mean), abs(target.mean), 1e-10)
            mean_sim = 1 - min(abs(source.mean - target.mean) / max_mean, 1.0)
        breakdown["mean_similarity"] = mean_sim

        # Std similarity
        std_sim = 0.0
        if source.std is not None and target.std is not None:
            max_std = max(source.std, target.std, 1e-10)
            std_sim = 1 - min(abs(source.std - target.std) / max_std, 1.0)
        breakdown["std_similarity"] = std_sim

        # Range similarity
        range_sim = 0.0
        if all(v is not None for v in [source.min_value, source.max_value,
                                        target.min_value, target.max_value]):
            source_range = source.max_value - source.min_value
            target_range = target.max_value - target.min_value
            max_range = max(source_range, target_range, 1e-10)
            range_sim = 1 - min(abs(source_range - target_range) / max_range, 1.0)
        breakdown["range_similarity"] = range_sim

        # Null ratio similarity
        null_sim = 0.0
        if source.null_ratio is not None and target.null_ratio is not None:
            null_sim = 1 - abs(source.null_ratio - target.null_ratio)
        breakdown["null_similarity"] = null_sim

        # Weighted combination
        weights = [0.1, 0.3, 0.2, 0.2, 0.2]
        scores = [type_sim, mean_sim, std_sim, range_sim, null_sim]
        total = sum(w * s for w, s in zip(weights, scores))

        return total, breakdown

    def _compute_correlation_similarity(
        self,
        source: FeatureProfile,
        target: FeatureProfile,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute similarity based on correlation of sample values."""
        breakdown = {}

        if source.sample_values is None or target.sample_values is None:
            return 0.0, {"correlation": 0.0, "note": "no_samples"}

        # Align sample lengths
        min_len = min(len(source.sample_values), len(target.sample_values))
        if min_len < 10:
            return 0.0, {"correlation": 0.0, "note": "insufficient_samples"}

        x = source.sample_values[:min_len]
        y = target.sample_values[:min_len]

        # Compute Pearson correlation
        try:
            corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        # Convert correlation to similarity (correlation ranges -1 to 1)
        # We care about absolute correlation (both positive and negative)
        similarity = abs(corr)
        breakdown["correlation"] = corr
        breakdown["absolute_correlation"] = similarity

        return similarity, breakdown

    def _compute_name_similarity(
        self,
        source: FeatureProfile,
        target: FeatureProfile,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute similarity based on feature names and descriptions."""
        breakdown = {}

        # Name similarity (Jaccard on tokens)
        source_tokens = self._tokenize(source.name)
        target_tokens = self._tokenize(target.name)

        name_sim = self._jaccard_similarity(source_tokens, target_tokens)
        breakdown["name_jaccard"] = name_sim

        # Description similarity
        desc_sim = 0.0
        if source.description and target.description:
            source_desc_tokens = self._tokenize(source.description)
            target_desc_tokens = self._tokenize(target.description)
            desc_sim = self._jaccard_similarity(source_desc_tokens, target_desc_tokens)
        breakdown["description_jaccard"] = desc_sim

        # Weighted combination
        total = 0.6 * name_sim + 0.4 * desc_sim
        return total, breakdown

    def _compute_metadata_similarity(
        self,
        source: FeatureProfile,
        target: FeatureProfile,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute similarity based on metadata (tags, owner, etc.)."""
        breakdown = {}

        # Tag similarity
        tag_sim = self._jaccard_similarity(
            set(t.lower() for t in source.tags),
            set(t.lower() for t in target.tags),
        )
        breakdown["tag_similarity"] = tag_sim

        # Owner match
        owner_sim = 0.0
        if source.owner and target.owner:
            owner_sim = 1.0 if source.owner.lower() == target.owner.lower() else 0.0
        breakdown["owner_match"] = owner_sim

        # Feature view match
        view_sim = 1.0 if source.feature_view == target.feature_view else 0.0
        breakdown["view_match"] = view_sim

        # Weighted combination
        total = 0.5 * tag_sim + 0.3 * owner_sim + 0.2 * view_sim
        return total, breakdown

    def _compute_combined_similarity(
        self,
        source: FeatureProfile,
        target: FeatureProfile,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute combined similarity using all methods."""
        stat_score, stat_breakdown = self._compute_statistical_similarity(source, target)
        corr_score, corr_breakdown = self._compute_correlation_similarity(source, target)
        name_score, name_breakdown = self._compute_name_similarity(source, target)
        meta_score, meta_breakdown = self._compute_metadata_similarity(source, target)

        combined = (
            self.stat_weight * stat_score +
            self.corr_weight * corr_score +
            self.name_weight * name_score +
            self.meta_weight * meta_score
        )

        breakdown = {
            "statistical": stat_score,
            "correlation": corr_score,
            "name": name_score,
            "metadata": meta_score,
            **{f"stat_{k}": v for k, v in stat_breakdown.items()},
            **{f"corr_{k}": v for k, v in corr_breakdown.items()},
            **{f"name_{k}": v for k, v in name_breakdown.items()},
            **{f"meta_{k}": v for k, v in meta_breakdown.items()},
        }

        return combined, breakdown

    def _tokenize(self, text: str) -> set:
        """Tokenize text into lowercase words."""
        import re
        tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
        return set(t for t in tokens if t and len(t) > 1)

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
