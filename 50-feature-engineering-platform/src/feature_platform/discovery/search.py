"""Feature search engine for discovering features in the registry."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import re
from collections import defaultdict


class SortOrder(Enum):
    """Sort order for search results."""

    RELEVANCE = "relevance"
    NAME = "name"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    POPULARITY = "popularity"


@dataclass
class SearchFilters:
    """Filters for feature search."""

    data_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    owners: Optional[List[str]] = None
    entity_types: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    min_quality_score: Optional[float] = None
    has_description: Optional[bool] = None
    is_deprecated: Optional[bool] = None


@dataclass
class SearchQuery:
    """Query for feature search."""

    text: str
    filters: Optional[SearchFilters] = None
    limit: int = 20
    offset: int = 0
    sort_by: SortOrder = SortOrder.RELEVANCE
    include_deprecated: bool = False


@dataclass
class SearchResult:
    """Result from feature search."""

    feature_name: str
    feature_view: str
    data_type: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    entity_type: Optional[str] = None
    quality_score: Optional[float] = None
    relevance_score: float = 0.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    usage_count: int = 0
    is_deprecated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureMetadata:
    """Internal representation of feature metadata for indexing."""

    name: str
    feature_view: str
    data_type: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    entity_type: Optional[str] = None
    quality_score: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    usage_count: int = 0
    is_deprecated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureSearchEngine:
    """
    Search engine for discovering features in the feature registry.

    Supports:
    - Full-text search on feature names and descriptions
    - Filtering by data type, tags, owners, entity types
    - Sorting by relevance, name, date, or popularity
    - Fuzzy matching for typo tolerance
    """

    def __init__(self, fuzzy_threshold: float = 0.7):
        self.fuzzy_threshold = fuzzy_threshold
        self._features: Dict[str, FeatureMetadata] = {}
        self._name_index: Dict[str, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._type_index: Dict[str, Set[str]] = defaultdict(set)
        self._owner_index: Dict[str, Set[str]] = defaultdict(set)
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)

    def index_feature(self, feature: FeatureMetadata) -> None:
        """Index a feature for searching."""
        key = f"{feature.feature_view}:{feature.name}"
        self._features[key] = feature

        # Index by name tokens
        tokens = self._tokenize(feature.name)
        for token in tokens:
            self._name_index[token].add(key)

        # Index description tokens
        if feature.description:
            desc_tokens = self._tokenize(feature.description)
            for token in desc_tokens:
                self._name_index[token].add(key)

        # Index by tags
        for tag in feature.tags:
            self._tag_index[tag.lower()].add(key)

        # Index by type
        self._type_index[feature.data_type.lower()].add(key)

        # Index by owner
        if feature.owner:
            self._owner_index[feature.owner.lower()].add(key)

        # Index by entity type
        if feature.entity_type:
            self._entity_index[feature.entity_type.lower()].add(key)

    def remove_feature(self, feature_view: str, feature_name: str) -> bool:
        """Remove a feature from the index."""
        key = f"{feature_view}:{feature_name}"
        if key not in self._features:
            return False

        feature = self._features[key]

        # Remove from name index
        tokens = self._tokenize(feature.name)
        for token in tokens:
            self._name_index[token].discard(key)

        if feature.description:
            desc_tokens = self._tokenize(feature.description)
            for token in desc_tokens:
                self._name_index[token].discard(key)

        # Remove from other indexes
        for tag in feature.tags:
            self._tag_index[tag.lower()].discard(key)

        self._type_index[feature.data_type.lower()].discard(key)

        if feature.owner:
            self._owner_index[feature.owner.lower()].discard(key)

        if feature.entity_type:
            self._entity_index[feature.entity_type.lower()].discard(key)

        del self._features[key]
        return True

    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Search for features matching the query.

        Returns list of SearchResult sorted by relevance or specified order.
        """
        # Get candidate keys
        candidates = self._get_candidates(query)

        # Apply filters
        filtered = self._apply_filters(candidates, query)

        # Score and rank
        scored = self._score_results(filtered, query)

        # Sort
        sorted_results = self._sort_results(scored, query.sort_by)

        # Paginate
        start = query.offset
        end = query.offset + query.limit
        return sorted_results[start:end]

    def get_popular_features(self, limit: int = 10) -> List[SearchResult]:
        """Get most popular features by usage count."""
        results = [
            self._to_search_result(f, relevance_score=0.0)
            for f in self._features.values()
            if not f.is_deprecated
        ]
        results.sort(key=lambda x: x.usage_count, reverse=True)
        return results[:limit]

    def get_recent_features(self, limit: int = 10) -> List[SearchResult]:
        """Get most recently created or updated features."""
        results = [
            self._to_search_result(f, relevance_score=0.0)
            for f in self._features.values()
            if not f.is_deprecated
        ]
        results.sort(
            key=lambda x: x.updated_at or x.created_at or datetime.min,
            reverse=True,
        )
        return results[:limit]

    def get_features_by_tag(self, tag: str) -> List[SearchResult]:
        """Get all features with a specific tag."""
        keys = self._tag_index.get(tag.lower(), set())
        return [
            self._to_search_result(self._features[k], relevance_score=1.0)
            for k in keys
            if not self._features[k].is_deprecated
        ]

    def get_features_by_owner(self, owner: str) -> List[SearchResult]:
        """Get all features owned by a specific user/team."""
        keys = self._owner_index.get(owner.lower(), set())
        return [
            self._to_search_result(self._features[k], relevance_score=1.0)
            for k in keys
            if not self._features[k].is_deprecated
        ]

    def get_all_tags(self) -> List[str]:
        """Get all unique tags in the index."""
        return list(self._tag_index.keys())

    def get_all_owners(self) -> List[str]:
        """Get all unique owners in the index."""
        return list(self._owner_index.keys())

    def get_all_entity_types(self) -> List[str]:
        """Get all unique entity types in the index."""
        return list(self._entity_index.keys())

    def get_feature_count(self) -> int:
        """Get total number of indexed features."""
        return len(self._features)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for indexing/searching."""
        # Split on non-alphanumeric, convert to lowercase
        tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
        return [t for t in tokens if t and len(t) > 1]

    def _get_candidates(self, query: SearchQuery) -> Set[str]:
        """Get candidate feature keys based on query text."""
        if not query.text.strip():
            return set(self._features.keys())

        tokens = self._tokenize(query.text)
        if not tokens:
            return set(self._features.keys())

        candidates: Set[str] = set()

        for token in tokens:
            # Exact matches
            if token in self._name_index:
                candidates.update(self._name_index[token])

            # Fuzzy matches
            for indexed_token, keys in self._name_index.items():
                if self._fuzzy_match(token, indexed_token):
                    candidates.update(keys)

        return candidates

    def _fuzzy_match(self, query_token: str, indexed_token: str) -> bool:
        """Check if tokens match with fuzzy tolerance."""
        if query_token == indexed_token:
            return True

        # Prefix match
        if indexed_token.startswith(query_token):
            return True

        # Substring match
        if query_token in indexed_token:
            return True

        # Levenshtein distance for short tokens
        if len(query_token) <= 5 and len(indexed_token) <= 10:
            distance = self._levenshtein_distance(query_token, indexed_token)
            max_len = max(len(query_token), len(indexed_token))
            similarity = 1 - (distance / max_len)
            return similarity >= self.fuzzy_threshold

        return False

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _apply_filters(self, candidates: Set[str], query: SearchQuery) -> Set[str]:
        """Apply search filters to candidates."""
        if not query.filters:
            if not query.include_deprecated:
                return {k for k in candidates if not self._features[k].is_deprecated}
            return candidates

        filters = query.filters
        result = candidates.copy()

        # Filter by deprecated status
        if not query.include_deprecated:
            result = {k for k in result if not self._features[k].is_deprecated}

        if filters.is_deprecated is not None:
            result = {
                k for k in result
                if self._features[k].is_deprecated == filters.is_deprecated
            }

        # Filter by data types
        if filters.data_types:
            type_keys: Set[str] = set()
            for dt in filters.data_types:
                type_keys.update(self._type_index.get(dt.lower(), set()))
            result &= type_keys

        # Filter by tags (OR logic - match any tag)
        if filters.tags:
            tag_keys: Set[str] = set()
            for tag in filters.tags:
                tag_keys.update(self._tag_index.get(tag.lower(), set()))
            result &= tag_keys

        # Filter by owners
        if filters.owners:
            owner_keys: Set[str] = set()
            for owner in filters.owners:
                owner_keys.update(self._owner_index.get(owner.lower(), set()))
            result &= owner_keys

        # Filter by entity types
        if filters.entity_types:
            entity_keys: Set[str] = set()
            for entity in filters.entity_types:
                entity_keys.update(self._entity_index.get(entity.lower(), set()))
            result &= entity_keys

        # Filter by date ranges
        if filters.created_after:
            result = {
                k for k in result
                if self._features[k].created_at and
                self._features[k].created_at >= filters.created_after
            }

        if filters.created_before:
            result = {
                k for k in result
                if self._features[k].created_at and
                self._features[k].created_at <= filters.created_before
            }

        if filters.updated_after:
            result = {
                k for k in result
                if self._features[k].updated_at and
                self._features[k].updated_at >= filters.updated_after
            }

        # Filter by quality score
        if filters.min_quality_score is not None:
            result = {
                k for k in result
                if self._features[k].quality_score is not None and
                self._features[k].quality_score >= filters.min_quality_score
            }

        # Filter by has description
        if filters.has_description is not None:
            if filters.has_description:
                result = {
                    k for k in result
                    if self._features[k].description
                }
            else:
                result = {
                    k for k in result
                    if not self._features[k].description
                }

        return result

    def _score_results(
        self,
        candidates: Set[str],
        query: SearchQuery,
    ) -> List[SearchResult]:
        """Score candidates for relevance ranking."""
        results = []
        query_tokens = self._tokenize(query.text) if query.text else []

        for key in candidates:
            feature = self._features[key]
            score = self._calculate_relevance(feature, query_tokens)
            results.append(self._to_search_result(feature, score))

        return results

    def _calculate_relevance(
        self,
        feature: FeatureMetadata,
        query_tokens: List[str],
    ) -> float:
        """Calculate relevance score for a feature."""
        if not query_tokens:
            return 0.0

        score = 0.0
        name_tokens = self._tokenize(feature.name)
        desc_tokens = self._tokenize(feature.description) if feature.description else []

        for qt in query_tokens:
            # Exact name match (highest weight)
            if qt in name_tokens:
                score += 10.0
            # Prefix match in name
            elif any(nt.startswith(qt) for nt in name_tokens):
                score += 5.0
            # Description match
            elif qt in desc_tokens:
                score += 2.0
            # Tag match
            elif any(qt == tag.lower() for tag in feature.tags):
                score += 3.0

        # Boost by quality score
        if feature.quality_score:
            score *= (1 + feature.quality_score * 0.1)

        # Boost by popularity
        if feature.usage_count > 0:
            score *= (1 + min(feature.usage_count, 100) * 0.01)

        return score

    def _sort_results(
        self,
        results: List[SearchResult],
        sort_by: SortOrder,
    ) -> List[SearchResult]:
        """Sort results by specified order."""
        if sort_by == SortOrder.RELEVANCE:
            return sorted(results, key=lambda x: x.relevance_score, reverse=True)
        elif sort_by == SortOrder.NAME:
            return sorted(results, key=lambda x: x.feature_name.lower())
        elif sort_by == SortOrder.CREATED_AT:
            return sorted(
                results,
                key=lambda x: x.created_at or datetime.min,
                reverse=True,
            )
        elif sort_by == SortOrder.UPDATED_AT:
            return sorted(
                results,
                key=lambda x: x.updated_at or datetime.min,
                reverse=True,
            )
        elif sort_by == SortOrder.POPULARITY:
            return sorted(results, key=lambda x: x.usage_count, reverse=True)
        return results

    def _to_search_result(
        self,
        feature: FeatureMetadata,
        relevance_score: float,
    ) -> SearchResult:
        """Convert feature metadata to search result."""
        return SearchResult(
            feature_name=feature.name,
            feature_view=feature.feature_view,
            data_type=feature.data_type,
            description=feature.description,
            tags=feature.tags.copy(),
            owner=feature.owner,
            entity_type=feature.entity_type,
            quality_score=feature.quality_score,
            relevance_score=relevance_score,
            created_at=feature.created_at,
            updated_at=feature.updated_at,
            usage_count=feature.usage_count,
            is_deprecated=feature.is_deprecated,
            metadata=feature.metadata.copy(),
        )
