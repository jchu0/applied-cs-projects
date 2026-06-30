"""Text feature transformers."""

import re
import math
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import numpy as np

from feature_platform.transformers.base import BaseTransformer


class TextCleaner(BaseTransformer):
    """
    Clean and normalize text data.

    Parameters:
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation characters
        remove_numbers: Remove numeric characters
        remove_whitespace: Normalize whitespace
        remove_stopwords: Remove common stopwords
        min_length: Minimum token length
        custom_stopwords: Additional stopwords to remove
    """

    DEFAULT_STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "i", "you", "he", "she", "it",
        "we", "they", "what", "which", "who", "when", "where", "why", "how",
    }

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_whitespace: bool = True,
        remove_stopwords: bool = False,
        min_length: int = 1,
        custom_stopwords: Optional[Set[str]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_whitespace = remove_whitespace
        self.remove_stopwords = remove_stopwords
        self.min_length = min_length
        self.custom_stopwords = custom_stopwords or set()
        self.stopwords_: Set[str] = set()
        self.n_features_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "TextCleaner":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"{col}_cleaned" for col in self._input_columns]

        if self.remove_stopwords:
            self.stopwords_ = self.DEFAULT_STOPWORDS | self.custom_stopwords

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def _clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not isinstance(text, str):
            text = str(text)

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        if self.remove_whitespace:
            text = ' '.join(text.split())

        if self.remove_stopwords or self.min_length > 1:
            tokens = text.split()
            tokens = [t for t in tokens if len(t) >= self.min_length]
            if self.remove_stopwords:
                tokens = [t for t in tokens if t not in self.stopwords_]
            text = ' '.join(tokens)

        return text

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.empty((X.shape[0], self.n_features_), dtype=object)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                result[j, i] = self._clean_text(str(X[j, i]))

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "remove_numbers": self.remove_numbers,
            "remove_whitespace": self.remove_whitespace,
            "remove_stopwords": self.remove_stopwords,
            "min_length": self.min_length,
        }


class CountVectorizer(BaseTransformer):
    """
    Convert text to a matrix of token counts.

    Parameters:
        max_features: Maximum number of features (vocabulary size)
        min_df: Minimum document frequency (int or float)
        max_df: Maximum document frequency (int or float)
        binary: If True, output binary indicators instead of counts
        ngram_range: Range of n-grams (min_n, max_n)
    """

    def __init__(
        self,
        max_features: Optional[int] = None,
        min_df: float = 1,
        max_df: float = 1.0,
        binary: bool = False,
        ngram_range: tuple = (1, 1),
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.ngram_range = ngram_range
        self.vocabulary_: Optional[Dict[str, int]] = None
        self.n_features_: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return text.lower().split()

    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        """Generate n-grams from tokens."""
        ngrams = []
        min_n, max_n = self.ngram_range

        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i + n])
                ngrams.append(ngram)

        return ngrams

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "CountVectorizer":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Count document frequencies
        doc_freq: Dict[str, int] = Counter()
        n_docs = X.shape[0] * self.n_features_

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                text = str(X[j, i])
                tokens = self._tokenize(text)
                ngrams = self._get_ngrams(tokens)
                # Count unique ngrams per document
                for ngram in set(ngrams):
                    doc_freq[ngram] += 1

        # Filter by document frequency
        min_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        max_count = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)

        filtered_terms = [
            term for term, count in doc_freq.items()
            if min_count <= count <= max_count
        ]

        # Sort by frequency and limit features
        filtered_terms = sorted(filtered_terms, key=lambda t: doc_freq[t], reverse=True)
        if self.max_features:
            filtered_terms = filtered_terms[:self.max_features]

        # Build vocabulary
        self.vocabulary_ = {term: idx for idx, term in enumerate(filtered_terms)}

        # Generate output column names
        self._output_columns = [f"term_{term}" for term in filtered_terms]

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        n_vocab = len(self.vocabulary_)
        result = np.zeros((X.shape[0], n_vocab), dtype=np.float64)

        for j in range(X.shape[0]):
            # Concatenate all text columns for this row
            all_text = ' '.join(str(X[j, i]) for i in range(self.n_features_))
            tokens = self._tokenize(all_text)
            ngrams = self._get_ngrams(tokens)

            for ngram in ngrams:
                if ngram in self.vocabulary_:
                    idx = self.vocabulary_[ngram]
                    if self.binary:
                        result[j, idx] = 1.0
                    else:
                        result[j, idx] += 1.0

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "binary": self.binary,
            "ngram_range": self.ngram_range,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "vocabulary": self.vocabulary_,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.vocabulary_ = stats.get("vocabulary")
        self.n_features_ = stats.get("n_features", 0)


class TfidfVectorizer(BaseTransformer):
    """
    Convert text to TF-IDF representation.

    Parameters:
        max_features: Maximum number of features
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        norm: Normalization method ('l1', 'l2', None)
        use_idf: If True, use IDF weighting
        smooth_idf: If True, add 1 to document frequencies to prevent division by zero
        sublinear_tf: If True, use log(tf) + 1
        ngram_range: Range of n-grams
    """

    def __init__(
        self,
        max_features: Optional[int] = None,
        min_df: float = 1,
        max_df: float = 1.0,
        norm: str = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        ngram_range: tuple = (1, 1),
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.ngram_range = ngram_range
        self.vocabulary_: Optional[Dict[str, int]] = None
        self.idf_: Optional[np.ndarray] = None
        self.n_features_: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return text.lower().split()

    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        """Generate n-grams from tokens."""
        ngrams = []
        min_n, max_n = self.ngram_range

        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i + n])
                ngrams.append(ngram)

        return ngrams

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "TfidfVectorizer":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Count document frequencies
        doc_freq: Dict[str, int] = Counter()
        n_docs = X.shape[0]

        for j in range(X.shape[0]):
            all_text = ' '.join(str(X[j, i]) for i in range(self.n_features_))
            tokens = self._tokenize(all_text)
            ngrams = self._get_ngrams(tokens)
            for ngram in set(ngrams):
                doc_freq[ngram] += 1

        # Filter by document frequency
        min_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        max_count = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)

        filtered_terms = [
            term for term, count in doc_freq.items()
            if min_count <= count <= max_count
        ]

        # Sort and limit features
        filtered_terms = sorted(filtered_terms, key=lambda t: doc_freq[t], reverse=True)
        if self.max_features:
            filtered_terms = filtered_terms[:self.max_features]

        # Build vocabulary
        self.vocabulary_ = {term: idx for idx, term in enumerate(filtered_terms)}

        # Compute IDF
        if self.use_idf:
            n_vocab = len(self.vocabulary_)
            self.idf_ = np.zeros(n_vocab)

            for term, idx in self.vocabulary_.items():
                df = doc_freq[term]
                if self.smooth_idf:
                    self.idf_[idx] = math.log((n_docs + 1) / (df + 1)) + 1
                else:
                    self.idf_[idx] = math.log(n_docs / df) + 1

        # Generate output column names
        self._output_columns = [f"tfidf_{term}" for term in filtered_terms]

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        n_vocab = len(self.vocabulary_)
        result = np.zeros((X.shape[0], n_vocab), dtype=np.float64)

        for j in range(X.shape[0]):
            all_text = ' '.join(str(X[j, i]) for i in range(self.n_features_))
            tokens = self._tokenize(all_text)
            ngrams = self._get_ngrams(tokens)
            term_counts = Counter(ngrams)

            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]

                    # Compute TF
                    if self.sublinear_tf:
                        tf = math.log(count) + 1 if count > 0 else 0
                    else:
                        tf = count

                    # Apply IDF
                    if self.use_idf:
                        result[j, idx] = tf * self.idf_[idx]
                    else:
                        result[j, idx] = tf

            # Normalize
            if self.norm:
                row_norm = np.linalg.norm(result[j], ord=2 if self.norm == "l2" else 1)
                if row_norm > 0:
                    result[j] /= row_norm

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "norm": self.norm,
            "use_idf": self.use_idf,
            "smooth_idf": self.smooth_idf,
            "sublinear_tf": self.sublinear_tf,
            "ngram_range": self.ngram_range,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "vocabulary": self.vocabulary_,
            "idf": self.idf_.tolist() if self.idf_ is not None else None,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.vocabulary_ = stats.get("vocabulary")
        if stats.get("idf"):
            self.idf_ = np.array(stats["idf"])
        self.n_features_ = stats.get("n_features", 0)


class HashingVectorizer(BaseTransformer):
    """
    Convert text to a fixed-size feature space using hashing.

    Parameters:
        n_features: Number of output features (hash buckets)
        alternate_sign: If True, alternate between +1 and -1 based on hash
        norm: Normalization method ('l1', 'l2', None)
        ngram_range: Range of n-grams
    """

    def __init__(
        self,
        n_features: int = 1024,
        alternate_sign: bool = True,
        norm: Optional[str] = "l2",
        ngram_range: tuple = (1, 1),
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.n_output_features = n_features
        self.alternate_sign = alternate_sign
        self.norm = norm
        self.ngram_range = ngram_range
        self.n_features_: int = 0

    def _hash(self, value: str) -> int:
        """Hash a string value."""
        h = 0
        for char in value:
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        return h

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return text.lower().split()

    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        """Generate n-grams from tokens."""
        ngrams = []
        min_n, max_n = self.ngram_range

        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i + n])
                ngrams.append(ngram)

        return ngrams

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "HashingVectorizer":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]
        self._output_columns = [f"hash_{i}" for i in range(self.n_output_features)]

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        result = np.zeros((X.shape[0], self.n_output_features), dtype=np.float64)

        for j in range(X.shape[0]):
            all_text = ' '.join(str(X[j, i]) for i in range(self.n_features_))
            tokens = self._tokenize(all_text)
            ngrams = self._get_ngrams(tokens)

            for ngram in ngrams:
                h = self._hash(ngram)
                idx = h % self.n_output_features

                if self.alternate_sign:
                    sign = 1 if (self._hash(ngram + "_s") % 2) == 0 else -1
                    result[j, idx] += sign
                else:
                    result[j, idx] += 1

            # Normalize
            if self.norm:
                row_norm = np.linalg.norm(result[j], ord=2 if self.norm == "l2" else 1)
                if row_norm > 0:
                    result[j] /= row_norm

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_output_features,
            "alternate_sign": self.alternate_sign,
            "norm": self.norm,
            "ngram_range": self.ngram_range,
        }


class NGramExtractor(BaseTransformer):
    """
    Extract character or word n-grams from text.

    Parameters:
        n: N-gram size
        level: Level of n-grams ('char' or 'word')
        max_features: Maximum number of n-grams to keep
    """

    def __init__(
        self,
        n: int = 3,
        level: str = "char",
        max_features: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.n = n
        self.level = level
        self.max_features = max_features
        self.vocabulary_: Optional[Dict[str, int]] = None
        self.n_features_: int = 0

    def _get_ngrams(self, text: str) -> List[str]:
        """Extract n-grams from text."""
        if self.level == "char":
            return [text[i:i + self.n] for i in range(len(text) - self.n + 1)]
        else:  # word
            tokens = text.lower().split()
            return [' '.join(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)]

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "NGramExtractor":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Count n-gram frequencies
        ngram_freq: Dict[str, int] = Counter()

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                text = str(X[j, i])
                ngrams = self._get_ngrams(text)
                ngram_freq.update(ngrams)

        # Sort by frequency and limit
        sorted_ngrams = sorted(ngram_freq.keys(), key=lambda x: ngram_freq[x], reverse=True)
        if self.max_features:
            sorted_ngrams = sorted_ngrams[:self.max_features]

        self.vocabulary_ = {ngram: idx for idx, ngram in enumerate(sorted_ngrams)}
        self._output_columns = [f"ngram_{i}" for i in range(len(self.vocabulary_))]

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        n_vocab = len(self.vocabulary_)
        result = np.zeros((X.shape[0], n_vocab), dtype=np.float64)

        for j in range(X.shape[0]):
            all_text = ' '.join(str(X[j, i]) for i in range(self.n_features_))
            ngrams = self._get_ngrams(all_text)
            ngram_counts = Counter(ngrams)

            for ngram, count in ngram_counts.items():
                if ngram in self.vocabulary_:
                    idx = self.vocabulary_[ngram]
                    result[j, idx] = count

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "level": self.level,
            "max_features": self.max_features,
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "vocabulary": self.vocabulary_,
            "n_features": self.n_features_,
        }

    def _set_statistics(self, stats: Dict[str, Any]) -> None:
        self.vocabulary_ = stats.get("vocabulary")
        self.n_features_ = stats.get("n_features", 0)


class TextStatistics(BaseTransformer):
    """
    Extract statistical features from text.

    Parameters:
        features: List of statistics to compute
            ('length', 'word_count', 'avg_word_length', 'char_count',
             'digit_count', 'upper_count', 'lower_count', 'punct_count',
             'unique_words', 'sentence_count')
    """

    VALID_FEATURES = [
        "length", "word_count", "avg_word_length", "char_count",
        "digit_count", "upper_count", "lower_count", "punct_count",
        "unique_words", "sentence_count", "avg_sentence_length",
    ]

    def __init__(
        self,
        features: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.features = features or ["length", "word_count", "avg_word_length"]
        self.n_features_: int = 0

        for feat in self.features:
            if feat not in self.VALID_FEATURES:
                raise ValueError(f"Invalid feature: {feat}")

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        columns: Optional[List[str]] = None,
    ) -> "TextStatistics":
        X = self._validate_input(X)
        X = self._ensure_2d(X)

        self.n_features_ = X.shape[1]
        self._input_columns = columns or [f"x{i}" for i in range(self.n_features_)]

        # Generate output column names
        self._output_columns = []
        for col in self._input_columns:
            for feat in self.features:
                self._output_columns.append(f"{col}_{feat}")

        self._fitted_at = datetime.utcnow()
        self.is_fitted = True
        return self

    def _compute_stat(self, text: str, stat: str) -> float:
        """Compute a single statistic for text."""
        if stat == "length":
            return len(text)
        elif stat == "word_count":
            return len(text.split())
        elif stat == "avg_word_length":
            words = text.split()
            return sum(len(w) for w in words) / len(words) if words else 0
        elif stat == "char_count":
            return len(text.replace(" ", ""))
        elif stat == "digit_count":
            return sum(c.isdigit() for c in text)
        elif stat == "upper_count":
            return sum(c.isupper() for c in text)
        elif stat == "lower_count":
            return sum(c.islower() for c in text)
        elif stat == "punct_count":
            return sum(c in '.,!?;:"\'-()[]{}' for c in text)
        elif stat == "unique_words":
            return len(set(text.lower().split()))
        elif stat == "sentence_count":
            return len(re.split(r'[.!?]+', text.strip())) if text.strip() else 0
        elif stat == "avg_sentence_length":
            sentences = re.split(r'[.!?]+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                return 0
            return sum(len(s.split()) for s in sentences) / len(sentences)
        else:
            return 0

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_input(X, require_fitted=True)
        X = self._ensure_2d(X)

        n_output_cols = self.n_features_ * len(self.features)
        result = np.zeros((X.shape[0], n_output_cols), dtype=np.float64)

        for i in range(self.n_features_):
            for j in range(X.shape[0]):
                text = str(X[j, i])
                for k, stat in enumerate(self.features):
                    col_idx = i * len(self.features) + k
                    result[j, col_idx] = self._compute_stat(text, stat)

        return result

    def _get_parameters(self) -> Dict[str, Any]:
        return {"features": self.features}
