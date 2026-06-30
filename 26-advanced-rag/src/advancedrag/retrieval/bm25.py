"""BM25 retrieval implementation."""

import math
import re
from collections import defaultdict
from typing import Optional
import numpy as np

from ..schemas import Document, RetrievalResult


class BM25Index:
    """BM25 index for lexical retrieval."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b

        # Index data
        self.documents: list[Document] = []
        self.doc_ids: list[str] = []

        # BM25 statistics
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.doc_lens: list[int] = []
        self.avg_doc_len: float = 0
        self.term_freqs: list[dict[str, int]] = []
        self.n_docs: int = 0

        # Inverted index for efficiency
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def add_documents(self, documents: list[Document]):
        """Add documents to BM25 index."""
        for doc in documents:
            self.documents.append(doc)
            self.doc_ids.append(doc.id)

            # Tokenize
            terms = self._tokenize(doc.content)
            self.doc_lens.append(len(terms))

            # Count term frequencies
            term_freq = defaultdict(int)
            for term in terms:
                term_freq[term] += 1
            self.term_freqs.append(dict(term_freq))

            # Update document frequencies and inverted index
            doc_idx = len(self.documents) - 1
            for term in set(terms):
                self.doc_freqs[term] += 1
                self.inverted_index[term].append((doc_idx, term_freq[term]))

        self.n_docs = len(self.documents)
        self.avg_doc_len = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 0

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict = None,
    ) -> list[RetrievalResult]:
        """Search using BM25 scoring."""
        if self.n_docs == 0:
            return []

        query_terms = self._tokenize(query)

        # Use inverted index for efficient scoring
        doc_scores = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            # IDF for this term
            df = self.doc_freqs[term]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

            # Score each document containing this term
            for doc_idx, tf in self.inverted_index[term]:
                doc_len = self.doc_lens[doc_idx]

                # BM25 score
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                )

                doc_scores[doc_idx] += idf * tf_norm

        # Apply filter if provided
        if filter_dict:
            filtered_scores = {}
            for doc_idx, score in doc_scores.items():
                if self._match_filter(self.documents[doc_idx].metadata, filter_dict):
                    filtered_scores[doc_idx] = score
            doc_scores = filtered_scores

        # Get top-k results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (doc_idx, score) in enumerate(sorted_docs):
            results.append(RetrievalResult(
                document=self.documents[doc_idx],
                score=float(score),
                retriever_type="bm25",
                rank=rank,
            ))

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization with stopword removal."""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())

        # Basic stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
            'neither', 'not', 'only', 'same', 'than', 'too', 'very',
            'just', 'also', 'now', 'here', 'there', 'where', 'when',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'any', 'this', 'that',
        }

        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _match_filter(self, metadata: dict, filter_dict: dict) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False

            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$in":
                        if metadata[key] not in op_value:
                            return False
                    elif op == "$eq":
                        if metadata[key] != op_value:
                            return False
                    elif op == "$ne":
                        if metadata[key] == op_value:
                            return False
                    elif op == "$gte":
                        if metadata[key] < op_value:
                            return False
                    elif op == "$gt":
                        if metadata[key] <= op_value:
                            return False
                    elif op == "$lte":
                        if metadata[key] > op_value:
                            return False
                    elif op == "$lt":
                        if metadata[key] >= op_value:
                            return False
            else:
                if metadata[key] != value:
                    return False

        return True

    def delete(self, doc_ids: list[str]):
        """Delete documents by ID."""
        ids_to_delete = set(doc_ids)
        indices_to_keep = [
            i for i, doc_id in enumerate(self.doc_ids)
            if doc_id not in ids_to_delete
        ]

        # Rebuild index with remaining documents
        remaining_docs = [self.documents[i] for i in indices_to_keep]

        # Clear and rebuild
        self.documents = []
        self.doc_ids = []
        self.doc_freqs = defaultdict(int)
        self.doc_lens = []
        self.term_freqs = []
        self.inverted_index = defaultdict(list)

        if remaining_docs:
            self.add_documents(remaining_docs)

    @property
    def count(self) -> int:
        """Get number of indexed documents."""
        return self.n_docs
