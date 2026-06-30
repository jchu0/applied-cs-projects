"""Multi-hop retrieval for complex queries requiring iterative search."""

import re
from typing import Optional

from ..schemas import RetrievalResult


class MultiHopRetriever:
    """Iterative retrieval that performs multiple search rounds for complex queries.

    Each hop retrieves documents, checks completeness, and optionally
    generates follow-up queries to fill information gaps.
    """

    def __init__(self, retriever, llm_client=None, max_hops: int = 3):
        self.retriever = retriever
        self.llm = llm_client
        self.max_hops = max_hops

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict = None,
    ) -> list[RetrievalResult]:
        """Perform multi-hop retrieval.

        Args:
            query: Initial user query
            top_k: Target number of unique results
            filter_dict: Metadata filter

        Returns:
            Merged and deduplicated retrieval results
        """
        all_results: list[RetrievalResult] = []
        current_query = query
        queries_used = [query]

        for hop in range(self.max_hops):
            # Retrieve for current query
            hop_results = self.retriever.search(
                current_query,
                top_k=top_k,
                filter_dict=filter_dict,
            )

            all_results = self._merge_results(all_results, hop_results)

            # Check if we have enough unique results
            if self._check_completeness(all_results, top_k):
                break

            # Generate follow-up query for next hop
            followup = await self._generate_followup(
                query, current_query, all_results, queries_used
            )
            if not followup or followup in queries_used:
                break

            current_query = followup
            queries_used.append(followup)

        # Re-rank merged results and return top_k
        all_results.sort(key=lambda r: r.score, reverse=True)
        for i, result in enumerate(all_results):
            result.rank = i

        return all_results[:top_k]

    def _merge_results(
        self,
        existing: list[RetrievalResult],
        new: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Merge new results into existing, deduplicating by doc ID and keeping highest score."""
        seen: dict[str, RetrievalResult] = {}

        for result in existing:
            doc_id = result.document.id
            if doc_id not in seen or result.score > seen[doc_id].score:
                seen[doc_id] = result

        for result in new:
            doc_id = result.document.id
            if doc_id not in seen or result.score > seen[doc_id].score:
                seen[doc_id] = result

        return list(seen.values())

    def _check_completeness(
        self,
        results: list[RetrievalResult],
        target_k: int,
    ) -> bool:
        """Check if we have enough unique results."""
        return len(results) >= target_k

    async def _generate_followup(
        self,
        original_query: str,
        current_query: str,
        results: list[RetrievalResult],
        queries_used: list[str],
    ) -> Optional[str]:
        """Generate a follow-up query to fill information gaps."""
        if self.llm:
            return await self._llm_followup(
                original_query, current_query, results, queries_used
            )
        return self._rule_based_followup(original_query, results)

    async def _llm_followup(
        self,
        original_query: str,
        current_query: str,
        results: list[RetrievalResult],
        queries_used: list[str],
    ) -> Optional[str]:
        """Generate follow-up query using LLM."""
        context_snippets = "\n".join(
            f"- {r.document.content[:100]}" for r in results[:5]
        )
        prev_queries = "\n".join(f"- {q}" for q in queries_used)

        prompt = f"""Given the original question and retrieved information, generate a follow-up search query to find missing information.

Original question: {original_query}
Previous queries:
{prev_queries}

Retrieved so far:
{context_snippets}

Generate a single follow-up search query that would help answer the original question more completely. Return ONLY the query text."""

        response = await self.llm.generate(prompt)
        followup = response.strip().strip('"').strip("'")
        return followup if followup else None

    def _rule_based_followup(
        self,
        original_query: str,
        results: list[RetrievalResult],
    ) -> Optional[str]:
        """Generate follow-up query using rule-based heuristics."""
        # Extract key nouns/terms from original query (words > 3 chars, not stopwords)
        stopwords = {"what", "when", "where", "which", "that", "this", "with", "from",
                      "have", "does", "about", "more", "than", "been", "were", "will"}
        words = re.findall(r'\b[a-zA-Z]{4,}\b', original_query.lower())
        key_terms = [w for w in words if w not in stopwords]

        if not key_terms:
            return None

        return f"more details about {' '.join(key_terms[:3])}"
