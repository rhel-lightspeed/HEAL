"""Content Relevance Agent - evaluates semantic relevance of retrieved docs.

This agent provides content-based scoring to complement URL-based metrics:
- Uses LLM to judge if retrieved content answers the query
- Provides relevance score even when URLs don't match expected ones
- Helps understand if search is retrieving semantically similar docs
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContentRelevanceAgent:
    """Content-based relevance evaluation agent.

    Evaluates retrieved documents for semantic relevance to query,
    independent of whether they match expected URLs.
    """

    def evaluate_relevance(
        self, query: str, retrieved_docs: List[Dict[str, Any]], top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate semantic relevance of retrieved documents.

        Args:
            query: Original user query
            retrieved_docs: Documents retrieved by search agent
            top_k: How many top docs to evaluate

        Returns:
            Dictionary with:
            - avg_relevance (float): Average relevance score (0.0-1.0)
            - doc_scores (List[float]): Per-document relevance scores
            - evaluation_method (str): How scores were computed
        """
        logger.info(
            f"ContentRelevanceAgent: Evaluating {min(top_k, len(retrieved_docs))} "
            f"docs for query '{query[:50]}...'"
        )

        # For now, use heuristic-based scoring (could be replaced with LLM)
        # Check if doc content contains key query terms

        if not retrieved_docs:
            return {
                "avg_relevance": 0.0,
                "doc_scores": [],
                "evaluation_method": "no_docs_retrieved",
            }

        doc_scores = []
        for doc in retrieved_docs[:top_k]:
            score = self._score_doc_heuristic(query, doc)
            doc_scores.append(score)

        avg_relevance = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0

        return {
            "avg_relevance": avg_relevance,
            "doc_scores": doc_scores,
            "evaluation_method": "keyword_overlap_heuristic",
        }

    def _score_doc_heuristic(self, query: str, doc: Dict[str, Any]) -> float:
        """
        Score document relevance using keyword overlap heuristic.

        Args:
            query: User query
            doc: Document dictionary with title, content, main_content

        Returns:
            Relevance score 0.0-1.0
        """
        # Extract query terms (simple tokenization)
        query_terms = set(query.lower().split())

        # Remove common stop words
        stop_words = {
            "how",
            "do",
            "i",
            "is",
            "the",
            "a",
            "an",
            "on",
            "in",
            "to",
            "for",
            "of",
            "and",
            "or",
        }
        query_terms = query_terms - stop_words

        if not query_terms:
            return 0.0

        # Get document text
        title = doc.get("title", "").lower()
        content = doc.get("content", "").lower()
        main_content = doc.get("main_content", "").lower()

        combined_text = f"{title} {content} {main_content}"

        # Count how many query terms appear in doc
        matches = sum(1 for term in query_terms if term in combined_text)

        # Normalize by number of query terms
        score = matches / len(query_terms)

        # Bonus if query terms appear in title
        title_matches = sum(1 for term in query_terms if term in title)
        if title_matches > 0:
            score = min(1.0, score + 0.2)  # Up to 20% bonus for title matches

        return score
