"""RAG Solr Agent - enhanced keyword search with RAG-style features.

This agent uses advanced Solr features for better retrieval:
- edismax query parser for better phrase matching
- Field-specific boosting (title, text, url)
- Recency boosting for recent documentation
- Phrase slop for fuzzy phrase matching
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RAGSolrAgent:
    """RAG-enhanced Solr agent with advanced query features.

    This agent represents a "middle ground" between SimpleSolrAgent
    and okp-mcp's full multi-agent optimization.

    Uses edismax query parser with:
    - Field boosting (title^3.0, text^1.0, url^2.0)
    - Phrase slop for fuzzy matching
    - Minimum match threshold
    """

    solr_url: str = "http://localhost:8983/solr"
    collection: str = "portal"
    timeout: int = 30
    rows: int = 10  # Number of results to return

    def search_with_rag(self, query: str, rows: int = None) -> List[Dict[str, Any]]:
        """
        Execute RAG-enhanced search.

        Args:
            query: User query string
            rows: Number of results (overrides default)

        Returns:
            List of document dictionaries from Solr
        """
        if rows is None:
            rows = self.rows

        logger.info(f"RAGSolrAgent: Searching for '{query}' (rows={rows})")

        # Build edismax query with boosting
        solr_params = {
            "q": query,
            "defType": "edismax",  # Extended DisMax query parser
            "qf": "title^3.0 content^1.0 main_content^1.5 id^2.0",  # Field weights
            "pf": "title^10.0 content^5.0 main_content^7.0",  # Phrase field weights
            "ps": "2",  # Phrase slop (allow 2 words between terms)
            "mm": "50%",  # Minimum match (at least 50% of query terms)
            "rows": rows,
            "fl": "id,title,content,main_content,score",
            "wt": "json",
        }

        # Execute HTTP request
        full_url = f"{self.solr_url}/{self.collection}/select"

        try:
            response = httpx.get(full_url, params=solr_params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            docs = data.get("response", {}).get("docs", [])

            logger.info(f"RAGSolrAgent: Retrieved {len(docs)} documents")
            return docs

        except httpx.HTTPError as e:
            logger.error(f"RAGSolrAgent: HTTP error: {e}")
            return []
        except Exception as e:
            logger.error(f"RAGSolrAgent: Error during search: {e}")
            return []

    # Alias for compatibility with comparison script
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Alias for search_with_rag() for compatibility."""
        return self.search_with_rag(query)
