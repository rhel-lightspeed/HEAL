"""Simple Solr Agent - basic keyword search without advanced features.

This agent provides a baseline for comparison with okp-mcp and RAG approaches.
It uses simple keyword matching without query expansion, boosting, or RAG features.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SimpleSolrAgent:
    """Simple Solr agent with basic keyword search.

    This is a baseline retrieval approach for comparison purposes.
    Uses standard Solr query parser (not edismax) with minimal configuration.
    """

    solr_url: str = "http://localhost:8983/solr"
    collection: str = "portal"
    timeout: int = 30
    rows: int = 10  # Number of results to return

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute simple keyword search.

        Args:
            query: User query string

        Returns:
            List of document dictionaries from Solr
        """
        logger.info(f"SimpleSolrAgent: Searching for '{query}'")

        # Build simple Solr query
        solr_params = {
            "q": query,  # Simple keyword match
            "rows": self.rows,
            "fl": "id,title,content,main_content,score",  # Fields to return
            "wt": "json",
        }

        # Execute HTTP request
        full_url = f"{self.solr_url}/{self.collection}/select"

        try:
            response = httpx.get(full_url, params=solr_params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            docs = data.get("response", {}).get("docs", [])

            logger.info(f"SimpleSolrAgent: Retrieved {len(docs)} documents")
            return docs

        except httpx.HTTPError as e:
            logger.error(f"SimpleSolrAgent: HTTP error: {e}")
            return []
        except Exception as e:
            logger.error(f"SimpleSolrAgent: Error during search: {e}")
            return []
