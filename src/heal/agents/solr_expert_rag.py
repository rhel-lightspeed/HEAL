"""RAG-Enhanced Solr Expert Agent - improved retrieval with proven edismax config.

This is a drop-in replacement for SolrExpertAgent that uses RAG-style field boosting
proven to achieve 87.4% content relevance (vs 63.3% for basic keyword search).

Key improvements:
- edismax with optimized field weights (title^3.0, main_content^1.5, id^2.0)
- Phrase field boosting (pf) for better phrase matching
- Phrase slop (ps=2) for fuzzy matching
- Minimum match (mm=50%) to reduce false positives

Based on findings from RETRIEVAL_OPTIMIZATION_FINDINGS.md.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VerificationQuery(BaseModel):
    """Verification query for Solr Expert."""

    query: str
    context: str  # Why we're searching for this
    expected_doc_type: str  # solution|article|documentation


class VerificationResult(BaseModel):
    """Results from Solr Expert verification."""

    found_docs: list[dict[str, Any]]
    key_facts: list[str]
    confidence: str  # HIGH|MEDIUM|LOW
    source_urls: list[str]
    reasoning: str


@dataclass
class SolrExpertRAGAgent:
    """RAG-Enhanced Solr Expert Agent for better document retrieval.

    Same interface as SolrExpertAgent but with proven RAG-style parameters
    that achieve 87.4% content relevance.

    Uses direct Solr HTTP queries with:
    - edismax query parser
    - Optimized field boosting (title^3.0, main_content^1.5, id^2.0)
    - Phrase field boosting (pf)
    - Phrase slop (ps=2)
    - Minimum match (mm=50%)

    Features:
    - Logs all searches to search intelligence database
    - Shares knowledge with okp_mcp_agent for better diagnosis
    - Drop-in replacement for SolrExpertAgent
    """

    solr_url: str = "http://localhost:8983/solr/portal"
    timeout: int = 30
    search_intelligence_mgr: Optional[Any] = None  # SearchIntelligenceManager
    ticket_key: Optional[str] = None  # Current ticket being processed

    def __post_init__(self):
        """Validate Solr URL and initialize search intelligence."""
        # Allow override via environment variable
        env_url = os.getenv("SOLR_URL")
        if env_url:
            self.solr_url = env_url.rstrip("/")

        # Initialize search intelligence if not provided
        if self.search_intelligence_mgr is None:
            try:
                from heal.core.search_intelligence import (
                    SearchIntelligenceManager,
                )

                # Default location in project directory
                db_path = Path(".claude/search_intelligence")
                self.search_intelligence_mgr = SearchIntelligenceManager(db_path)
                logger.info(f"Initialized RAG search intelligence: {db_path}")
            except Exception as e:
                logger.warning(f"Could not initialize search intelligence: {e}")
                self.search_intelligence_mgr = None

    async def search_for_verification(
        self,
        search_queries: list[VerificationQuery],
    ) -> VerificationResult:
        """Search Solr for verification of Linux Expert's hypothesis.

        Uses RAG-enhanced query parameters proven to achieve 87.4% content relevance.

        Args:
            search_queries: List of verification queries to search

        Returns:
            VerificationResult with found docs, key facts, confidence
        """
        logger.info(f"RAG Solr Expert: Searching for {len(search_queries)} verification queries")

        all_docs = []
        all_urls = set()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Search for each query
            for vq in search_queries:
                logger.debug(f"  Query: {vq.query} (context: {vq.context})")
                docs = await self._query_solr_rag(client, vq.query, num_results=10)
                all_docs.extend(docs)
                for doc in docs:
                    if "url" in doc:
                        all_urls.add(doc["url"])

                # Log search intelligence
                if self.search_intelligence_mgr and self.ticket_key:
                    self._log_search_intelligence(vq, docs)

        # Extract key facts from top documents
        key_facts = self._extract_key_facts(all_docs, search_queries)

        # Determine confidence based on retrieval success
        confidence = self._determine_confidence(all_docs, search_queries)

        reasoning = (
            f"Found {len(all_docs)} documents across {len(search_queries)} verification queries "
            f"using RAG-enhanced edismax (87.4% content relevance)."
        )
        if all_docs:
            reasoning += f" Top result: {all_docs[0].get('title', 'N/A')}"

        return VerificationResult(
            found_docs=all_docs[:10],  # Return top 10
            key_facts=key_facts,
            confidence=confidence,
            source_urls=list(all_urls)[:5],  # Top 5 unique URLs
            reasoning=reasoning,
        )

    async def _query_solr_rag(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Query Solr with RAG-enhanced parameters.

        Uses proven configuration from RETRIEVAL_OPTIMIZATION_FINDINGS.md:
        - edismax with field boosting: title^3.0, main_content^1.5, id^2.0
        - Phrase boosting: pf for title^10.0, main_content^7.0
        - Phrase slop: ps=2 (allow 2 words between terms)
        - Minimum match: mm=50% (reduce false positives)

        Args:
            client: HTTP client
            query: Search query
            num_results: Number of results to return

        Returns:
            List of document dictionaries with title, url, content
        """
        # RAG-enhanced parameters (proven: 87.4% content relevance)
        params: Dict[str, Any] = {
            "q": query,
            "defType": "edismax",
            # Field weights (keyword matching)
            "qf": "title^3.0 content^1.0 main_content^1.5 id^2.0",
            # Phrase field weights (phrase matching - highest signal)
            "pf": "title^10.0 content^5.0 main_content^7.0",
            # Phrase slop (allow 2 words between terms)
            "ps": "2",
            # Minimum match (at least 50% of query terms)
            "mm": "50%",
            "rows": num_results,
            "fl": "title,resourceName,main_content,documentKind,product,documentation_version",
            "wt": "json",
        }

        try:
            response = await client.get(
                f"{self.solr_url}/select",
                params=params,
            )
            response.raise_for_status()

            data = response.json()
            docs = data.get("response", {}).get("docs", [])

            # Format for consistency
            formatted_docs = []
            base_url = "https://access.redhat.com"
            for doc in docs:
                # Extract scalar values from lists if needed
                title = doc.get("title", "Untitled")
                if isinstance(title, list):
                    title = title[0] if title else "Untitled"

                content = doc.get("main_content", "")
                if isinstance(content, list):
                    content = content[0] if content else ""

                # Build full URL from resourceName
                resource_name = doc.get("resourceName", "")
                url = f"{base_url}{resource_name}" if resource_name else ""

                formatted_docs.append(
                    {
                        "title": title,
                        "url": url,
                        "content": content,
                        "documentKind": doc.get("documentKind", "unknown"),
                    }
                )

            logger.debug(f"  Found {len(formatted_docs)} documents for query: {query}")
            return formatted_docs

        except httpx.HTTPError as e:
            logger.error(f"Solr HTTP error: {e}")
            return []
        except Exception as e:
            logger.error(f"Solr query error: {e}")
            return []

    def _extract_key_facts(
        self,
        docs: list[dict[str, Any]],
        queries: list[VerificationQuery],
    ) -> list[str]:
        """Extract key facts from retrieved documents.

        Args:
            docs: Retrieved documents
            queries: Original verification queries

        Returns:
            List of key facts extracted from docs
        """
        facts = []

        # Extract from top 3 documents
        for doc in docs[:3]:
            title = doc.get("title", "")
            content = doc.get("content", "")[:500]  # First 500 chars

            if title:
                facts.append(f"Document: {title}")

            # Extract key sentences from content
            if content:
                sentences = content.split(".")[:2]  # First 2 sentences
                for sentence in sentences:
                    cleaned = sentence.strip()
                    if len(cleaned) > 20:  # Meaningful sentences only
                        facts.append(cleaned)

        return facts[:5]  # Return top 5 facts

    def _determine_confidence(
        self,
        docs: list[dict[str, Any]],
        queries: list[VerificationQuery],
    ) -> str:
        """Determine confidence level based on retrieval success.

        Args:
            docs: Retrieved documents
            queries: Original verification queries

        Returns:
            Confidence level: HIGH|MEDIUM|LOW
        """
        if not docs:
            return "LOW"

        # High confidence: Found docs for most queries
        docs_per_query = len(docs) / len(queries)
        if docs_per_query >= 5:
            return "HIGH"
        elif docs_per_query >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _log_search_intelligence(
        self,
        query: VerificationQuery,
        docs: list[dict[str, Any]],
    ) -> None:
        """Log search to search intelligence database.

        Args:
            query: Verification query
            docs: Retrieved documents
        """
        if not self.search_intelligence_mgr or not self.ticket_key:
            return

        try:
            from heal.core.search_intelligence import SearchResult

            # Determine confidence based on results
            if len(docs) >= 5:
                confidence = "HIGH"
            elif len(docs) >= 2:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            # Create SearchResult object
            result = SearchResult.from_verification(
                query=query.query,
                topic=query.context,  # Use context as topic
                ticket_key=self.ticket_key,
                found_docs=docs,
                confidence=confidence,
            )

            # Log to database
            self.search_intelligence_mgr.log_search(result)

        except Exception as e:
            logger.warning(f"Failed to log search intelligence: {e}")
