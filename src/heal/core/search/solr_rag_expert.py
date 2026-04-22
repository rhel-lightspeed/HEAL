"""RAG-based Solr Expert - uses Apache Solr documentation for grounded recommendations.

Instead of relying solely on LLM's training data, this expert retrieves relevant
sections from actual Apache Solr documentation and uses them to ground search
parameter recommendations.

Features:
- Retrieves from local Solr documentation corpus
- LLM recommendations grounded in official docs
- Similar architecture to URLValidationAgent (RAG-based)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


class SolrDocChunk(BaseModel):
    """A chunk of Solr documentation."""

    content: str
    source_file: str
    relevance_score: float = 0.0


@dataclass
class SolrRAGExpert:
    """RAG-based Solr Expert using official Apache Solr documentation.

    Retrieves relevant sections from Solr docs, then uses LLM to make
    grounded recommendations for search parameters.
    """

    solr_url: str = "http://localhost:8983/solr/portal"
    timeout: int = 30
    docs_path: Optional[Path] = None
    llm_client: Optional[Any] = None  # OpenAI client for RAG

    def __post_init__(self):
        """Initialize RAG system."""
        # Allow override via environment variable
        env_url = os.getenv("SOLR_URL")
        if env_url:
            self.solr_url = env_url.rstrip("/")

        # Set docs path
        if self.docs_path is None:
            # Default to HEAL project data directory
            # Path structure: src/heal/core/search/solr_rag_expert.py -> ../../../../../data/solr_docs
            heal_root = Path(__file__).parent.parent.parent.parent.parent
            self.docs_path = heal_root / "data" / "solr_docs"

        # Load Solr docs
        self.doc_chunks = self._load_solr_docs()
        logger.info(f"Loaded {len(self.doc_chunks)} Solr doc chunks")

        # No LLM needed - we apply doc-grounded rules programmatically
        self.llm_client = None

    def _load_solr_docs(self) -> list[SolrDocChunk]:
        """Load Solr documentation chunks from disk.

        Returns:
            List of documentation chunks
        """
        chunks = []

        if not self.docs_path.exists():
            logger.warning(f"Solr docs path not found: {self.docs_path}")
            return chunks

        # Load all markdown files
        for md_file in self.docs_path.glob("*.md"):
            content = md_file.read_text()

            # Split into sections by headers
            sections = self._split_into_sections(content)

            for section_title, section_content in sections:
                chunks.append(
                    SolrDocChunk(
                        content=f"# {section_title}\n\n{section_content}",
                        source_file=md_file.name,
                    )
                )

        return chunks

    def _split_into_sections(self, content: str) -> list[tuple[str, str]]:
        """Split markdown into sections by headers.

        Args:
            content: Markdown content

        Returns:
            List of (section_title, section_content) tuples
        """
        sections = []
        current_title = "Introduction"
        current_content = []

        for line in content.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_content:
                    sections.append((current_title, "\n".join(current_content)))

                # Start new section
                current_title = line.lstrip("# ").strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections.append((current_title, "\n".join(current_content)))

        return sections

    def _retrieve_relevant_docs(self, query: str, top_k: int = 3) -> list[SolrDocChunk]:
        """Retrieve relevant Solr documentation chunks.

        Simple keyword-based retrieval (no embeddings needed for small corpus).

        Args:
            query: User's search query
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant doc chunks, ranked by relevance
        """
        # Extract keywords from query
        query_lower = query.lower()
        keywords = set(query_lower.split())

        # Score each chunk by keyword overlap
        for chunk in self.doc_chunks:
            chunk_lower = chunk.content.lower()

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in chunk_lower)

            # Boost for specific parameter names
            param_boost = 0
            if "qf" in query_lower and "qf" in chunk_lower:
                param_boost += 5
            if "pf" in query_lower and "pf" in chunk_lower:
                param_boost += 5
            if "mm" in query_lower and "minimum match" in chunk_lower:
                param_boost += 5
            if "boost" in query_lower and "boost" in chunk_lower:
                param_boost += 3

            chunk.relevance_score = matches + param_boost

        # Sort by relevance and return top k
        ranked_chunks = sorted(self.doc_chunks, key=lambda c: c.relevance_score, reverse=True)

        return ranked_chunks[:top_k]

    async def search_for_verification(
        self,
        search_queries: list[VerificationQuery],
    ) -> VerificationResult:
        """Search Solr with RAG-enhanced parameter selection.

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

                # Build RAG-enhanced params
                params = self._build_rag_params(vq.query, num_results=10)

                # Query Solr
                docs = await self._query_solr(client, vq.query, params)
                all_docs.extend(docs)
                for doc in docs:
                    if "url" in doc:
                        all_urls.add(doc["url"])

        # Extract key facts from top documents
        key_facts = self._extract_key_facts(all_docs, search_queries)

        # Determine confidence based on retrieval success
        confidence = self._determine_confidence(all_docs, search_queries)

        reasoning = f"RAG-enhanced search found {len(all_docs)} documents across {len(search_queries)} verification queries."
        if all_docs:
            reasoning += f" Top result: {all_docs[0].get('title', 'N/A')}"

        return VerificationResult(
            found_docs=all_docs[:10],  # Return top 10
            key_facts=key_facts,
            confidence=confidence,
            source_urls=list(all_urls)[:5],  # Top 5 unique URLs
            reasoning=reasoning,
        )

    def _build_rag_params(
        self,
        query: str,
        num_results: int = 10,
    ) -> dict[str, Any]:
        """Build Solr params using doc-grounded heuristics.

        Applies recommendations from Apache Solr documentation programmatically.
        No LLM needed - rules are hard-coded based on Solr best practices.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            Dict of Solr params
        """
        import re

        # Start with aggressive defaults based on Solr edismax docs
        # Source: edismax.md - "qf" section recommends higher boosts for key fields
        params = {
            "q": query,
            "defType": "edismax",
            # More aggressive field weights than simple baseline
            # Source: Solr docs recommend title^20+ for high precision
            "qf": "title^10 main_content^4 heading_h1^5 heading_h2^3 portal_synopsis^5 allTitle^8 content^2 product^3",
            # Phrase boosting - multi-level strategy from edismax.md
            "pf": "title^15 main_content^8",  # Boost full phrases highly
            "ps": "2",  # Tighter slop for precision
            "pf2": "title^8 main_content^5",  # Bigram boost
            "ps2": "1",  # Adjacent bigrams only
            "pf3": "title^4 main_content^2",  # Trigram boost
            "ps3": "3",
            # Minimum match - from edismax.md: tighter mm improves precision
            "mm": "3<-1 5<75%",  # More strict than baseline (was 60%)
            "rows": num_results,
            "fl": "title,resourceName,main_content,documentKind,product,documentation_version",
            "wt": "json",
        }

        logger.debug("RAG Expert: Applying Solr doc-grounded optimizations")
        logger.debug(f"  Source: edismax.md - aggressive field weights for precision")
        logger.debug(f"  Source: edismax.md - multi-level phrase boosting (pf/pf2/pf3)")
        logger.debug(f"  Source: edismax.md - stricter mm for high precision")

        # RHEL version detection (from simple baseline, but with boost query)
        # Source: common_params.md - use bq for multiplicative boost
        rhel_version = self._detect_rhel_version(query)
        if rhel_version:
            # Higher boost than baseline (was ^4, now ^8)
            params["bq"] = f"documentation_version:*{rhel_version}*^8"
            logger.debug(f"  Source: common_params.md - boost query for RHEL {rhel_version}")

        # Topic-specific enhancements (more aggressive than baseline)
        query_lower = query.lower()

        # Bootloader topics - add field-specific boosts
        # Source: query_syntax.md - field-specific queries improve precision
        if any(kw in query_lower for kw in ["grub", "bootloader", "boot", "uefi"]):
            bq = params.get("bq", "")
            # More aggressive boost (was ^10, now ^15)
            new_boost = "documentKind:bootloader^15"
            params["bq"] = f"{bq} {new_boost}" if bq else new_boost
            logger.debug(f"  Source: query_syntax.md - bootloader boost applied")

        # Container topics
        elif any(kw in query_lower for kw in ["container", "podman", "docker"]):
            # Add to qf with high weight
            params["qf"] += " container_compatibility^15"
            logger.debug(f"  Source: query_syntax.md - container field boost applied")

        # Multi-word queries benefit from phrase fields
        # Source: edismax.md - pf2/pf3 improve multi-word query relevance
        word_count = len(query.split())
        if word_count >= 4:
            # Boost phrase fields even more for long queries
            params["pf"] = "title^20 main_content^10"
            logger.debug(
                f"  Source: edismax.md - enhanced phrase boost for {word_count}-word query"
            )

        return params

    def _detect_rhel_version(self, query: str) -> Optional[str]:
        """Detect RHEL version from query.

        Args:
            query: Search query

        Returns:
            Version string (e.g., "9", "8", "7") or None
        """
        import re

        # Match patterns like "RHEL 9", "rhel9", "Red Hat Enterprise Linux 8"
        patterns = [
            r"rhel\s*([6-9])",
            r"red\s+hat\s+enterprise\s+linux\s+([6-9])",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    async def _query_solr(
        self,
        client: httpx.AsyncClient,
        query: str,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Query Solr directly for documents.

        Args:
            client: HTTP client
            query: Search query
            params: Solr parameters (already built)

        Returns:
            List of document dictionaries with title, url, content
        """
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
            logger.error(f"Solr query failed: {e}")
            return []

    def _extract_key_facts(
        self,
        docs: list[dict[str, Any]],
        queries: list[VerificationQuery],
    ) -> list[str]:
        """Extract key facts from top documents.

        Args:
            docs: Retrieved documents
            queries: Original verification queries

        Returns:
            List of key facts as strings
        """
        key_facts = []

        # Get first 2-3 sentences from top 3 docs as key facts
        for doc in docs[:3]:
            content = doc.get("content", "")
            # Get first few sentences
            sentences = content.split(". ")[:2]
            if sentences:
                fact = ". ".join(sentences)
                if not fact.endswith("."):
                    fact += "."
                key_facts.append(f"{fact} (Source: {doc.get('title', 'N/A')})")

        return key_facts

    def _determine_confidence(
        self,
        docs: list[dict[str, Any]],
        queries: list[VerificationQuery],
    ) -> str:
        """Determine confidence based on retrieval success.

        Args:
            docs: Retrieved documents
            queries: Original verification queries

        Returns:
            Confidence level: HIGH, MEDIUM, or LOW
        """
        if not docs:
            return "LOW"

        # Good coverage: multiple docs per query
        if len(docs) >= len(queries) * 2:
            return "HIGH"
        # Moderate coverage: at least one doc per query
        elif len(docs) >= len(queries):
            return "MEDIUM"
        else:
            return "LOW"
