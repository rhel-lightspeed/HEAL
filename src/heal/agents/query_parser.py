"""Query Parser - extracts key semantic elements from natural language queries.

Uses dependency parsing (similar to AST for code) to extract:
- Subject: What the query is about
- Verb: What action is being asked
- Object/Predicate: Additional context

This helps reformulate verbose queries like:
  "How do I recreate the GRUB configuration file on Red Hat Enterprise Linux?"
Into concise search terms:
  "grub configuration file recreate RHEL"
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Set, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Parsed query with extracted semantic components."""

    original: str
    technical_terms: List[str]  # GRUB, Secure Boot, systemd, etc.
    action_verbs: List[str]  # disable, configure, install, etc.
    core_concepts: List[str]  # bootloader, firewall, partition, etc.
    reformulated: str  # Concise search query


class QueryParser:
    """
    Parse queries to extract key semantic elements.

    Uses rule-based approach (no external dependencies needed).
    For production, could use spaCy dependency parsing for better accuracy.
    """

    # Common procedural phrases to remove
    STOP_PHRASES = {
        "how do i",
        "how to",
        "what is the",
        "what is a",
        "can i",
        "is it possible to",
        "how can i",
        "what's the way to",
        "how would i",
    }

    # Generic verbs that don't add search value
    GENERIC_VERBS = {
        "do",
        "be",
        "have",
        "get",
        "make",
        "use",
        "see",
        "know",
        "need",
    }

    # Technical action verbs worth keeping
    TECHNICAL_VERBS = {
        "install",
        "configure",
        "disable",
        "enable",
        "recreate",
        "rebuild",
        "update",
        "upgrade",
        "remove",
        "delete",
        "check",
        "verify",
        "troubleshoot",
        "diagnose",
        "fix",
        "repair",
        "mount",
        "unmount",
        "start",
        "stop",
        "restart",
        "reboot",
    }

    def __init__(self):
        """Initialize query parser."""
        pass

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse query to extract semantic components.

        Args:
            query: Natural language query

        Returns:
            ParsedQuery with extracted components
        """
        logger.info(f"Parsing query: {query[:100]}...")

        # Normalize
        query_lower = query.lower()

        # Extract components
        technical_terms = self._extract_technical_terms(query)
        action_verbs = self._extract_action_verbs(query_lower)
        core_concepts = self._extract_core_concepts(query_lower)

        # Reformulate
        reformulated = self._reformulate(query_lower, technical_terms, action_verbs, core_concepts)

        return ParsedQuery(
            original=query,
            technical_terms=technical_terms,
            action_verbs=action_verbs,
            core_concepts=core_concepts,
            reformulated=reformulated,
        )

    def _extract_technical_terms(self, query: str) -> List[str]:
        """
        Extract technical terms (capitalized phrases, acronyms, commands).

        Args:
            query: Query string

        Returns:
            List of technical terms
        """
        terms = []

        # 1. Capitalized multi-word phrases (Red Hat Enterprise Linux, Secure Boot)
        capitalized_phrase_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
        for match in re.finditer(capitalized_phrase_pattern, query):
            terms.append(match.group(0))

        # 2. All-caps acronyms (GRUB, RHEL, UEFI, VDO)
        acronym_pattern = r"\b[A-Z]{2,}\b"
        for match in re.finditer(acronym_pattern, query):
            acronym = match.group(0)
            # Filter out common words that happen to be all-caps
            if acronym not in {"I", "A", "IS", "IT"}:
                terms.append(acronym)

        # 3. Hyphenated technical terms (grub2-mkconfig, x86_64)
        hyphenated_pattern = r"\b[a-z0-9]+(?:[-_][a-z0-9]+)+\b"
        for match in re.finditer(hyphenated_pattern, query.lower()):
            terms.append(match.group(0))

        # 4. File paths and extensions
        file_pattern = r"(?:/[\w/-]+\.[\w]+|\.[\w]+|/etc/[\w/]+)"
        for match in re.finditer(file_pattern, query):
            terms.append(match.group(0))

        return list(set(terms))  # Remove duplicates

    def _extract_action_verbs(self, query_lower: str) -> List[str]:
        """
        Extract meaningful action verbs.

        Args:
            query_lower: Lowercased query

        Returns:
            List of action verbs
        """
        verbs = []

        # Find technical verbs
        words = re.findall(r"\b\w+\b", query_lower)
        for word in words:
            if word in self.TECHNICAL_VERBS:
                verbs.append(word)

        return verbs

    def _extract_core_concepts(self, query_lower: str) -> List[str]:
        """
        Extract core concepts (nouns/noun phrases).

        Args:
            query_lower: Lowercased query

        Returns:
            List of core concepts
        """
        # Common technical nouns in RHEL domain
        TECH_NOUNS = {
            "bootloader",
            "grub",
            "configuration",
            "file",
            "system",
            "boot",
            "partition",
            "disk",
            "firewall",
            "network",
            "service",
            "package",
            "kernel",
            "module",
            "firmware",
            "secure boot",
            "uefi",
            "bios",
            "virtualization",
            "container",
            "selinux",
        }

        concepts = []
        query_words = set(query_lower.split())

        # Check for known technical nouns
        for noun in TECH_NOUNS:
            if noun in query_lower:
                concepts.append(noun)

        return concepts

    def _reformulate(
        self,
        query_lower: str,
        technical_terms: List[str],
        action_verbs: List[str],
        core_concepts: List[str],
    ) -> str:
        """
        Reformulate query by combining key components.

        Args:
            query_lower: Lowercased original query
            technical_terms: Extracted technical terms
            action_verbs: Action verbs
            core_concepts: Core concepts

        Returns:
            Reformulated query string
        """
        # Start with original, remove stop phrases
        cleaned = query_lower
        for phrase in self.STOP_PHRASES:
            cleaned = cleaned.replace(phrase, "")

        # Build reformulated query from components
        components = []

        # 1. Technical terms (highest priority)
        components.extend(technical_terms)

        # 2. Core concepts (unless already in technical terms)
        for concept in core_concepts:
            if not any(concept.lower() in term.lower() for term in technical_terms):
                components.append(concept)

        # 3. Action verbs
        components.extend(action_verbs)

        # If we extracted good components, use them
        if components:
            reformulated = " ".join(components)
        else:
            # Fallback: just clean up the original
            # Remove question marks, extra spaces
            reformulated = cleaned.strip()
            reformulated = re.sub(r"[?!]", "", reformulated)
            reformulated = re.sub(r"\s+", " ", reformulated)

        return reformulated


# Example usage and testing
if __name__ == "__main__":
    parser = QueryParser()

    test_queries = [
        "How do I recreate the GRUB configuration file on Red Hat Enterprise Linux?",
        "How do I disable Secure Boot on a Red Hat Enterprise Linux system?",
        "How to check if Secure Boot is enabled on a running RHEL system",
        "What is the command to update the GRUB bootloader package in RHEL?",
        "How can I determine if my RHEL system is using BIOS or UEFI firmware?",
    ]

    print("Query Parsing Examples\n" + "=" * 80)

    for query in test_queries:
        result = parser.parse(query)
        print(f"\nOriginal: {result.original}")
        print(f"Technical Terms: {result.technical_terms}")
        print(f"Action Verbs: {result.action_verbs}")
        print(f"Core Concepts: {result.core_concepts}")
        print(f"Reformulated: {result.reformulated}")
        print("-" * 80)
