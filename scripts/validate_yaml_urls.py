#!/usr/bin/env python3
"""Validate and update expected_urls in pattern YAMLs without full re-extraction.

This script validates that expected_urls in existing pattern YAML files actually
answer the questions. It uses URLValidationAgent to check relevance and can
optionally update the YAMLs with better URLs from SolrExpert.

Usage:
    # Validate all patterns (read-only, report issues)
    python scripts/validate_yaml_urls.py

    # Validate specific pattern
    python scripts/validate_yaml_urls.py --pattern BOOTLOADER_GRUB_ISSUES

    # Validate and auto-fix (search for better URLs if validation fails)
    python scripts/validate_yaml_urls.py --pattern BOOTLOADER_GRUB_ISSUES --auto-fix

    # Dry-run mode (show what would be changed)
    python scripts/validate_yaml_urls.py --pattern BOOTLOADER_GRUB_ISSUES --auto-fix --dry-run
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add src/ to sys.path
SRC_ROOT = Path(__file__).parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from heal.core import SolrExpertAgent, URLValidationAgent  # noqa: E402
from heal.core.solr_expert import VerificationQuery  # noqa: E402

REPO_ROOT = SRC_ROOT
PATTERNS_DIR = REPO_ROOT / "config" / "patterns"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def validate_ticket_urls(
    ticket: Dict[str, Any],
    url_validator: URLValidationAgent,
    solr_expert: SolrExpertAgent,
    auto_fix: bool = False,
) -> Dict[str, Any]:
    """Validate URLs for a single ticket by searching Solr.

    Strategy:
    1. Search Solr with the query to get REAL docs
    2. Validate those docs match the query
    3. If validation passes, use those URLs
    4. If validation fails and auto_fix=True, retry with suggested queries

    Args:
        ticket: Ticket dict from pattern YAML
        url_validator: URL validation agent
        solr_expert: Solr expert agent
        auto_fix: If True, retry search with better queries if validation fails

    Returns:
        Dict with validation results and optionally updated URLs
    """
    ticket_id = ticket.get("conversation_group_id", "UNKNOWN")
    turns = ticket.get("turns", [])

    if not turns:
        return {
            "ticket_id": ticket_id,
            "status": "skipped",
            "reason": "No turns",
        }

    first_turn = turns[0]
    query = first_turn.get("query", "")
    expected_response = first_turn.get("expected_response", "")
    current_urls = first_turn.get("expected_urls", [])

    if not query:
        return {
            "ticket_id": ticket_id,
            "status": "skipped",
            "reason": "No query",
        }

    logger.info(f"\n[{ticket_id}] Searching Solr and validating URLs...")
    logger.info(f"  Query: {query[:80]}...")
    if current_urls:
        logger.info(f"  Current URLs: {len(current_urls)}")

    # Search Solr with the query to get REAL docs
    solr_expert.ticket_key = ticket_id

    verification_queries = [
        VerificationQuery(
            query=query,
            context="URL validation via Solr search",
            expected_doc_type="documentation"
        )
    ]

    verification = await solr_expert.search_for_verification(verification_queries)

    if not verification.found_docs:
        logger.warning(f"  ⚠️  Solr search returned no documents")
        return {
            "ticket_id": ticket_id,
            "status": "no_docs",
            "reason": "Solr search found no documents",
            "expected_urls": current_urls,
        }

    logger.info(f"  Found {len(verification.found_docs)} docs from Solr")

    # Validate the retrieved docs
    validation = await url_validator.validate_urls(
        query=query,
        hypothesis=expected_response[:300] if expected_response else query,
        retrieved_docs=verification.found_docs,
    )

    logger.info(f"  Validation score: {validation.score:.2f}")
    logger.info(f"  Validation passes: {validation.passes}")

    # Check if Solr URLs are better than current URLs
    solr_urls = verification.source_urls or []

    if validation.passes:
        # Validation passed - use Solr URLs
        if set(solr_urls) != set(current_urls):
            logger.info(f"  ✅ Validation passed, URLs differ from current")
            return {
                "ticket_id": ticket_id,
                "status": "updated",
                "score": validation.score,
                "old_urls": current_urls,
                "new_urls": solr_urls,
                "changed": True,
            }
        else:
            logger.info(f"  ✅ Validation passed, URLs unchanged")
            return {
                "ticket_id": ticket_id,
                "status": "passed",
                "score": validation.score,
                "expected_urls": current_urls,
                "changed": False,
            }

    # Validation failed
    logger.warning(f"  ❌ Validation failed:")
    for issue in validation.issues:
        logger.warning(f"     - {issue}")

    if not auto_fix:
        return {
            "ticket_id": ticket_id,
            "status": "failed",
            "score": validation.score,
            "issues": validation.issues,
            "expected_urls": current_urls,
            "changed": False,
        }

    # Auto-fix: Retry with suggested queries
    logger.info(f"  🔧 Retrying search with suggested queries...")

    if not validation.suggested_search_queries:
        logger.warning(f"  ⚠️  No suggested queries available")
        return {
            "ticket_id": ticket_id,
            "status": "unfixable",
            "score": validation.score,
            "issues": validation.issues,
            "expected_urls": current_urls,
            "changed": False,
        }

    # Search with suggested queries
    refinement_queries = [
        VerificationQuery(
            query=sq,
            context="URL validation auto-fix retry",
            expected_doc_type="documentation"
        )
        for sq in validation.suggested_search_queries[:3]
    ]

    retry_verification = await solr_expert.search_for_verification(refinement_queries)

    if not retry_verification.found_docs:
        logger.warning(f"  ⚠️  Retry search found no documents")
        return {
            "ticket_id": ticket_id,
            "status": "unfixable",
            "score": validation.score,
            "issues": validation.issues,
            "expected_urls": current_urls,
            "changed": False,
        }

    # Validate retry results
    retry_validation = await url_validator.validate_urls(
        query=query,
        hypothesis=expected_response[:300] if expected_response else query,
        retrieved_docs=retry_verification.found_docs,
    )

    logger.info(f"  Retry validation score: {retry_validation.score:.2f}")

    if retry_validation.passes:
        logger.info(f"  ✅ Retry successful! Found better URLs")
        return {
            "ticket_id": ticket_id,
            "status": "fixed",
            "score": retry_validation.score,
            "old_urls": current_urls,
            "new_urls": retry_verification.source_urls or [],
            "changed": True,
        }
    else:
        logger.warning(f"  ⚠️  Retry validation also failed")
        return {
            "ticket_id": ticket_id,
            "status": "unfixable",
            "score": validation.score,
            "issues": validation.issues,
            "expected_urls": current_urls,
            "changed": False,
        }


async def validate_pattern(
    pattern_file: Path,
    url_validator: URLValidationAgent,
    solr_expert: SolrExpertAgent,
    auto_fix: bool = False,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Validate all tickets in a pattern YAML.

    Args:
        pattern_file: Path to pattern YAML file
        url_validator: URL validation agent
        solr_expert: Solr expert agent
        auto_fix: If True, search for better URLs
        dry_run: If True, don't save changes

    Returns:
        List of validation results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Validating: {pattern_file.name}")
    logger.info(f"{'='*80}")

    with open(pattern_file) as f:
        tickets = yaml.safe_load(f)

    if not tickets:
        logger.warning("No tickets found in YAML")
        return []

    results = []
    updates_needed = False

    for ticket in tickets:
        result = await validate_ticket_urls(
            ticket, url_validator, solr_expert, auto_fix
        )
        results.append(result)

        # Update ticket if URLs changed (status: "updated" or "fixed")
        if result.get("changed") and result.get("new_urls"):
            if dry_run:
                logger.info(f"  [DRY RUN] Would update {result['ticket_id']}")
                logger.info(f"    Old: {result.get('old_urls', [])}")
                logger.info(f"    New: {result['new_urls']}")
            else:
                ticket["turns"][0]["expected_urls"] = result["new_urls"]
                updates_needed = True
                logger.info(f"  📝 Updated {result['ticket_id']} URLs in memory")

    # Save updated YAML
    if updates_needed:
        backup_file = pattern_file.with_suffix(".yaml.bak")
        logger.info(f"\n📝 Saving backup to: {backup_file}")
        with open(backup_file, "w") as f:
            yaml.dump(tickets, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"📝 Saving updates to: {pattern_file}")
        with open(pattern_file, "w") as f:
            yaml.dump(tickets, f, default_flow_style=False, allow_unicode=True)

    # Print summary
    print_validation_summary(results, pattern_file.stem, dry_run)

    return results


def print_validation_summary(
    results: List[Dict[str, Any]], pattern_id: str, dry_run: bool
):
    """Print validation summary."""
    passed = sum(1 for r in results if r["status"] == "passed")
    updated = sum(1 for r in results if r["status"] == "updated")
    failed = sum(1 for r in results if r["status"] == "failed")
    fixed = sum(1 for r in results if r["status"] == "fixed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    unfixable = sum(1 for r in results if r["status"] == "unfixable")
    no_docs = sum(1 for r in results if r["status"] == "no_docs")

    print(f"\n{'='*80}")
    print(f"Validation Summary: {pattern_id}")
    print(f"{'='*80}")
    print(f"Total tickets: {len(results)}")
    print(f"  ✅ Passed (unchanged): {passed}")
    print(f"  📝 Updated: {updated}" + (" (would update)" if dry_run else ""))
    print(f"  🔧 Fixed (retry succeeded): {fixed}" + (" (would fix)" if dry_run else ""))
    print(f"  ❌ Failed: {failed}")
    print(f"  ⚠️  Unfixable: {unfixable}")
    print(f"  ⚠️  No docs found: {no_docs}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"{'='*80}\n")

    # Show changes
    changes = [r for r in results if r["status"] in ["updated", "fixed"]]
    if changes:
        print("URL Changes:")
        for result in changes:
            print(f"\n  {result['ticket_id']}: {result['status']}")
            print(f"    Score: {result['score']:.2f}")
            old_urls = result.get("old_urls", [])
            new_urls = result.get("new_urls", [])
            print(f"    Old URLs ({len(old_urls)}): {old_urls[:2]}{'...' if len(old_urls) > 2 else ''}")
            print(f"    New URLs ({len(new_urls)}): {new_urls[:2]}{'...' if len(new_urls) > 2 else ''}")

    # Show failures
    failures = [r for r in results if r["status"] in ["failed", "unfixable"]]
    if failures:
        print("\nIssues found:")
        for result in failures:
            print(f"\n  {result['ticket_id']}: {result['status']}")
            print(f"    Score: {result['score']:.2f}")
            for issue in result.get("issues", [])[:3]:
                print(f"    - {issue}")


async def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate expected_urls in pattern YAMLs"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Specific pattern to validate (e.g., BOOTLOADER_GRUB_ISSUES)",
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically search for better URLs if validation fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without saving",
    )

    args = parser.parse_args()

    # Initialize agents
    logger.info("Initializing agents...")
    url_validator = URLValidationAgent()
    solr_expert = SolrExpertAgent()

    # Find patterns to validate
    if args.pattern:
        pattern_file = PATTERNS_DIR / f"{args.pattern}.yaml"
        if not pattern_file.exists():
            logger.error(f"Pattern file not found: {pattern_file}")
            sys.exit(1)
        patterns = [pattern_file]
    else:
        patterns = list(PATTERNS_DIR.glob("*.yaml"))

    if not patterns:
        logger.error(f"No patterns found in {PATTERNS_DIR}")
        sys.exit(1)

    # Validate each pattern
    all_results = []
    for pattern_file in patterns:
        results = await validate_pattern(
            pattern_file,
            url_validator,
            solr_expert,
            auto_fix=args.auto_fix,
            dry_run=args.dry_run,
        )
        all_results.extend(results)

    # Overall summary if validating multiple patterns
    if len(patterns) > 1:
        print(f"\n{'='*80}")
        print("Overall Summary")
        print(f"{'='*80}")
        print(f"Patterns validated: {len(patterns)}")
        print(f"Total tickets: {len(all_results)}")
        print(f"  ✅ Passed: {sum(1 for r in all_results if r['status'] == 'passed')}")
        print(f"  📝 Updated: {sum(1 for r in all_results if r['status'] == 'updated')}")
        print(f"  🔧 Fixed: {sum(1 for r in all_results if r['status'] == 'fixed')}")
        print(f"  ❌ Failed: {sum(1 for r in all_results if r['status'] == 'failed')}")
        print(f"  ⚠️  Unfixable: {sum(1 for r in all_results if r['status'] == 'unfixable')}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
