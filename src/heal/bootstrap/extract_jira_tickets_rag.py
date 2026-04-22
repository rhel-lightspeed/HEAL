#!/usr/bin/env python3
"""Extract JIRA tickets with RAG-enhanced Solr Expert for testing.

This is a variant of extract_jira_tickets.py that uses SolrExpertRAGAgent
instead of SolrExpertAgent to test if RAG-enhanced retrieval improves
the quality of expected answers.

Differences from standard extraction:
- Uses SolrExpertRAGAgent (87.4% content relevance vs 63.3% baseline)
- Outputs to extracted_tickets_rag.yaml for comparison
- Otherwise identical workflow

Usage:
    # Test RAG extraction on specific tickets
    python src/heal/bootstrap/extract_jira_tickets_rag.py --tickets RSPEED-2482,RSPEED-2511

    # Test on recent tickets
    python src/heal/bootstrap/extract_jira_tickets_rag.py --limit 5

Compare with original:
    diff config/extracted_tickets.yaml config/extracted_tickets_rag.yaml
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml

# Add src/ to sys.path for imports
SRC_ROOT = Path(__file__).parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from heal.core import (  # noqa: E402
    AnswerReviewAgent,
    LinuxExpertAgent,
    URLValidationAgent,
)

# Import RAG-enhanced Solr Expert
from heal.agents.solr_expert_rag import SolrExpertRAGAgent  # noqa: E402

# HEAL repository root for file paths
REPO_ROOT = SRC_ROOT.parent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default JQL query for RSPEED CLA incorrect-answer tickets
DEFAULT_JQL = (
    "project = RSPEED AND "
    'component = "command-line-assistant" AND '
    "resolution = Unresolved AND "
    'labels = "cla-incorrect-answer" '
    "ORDER BY created DESC"
)

# Output to different file for comparison
DEFAULT_OUTPUT = REPO_ROOT / "config" / "extracted_tickets_rag.yaml"


def get_jira_token() -> str:
    """Get JIRA API token from secret-tool."""
    result = subprocess.run(
        ["secret-tool", "lookup", "application", "jira"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def fetch_tickets_from_jira(jql: str, limit: int = 200) -> list[dict[str, Any]]:
    """Fetch tickets using JIRA REST API directly with pagination.

    Args:
        jql: JQL query
        limit: Maximum results

    Returns:
        List of ticket dictionaries with full details
    """
    token = get_jira_token()

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    all_issues: List[Dict[str, Any]] = []
    start_at = 0
    max_results = 100  # JIRA API limit per request

    logger.info(f"Fetching tickets with JQL: {jql}")

    while len(all_issues) < limit:
        # Use GET with query parameters for the new /search/jql endpoint
        params: Dict[str, Any] = {
            "jql": jql,
            "startAt": start_at,
            "maxResults": min(max_results, limit - len(all_issues)),
            "fields": ",".join(
                [
                    "summary",
                    "description",
                    "assignee",
                    "status",
                    "created",
                    "updated",
                    "labels",
                    "issuetype",
                ]
            ),
        }

        response = requests.get(
            "https://redhat.atlassian.net/rest/api/3/search/jql",
            headers=headers,
            auth=("emackey@redhat.com", token),
            params=params,
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f"Error fetching tickets: {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            break

        data = response.json()
        issues = data.get("issues", [])

        if not issues:
            break  # No more results

        # Add issues but don't exceed limit
        remaining = limit - len(all_issues)
        all_issues.extend(issues[:remaining])
        start_at += len(issues)

        logger.info(f"Fetched {len(all_issues)} tickets so far...")

        # Stop if we've reached the limit or no more pages
        if len(all_issues) >= limit or len(issues) < max_results:
            break

    logger.info(f"Total tickets fetched: {len(all_issues)}")
    return all_issues


def load_existing_yaml(path: Path) -> list[dict[str, Any]]:
    """Load existing extracted tickets from YAML.

    Args:
        path: Path to YAML file

    Returns:
        List of extracted ticket dictionaries
    """
    if not path.exists():
        logger.info(f"No existing YAML found at {path}")
        return []

    logger.info(f"Loading existing tickets from {path}")
    with open(path) as f:
        data = yaml.safe_load(f)

    if not data or "tickets" not in data:
        logger.warning("YAML file exists but has no 'tickets' key")
        return []

    tickets = data["tickets"]
    logger.info(f"Loaded {len(tickets)} existing tickets")
    return tickets


def save_yaml(tickets: list[dict[str, Any]], path: Path) -> None:
    """Save extracted tickets to YAML.

    Args:
        tickets: List of extracted ticket dictionaries
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "total_tickets": len(tickets),
            "extraction_method": "rag_enhanced_solr_expert",
            "note": "Uses RAG-enhanced edismax (87.4% content relevance)",
        },
        "tickets": tickets,
    }

    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved {len(tickets)} tickets to {path}")


async def extract_ticket(
    ticket: dict[str, Any],
    linux_expert: LinuxExpertAgent,
    solr_expert: SolrExpertRAGAgent,
    reviewer: AnswerReviewAgent,
    url_validator: URLValidationAgent,
    max_review_iterations: int = 3,
) -> dict[str, Any]:
    """Extract query/answer from a single JIRA ticket with autonomous quality review.

    Args:
        ticket: JIRA ticket dictionary
        linux_expert: Linux Expert Agent
        solr_expert: RAG-Enhanced Solr Expert Agent
        reviewer: Answer Review Agent for quality checks
        url_validator: URL Validation Agent for verifying retrieved docs
        max_review_iterations: Maximum refinement iterations (default: 3)

    Returns:
        Conversation as dict ready for YAML output
    """
    from dataclasses import asdict

    key = ticket.get("key", "UNKNOWN")

    # Set ticket key for search intelligence logging
    solr_expert.ticket_key = key

    # Extract with autonomous review + URL validation - returns Conversation object
    conversation = await linux_expert.extract_with_autonomous_review(
        ticket, solr_expert, reviewer, url_validator, max_iterations=max_review_iterations
    )

    # Convert to dict for YAML and filter None values
    def filter_none(d):
        """Recursively filter None values from dict."""
        if isinstance(d, dict):
            return {k: filter_none(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [filter_none(item) for item in d]
        return d

    return filter_none(asdict(conversation))


async def main():
    """Main extraction workflow with RAG-enhanced Solr Expert."""
    parser = argparse.ArgumentParser(
        description="Extract JIRA tickets with RAG-enhanced Solr Expert (testing)"
    )
    parser.add_argument(
        "--jql",
        type=str,
        default=DEFAULT_JQL,
        help=f"JQL query (default: {DEFAULT_JQL})",
    )
    parser.add_argument(
        "--tickets",
        type=str,
        help="Comma-separated ticket keys to process (e.g., RSPEED-2482,RSPEED-2511)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output YAML path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-extract all tickets (ignore existing YAML)",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Force re-extract tickets even if already processed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum tickets to fetch from JIRA (default: 200)",
    )
    parser.add_argument(
        "--max-review-iterations",
        type=int,
        default=3,
        help="Maximum autonomous review refinement iterations (default: 3)",
    )

    args = parser.parse_args()

    # Initialize agents - Use RAG-enhanced Solr Expert!
    logger.info("Initializing agents with RAG-enhanced Solr Expert...")
    solr_expert = SolrExpertRAGAgent()  # ← RAG agent instead of SolrExpertAgent
    linux_expert = LinuxExpertAgent()
    reviewer = AnswerReviewAgent()
    url_validator = URLValidationAgent()

    logger.info("✨ Using RAG-enhanced edismax (87.4% content relevance)")

    # Load existing tickets (unless force rebuild)
    existing_tickets = [] if args.force_rebuild else load_existing_yaml(args.output)
    existing_keys = {
        t.get("conversation_group_id") or t.get("ticket_key") for t in existing_tickets
    }

    # Fetch tickets
    if args.tickets:
        # Process specific tickets
        ticket_keys = [k.strip() for k in args.tickets.split(",")]
        logger.info(f"Processing specific tickets: {ticket_keys}")

        jira_tickets = []
        for key in ticket_keys:
            # Fetch individual ticket
            tickets = fetch_tickets_from_jira(f"key = {key}", limit=1)
            if tickets:
                jira_tickets.extend(tickets)
            else:
                logger.warning(f"Ticket not found: {key}")
    else:
        # Fetch via JQL
        jira_tickets = fetch_tickets_from_jira(args.jql, limit=args.limit)

    # Filter to new tickets (unless force re-extract)
    if args.force_reextract:
        tickets_to_process = jira_tickets
        logger.info(f"Force re-extract: Processing {len(tickets_to_process)} tickets")
    else:
        tickets_to_process = [t for t in jira_tickets if t["key"] not in existing_keys]
        skipped = len(jira_tickets) - len(tickets_to_process)
        logger.info(f"Found {len(tickets_to_process)} new tickets ({skipped} already extracted)")

    if not tickets_to_process:
        logger.info("No new tickets to process!")
        return

    # Process tickets
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {len(tickets_to_process)} tickets with RAG-enhanced retrieval")
    logger.info(f"{'='*80}\n")

    # Start with existing tickets
    all_tickets = existing_tickets.copy()

    # Remove old versions if force re-extract
    if args.force_reextract:
        tickets_to_remove = {t["key"] for t in tickets_to_process}
        all_tickets = [
            t for t in all_tickets if t.get("conversation_group_id") not in tickets_to_remove
        ]

    newly_extracted_count = 0
    skipped_jailbreak = 0

    for i, ticket in enumerate(tickets_to_process, 1):
        logger.info(f"\n[{i}/{len(tickets_to_process)}] Processing {ticket['key']}")

        try:
            # Quick jailbreak check - skip tickets that are clearly out of scope
            # This is a simplified filter for the RAG testing demo
            summary = ticket.get("fields", {}).get("summary", "")
            if isinstance(summary, str):
                summary = summary.lower()
            else:
                summary = ""

            description = ticket.get("fields", {}).get("description", "")
            # Description can be ADF (dict) or string
            if isinstance(description, dict):
                # Extract text from ADF format if possible
                description = str(description).lower()
            elif isinstance(description, str):
                description = description.lower()
            else:
                description = ""

            jailbreak_patterns = [
                "ignore instructions",
                "ignore previous",
                "disregard",
                "override",
                "system prompt",
                "you are now",
                "pretend you are",
                "act as",
                "roleplay",
                "<|start_of_role|>",  # Prompt injection token
                "windows server",  # Non-RHEL OS
            ]

            is_jailbreak = any(
                pattern in summary or pattern in description for pattern in jailbreak_patterns
            )

            if is_jailbreak:
                logger.warning(f"  ⚠️  Skipping {ticket['key']}: Detected jailbreak attempt")
                skipped_jailbreak += 1
                continue

            extracted = await extract_ticket(
                ticket,
                linux_expert,
                solr_expert,
                reviewer,
                url_validator,
                max_review_iterations=args.max_review_iterations,
            )

            # Check if extraction marked as out-of-scope after processing
            metadata = extracted.get("metadata", {})
            if not metadata.get("in_scope", True):
                logger.warning(f"  ⚠️  Skipping {ticket['key']}: Out of scope after extraction")
                skipped_jailbreak += 1
                continue

            # Append and save immediately (incremental progress saving)
            all_tickets.append(extracted)
            save_yaml(all_tickets, args.output)
            newly_extracted_count += 1

            logger.info(f"  ✅ Extracted: {extracted['conversation_group_id']}")
            logger.info(f"  💾 Saved to {args.output} ({len(all_tickets)} total tickets)")

        except Exception as e:
            logger.error(f"  ❌ Failed to extract {ticket['key']}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            continue

    # Show search intelligence stats
    if solr_expert.search_intelligence_mgr:
        logger.info(f"\n{'='*80}")
        logger.info("SEARCH INTELLIGENCE STATS (RAG-enhanced)")
        logger.info(f"{'='*80}")
        stats = solr_expert.search_intelligence_mgr.get_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("EXTRACTION COMPLETE (RAG-enhanced)")
    logger.info(f"{'='*80}")
    logger.info(f"Total tickets in YAML: {len(all_tickets)}")
    logger.info(f"Newly extracted: {newly_extracted_count}")
    if skipped_jailbreak > 0:
        logger.info(f"Skipped (jailbreak/out-of-scope): {skipped_jailbreak}")
    logger.info(f"Output: {args.output}")
    logger.info(f"\nCompare with baseline:")
    logger.info(f"  diff config/extracted_tickets.yaml {args.output}")


if __name__ == "__main__":
    import os
    import sys

    exit_code = 0
    interrupted = False

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        interrupted = True
        exit_code = 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        # Aggressive cleanup: Force kill any remaining Claude SDK subprocess tasks
        # This is necessary because Claude SDK spawns background tasks that don't
        # get properly cleaned up when the query finishes
        #
        # IMPORTANT: We use os._exit() instead of sys.exit() because sys.exit()
        # waits for background threads to finish, which causes hangs with Claude SDK
        if not interrupted:
            # On normal completion, try gentle cleanup first
            try:
                loop = asyncio.get_event_loop()
                if loop and not loop.is_closed():
                    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                    if pending:
                        for task in pending:
                            task.cancel()
                        try:
                            loop.run_until_complete(asyncio.wait(pending, timeout=0.5))
                        except Exception:
                            pass
            except Exception:
                pass

        # Force exit to kill any lingering subprocess threads
        # This works for both normal completion AND Ctrl+C interruption
        os._exit(exit_code)
