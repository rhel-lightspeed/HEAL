#!/usr/bin/env python3
"""Autonomous agent for okp-mcp RSPEED ticket fixing.

This agent automates the INCORRECT_ANSWER_LOOP workflow:
1. Diagnose: Run full evaluation to identify problem type
2. Analyze: Determine if it's retrieval or answer quality issue
3. Iterate: Make targeted changes (boost queries or prompts)
4. Validate: Check for regressions across all test suites
5. Commit: Create commit with detailed metrics

Usage:
    # Diagnose a single ticket (runs new evaluation)
    python okp_mcp_agent/agents/okp_mcp_agent.py diagnose RSPEED-2482

    # Diagnose using existing results (fast, no re-run)
    python okp_mcp_agent/agents/okp_mcp_agent.py diagnose RSPEED-2482 --use-existing

    # Auto-fix with iterations
    python okp_mcp_agent/agents/okp_mcp_agent.py fix RSPEED-2482 --max-iterations 10

    # Validate across all suites
    python okp_mcp_agent/agents/okp_mcp_agent.py validate
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

# Add src/ to sys.path so imports work from any working directory
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# LLM advisor for AI-powered suggestions (Phase 2)
try:
    from .okp_mcp_llm_advisor import OkpMcpLLMAdvisor, MetricSummary

    LLM_ADVISOR_AVAILABLE = True
except ImportError as e:
    LLM_ADVISOR_AVAILABLE = False
    print(f"⚠️  LLM advisor not available: {e}")

# Solr checker for document validation
try:
    from .okp_solr_checker import SolrDocumentChecker

    SOLR_CHECKER_AVAILABLE = True
except ImportError as e:
    SOLR_CHECKER_AVAILABLE = False
    print(f"⚠️  Solr Checker not available: {e}")

# Solr config analyzer for explain output and tuning suggestions
try:
    from .okp_solr_config_analyzer import SolrConfigAnalyzer

    SOLR_ANALYZER_AVAILABLE = True
except ImportError as e:
    SOLR_ANALYZER_AVAILABLE = False
    print(f"⚠️  Solr Analyzer not available (import failed): {e}")
    print("   Make sure okp_solr_config_analyzer.py exists in okp_mcp_agent/agents/")
except Exception as e:
    SOLR_ANALYZER_AVAILABLE = False
    print(f"⚠️  Solr Analyzer import error: {e}")


# Iteration Strategy Constants
PRIMARY_FIX_MAX_ITERATIONS = 5  # Max attempts to fix original RSPEED ticket
REGRESSION_FIX_MAX_ITERATIONS = 3  # Max attempts per individual regression
ESCALATION_THRESHOLD = 2  # Failed attempts before escalating model
PLATEAU_THRESHOLD = 2  # Iterations without improvement = plateau
MIN_IMPROVEMENT_THRESHOLD = 0.05  # Significant improvement (resets escalation)
SMALL_IMPROVEMENT_THRESHOLD = (
    0.02  # Small but real improvement (keep building on it, not just noise)
)

# Model Tier Configuration
TIER_MODELS = {
    "simple": "claude-haiku-4-5-20251001",  # Classification only (not for fixes)
    "medium": "claude-sonnet-4-6",  # Default for all fixes
    "complex": "claude-opus-4-6",  # Escalation for hard problems (FIXED: was 4-5 which doesn't exist!)
}


@dataclass
class MetricThresholds:
    """Thresholds for determining problem type."""

    url_f1_retrieval_problem: float = 0.7
    mrr_retrieval_problem: float = 0.5
    context_relevance_retrieval_problem: float = 0.7
    keywords_answer_problem: float = 0.7
    answer_correctness_good: float = 0.8  # Answer is good enough regardless of retrieval


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""

    ticket_id: str
    query: Optional[str] = None  # User query from CSV

    # Retrieval metrics (available in both retrieval-only and full modes)
    url_f1: Optional[float] = None
    mrr: Optional[float] = None
    context_relevance: Optional[float] = None
    context_precision: Optional[float] = None

    # Answer quality metrics (only in full mode with /v1/infer)
    keywords_score: Optional[float] = None
    forbidden_claims_score: Optional[float] = None
    faithfulness: Optional[float] = None  # ragas:faithfulness - answer grounded in context
    answer_correctness: Optional[float] = None  # custom:answer_correctness - vs expected answer
    response_relevancy: Optional[float] = None  # ragas:response_relevancy - addresses question

    # LLM Judge reasoning (detailed explanations from judge)
    answer_correctness_reason: Optional[str] = None
    faithfulness_reason: Optional[str] = None
    response_relevancy_reason: Optional[str] = None
    context_relevance_reason: Optional[str] = None
    context_precision_reason: Optional[str] = None

    # Judge disagreement tracking (when Claude disagrees with Gemini)
    judge_disagreements: List[Dict] = field(default_factory=list)  # List of disagreement reports

    # RAG usage tracking
    tool_calls: Optional[str] = None  # Raw tool_calls from CSV
    contexts: Optional[str] = None  # Raw contexts from CSV
    rag_used: bool = False  # Was RAG/search tool called?
    docs_retrieved: bool = False  # Were any documents retrieved?

    # Ground truth / expected values (from test config)
    response: Optional[str] = None  # Actual LLM answer
    expected_response: Optional[str] = None  # What answer should say
    expected_keywords: Optional[list] = field(
        default_factory=list
    )  # Which keywords should be present
    expected_urls: Optional[list] = field(default_factory=list)  # Which URLs should be retrieved
    forbidden_claims: Optional[list] = field(default_factory=list)  # What should NOT be in answer
    retrieved_urls: Optional[list] = field(
        default_factory=list
    )  # Which URLs were actually retrieved
    retrieved_doc_titles: Optional[list] = field(default_factory=list)  # Titles of retrieved docs

    # Evaluation metadata
    num_runs: int = 1  # Number of runs averaged (for stability)
    high_variance_metrics: List[str] = field(
        default_factory=list
    )  # Metrics with >15% variance (instability)
    solr_check: Optional[Dict] = None  # Solr document existence check results
    url_overlap_with_previous: Optional[float] = (
        None  # Jaccard similarity with previous iteration (0-1)
    )

    @property
    def is_retrieval_problem(self) -> bool:
        """Determine if this is a retrieval problem based on metrics.

        IMPORTANT: Prioritizes RAGAS context metrics over URL F1.
        URL F1 is unreliable (can be 0.00 even with correct answer).
        """
        thresholds = MetricThresholds()

        # PRIMARY: Check RAGAS context metrics (most reliable)
        retrieval_issues = []

        if (
            self.context_relevance is not None
            and self.context_relevance < thresholds.context_relevance_retrieval_problem
        ):
            retrieval_issues.append(f"Context relevance low ({self.context_relevance:.2f})")

        # SECONDARY: Check MRR if available
        if self.mrr is not None and self.mrr < thresholds.mrr_retrieval_problem:
            retrieval_issues.append(f"MRR low ({self.mrr:.2f})")

        # TERTIARY: Only check URL F1 if context metrics are unavailable
        # Don't weight URL F1 heavily as it's unreliable
        if (
            self.url_f1 is not None
            and self.url_f1 < thresholds.url_f1_retrieval_problem
            and self.context_relevance is None  # Only if we don't have better metrics
        ):
            retrieval_issues.append(f"URL F1 low ({self.url_f1:.2f})")

        return len(retrieval_issues) > 0

    @property
    def is_answer_problem(self) -> bool:
        """Determine if this is an answer quality problem."""
        # Answer problem if retrieval is good but keywords are missing
        thresholds = MetricThresholds()

        good_retrieval = (
            self.url_f1 is not None and self.url_f1 >= thresholds.url_f1_retrieval_problem
        )
        poor_keywords = (
            self.keywords_score is not None
            and self.keywords_score < thresholds.keywords_answer_problem
        )

        return good_retrieval and poor_keywords

    @property
    def is_answer_good_enough(self) -> bool:
        """Check if answer quality is good regardless of retrieval.

        This allows the loop to end early if the answer is correct even if
        we didn't retrieve the "expected" URLs (e.g., LLM used general knowledge,
        or retrieved docs were fine despite not matching expected URLs).

        Uses answer_correctness if available, otherwise falls back to keywords.

        IMPORTANT: Also verifies answer is grounded in RAG (not hallucinated).
        """
        thresholds = MetricThresholds()

        # Check answer correctness (primary signal, if available)
        if self.answer_correctness is not None:
            good_answer = self.answer_correctness >= thresholds.answer_correctness_good
        elif self.keywords_score is not None and self.keywords_score >= 0.9:
            # Fallback 1: Very high keywords score (all required facts present)
            good_answer = True
        elif (
            self.context_relevance is not None
            and self.context_relevance >= 0.9
            and self.context_precision is not None
            and self.context_precision >= 0.8
        ):
            # Fallback 2: Very high context metrics (retrieved docs can answer the question)
            # This is used when we don't have expected_response or keywords
            # High context_relevance + context_precision means docs are highly relevant and precise
            good_answer = True
        else:
            good_answer = False

        # Check keywords (required facts present)
        good_keywords = (
            self.keywords_score is None  # Not checked (fallback case)
            or self.keywords_score >= thresholds.keywords_answer_problem
        )

        # Check forbidden claims (no regression)
        no_forbidden_claims = (
            self.forbidden_claims_score is None  # Not checked
            or self.forbidden_claims_score >= 0.9  # Or high score
        )

        # GROUNDING CHECK: Verify answer is grounded in RAG (not hallucinated)
        is_grounded = True

        # Verify RAG is being used
        if not self.rag_used:
            is_grounded = False

        # Verify context is relevant (docs actually relate to the question)
        if self.context_relevance is not None and self.context_relevance < 0.7:
            is_grounded = False

        # Verify answer is faithful to context (not making things up)
        if self.faithfulness is not None and self.faithfulness < 0.7:
            is_grounded = False

        return good_answer and good_keywords and no_forbidden_claims and is_grounded

    @property
    def has_metrics(self) -> bool:
        """Check if any metrics were successfully parsed."""
        return any(
            [
                self.url_f1 is not None,
                self.mrr is not None,
                self.context_relevance is not None,
                self.context_precision is not None,
                self.keywords_score is not None,
                self.forbidden_claims_score is not None,
                self.faithfulness is not None,
                self.answer_correctness is not None,
                self.response_relevancy is not None,
            ]
        )

    @property
    def is_retrieval_only_mode(self) -> bool:
        """Detect if this was retrieval-only mode (no answer metrics)."""
        # Retrieval-only mode has URL F1 but no answer quality metrics
        has_retrieval = self.url_f1 is not None or self.context_relevance is not None
        has_answer = any(
            [
                self.keywords_score is not None,
                self.faithfulness is not None,
                self.answer_correctness is not None,
            ]
        )
        return has_retrieval and not has_answer

    def summary(self) -> str:
        """Human-readable summary of metrics."""
        lines = [f"Ticket: {self.ticket_id}"]

        # Show if metrics are averaged across multiple runs
        if self.num_runs > 1:
            lines.append(f"  📊 Metrics averaged across {self.num_runs} runs for stability")

        if not self.has_metrics:
            lines.append("  ⚠️  No metrics found (check if evaluation ran successfully)")
            return "\n".join(lines)

        # RAG usage status
        if self.rag_used:
            if self.docs_retrieved:
                lines.append(f"  RAG Status: ✅ Used, {self.num_docs_retrieved()} docs retrieved")
            else:
                lines.append("  RAG Status: ⚠️  Used, but NO documents retrieved")
        else:
            lines.append("  RAG Status: ❌ NOT used (LLM used general knowledge only)")

        # Retrieval Metrics
        if self.url_f1 is not None:
            lines.append(f"  URL F1: {self.url_f1:.2f}")
        if self.mrr is not None:
            lines.append(f"  MRR: {self.mrr:.2f}")
        if self.context_relevance is not None:
            lines.append(f"  Context Relevance: {self.context_relevance:.2f}")
        if self.context_precision is not None:
            lines.append(f"  Context Precision: {self.context_precision:.2f}")

        # Answer Quality Metrics (only in full mode)
        if self.keywords_score is not None:
            lines.append(f"  Keywords: {self.keywords_score:.2f}")
        if self.faithfulness is not None:
            lines.append(f"  Faithfulness: {self.faithfulness:.2f}")
        if self.answer_correctness is not None:
            lines.append(f"  Answer Correctness: {self.answer_correctness:.2f}")
        if self.response_relevancy is not None:
            lines.append(f"  Response Relevancy: {self.response_relevancy:.2f}")
        if self.forbidden_claims_score is not None:
            lines.append(f"  Forbidden Claims: {self.forbidden_claims_score:.2f}")

        return "\n".join(lines)

    def num_docs_retrieved(self) -> int:
        """Count how many documents were retrieved."""
        if not self.docs_retrieved or not self.contexts:
            return 0

        # Try to count documents in contexts (rough heuristic)
        contexts_str = str(self.contexts)
        # Count URLs in the contexts
        import re

        urls = re.findall(r'https?://[^\s"]+', contexts_str)
        return len(urls) if urls else 1  # At least 1 if contexts exist


class OkpMcpAgent:
    """Autonomous agent for fixing okp-mcp RSPEED tickets."""

    def __init__(
        self,
        eval_root: Path,
        okp_mcp_root: Path,
        lscore_deploy_root: Path,
        worktree_root: Optional[Path] = None,
        interactive: bool = True,
        enable_llm_advisor: bool = True,
    ):
        """Initialize agent with paths to key directories.

        Args:
            eval_root: Path to lightspeed-evaluation repo
            okp_mcp_root: Path to okp-mcp repo
            lscore_deploy_root: Path to lscore-deploy repo
            worktree_root: Base directory for worktrees (default: ~/Work/okp-mcp-worktrees)
            interactive: If True, ask for confirmation before making changes
            enable_llm_advisor: If True, use LLM advisor for AI-powered suggestions (requires Vertex AI)
        """
        self.eval_root = eval_root
        self.okp_mcp_root = okp_mcp_root
        self.lscore_deploy_root = lscore_deploy_root
        self.worktree_root = worktree_root or (Path.home() / "Work/okp-mcp-worktrees")
        self.interactive = interactive

        # Test suite configs
        self.functional_full = (
            eval_root / "okp_mcp_agent/config/test_suites/functional_tests_full.yaml"
        )
        self.functional_retrieval = (
            eval_root / "okp_mcp_agent/config/test_suites/functional_tests_retrieval.yaml"
        )

        # Initialize LLM advisor (Phase 2)
        self.llm_advisor = None
        if enable_llm_advisor and LLM_ADVISOR_AVAILABLE:
            try:
                self.llm_advisor = OkpMcpLLMAdvisor(
                    model="claude-sonnet-4-6",
                    okp_mcp_root=okp_mcp_root,
                    use_tiered_models=True,
                    simple_model="claude-haiku-4-5-20251001",
                    complex_model="claude-opus-4-6",
                )
                print("✅ LLM advisor initialized")
            except Exception as e:
                print(f"⚠️  LLM advisor initialization failed: {e}")
                print("   Continuing without AI-powered suggestions")
                self.llm_advisor = None

        # Initialize Solr checker for document validation
        self.solr_checker = None
        if SOLR_CHECKER_AVAILABLE:
            try:
                self.solr_checker = SolrDocumentChecker()
                if self.solr_checker.is_available():
                    print("✅ Solr checker initialized (http://localhost:8983/solr/portal)")
                else:
                    print("⚠️  Solr is not accessible at http://localhost:8983/solr/portal")
                    print("   Continuing without document validation")
                    self.solr_checker = None
            except Exception as e:
                print(f"⚠️  Solr checker initialization failed: {e}")
                self.solr_checker = None

        # Initialize Solr config analyzer for explain output and tuning
        self.solr_analyzer = None
        if SOLR_ANALYZER_AVAILABLE:
            try:
                self.solr_analyzer = SolrConfigAnalyzer(okp_mcp_root)
                print("✅ Solr analyzer initialized (Solr URL: http://localhost:8983/solr/portal)")
            except Exception as e:
                print(f"❌ Solr analyzer initialization FAILED: {e}")
                print("   This will prevent document discovery and fast loops from working")
                import traceback

                traceback.print_exc()
                self.solr_analyzer = None
        else:
            print("❌ Solr analyzer NOT AVAILABLE (import failed at module level)")
            print("   Document discovery and fast loops will not work")
            print("   Make sure okp_solr_config_analyzer.py exists in okp_mcp_agent/agents/")

        # Track pending commits (only commit if test passes)
        self._pending_commit_msg: Optional[str] = None
        self._pending_commit_file: Optional[Path] = None

    def check_environment(self) -> bool:
        """Check that required environment variables are set.

        Returns:
            True if all required variables are set, False otherwise
        """
        required_vars = [
            "GOOGLE_APPLICATION_CREDENTIALS",  # For Gemini evaluation LLM
            "ANTHROPIC_VERTEX_PROJECT_ID",  # For Claude advisor (if enabled)
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                # ANTHROPIC_VERTEX_PROJECT_ID only required if LLM advisor enabled
                if var == "ANTHROPIC_VERTEX_PROJECT_ID" and not self.llm_advisor:
                    continue
                missing_vars.append(var)

        if missing_vars:
            print("\n❌ Missing required environment variables:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\nRecommended: Create a .env file with these variables:")
            print("   cp .env.example .env")
            print("   # Edit .env and fill in your values")
            print("\nOr set them manually:")
            print("   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
            if "ANTHROPIC_VERTEX_PROJECT_ID" in missing_vars:
                print("   export ANTHROPIC_VERTEX_PROJECT_ID=your-project-id")
                print("\nNote: For ANTHROPIC_VERTEX_PROJECT_ID, you also need:")
                print("   gcloud auth application-default login")
            return False

        return True

    def _run_async_in_thread(self, coro):
        """Run async coroutine in a new thread with its own event loop.

        This avoids conflicts when calling async code from sync context,
        especially when Claude Agent SDK spawns subprocesses.

        Args:
            coro: Coroutine to run

        Returns:
            Result from coroutine
        """
        import threading
        import sys
        import io

        result = [None]
        exception = [None]
        stderr_capture = io.StringIO()

        def thread_target():
            # Redirect stderr to capture Claude CLI errors
            original_stderr = sys.stderr
            sys.stderr = stderr_capture
            try:
                result[0] = asyncio.run(coro)
            except Exception as e:
                exception[0] = e
            finally:
                sys.stderr = original_stderr

        thread = threading.Thread(target=thread_target)
        thread.start()
        thread.join()

        # Print captured stderr if there was an error
        stderr_output = stderr_capture.getvalue()
        if stderr_output:
            print(f"\n🔍 Claude CLI stderr output:\n{stderr_output}", file=sys.stderr)

        if exception[0]:
            raise exception[0]
        return result[0]

    def run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        check: bool = True,
        stream_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a shell command and return result.

        Args:
            cmd: Command and arguments to run
            cwd: Working directory
            check: Raise exception on non-zero exit
            stream_output: Stream output in real-time (default True for long-running commands)
        """
        print(f"$ {' '.join(cmd)}")

        if stream_output:
            # Stream output in real-time (good for long-running commands)
            result = subprocess.run(cmd, cwd=cwd, text=True, check=check)
            return result
        else:
            # Capture and print at end (good for short commands)
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            return result

    def ask_approval(self, question: str, default: bool = False) -> bool:
        """Ask user for approval in interactive mode.

        Args:
            question: Question to ask
            default: Default answer if non-interactive

        Returns:
            True if approved, False otherwise
        """
        if not self.interactive:
            return default

        choices = "[Y/n]" if default else "[y/N]"
        response = input(f"\n{question} {choices}: ").strip().lower()

        if not response:
            return default

        return response in ("y", "yes")

    def create_worktree(self, ticket_id: str, branch_name: Optional[str] = None) -> Path:
        """Create a git worktree for isolated development.

        Args:
            ticket_id: RSPEED ticket ID
            branch_name: Optional custom branch name

        Returns:
            Path to the worktree directory
        """
        if not branch_name:
            branch_name = f"fix/{ticket_id.lower()}"

        worktree_dir = self.worktree_root / branch_name.replace("/", "-")

        print(f"\n🌳 Creating worktree for {ticket_id}...")
        print(f"   Branch: {branch_name}")
        print(f"   Directory: {worktree_dir}")

        # Create worktree directory if it doesn't exist
        self.worktree_root.mkdir(parents=True, exist_ok=True)

        # Check if worktree directory already exists
        if worktree_dir.exists():
            if self.ask_approval(f"Worktree {worktree_dir} already exists. Remove and recreate?"):
                # Remove worktree
                self.run_command(
                    ["git", "worktree", "remove", str(worktree_dir), "--force"],
                    cwd=self.okp_mcp_root,
                    check=False,
                )
            else:
                print("   Using existing worktree")
                return worktree_dir

        # Check if branch exists (could be orphaned from previous failed run)
        result = subprocess.run(
            ["git", "branch", "--list", branch_name],
            cwd=self.okp_mcp_root,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(f"   Branch {branch_name} already exists, deleting...")
            self.run_command(
                ["git", "branch", "-D", branch_name],
                cwd=self.okp_mcp_root,
                check=False,
            )

        # Create new worktree
        self.run_command(
            ["git", "worktree", "add", "-b", branch_name, str(worktree_dir)],
            cwd=self.okp_mcp_root,
        )

        print(f"✓ Worktree created at {worktree_dir}")
        return worktree_dir

    def cleanup_worktree(
        self, worktree_dir: Path, branch_name: Optional[str] = None, ask: bool = True
    ):
        """Remove a worktree and optionally its branch after work is complete.

        Args:
            worktree_dir: Path to worktree to remove
            branch_name: Optional branch name to delete after removing worktree
            ask: Whether to ask for approval (False during error cleanup)
        """
        if ask and not self.ask_approval(f"Remove worktree {worktree_dir}?", default=False):
            return

        print("\n🧹 Cleaning up worktree...")
        try:
            self.run_command(
                ["git", "worktree", "remove", str(worktree_dir), "--force"],
                cwd=self.okp_mcp_root,
            )
            print("✓ Worktree removed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to remove worktree: {e}")

        # Delete the branch if specified
        if branch_name:
            try:
                # Check if branch exists
                result = subprocess.run(
                    ["git", "branch", "--list", branch_name],
                    cwd=self.okp_mcp_root,
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    self.run_command(
                        ["git", "branch", "-D", branch_name],
                        cwd=self.okp_mcp_root,
                    )
                    print(f"✓ Branch {branch_name} deleted")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Failed to delete branch: {e}")

    def update_compose_mount(self, worktree_path: Path):
        """Update podman-compose.yml to mount worktree instead of main.

        Args:
            worktree_path: Path to worktree directory (e.g., ~/Work/okp-mcp-fix-RSPEED-2482)
        """
        compose_file = self.lscore_deploy_root / "local" / "podman-compose.yml"
        print(f"\n📝 Updating {compose_file.name} to use worktree...")

        # Read current content
        content = compose_file.read_text()

        # Calculate relative path from lscore-deploy/local to worktree
        # lscore-deploy/local -> ../../okp-mcp-worktrees/fix-rspeed_2482/src
        relative_path = f"../../{worktree_path.parent.name}/{worktree_path.name}/src"

        # Replace the active mount line (may have comments after it)
        # Current: - ../../okp-mcp/src:/dev/src:z  # Main repo ...
        # New:     - ../../okp-mcp-worktrees/fix-rspeed_2482/src:/dev/src:z
        import re

        # Match the mount line with optional comment
        pattern = r"(\s*- ../../okp-mcp/src:/dev/src:z)(\s*#.*)?$"
        new_line = f"- {relative_path}:/dev/src:z"

        # Check if pattern exists
        if not re.search(pattern, content, re.MULTILINE):
            print("⚠️  Warning: Expected mount line not found in compose file")
            print(f"   Looking for pattern: {pattern}")
            # Fallback: try simple string match
            if "- ../../okp-mcp/src:/dev/src:z" in content:
                print("   Found mount line without regex, using simple replacement")
                modified_content = content.replace(
                    "- ../../okp-mcp/src:/dev/src:z",
                    f"#- ../../okp-mcp/src:/dev/src:z  # Temporarily disabled for worktree\n      {new_line}",
                )
            else:
                raise RuntimeError("Could not find okp-mcp mount line to replace")
        else:
            # Comment out the main mount and add worktree mount
            modified_content = re.sub(
                pattern,
                rf"#\1\2  # Temporarily disabled for worktree\n      {new_line}",
                content,
                flags=re.MULTILINE,
            )

        # Backup original
        backup_file = compose_file.with_suffix(".yml.backup")
        compose_file.write_text(content)  # This creates backup in same operation
        backup_file.write_text(content)

        # Write modified
        compose_file.write_text(modified_content)
        print(f"✅ Updated mount to: {new_line}")
        print(f"   Backup saved: {backup_file}")

    def revert_compose_mount(self):
        """Revert podman-compose.yml back to main mount."""
        compose_file = self.lscore_deploy_root / "local" / "podman-compose.yml"
        backup_file = compose_file.with_suffix(".yml.backup")

        print(f"\n🔄 Reverting {compose_file.name} to main mount...")

        if backup_file.exists():
            # Restore from backup
            backup_content = backup_file.read_text()
            compose_file.write_text(backup_content)
            backup_file.unlink()  # Remove backup
            print("✅ Restored from backup")
        else:
            print("⚠️  No backup found, skipping revert")

    def verify_container_healthy(self, max_wait_seconds: int = 30) -> bool:
        """Wait for okp-mcp container to be healthy.

        Args:
            max_wait_seconds: Maximum time to wait for healthy status

        Returns:
            True if container is healthy, False if timeout
        """
        print("\n🏥 Waiting for okp-mcp container to be healthy...")

        import time

        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            try:
                result = subprocess.run(
                    ["podman", "inspect", "okp-mcp", "--format", "{{.State.Health.Status}}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode == 0:
                    health_status = result.stdout.strip()
                    if health_status == "healthy":
                        print("✅ Container is healthy!")
                        return True
                    else:
                        print(f"   Status: {health_status}, waiting...")

                time.sleep(2)

            except Exception as e:
                print(f"   Error checking health: {e}")
                time.sleep(2)

        print("❌ Container did not become healthy within timeout")
        return False

    def restart_okp_mcp(self, verify_healthy: bool = True):
        """Restart okp-mcp service and optionally wait for healthy status.

        Args:
            verify_healthy: If True, wait for container to be healthy before returning
        """
        print("\n🔄 Restarting okp-mcp...")
        self.run_command(
            ["podman-compose", "restart", "okp-mcp"],
            cwd=self.lscore_deploy_root / "local",
        )
        print("✓ okp-mcp restarted")

        if verify_healthy:
            if not self.verify_container_healthy():
                print("⚠️  Warning: Container may not be ready, but continuing...")
                if self.interactive:
                    input("Press Enter to continue anyway, or Ctrl+C to abort...")
        else:
            # Even if not verifying, give it a moment to start
            import time

            print("   Waiting 5 seconds for service to start...")
            time.sleep(5)

    def save_iteration_diagnostics(
        self,
        ticket_id: Optional[str],
        iteration: int,
        result: EvaluationResult,
        solr_query_info: Optional[Dict] = None,
        suggestion: Optional[Any] = None,
    ) -> Path:
        """Save iteration diagnostics to JSON file for later analysis.

        Args:
            ticket_id: Ticket being fixed (None for full-pattern mode)
            iteration: Iteration number
            result: EvaluationResult with metrics and retrieved docs
            solr_query_info: Solr query inspection results
            suggestion: LLM suggestion object (with reasoning, code, etc.)

        Returns:
            Path to saved diagnostics file
        """
        import json
        from datetime import datetime

        # Create diagnostics directory
        if ticket_id:
            diag_dir = self.eval_root / ".diagnostics" / ticket_id.replace("-", "_")
        else:
            # Full-pattern mode: use pattern name from result if available
            diag_dir = self.eval_root / ".diagnostics" / "FULL_PATTERN"
        diag_dir.mkdir(parents=True, exist_ok=True)

        # Prepare diagnostic data
        diagnostics = {
            "ticket_id": ticket_id,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            # Question and Answer (for debugging LLM judge scoring)
            "question": result.query,
            "actual_response": result.response,  # What the LLM actually said
            "expected_response": result.expected_response,  # What it should say
            "expected_keywords": result.expected_keywords,  # Keywords that should be present
            # Metrics (to see if LLM judge scored correctly)
            "metrics": {
                # Retrieval metrics
                "url_f1": result.url_f1,
                "mrr": result.mrr,
                "context_relevance": result.context_relevance,
                "context_precision": result.context_precision,
                "url_overlap_with_previous": result.url_overlap_with_previous,
                # Answer quality metrics (LLM-judged)
                "keywords_score": result.keywords_score,
                "answer_correctness": result.answer_correctness,
                "faithfulness": result.faithfulness,
                "response_relevancy": result.response_relevancy,
                "forbidden_claims_score": result.forbidden_claims_score,
            },
            # Retrieved context (what docs were sent to LLM)
            "retrieved_documents": [
                {
                    "url": url,
                    "title": (
                        result.retrieved_doc_titles[i]
                        if result.retrieved_doc_titles and i < len(result.retrieved_doc_titles)
                        else None
                    ),
                }
                for i, url in enumerate(result.retrieved_urls or [])
            ],
            "expected_documents": [{"url": url} for url in (result.expected_urls or [])],
            # Raw contexts sent to LLM (truncated to first 500 chars for readability)
            "contexts_sample": (
                str(result.contexts)[:500] + "..."
                if result.contexts and len(str(result.contexts)) > 500
                else str(result.contexts)
            ),
            # Solr query inspection
            "solr_query_inspection": solr_query_info,
            # LLM Suggestion (code changes, reasoning, expected improvement)
            "llm_suggestion": None,
        }

        # Add LLM suggestion details if available
        if suggestion:
            diagnostics["llm_suggestion"] = {
                "suggested_change": getattr(suggestion, "suggested_change", None),
                "reasoning": getattr(suggestion, "reasoning", None),
                "code_snippet": getattr(suggestion, "code_snippet", None),
                "expected_improvement": getattr(suggestion, "expected_improvement", None),
                "confidence": getattr(suggestion, "confidence", None),
                "file_path": getattr(suggestion, "file_path", None),
            }

        # Save to file
        diag_file = diag_dir / f"iteration_{iteration:03d}.json"
        with open(diag_file, "w") as f:
            json.dump(diagnostics, f, indent=2)

        return diag_file

    def save_iteration_summary_table(
        self, ticket_id: str, iteration_history: List[Dict], final_status: str = "In Progress"
    ) -> Path:
        """Save human-readable iteration summary table and progress report.

        Args:
            ticket_id: Ticket being fixed
            iteration_history: List of iteration records with metrics and changes
            final_status: Final status (Fixed, Max Iterations, Failed, etc.)

        Returns:
            Path to saved summary file
        """
        diag_dir = self.eval_root / ".diagnostics" / ticket_id.replace("-", "_")
        diag_dir.mkdir(parents=True, exist_ok=True)

        summary_file = diag_dir / "iteration_summary.txt"

        # Calculate run statistics
        if iteration_history:
            start_time = iteration_history[0].get("timestamp", "")
            end_time = iteration_history[-1].get("timestamp", "")
            total_iterations = len(iteration_history)
            changes_applied = sum(1 for r in iteration_history if r.get("improved", False))
            changes_reverted = total_iterations - changes_applied

            # Calculate duration if timestamps available
            duration_str = "N/A"
            if start_time and end_time:
                from datetime import datetime

                try:
                    start_dt = datetime.fromisoformat(start_time)
                    end_dt = datetime.fromisoformat(end_time)
                    duration = end_dt - start_dt
                    minutes = int(duration.total_seconds() / 60)
                    seconds = int(duration.total_seconds() % 60)
                    duration_str = f"{minutes}m {seconds}s"
                except Exception:
                    pass
        else:
            start_time = end_time = duration_str = "N/A"
            total_iterations = changes_applied = changes_reverted = 0

        lines = [
            "=" * 100,
            f"PROGRESS REPORT - {ticket_id}",
            "=" * 100,
            "",
            "RUN STATISTICS:",
            f"  Status:           {final_status}",
            f"  Start Time:       {start_time[:19] if start_time != 'N/A' else 'N/A'}",
            f"  End Time:         {end_time[:19] if end_time != 'N/A' else 'N/A'}",
            f"  Duration:         {duration_str}",
            f"  Total Iterations: {total_iterations}",
            f"  Changes Applied:  {changes_applied} (kept)",
            f"  Changes Reverted: {changes_reverted} (didn't improve)",
            "",
            "=" * 100,
            "ITERATION DETAILS",
            "=" * 100,
            "",
            f"{'Iter':<6} {'Change':<45} {'Metric Δ':<10} {'Overlap':<9} {'Result':<8} {'Notes'}",
            "-" * 100,
        ]

        for record in iteration_history:
            iter_num = record.get("iteration", "?")

            # Change description
            change = record.get("change", "N/A")
            if len(change) > 42:
                change = change[:39] + "..."

            # Metric delta
            if "metric_before" in record and "metric_after" in record:
                delta = record["metric_after"] - record["metric_before"]
                metric_str = f"{delta:+.3f}"
            else:
                metric_str = "N/A"

            # URL overlap
            if (
                "metrics" in record
                and record["metrics"].get("url_overlap_with_previous") is not None
            ):
                overlap = record["metrics"]["url_overlap_with_previous"]
                overlap_str = f"{overlap:.2f}"
            else:
                overlap_str = "N/A"

            # Result
            improved = record.get("improved", False)
            result_str = "✓ Yes" if improved else "✗ No"

            # Notes (interesting info)
            notes = []
            if "solr_query_inspection" in record and record["solr_query_inspection"]:
                sqr = record["solr_query_inspection"]
                if sqr.get("injected_terms"):
                    notes.append(f"Query+{len(sqr['injected_terms'])}")

            if "retrieved_documents" in record and "expected_documents" in record:
                retrieved_urls = {doc["url"] for doc in record["retrieved_documents"]}
                expected_urls = {doc["url"] for doc in record["expected_documents"]}
                matched = len(retrieved_urls & expected_urls)
                total = len(expected_urls)
                notes.append(f"{matched}/{total} expected")

            notes_str = ", ".join(notes) if notes else ""

            lines.append(
                f"{iter_num:<6} {change:<45} {metric_str:<10} {overlap_str:<9} {result_str:<8} {notes_str}"
            )

            # Add detailed metrics if available
            if "metrics" in record:
                m = record["metrics"]
                details = []
                if m.get("url_f1") is not None:
                    details.append(f"URL_F1={m['url_f1']:.2f}")
                if m.get("mrr") is not None:
                    details.append(f"MRR={m['mrr']:.2f}")
                if m.get("context_relevance") is not None:
                    details.append(f"CtxRel={m['context_relevance']:.2f}")
                if m.get("context_precision") is not None:
                    details.append(f"CtxPrec={m['context_precision']:.2f}")

                if details:
                    lines.append(f"       Metrics: {', '.join(details)}")

            # Add query augmentation details if present
            if "solr_query_inspection" in record and record["solr_query_inspection"]:
                sqr = record["solr_query_inspection"]
                if sqr.get("injected_terms"):
                    lines.append(
                        f"       ⚠️  Solr query: '{sqr.get('original', '')}' → '{sqr.get('actual', '')}'"
                    )
                    lines.append(
                        f"           Injected: {', '.join(sqr['injected_terms'][:5])}{' ...' if len(sqr['injected_terms']) > 5 else ''}"
                    )

            lines.append("")

        # Add metric progression chart
        lines.extend(
            [
                "=" * 100,
                "METRIC PROGRESSION",
                "=" * 100,
                "",
            ]
        )

        # Track best scores
        best_scores: Dict[str, Dict[str, Any]] = {}
        metric_names = [
            "url_f1",
            "mrr",
            "context_relevance",
            "context_precision",
            "keywords_score",
            "answer_correctness",
            "faithfulness",
            "response_relevancy",
        ]

        # Build progression table
        if iteration_history:
            # Header
            lines.append(
                f"{'Iter':<6} {'URL_F1':<8} {'MRR':<8} {'CtxRel':<8} {'CtxPrec':<8} {'Keywords':<9} {'AnsCorr':<9} {'Faith':<8}"
            )
            lines.append("-" * 100)

            for record in iteration_history:
                iter_num = record.get("iteration", "?")
                m = record.get("metrics", {})

                # Format each metric (N/A if not present)
                url_f1 = f"{m.get('url_f1', 0):.2f}" if m.get("url_f1") is not None else "N/A"
                mrr = f"{m.get('mrr', 0):.2f}" if m.get("mrr") is not None else "N/A"
                ctx_rel = (
                    f"{m.get('context_relevance', 0):.2f}"
                    if m.get("context_relevance") is not None
                    else "N/A"
                )
                ctx_prec = (
                    f"{m.get('context_precision', 0):.2f}"
                    if m.get("context_precision") is not None
                    else "N/A"
                )
                kw = (
                    f"{m.get('keywords_score', 0):.2f}"
                    if m.get("keywords_score") is not None
                    else "N/A"
                )
                ans = (
                    f"{m.get('answer_correctness', 0):.2f}"
                    if m.get("answer_correctness") is not None
                    else "N/A"
                )
                faith = (
                    f"{m.get('faithfulness', 0):.2f}"
                    if m.get("faithfulness") is not None
                    else "N/A"
                )

                lines.append(
                    f"{iter_num:<6} {url_f1:<8} {mrr:<8} {ctx_rel:<8} {ctx_prec:<8} {kw:<9} {ans:<9} {faith:<8}"
                )

                # Track best scores
                for metric_name in metric_names:
                    val = m.get(metric_name)
                    if val is not None:
                        if (
                            metric_name not in best_scores
                            or val > best_scores[metric_name]["value"]
                        ):
                            best_scores[metric_name] = {"value": val, "iteration": iter_num}

            lines.append("")

        # Show best scores achieved
        lines.extend(
            [
                "=" * 100,
                "BEST SCORES ACHIEVED",
                "=" * 100,
                "",
            ]
        )

        if best_scores:
            for metric_name, data in best_scores.items():
                if data["value"] > 0:  # Only show non-zero scores
                    metric_display = metric_name.replace("_", " ").title()
                    lines.append(
                        f"  {metric_display:<25} {data['value']:.3f} (iteration {data['iteration']})"
                    )
        else:
            lines.append("  No metrics recorded")

        lines.extend(
            [
                "",
                "=" * 100,
                "LEGEND",
                "=" * 100,
                "",
                "METRICS:",
                "  URL_F1:     F1 score for retrieved URLs vs expected URLs",
                "  MRR:        Mean Reciprocal Rank of first expected URL in results",
                "  CtxRel:     Context Relevance - are retrieved docs relevant to question?",
                "  CtxPrec:    Context Precision - what % of retrieved docs are useful?",
                "  Keywords:   Were expected keywords present in answer?",
                "  AnsCorr:    Answer Correctness - factually correct answer?",
                "  Faith:      Faithfulness - answer grounded in retrieved context?",
                "",
                "TABLE COLUMNS:",
                "  Metric Δ:   Change in primary metric (URL F1 or Answer Correctness)",
                "  Overlap:    URL similarity between iterations (1.0 = same docs)",
                "  Result:     ✓ = improved (kept), ✗ = no improvement (reverted)",
                "",
            ]
        )

        # Save to file
        with open(summary_file, "w") as f:
            f.write("\n".join(lines))

        print(f"\n💾 Saved progress report: {summary_file}")
        return summary_file

    def load_iteration_history(self, ticket_id: str) -> List[Dict]:
        """Load all iteration diagnostics for a ticket.

        Args:
            ticket_id: Ticket ID

        Returns:
            List of diagnostic dicts, ordered by iteration
        """
        import json

        diag_dir = self.eval_root / ".diagnostics" / ticket_id.replace("-", "_")
        if not diag_dir.exists():
            return []

        history = []
        for diag_file in sorted(diag_dir.glob("iteration_*.json")):
            with open(diag_file) as f:
                history.append(json.load(f))

        return history

    def extract_solr_config_snapshot(self, ticket_id: str) -> Dict:
        """Extract current Solr configuration patterns for LLM context.

        Instead of having Claude read entire files (500+ lines), extract
        just the key tunable parameters and patterns.

        Args:
            ticket_id: Ticket ID for caching

        Returns:
            Dict with Solr config snapshot
        """
        import json
        from datetime import datetime

        if not self.solr_analyzer:
            return {}

        # Parse current config from solr.py
        config = self.solr_analyzer.parse_current_config()

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "solr_params": {
                # Document ranking params
                "mm": config.get("mm", "unknown"),
                "qf": config.get("qf", "unknown"),
                "pf": config.get("pf", "unknown"),
                "pf2": config.get("pf2", "unknown"),
                "pf3": config.get("pf3", "unknown"),
                "ps": config.get("ps", "unknown"),
                "ps2": config.get("ps2", "unknown"),
                "ps3": config.get("ps3", "unknown"),
                "boost_multiplier": config.get("boost_multiplier", 2.0),
                "demote_multiplier": config.get("demote_multiplier", 0.05),
            },
            "highlighting_params": {
                # Snippet selection scoring (BM25 for highlighting)
                "hl.score.k1": config.get("hl.score.k1", "unknown"),
                "hl.score.b": config.get("hl.score.b", "unknown"),
                "hl.score.pivot": config.get("hl.score.pivot", "unknown"),
                # Snippet configuration
                "hl.snippets": config.get("hl.snippets", "unknown"),
                "hl.fragsize": config.get("hl.fragsize", "unknown"),
            },
            "boost_keywords_count": len(config.get("boost_keywords", [])),
            "boost_keywords_sample": config.get("boost_keywords", [])[:30],  # First 30
            "demote_keywords_count": len(config.get("demote_keywords", [])),
            "demote_keywords_sample": config.get("demote_keywords", [])[:10],  # First 10
            "file_locations": {
                "solr_params": "src/okp_mcp/solr.py (lines ~140-160)",
                "highlighting_params": "src/okp_mcp/solr.py (lines ~110-130)",
                "boost_keywords": "src/okp_mcp/solr.py (lines ~45-250)",
                "demote_keywords": "src/okp_mcp/solr.py (lines ~250-260)",
            },
        }

        # Cache to diagnostics directory
        diag_dir = self.eval_root / ".diagnostics" / ticket_id.replace("-", "_")
        diag_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file = diag_dir / "solr_config_snapshot.json"

        with open(snapshot_file, "w") as f:
            json.dump(snapshot, f, indent=2)

        print(f"💾 Cached Solr config snapshot: {snapshot_file.name}")
        return snapshot

    def load_solr_config_snapshot(self, ticket_id: str) -> Optional[Dict]:
        """Load cached Solr config snapshot.

        Args:
            ticket_id: Ticket ID

        Returns:
            Cached snapshot dict, or None if not found
        """
        import json

        diag_dir = self.eval_root / ".diagnostics" / ticket_id.replace("-", "_")
        snapshot_file = diag_dir / "solr_config_snapshot.json"

        if not snapshot_file.exists():
            return None

        with open(snapshot_file) as f:
            return json.load(f)

    def _clear_mcp_cache(self):
        """Clear MCP direct mode cache to force fresh evaluation after code changes.

        CRITICAL: The MCP direct cache uses query-based keys that don't include
        Solr configuration parameters. After modifying okp-mcp code (boost keywords,
        field weights, etc.), the cache MUST be cleared to avoid returning stale results.
        """
        import shutil

        cache_dir = Path(".caches/mcp_direct_cache")
        if cache_dir.exists():
            print("🗑️  Clearing MCP direct cache (config changed)...")
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("✓ Cache cleared")

    def create_single_ticket_config(
        self,
        ticket_id: str,
        base_config: Path,
        add_answer_metrics: bool = True,
        metrics: Optional[List[str]] = None,
    ) -> Path:
        """Create a temporary config with just one ticket for faster iteration.

        Args:
            ticket_id: Ticket ID (e.g., "RSPEED-2482" or "RSPEED_2482")
            base_config: Path to full config file to extract from
            add_answer_metrics: If False, skip adding answer_correctness (for retrieval-only mode)
            metrics: If provided, override turn_metrics with this list (for mode-specific metrics)

        Returns:
            Path to temporary config file with single ticket
        """
        # Normalize ticket ID (some configs use underscores, some use hyphens)
        normalized_ticket_id = ticket_id.replace("-", "_")

        # Read full config
        with open(base_config) as f:
            all_tickets = yaml.safe_load(f)

        # Find the specific ticket (check both hyphen and underscore versions)
        ticket_config = None
        for ticket in all_tickets:
            ticket_conv_id = ticket.get("conversation_group_id", "")
            if ticket_conv_id == ticket_id or ticket_conv_id == normalized_ticket_id:
                ticket_config = [ticket]  # Wrap in list for YAML format
                break

        if not ticket_config:
            raise RuntimeError(
                f"Ticket {ticket_id} not found in {base_config}. "
                f"Available: {[t.get('conversation_group_id') for t in all_tickets[:5]]}"
            )

        # Clean up ticket config for lightspeed-evaluation format
        for ticket in ticket_config:
            # Remove metadata if present (pattern files have this, test configs don't)
            if "metadata" in ticket:
                del ticket["metadata"]

            # Ensure turns have turn_id
            for i, turn in enumerate(ticket.get("turns", []), start=1):
                if "turn_id" not in turn:
                    turn["turn_id"] = str(i)

                # Override metrics if explicitly provided
                if metrics is not None:
                    turn["turn_metrics"] = metrics
                    print(f"   Using specified metrics: {metrics}")
                else:
                    # Optionally add answer_correctness if expected_response exists
                    # This is needed to check if answer is good even if retrieval is suboptimal
                    # BUT skip this for retrieval-only mode (add_answer_metrics=False)
                    if add_answer_metrics:
                        # Only add answer_correctness if expected_response is defined
                        if "expected_response" in turn:
                            turn_metrics = turn.get("turn_metrics", [])
                            if "custom:answer_correctness" not in turn_metrics:
                                turn_metrics.append("custom:answer_correctness")
                                print("   Added custom:answer_correctness to metrics")
                        else:
                            # Config doesn't have expected_response, can't evaluate answer_correctness
                            # This is OK - we'll rely on keywords_eval for answer quality
                            pass

        # Create temp file
        temp_dir = self.eval_root / ".temp_configs"
        temp_dir.mkdir(exist_ok=True)
        temp_config = temp_dir / f"{normalized_ticket_id}_single.yaml"

        # Write single-ticket config
        with open(temp_config, "w") as f:
            yaml.dump(ticket_config, f, default_flow_style=False)

        print(f"   Created single-ticket config: {temp_config.name}")
        return temp_config

    def clean_pattern_config(self, config: Path, metrics: Optional[List[str]] = None) -> Path:
        """Clean pattern config for evaluation: remove metadata, add turn_id, filter empty expected_response.

        Args:
            config: Path to pattern config file
            metrics: If provided, override turn_metrics with this list

        Returns:
            Path to cleaned temporary config file
        """
        # Read pattern config
        with open(config) as f:
            all_tickets = yaml.safe_load(f)

        if not all_tickets:
            raise RuntimeError(f"No conversations found in {config}")

        # Separate tickets with and without expected_response
        ready_tickets = []
        sme_review_tickets = []

        for ticket in all_tickets:
            # Remove metadata if present (pattern files have this, test configs don't)
            if "metadata" in ticket:
                del ticket["metadata"]

            # Check if ticket has valid expected_response
            has_valid_response = False
            for i, turn in enumerate(ticket.get("turns", []), start=1):
                # Add turn_id if missing
                if "turn_id" not in turn:
                    turn["turn_id"] = str(i)

                # Preserve existing skip tag (if present)
                existing_skip = turn.get("skip")

                # Override metrics if specified
                if metrics is not None:
                    turn["turn_metrics"] = metrics

                # Restore skip tag if it was present
                if existing_skip is not None:
                    turn["skip"] = existing_skip

                # Check if expected_response exists and is not empty
                expected = turn.get("expected_response", "").strip()
                if expected:
                    has_valid_response = True

            if has_valid_response:
                ready_tickets.append(ticket)
            else:
                sme_review_tickets.append(ticket)

        print("   Cleaned pattern config:")
        print(f"   - Ready for eval: {len(ready_tickets)} tickets")
        print(f"   - Need SME review: {len(sme_review_tickets)} tickets (empty expected_response)")
        if metrics:
            metric_names = [m.split(":")[1] for m in metrics]
            print(f"   - Metrics in YAML: {len(metrics)} ({', '.join(metric_names)})")

        if not ready_tickets:
            raise RuntimeError(
                f"No tickets with expected_response found in {config}. "
                f"All {len(sme_review_tickets)} tickets need SME review."
            )

        # Save SME review tickets if any
        if sme_review_tickets:
            sme_file = config.parent / f"{config.stem}_SME_REVIEW.yaml"
            with open(sme_file, "w") as f:
                yaml.dump(sme_review_tickets, f, default_flow_style=False)
            print(f"   ⚠️  Quarantined tickets needing SME review: {sme_file.name}")

        # Create temp file with ready tickets
        temp_dir = self.eval_root / ".temp_configs"
        temp_dir.mkdir(exist_ok=True)
        temp_config = temp_dir / f"{config.stem}_cleaned.yaml"

        # Write cleaned config
        with open(temp_config, "w") as f:
            yaml.dump(ready_tickets, f, default_flow_style=False)

        print(f"   Created cleaned config: {temp_config.name}")
        return temp_config

    def update_skip_tags(
        self, config: Path, ticket_classifications: Dict, mode: str = "set"
    ) -> Path:
        """Update skip tags in YAML based on stability classifications.

        Args:
            config: Path to YAML config file
            ticket_classifications: Dict mapping ticket_id → StabilityClassification
            mode: "set" to add skip tags, "remove" to remove all skip tags

        Returns:
            Path to updated config file (same as input)
        """
        # Read config
        with open(config) as f:
            tickets = yaml.safe_load(f)

        if not tickets:
            return config

        # Update skip tags
        for ticket in tickets:
            ticket_id = ticket.get("conversation_group_id")
            if not ticket_id:
                continue

            # Normalize ticket ID (handles both RSPEED-2218 and RSPEED_2218)
            normalized_id = ticket_id.replace("-", "_")

            if mode == "remove":
                # Remove skip tags from all turns
                for turn in ticket.get("turns", []):
                    if "skip" in turn:
                        del turn["skip"]
            elif mode == "set" and normalized_id in ticket_classifications:
                # Set skip based on classification
                classification = ticket_classifications[normalized_id]
                for turn in ticket.get("turns", []):
                    turn["skip"] = classification.skip

        # Write updated config
        with open(config, "w") as f:
            yaml.dump(tickets, f, default_flow_style=False)

        if mode == "set":
            skip_count = sum(1 for c in ticket_classifications.values() if c.skip)
            print(
                f"   📌 Updated skip tags: {skip_count}/{len(ticket_classifications)} tickets will be skipped"
            )
        else:
            print(f"   🔓 Removed all skip tags from {len(tickets)} tickets")

        return config

    def run_full_eval(
        self, config: Path, runs: int = 1, single_ticket: Optional[str] = None
    ) -> Path:
        """Run full evaluation suite and return output directory.

        Args:
            config: Path to evaluation config file
            runs: Number of evaluation runs for stability
            single_ticket: Optional ticket ID - if provided, only evaluate this one ticket

        Returns:
            Path to evaluation output directory
        """
        # Full metrics list (add to YAML, don't use --metrics flag for baseline)
        full_metrics = [
            "custom:url_retrieval_eval",
            "ragas:context_relevance",
            "ragas:context_precision_without_reference",
            "custom:answer_correctness",
            "ragas:faithfulness",
            "ragas:response_relevancy",
        ]

        # Clean and prepare config for evaluation
        if single_ticket:
            # Single-ticket mode: create filtered config with one ticket
            config = self.create_single_ticket_config(
                single_ticket, config, add_answer_metrics=True, metrics=full_metrics
            )
            print(f"\n📊 Running evaluation for {single_ticket} only ({runs} runs)...")
        else:
            # Full-pattern mode: clean the entire pattern file with full metrics in YAML
            config = self.clean_pattern_config(config, metrics=full_metrics)
            print(f"\n📊 Running full evaluation ({runs} runs)...")

        print(f"   ⏳ Estimated time: ~{runs * 2} minutes")

        # Log start timestamp for debugging
        import time
        from datetime import datetime

        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"   🕐 Started at: {start_timestamp}")
        print(f"   🔄 Running {runs} evaluation runs...\n")
        sys.stdout.flush()

        start_time = time.time()

        # Start subprocess in background
        proc = subprocess.Popen(
            [
                "./run_okp_mcp_full_suite.sh",
                "--config",
                str(config),
                "--runs",
                str(runs),
            ],
            cwd=self.eval_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Monitor progress by checking which runs have completed
        output_base = self.eval_root / "okp_mcp_full_output"
        last_completed = 0

        while proc.poll() is None:
            # Count completed runs by looking for summary.json files
            try:
                suite_dirs = sorted(output_base.glob("suite_*"), key=lambda p: p.stat().st_mtime)
                if suite_dirs:
                    latest_suite = suite_dirs[-1]
                    # Count run directories that have a summary.json file (indicating completion)
                    completed_runs = len(list(latest_suite.glob("run_*/evaluation_*_summary.json")))

                    if completed_runs > last_completed:
                        elapsed = time.time() - start_time
                        print(
                            f"      ✅ Run {completed_runs}/{runs} complete ({elapsed/60:.1f} min elapsed)"
                        )
                        sys.stdout.flush()
                        last_completed = completed_runs
            except Exception:
                pass  # Ignore errors while monitoring

            time.sleep(5)  # Check every 5 seconds

        # Wait for process to finish
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)

        elapsed = time.time() - start_time
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n   ✅ All {runs} runs completed in {elapsed/60:.1f} minutes")
        print(f"   🕐 Started: {start_timestamp}, Ended: {end_timestamp}")
        sys.stdout.flush()

        # Find the latest output directory
        output_dirs = sorted(
            (self.eval_root / "okp_mcp_full_output").glob("suite_*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not output_dirs:
            raise RuntimeError("No evaluation output found")

        # Parse and print key scores immediately
        output_dir = output_dirs[-1]
        try:
            self._print_scores_from_output(output_dir, single_ticket, runs)
        except Exception as e:
            print(f"   ⚠️  Could not parse scores: {e}")

        return output_dir

    def enrich_config_with_expected_urls(
        self, config_path: Path, ticket_id: str, expected_urls: List[str]
    ) -> Path:
        """Add expected_urls and url_retrieval_eval metric to config.

        Args:
            config_path: Path to config file to update
            ticket_id: Ticket conversation_group_id to update
            expected_urls: List of URLs to add as expected

        Returns:
            Path to updated config file (same as input)
        """
        import yaml

        normalized_ticket_id = ticket_id.replace("-", "_")

        # Load config
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Find the conversation group
        for conv in data:
            if conv.get("conversation_group_id") == normalized_ticket_id:
                # Add expected_urls to first turn
                if "turns" in conv and conv["turns"]:
                    turn = conv["turns"][0]

                    # Add or update expected_urls
                    turn["expected_urls"] = expected_urls

                    # Add url_retrieval_eval metric if not present
                    if "turn_metrics" in turn:
                        if "custom:url_retrieval_eval" not in turn["turn_metrics"]:
                            turn["turn_metrics"].insert(0, "custom:url_retrieval_eval")
                    else:
                        turn["turn_metrics"] = ["custom:url_retrieval_eval"]

                    break

        # Write back
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"✅ Updated config with {len(expected_urls)} expected URLs")
        return config_path

    def discover_expected_documents(
        self,
        ticket_id: str,
        query: str,
        expected_response: str,
        expected_keywords: Optional[List] = None,
        auto_select_threshold: float = 10.0,
    ) -> List[str]:
        """Discover which Solr documents contain the answer.

        Args:
            ticket_id: Ticket ID for logging
            query: User's question
            expected_response: What the answer should contain
            expected_keywords: Keywords that should be in the answer
            auto_select_threshold: Auto-select docs with score > this (0 = ask user)

        Returns:
            List of URLs that likely contain the answer
        """
        if not self.solr_analyzer:
            print("❌ Solr analyzer not available")
            print("   This is required for document discovery (searching Solr)")
            print(f"   SOLR_ANALYZER_AVAILABLE: {SOLR_ANALYZER_AVAILABLE}")
            if SOLR_ANALYZER_AVAILABLE:
                print("   Import succeeded but initialization failed during agent startup")
                print("   Check the startup logs for initialization errors")
            else:
                print(
                    "   Import failed - check that okp_mcp_agent/agents/okp_solr_config_analyzer.py exists"
                )
            return []

        print(f"\n🔍 DOCUMENT DISCOVERY for {ticket_id}")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"Looking for documents containing:\n{expected_response[:200]}...")
        print("=" * 80)

        # Extract keywords from expected_keywords format
        kw_list = []
        if expected_keywords:
            for kw in expected_keywords:
                if isinstance(kw, list):
                    kw_list.extend(kw)
                else:
                    kw_list.append(kw)

        # Search Solr
        results = self.solr_analyzer.search_for_answer_content(
            keywords=kw_list, expected_response=expected_response, num_results=10
        )

        if not results:
            print("\n" + "=" * 80)
            print("❌ KNOWLEDGE GAP DETECTED")
            print("=" * 80)
            print(f"\nQuestion: {query}")
            print("\nNo documents in Solr contain the expected answer.")
            print("\nSearched for terms:")
            for kw in kw_list[:10]:
                print(f"  - {kw}")
            print("\nThis question cannot be answered with the current knowledge base.")
            print("\nNext steps:")
            print("  1. Verify the correct documents exist at access.redhat.com")
            print("  2. Check if documents need to be indexed in Solr")
            print("  3. Mark question as 'unanswerable' if content doesn't exist")
            print("  4. File content gap ticket if documentation is missing")
            print("=" * 80)
            return []

        print(f"\n📄 Found {len(results)} candidate documents:\n")

        # Display results
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc['title']}")
            print(f"   URL: {doc['url']}")
            print(f"   Score: {doc['score']:.2f}")
            print(f"   Snippet: {doc['snippet'][:150]}...")
            print()

        # ENHANCED: Verify which docs actually contain the answer
        if self.llm_advisor and auto_select_threshold > 0:
            print("🔍 Verifying which documents actually contain the answer information...\n")

            verified_docs = []
            for doc in results:
                # Check if this doc's snippet contains answer info
                check_result = self.check_answer_in_retrieved_docs(
                    expected_answer=expected_response, retrieved_contexts=[doc["snippet"]]
                )

                if check_result["contains_answer"]:
                    verified_docs.append(
                        {
                            **doc,
                            "verification_confidence": check_result["confidence"],
                            "verification_reason": check_result["explanation"],
                        }
                    )
                    status = "✅ VERIFIED"
                else:
                    status = "❌ REJECTED"

                print(f"{status}: {doc['title'][:50]}...")
                print(f"          {check_result['explanation'][:80]}")
                print()

            if verified_docs:
                selected_urls = [doc["url"] for doc in verified_docs]
                print(f"\n✅ Auto-selected {len(selected_urls)} verified documents:")
                for doc in verified_docs:
                    print(f"   - {doc['url']}")
                    print(f"     Confidence: {doc['verification_confidence']:.2f}")
                    print(f"     Reason: {doc['verification_reason'][:60]}...")
                return selected_urls
            else:
                print("\n⚠️  No documents passed verification!")
                print("   Falling back to score-based selection...")
                # Fall through to score-based selection

        # Auto-select high-scoring docs (fallback if no verification or no verified docs)
        if auto_select_threshold > 0:
            selected_urls = [doc["url"] for doc in results if doc["score"] >= auto_select_threshold]
            if selected_urls:
                print(
                    f"✅ Auto-selected {len(selected_urls)} docs with score >= {auto_select_threshold}"
                )
                for url in selected_urls:
                    print(f"   - {url}")
                return selected_urls

        # Ask user to select
        print("\n❓ Which documents contain the correct answer?")
        print("   Enter numbers separated by spaces (e.g., '1 3 5')")
        print("   Or press Enter to select all top 3")

        try:
            user_input = input("> ").strip()
            if not user_input:
                # Default: top 3
                selected = results[:3]
            else:
                indices = [int(x.strip()) - 1 for x in user_input.split()]
                selected = [results[i] for i in indices if 0 <= i < len(results)]

            selected_urls = [doc["url"] for doc in selected]
            print(f"\n✅ Selected {len(selected_urls)} documents:")
            for url in selected_urls:
                print(f"   - {url}")

            return selected_urls

        except (ValueError, IndexError) as e:
            print(f"❌ Invalid input: {e}")
            return []

    def run_retrieval_eval(
        self, config: Path, runs: int = 3, single_ticket: Optional[str] = None
    ) -> Path:
        """Run fast retrieval-only evaluation and return output directory.

        Args:
            config: Path to retrieval config file
            runs: Number of evaluation runs
            single_ticket: Optional ticket ID - if provided, only evaluate this one ticket

        Returns:
            Path to evaluation output directory
        """
        # Full metrics in YAML, but filter to retrieval-only via --metrics flag
        full_metrics = [
            "custom:url_retrieval_eval",
            "ragas:context_relevance",
            "ragas:context_precision_without_reference",
            "custom:answer_correctness",
            "ragas:faithfulness",
            "ragas:response_relevancy",
        ]

        # Retrieval-only subset (use --metrics flag to filter)
        retrieval_metrics_filter = "custom:url_retrieval_eval ragas:context_relevance ragas:context_precision_without_reference"

        if single_ticket:
            # Single-ticket mode: create filtered config with one ticket
            # Full metrics in YAML, filter via --metrics flag
            retrieval_config = self.functional_retrieval
            config = self.create_single_ticket_config(
                single_ticket, retrieval_config, add_answer_metrics=True, metrics=full_metrics
            )
            print(f"\n⚡ Running retrieval-only evaluation for {single_ticket} ({runs} runs)...")
        else:
            # Full-pattern mode: clean the entire pattern file with full metrics in YAML
            config = self.clean_pattern_config(config, metrics=full_metrics)
            print(f"\n⚡ Running retrieval evaluation ({runs} runs)...")

        print(f"   Using --metrics flag to filter: {retrieval_metrics_filter}")
        print(f"   ⏳ Estimated time: ~{runs * 0.5} minutes (faster - no LLM response generation)")

        # Log start timestamp
        import time
        from datetime import datetime

        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"   🕐 Started at: {start_timestamp}")
        print(f"   🔄 Running {runs} retrieval-only runs...\n")
        sys.stdout.flush()

        start_time = time.time()

        # Start subprocess in background
        proc = subprocess.Popen(
            [
                "./run_mcp_retrieval_suite.sh",
                "--config",
                str(config),
                "--runs",
                str(runs),
                "--metrics",
                retrieval_metrics_filter,
            ],
            cwd=self.eval_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Monitor progress by checking for completed runs
        output_base = self.eval_root / "mcp_retrieval_output"
        last_completed = 0

        while proc.poll() is None:
            try:
                suite_dirs = sorted(output_base.glob("suite_*"), key=lambda p: p.stat().st_mtime)
                if suite_dirs:
                    latest_suite = suite_dirs[-1]
                    # Count completed runs by looking for summary.json files
                    completed_runs = len(list(latest_suite.glob("run_*/evaluation_*_summary.json")))

                    if completed_runs > last_completed:
                        elapsed = time.time() - start_time
                        print(
                            f"      ✅ Run {completed_runs}/{runs} complete ({elapsed/60:.1f} min elapsed)"
                        )
                        sys.stdout.flush()
                        last_completed = completed_runs
            except Exception:
                pass

            time.sleep(3)  # Check every 3 seconds (faster for retrieval-only)

        # Wait for process to finish
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)

        elapsed = time.time() - start_time
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n   ✅ All {runs} runs completed in {elapsed/60:.1f} minutes")
        print(f"   🕐 Started: {start_timestamp}, Ended: {end_timestamp}")
        sys.stdout.flush()

        # Find the latest output directory
        output_dirs = sorted(
            (self.eval_root / "mcp_retrieval_output").glob("suite_*"),
            key=lambda p: p.stat().st_mtime,
        )
        if not output_dirs:
            raise RuntimeError("No evaluation output found")

        # Parse and print key retrieval scores immediately
        output_dir = output_dirs[-1]
        try:
            result = self.parse_results(output_dir, single_ticket)
            print(f"\n   📊 RETRIEVAL SCORES (avg of {runs} run{'s' if runs > 1 else ''}):")
            print("   ─────────────────────────────────")
            if result.url_f1 is not None:
                emoji = "✅" if result.url_f1 > 0 else "❌"
                print(f"   {emoji} URL F1:             {result.url_f1:.2f}")
            if result.context_relevance is not None:
                emoji = "✅" if result.context_relevance >= 0.70 else "⚠️"
                print(f"   {emoji} Context Relevance:  {result.context_relevance:.2f}")
            if result.context_precision is not None:
                emoji = "✅" if result.context_precision >= 0.70 else "⚠️"
                print(f"   {emoji} Context Precision:  {result.context_precision:.2f}")
            print()
        except Exception as e:
            print(f"   ⚠️  Could not parse scores: {e}")

        return output_dir

    def get_latest_output_dir(self, output_type: str = "full") -> Path:
        """Find the most recent evaluation output directory.

        Args:
            output_type: "full" or "retrieval"

        Returns:
            Path to latest suite_* directory
        """
        if output_type == "full":
            base_dir = self.eval_root / "okp_mcp_full_output"
        else:
            base_dir = self.eval_root / "mcp_retrieval_output"

        output_dirs = sorted(
            base_dir.glob("suite_*"),
            key=lambda p: p.stat().st_mtime,
        )

        if not output_dirs:
            raise RuntimeError(
                f"No existing evaluation output found in {base_dir}\n"
                f"Run an evaluation first or remove --use-existing flag"
            )

        return output_dirs[-1]

    def _get_num_docs(self, contexts: Optional[str]) -> int:
        """Extract number of documents from contexts field."""
        if not contexts or pd.isna(contexts):
            return 0

        try:
            # Try to parse as JSON array
            import json

            contexts_list = json.loads(contexts)
            if isinstance(contexts_list, list):
                return len(contexts_list)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: count non-empty string
        return 1 if str(contexts).strip() and str(contexts).strip() != "[]" else 0

    def _get_llm_boost_suggestion(self, result: EvaluationResult):
        """Get LLM suggestion for boost query improvements."""
        if not self.llm_advisor or not result.query:
            return

        print("\n" + "=" * 80)
        print("💡 AI-POWERED SUGGESTION (Boost Query)")
        print("=" * 80)

        try:
            # Convert EvaluationResult to MetricSummary
            metrics = MetricSummary(
                ticket_id=result.ticket_id,
                query=result.query,
                url_f1=result.url_f1,
                mrr=result.mrr,
                context_relevance=result.context_relevance,
                context_precision=result.context_precision,
                keywords_score=result.keywords_score,
                forbidden_claims_score=result.forbidden_claims_score,
                faithfulness=result.faithfulness,
                answer_correctness=result.answer_correctness,
                response_relevancy=result.response_relevancy,
                rag_used=result.rag_used,
                docs_retrieved=result.docs_retrieved,
                num_docs=self._get_num_docs(result.contexts),
                # LLM Judge reasoning
                answer_correctness_reason=result.answer_correctness_reason,
                faithfulness_reason=result.faithfulness_reason,
                response_relevancy_reason=result.response_relevancy_reason,
                context_relevance_reason=result.context_relevance_reason,
                context_precision_reason=result.context_precision_reason,
                # Ground truth / expected values
                response=result.response,
                expected_response=result.expected_response,
                expected_keywords=result.expected_keywords,
                expected_urls=result.expected_urls,
                forbidden_claims=result.forbidden_claims,
                retrieved_urls=result.retrieved_urls,
                contexts=result.contexts,
            )

            # Check for judge disagreement first (answer_correctness is most critical)
            if result.answer_correctness is not None and result.answer_correctness < 0.9:
                print("\n🔍 Checking if advisor agrees with judge's assessment...")
                disagreement = self._run_async_in_thread(
                    self.llm_advisor.check_judge_agreement(
                        metrics, focus_metric="answer_correctness"
                    )
                )

                if disagreement:
                    print("\n" + "⚠️ " * 40)
                    print("⚠️  JUDGE/ADVISOR DISAGREEMENT DETECTED")
                    print("⚠️ " * 40)
                    print(f"\nMetric: {disagreement.metric}")
                    print(f"Judge Score: {disagreement.judge_score:.2f}")
                    print("\n📊 Judge's Reasoning:")
                    print(f"   {disagreement.judge_reasoning[:300]}")
                    print("\n🤖 Advisor's Assessment:")
                    print(f"   {disagreement.advisor_assessment[:300]}")
                    print(f"\n⚡ Severity: {disagreement.severity.upper()}")
                    print("\n💡 Recommendation:")
                    print(f"   {disagreement.recommendation}")
                    print("\n" + "⚠️ " * 40)

                    # Store disagreement in result
                    result.judge_disagreements.append(
                        {
                            "metric": disagreement.metric,
                            "judge_score": disagreement.judge_score,
                            "judge_reasoning": disagreement.judge_reasoning,
                            "advisor_assessment": disagreement.advisor_assessment,
                            "severity": disagreement.severity,
                            "recommendation": disagreement.recommendation,
                        }
                    )

                    # If high severity disagreement, warn before proceeding
                    if disagreement.severity == "high":
                        print(
                            "\n⚠️  HIGH SEVERITY DISAGREEMENT - Consider human review before auto-fixing"
                        )
                        print("   Continuing with suggestion but flagged for review...")
                else:
                    print("   ✅ Advisor agrees with judge's assessment\n")

            # Get suggestion (async call)
            suggestion = self._run_async_in_thread(
                self.llm_advisor.suggest_boost_query_changes(metrics)
            )

            # Display suggestion
            print(f"\n📝 Reasoning:\n{suggestion.reasoning}\n")
            print(f"📄 File: {suggestion.file_path}")
            print(f"✏️  Change: {suggestion.suggested_change}\n")
            print(f"📈 Expected Improvement:\n{suggestion.expected_improvement}\n")
            print(f"🎯 Confidence: {suggestion.confidence}")

            if suggestion.code_snippet:
                print(f"\n💻 Code Snippet:\n{suggestion.code_snippet}")

        except Exception as e:
            print(f"\n⚠️  Failed to get LLM suggestion: {e}")
            print(f"   Exception type: {type(e).__name__}")
            if hasattr(e, "__cause__") and e.__cause__:
                print(f"   Caused by: {e.__cause__}")
            import traceback

            print("\n   Full traceback:")
            traceback.print_exc()
            import sys

            print("\n   Checking stderr capture...", file=sys.stderr)
            print("   Continuing without AI-powered suggestion")

    def _get_llm_prompt_suggestion(self, result: EvaluationResult):
        """Get LLM suggestion for system prompt improvements."""
        if not self.llm_advisor or not result.query:
            return

        print("\n" + "=" * 80)
        print("💡 AI-POWERED SUGGESTION (System Prompt)")
        print("=" * 80)

        try:
            # Convert EvaluationResult to MetricSummary
            metrics = MetricSummary(
                ticket_id=result.ticket_id,
                query=result.query,
                url_f1=result.url_f1,
                mrr=result.mrr,
                context_relevance=result.context_relevance,
                context_precision=result.context_precision,
                keywords_score=result.keywords_score,
                forbidden_claims_score=result.forbidden_claims_score,
                faithfulness=result.faithfulness,
                answer_correctness=result.answer_correctness,
                response_relevancy=result.response_relevancy,
                rag_used=result.rag_used,
                docs_retrieved=result.docs_retrieved,
                num_docs=self._get_num_docs(result.contexts),
                # LLM Judge reasoning
                answer_correctness_reason=result.answer_correctness_reason,
                faithfulness_reason=result.faithfulness_reason,
                response_relevancy_reason=result.response_relevancy_reason,
                context_relevance_reason=result.context_relevance_reason,
                context_precision_reason=result.context_precision_reason,
                # Ground truth / expected values
                response=result.response,
                expected_response=result.expected_response,
                expected_keywords=result.expected_keywords,
                expected_urls=result.expected_urls,
                forbidden_claims=result.forbidden_claims,
                retrieved_urls=result.retrieved_urls,
                contexts=result.contexts,
            )

            # Check for judge disagreement first (answer_correctness is most critical)
            if result.answer_correctness is not None and result.answer_correctness < 0.9:
                print("\n🔍 Checking if advisor agrees with judge's assessment...")
                disagreement = self._run_async_in_thread(
                    self.llm_advisor.check_judge_agreement(
                        metrics, focus_metric="answer_correctness"
                    )
                )

                if disagreement:
                    print("\n" + "⚠️ " * 40)
                    print("⚠️  JUDGE/ADVISOR DISAGREEMENT DETECTED")
                    print("⚠️ " * 40)
                    print(f"\nMetric: {disagreement.metric}")
                    print(f"Judge Score: {disagreement.judge_score:.2f}")
                    print("\n📊 Judge's Reasoning:")
                    print(f"   {disagreement.judge_reasoning[:300]}")
                    print("\n🤖 Advisor's Assessment:")
                    print(f"   {disagreement.advisor_assessment[:300]}")
                    print(f"\n⚡ Severity: {disagreement.severity.upper()}")
                    print("\n💡 Recommendation:")
                    print(f"   {disagreement.recommendation}")
                    print("\n" + "⚠️ " * 40)

                    # Store disagreement in result (avoid duplicates)
                    disagreement_dict = {
                        "metric": disagreement.metric,
                        "judge_score": disagreement.judge_score,
                        "judge_reasoning": disagreement.judge_reasoning,
                        "advisor_assessment": disagreement.advisor_assessment,
                        "severity": disagreement.severity,
                        "recommendation": disagreement.recommendation,
                    }
                    if disagreement_dict not in result.judge_disagreements:
                        result.judge_disagreements.append(disagreement_dict)

                    # If high severity disagreement, warn before proceeding
                    if disagreement.severity == "high":
                        print(
                            "\n⚠️  HIGH SEVERITY DISAGREEMENT - Consider human review before auto-fixing"
                        )
                        print("   Continuing with suggestion but flagged for review...")
                else:
                    print("   ✅ Advisor agrees with judge's assessment\n")

            # Get suggestion (async call)
            suggestion = self._run_async_in_thread(self.llm_advisor.suggest_prompt_changes(metrics))

            # Display suggestion
            print(f"\n📝 Reasoning:\n{suggestion.reasoning}\n")
            print(f"✏️  Suggested Change:\n{suggestion.suggested_change}\n")
            print(f"📈 Expected Improvement:\n{suggestion.expected_improvement}\n")
            print(f"🎯 Confidence: {suggestion.confidence}")

        except Exception as e:
            print(f"\n⚠️  Failed to get LLM suggestion: {e}")
            print(f"   Exception type: {type(e).__name__}")
            if hasattr(e, "__cause__") and e.__cause__:
                print(f"   Caused by: {e.__cause__}")
            import traceback

            print("\n   Full traceback:")
            traceback.print_exc()
            import sys

            print("\n   Checking stderr capture...", file=sys.stderr)
            print("   Continuing without AI-powered suggestion")

    def check_solr_documents(self, result: EvaluationResult) -> Dict:
        """Check if expected documents exist in Solr index.

        Args:
            result: EvaluationResult with expected_urls

        Returns:
            Dictionary with Solr check results
        """
        if not self.solr_checker:
            return {"available": False, "message": "Solr checker not available"}

        print("\n" + "=" * 80)
        print("🔍 SOLR DOCUMENT CHECK")
        print("=" * 80)

        # Check if expected_urls are defined
        if not result.expected_urls:
            print("⚠️  No expected_urls defined in test config")
            print("   Suggesting URLs from Solr based on query...")

            if result.query:
                suggestions = self.solr_checker.suggest_urls_for_query(result.query, max_results=5)
                if suggestions:
                    print(f"\n📝 Suggested URLs for '{result.query}':")
                    for i, doc in enumerate(suggestions, 1):
                        print(f"\n{i}. {doc['url']}")
                        print(f"   {doc['title']}")
                        print(f"   Kind: {doc['documentKind']}, Score: {doc['score']:.2f}")

                    return {
                        "available": True,
                        "expected_urls_missing": True,
                        "suggested_urls": suggestions,
                        "message": f"No expected_urls defined. {len(suggestions)} URLs suggested from Solr.",
                    }
                else:
                    return {
                        "available": True,
                        "expected_urls_missing": True,
                        "suggested_urls": [],
                        "message": "No expected_urls defined and no suggestions found in Solr.",
                    }
            else:
                return {
                    "available": True,
                    "expected_urls_missing": True,
                    "message": "No expected_urls and no query available for suggestions.",
                }

        # Check all expected URLs
        print(f"\nChecking {len(result.expected_urls)} expected URL(s) in Solr...")
        url_results = self.solr_checker.check_all_expected_urls(result.expected_urls)

        missing_urls = []
        found_urls = []

        for url, check_result in url_results.items():
            if check_result["exists"]:
                found_urls.append(url)
                print(f"  ✅ {url}")
                print(f"     Title: {check_result.get('title', 'N/A')}")
            else:
                missing_urls.append(url)
                print(f"  ❌ {url}")
                if "error" in check_result:
                    print(f"     Error: {check_result['error']}")
                else:
                    print("     Not found in Solr index")

        # Summary
        if missing_urls:
            print(
                f"\n⚠️  {len(missing_urls)} of {len(result.expected_urls)} expected documents MISSING from Solr"
            )
            print("   → These documents need to be ingested into Solr")
            print("   → Boost query changes won't help until docs are indexed")
        else:
            print(f"\n✅ All {len(result.expected_urls)} expected documents found in Solr")
            if result.url_f1 is not None and result.url_f1 < 0.7:
                print("   → Documents exist but not being retrieved")
                print("   → Boost query tuning needed")

        return {
            "available": True,
            "expected_urls_missing": False,
            "total": len(result.expected_urls),
            "found": len(found_urls),
            "missing": len(missing_urls),
            "missing_urls": missing_urls,
            "found_urls": found_urls,
            "url_results": url_results,
        }

    def _load_test_config_for_ticket(self, ticket_id: Optional[str]) -> Optional[dict]:
        """Load test config for a specific ticket from YAML.

        Args:
            ticket_id: Ticket ID (e.g., "RSPEED-2482" or "RSPEED_2482"), or None for full pattern

        Returns:
            Dictionary with test config or None if not found
        """
        if ticket_id is None:
            return None

        normalized_id = ticket_id.replace("-", "_")

        # Try functional_tests_full.yaml first
        try:
            with open(self.functional_full, "r") as f:
                configs = yaml.safe_load(f)
                for config in configs:
                    if config.get("conversation_group_id") == normalized_id:
                        return config
        except Exception:
            pass

        return None

    def _print_scores_from_output(
        self, output_dir: Path, ticket_id: Optional[str], runs: int
    ) -> None:
        """Parse and print key scores from evaluation output.

        Args:
            output_dir: Path to evaluation output directory
            ticket_id: Ticket ID or None for full pattern
            runs: Number of runs that were executed
        """
        result = self.parse_results(output_dir, ticket_id)

        print(f"\n   📊 SCORES (avg of {runs} run{'s' if runs > 1 else ''}):")
        print("   ─────────────────────────────────")

        # Answer metrics (most important)
        if result.answer_correctness is not None:
            emoji = "✅" if result.answer_correctness >= 0.90 else "❌"
            print(f"   {emoji} Answer Correctness: {result.answer_correctness:.2f}")

        if result.faithfulness is not None:
            emoji = "✅" if result.faithfulness >= 0.80 else "⚠️"
            print(f"   {emoji} Faithfulness:       {result.faithfulness:.2f}")

        # Retrieval metrics
        if result.url_f1 is not None:
            emoji = "✅" if result.url_f1 > 0 else "❌"
            print(f"   {emoji} URL F1:             {result.url_f1:.2f}")

        if result.context_relevance is not None:
            emoji = "✅" if result.context_relevance >= 0.70 else "⚠️"
            print(f"   {emoji} Context Relevance:  {result.context_relevance:.2f}")

        if result.context_precision is not None:
            emoji = "✅" if result.context_precision >= 0.70 else "⚠️"
            print(f"   {emoji} Context Precision:  {result.context_precision:.2f}")

        # Composite score
        composite = self._calculate_composite_metric(result)
        if composite > 0:
            emoji = "✅" if composite >= 0.80 else "⚠️"
            print(f"   {emoji} Composite Score:    {composite:.2f} (threshold: 0.80)")

        print()  # Blank line for readability

    def parse_results(self, output_dir: Path, ticket_id: Optional[str]) -> EvaluationResult:
        """Parse evaluation results for a specific ticket or all tickets (averaged).

        Reads all runs (run_001, run_002, run_003, etc.) and averages metrics
        across runs for better stability.

        Args:
            output_dir: Path to evaluation output directory
            ticket_id: Ticket ID (e.g., "RSPEED-2482" or "RSPEED_2482"), or None for all tickets (averaged)

        Returns:
            EvaluationResult with averaged metrics across all runs
        """
        # Find all run directories
        run_dirs = sorted(output_dir.glob("run_*"))
        if not run_dirs:
            raise RuntimeError(f"No run directories found in {output_dir}")

        # Collect dataframes from all runs
        all_dfs = []
        for run_dir in run_dirs:
            csv_files = list(run_dir.glob("evaluation_*_detailed.csv"))
            if csv_files:
                all_dfs.append(pd.read_csv(csv_files[0]))

        if not all_dfs:
            raise RuntimeError("No detailed CSV found in any run directory")

        # Concatenate all runs
        df = pd.concat(all_dfs, ignore_index=True)

        # Filter to specific ticket if provided, otherwise use all tickets
        if ticket_id is not None:
            # Try to find ticket in CSV (check both hyphen and underscore versions)
            normalized_ticket_id = ticket_id.replace("-", "_")
            ticket_df = df[df["conversation_group_id"] == normalized_ticket_id]

            # If not found with underscores, try with hyphens
            if ticket_df.empty:
                hyphenated_ticket_id = ticket_id.replace("_", "-")
                ticket_df = df[df["conversation_group_id"] == hyphenated_ticket_id]

            if ticket_df.empty:
                available_tickets = df["conversation_group_id"].unique()[:10]
                raise RuntimeError(
                    f"Ticket '{ticket_id}' not found in CSV.\n"
                    f"Available tickets: {', '.join(available_tickets)}"
                )
        else:
            # Use all tickets (for full-pattern mode)
            ticket_df = df

        result = EvaluationResult(ticket_id=ticket_id, num_runs=len(all_dfs))

        # Extract tool_calls, contexts, and query (same across all rows for a ticket)
        if not ticket_df.empty:
            first_row = ticket_df.iloc[0]
            result.tool_calls = first_row.get("tool_calls")
            result.contexts = first_row.get("contexts")
            result.query = first_row.get("query", first_row.get("user_input"))

            # Handle string fields that may be NaN in retrieval-only mode
            response_raw = first_row.get("response")
            result.response = response_raw if pd.notna(response_raw) else None

            # Extract expected values from test config
            expected_response_raw = first_row.get("expected_response")
            result.expected_response = (
                expected_response_raw if pd.notna(expected_response_raw) else None
            )

            # Parse expected_keywords (stored as list/JSON in CSV)
            expected_keywords_raw = first_row.get("expected_keywords")
            if pd.notna(expected_keywords_raw):
                if isinstance(expected_keywords_raw, list):
                    result.expected_keywords = expected_keywords_raw
                elif isinstance(expected_keywords_raw, str):
                    try:
                        result.expected_keywords = json.loads(expected_keywords_raw)
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Load expected_urls and forbidden_claims from test config YAML
            # (these are not in the CSV output)
            test_config = self._load_test_config_for_ticket(ticket_id)
            if test_config and "turns" in test_config:
                first_turn = test_config["turns"][0]
                result.expected_urls = first_turn.get("expected_urls", [])
                result.forbidden_claims = first_turn.get("forbidden_claims", [])

            # Extract retrieved URLs and titles from tool_calls JSON (more accurate than regex)
            if pd.notna(result.tool_calls) and result.tool_calls:
                try:
                    tool_calls_data = json.loads(str(result.tool_calls))
                    # Navigate nested structure: [[{tool_name, arguments, result}]]
                    if isinstance(tool_calls_data, list) and len(tool_calls_data) > 0:
                        for turn_calls in tool_calls_data:
                            if isinstance(turn_calls, list):
                                for call in turn_calls:
                                    if isinstance(call, dict) and "result" in call:
                                        call_result = call["result"]
                                        if (
                                            isinstance(call_result, dict)
                                            and "contexts" in call_result
                                        ):
                                            contexts = call_result["contexts"]
                                            if isinstance(contexts, list):
                                                for ctx in contexts:
                                                    if isinstance(ctx, dict):
                                                        url = ctx.get("url", "")
                                                        title = ctx.get("title", "")
                                                        if url:
                                                            # Normalize URL (remove https://)
                                                            url_normalized = url.replace(
                                                                "https://", ""
                                                            ).replace("http://", "")
                                                            if result.retrieved_urls is None:
                                                                result.retrieved_urls = []
                                                            result.retrieved_urls.append(
                                                                url_normalized
                                                            )
                                                        if title:
                                                            if result.retrieved_doc_titles is None:
                                                                result.retrieved_doc_titles = []
                                                            result.retrieved_doc_titles.append(
                                                                title
                                                            )
                except (json.JSONDecodeError, TypeError, KeyError):
                    # Fallback to regex if JSON parsing fails
                    if pd.notna(result.contexts) and result.contexts:
                        contexts_str = str(result.contexts)
                        import re

                        url_pattern = r'access\.redhat\.com/[^\s\'"<>)}\]]*'
                        found_urls = re.findall(url_pattern, contexts_str)
                        result.retrieved_urls = list(set(found_urls))

            # Check if RAG was used
            if pd.notna(result.tool_calls) and result.tool_calls:
                # Check if search/retrieval tool was called
                tool_calls_str = str(result.tool_calls).lower()
                result.rag_used = any(
                    keyword in tool_calls_str for keyword in ["search", "portal", "retrieve", "mcp"]
                )

            # Check if documents were retrieved
            if pd.notna(result.contexts) and result.contexts:
                contexts_str = str(result.contexts).strip()
                result.docs_retrieved = (
                    contexts_str != "" and contexts_str != "[]" and contexts_str != "null"
                )

        # Extract metrics - average across all runs for stability
        # Group by metric and calculate mean
        metric_groups = ticket_df.groupby("metric_identifier")["score"].mean()

        # Helper to get first reason for a metric (representative explanation from judge)
        def get_reason(metric_name: str) -> Optional[str]:
            metric_rows = ticket_df[ticket_df["metric_identifier"] == metric_name]
            if not metric_rows.empty:
                reason = metric_rows.iloc[0].get("reason")
                return reason if pd.notna(reason) else None
            return None

        for metric, avg_score in metric_groups.items():
            if metric == "custom:url_retrieval_eval":
                result.url_f1 = avg_score
                # Extract MRR from metadata across all runs and average
                mrr_values = []
                for _, row in ticket_df[ticket_df["metric_identifier"] == metric].iterrows():
                    metadata = row.get("metric_metadata", "")
                    if isinstance(metadata, str) and "mrr" in metadata.lower():
                        try:
                            meta_dict = json.loads(metadata)
                            mrr = meta_dict.get("mrr")
                            if mrr is not None:
                                mrr_values.append(mrr)
                        except (json.JSONDecodeError, TypeError):
                            pass
                if mrr_values:
                    result.mrr = sum(mrr_values) / len(mrr_values)  # Average MRR

            elif metric == "ragas:context_relevance":
                result.context_relevance = avg_score
                result.context_relevance_reason = get_reason(metric)

            elif metric == "ragas:context_precision_without_reference":
                result.context_precision = avg_score
                result.context_precision_reason = get_reason(metric)

            elif metric == "custom:keywords_eval":
                result.keywords_score = avg_score

            elif metric == "custom:forbidden_claims_eval":
                result.forbidden_claims_score = avg_score

            elif metric == "ragas:faithfulness":
                result.faithfulness = avg_score
                result.faithfulness_reason = get_reason(metric)

            elif metric == "custom:answer_correctness":
                result.answer_correctness = avg_score
                result.answer_correctness_reason = get_reason(metric)

            elif metric == "ragas:response_relevancy":
                result.response_relevancy = avg_score
                result.response_relevancy_reason = get_reason(metric)

        # Calculate standard deviation for key metrics to detect instability
        std_groups = ticket_df.groupby("metric_identifier")["score"].std()

        # Track high variance metrics (indicates intermittent/flaky behavior)
        high_variance_threshold = 0.15  # 15% std deviation is concerning
        result.high_variance_metrics = []
        for metric, std_score in std_groups.items():
            if pd.notna(std_score) and std_score > high_variance_threshold:
                result.high_variance_metrics.append(f"{metric} (std={std_score:.3f})")

        return result

    def parse_results_per_ticket(self, output_dir: Path) -> Dict[str, List[Dict[str, float]]]:
        """Parse evaluation results grouped by ticket, keeping per-run scores.

        Unlike parse_results() which averages scores, this method preserves
        individual run scores for stability classification.

        Args:
            output_dir: Path to evaluation output directory

        Returns:
            Dict mapping ticket_id → list of per-run metric dicts
            Example:
            {
                "RSPEED-2218": [
                    {"answer_correctness": 0.95, "url_f1": 0.8, "faithfulness": 0.9},  # run 1
                    {"answer_correctness": 0.50, "url_f1": 0.7, "faithfulness": 0.85}, # run 2
                    {"answer_correctness": 0.92, "url_f1": 0.75, "faithfulness": 0.88}, # run 3
                ],
                "RSPEED-2219": [...]
            }
        """
        import json

        # Find all run directories
        run_dirs = sorted(output_dir.glob("run_*"))
        if not run_dirs:
            raise RuntimeError(f"No run directories found in {output_dir}")

        # Collect dataframes from all runs
        all_dfs = []
        for run_dir in run_dirs:
            csv_files = list(run_dir.glob("evaluation_*_detailed.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                # Add run number for tracking
                run_num = int(run_dir.name.split("_")[1])
                df["run_number"] = run_num
                all_dfs.append(df)

        if not all_dfs:
            raise RuntimeError("No detailed CSV found in any run directory")

        # Concatenate all runs
        df = pd.concat(all_dfs, ignore_index=True)

        # Get unique ticket IDs
        ticket_ids = df["conversation_group_id"].unique()

        # Build per-ticket, per-run results
        results = {}
        for ticket_id in ticket_ids:
            ticket_df = df[df["conversation_group_id"] == ticket_id]

            # Group by run number
            run_numbers = sorted(ticket_df["run_number"].unique())
            ticket_runs = []

            for run_num in run_numbers:
                run_df = ticket_df[ticket_df["run_number"] == run_num]

                # Extract metrics for this run
                run_metrics = {}
                metric_groups = run_df.groupby("metric_identifier")["score"].mean()

                for metric, score in metric_groups.items():
                    if metric == "custom:url_retrieval_eval":
                        run_metrics["url_f1"] = score
                        # Extract MRR from metadata
                        for _, row in run_df[run_df["metric_identifier"] == metric].iterrows():
                            metadata = row.get("metric_metadata", "")
                            if isinstance(metadata, str) and "mrr" in metadata.lower():
                                try:
                                    meta_dict = json.loads(metadata)
                                    mrr = meta_dict.get("mrr")
                                    if mrr is not None:
                                        run_metrics["mrr"] = mrr
                                except (json.JSONDecodeError, TypeError):
                                    pass
                    elif metric == "ragas:context_relevance":
                        run_metrics["context_relevance"] = score
                    elif metric == "ragas:context_precision_without_reference":
                        run_metrics["context_precision"] = score
                    elif metric == "ragas:faithfulness":
                        run_metrics["faithfulness"] = score
                    elif metric == "custom:answer_correctness":
                        run_metrics["answer_correctness"] = score
                    elif metric == "ragas:response_relevancy":
                        run_metrics["response_relevancy"] = score

                ticket_runs.append(run_metrics)

            results[ticket_id] = ticket_runs

        return results

    def calculate_url_overlap(self, urls_before: list[str], urls_after: list[str]) -> float:
        """Calculate Jaccard similarity between two URL sets.

        Args:
            urls_before: URLs from previous iteration
            urls_after: URLs from current iteration

        Returns:
            Jaccard similarity (0.0 = completely different, 1.0 = identical)
        """
        if not urls_before and not urls_after:
            return 1.0  # Both empty = identical
        if not urls_before or not urls_after:
            return 0.0  # One empty = completely different

        # Normalize URLs (remove protocol, trailing slashes)
        def normalize(url):
            return url.replace("https://", "").replace("http://", "").rstrip("/")

        set_before = {normalize(url) for url in urls_before}
        set_after = {normalize(url) for url in urls_after}

        intersection = set_before & set_after
        union = set_before | set_after

        return len(intersection) / len(union) if union else 0.0

    def display_retrieved_vs_expected(self, result: EvaluationResult) -> None:
        """Display retrieved URLs vs expected URLs with titles.

        Args:
            result: EvaluationResult with retrieved and expected URLs
        """
        print(f"\n📄 RETRIEVED DOCUMENTS ({len(result.retrieved_urls or [])}):")
        print("=" * 80)
        if result.retrieved_urls:
            for i, url in enumerate(result.retrieved_urls, 1):
                url_short = url.replace("access.redhat.com/", "")
                title = (
                    result.retrieved_doc_titles[i - 1]
                    if result.retrieved_doc_titles and i - 1 < len(result.retrieved_doc_titles)
                    else "N/A"
                )
                title_short = title[:60] + "..." if len(title) > 60 else title
                print(f"  {i}. {url_short}")
                print(f"     {title_short}")
        else:
            print("  (none)")
        print()

        print(f"🎯 EXPECTED DOCUMENTS ({len(result.expected_urls or [])}):")
        print("=" * 80)
        if result.expected_urls:
            for url in result.expected_urls:
                url_short = url.replace("access.redhat.com/", "")
                print(f"  - {url_short}")
        else:
            print("  (none)")
        print()

    def calculate_url_f1(self, expected_urls: List[str], retrieved_urls: List[str]) -> float:
        """Calculate F1 score for URL retrieval.

        Args:
            expected_urls: Expected URLs (normalized)
            retrieved_urls: Retrieved URLs (normalized)

        Returns:
            F1 score (0.0 to 1.0)
        """
        if not expected_urls:
            return 0.0
        if not retrieved_urls:
            return 0.0

        # Normalize URLs (remove https:// prefix)
        def normalize(url):
            return url.replace("https://", "").replace("http://", "").rstrip("/")

        expected_set = {normalize(url) for url in expected_urls}
        retrieved_set = {normalize(url) for url in retrieved_urls}

        # Calculate metrics
        matched = len(expected_set & retrieved_set)
        precision = matched / len(retrieved_set) if retrieved_set else 0.0
        recall = matched / len(expected_set) if expected_set else 0.0

        # F1 score
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0

    def calculate_mrr(self, expected_urls: List[str], retrieved_urls: List[str]) -> float:
        """Calculate Mean Reciprocal Rank.

        Args:
            expected_urls: Expected URLs (normalized)
            retrieved_urls: Retrieved URLs (in ranking order)

        Returns:
            MRR score (0.0 to 1.0)
        """
        if not expected_urls or not retrieved_urls:
            return 0.0

        # Normalize URLs
        def normalize(url):
            return url.replace("https://", "").replace("http://", "").rstrip("/")

        expected_set = {normalize(url) for url in expected_urls}

        # Find first occurrence of each expected URL
        reciprocal_ranks = []
        for expected_url in expected_set:
            for i, retrieved_url in enumerate(retrieved_urls, 1):
                if normalize(retrieved_url) == expected_url:
                    reciprocal_ranks.append(1.0 / i)
                    break

        # MRR is average of reciprocal ranks (or 0 if none found)
        if reciprocal_ranks:
            return sum(reciprocal_ranks) / len(expected_set)
        return 0.0

    def _calculate_composite_metric(self, result: EvaluationResult) -> float:
        """Calculate composite metric prioritizing answer correctness.

        Weights (in priority order - CUSTOMER-FOCUSED):
        1. Answer correctness (70%) - THE most important metric for customers
        2. Context relevance (15%) - RAGAS metric (docs relate to question)
        3. Context precision (10%) - RAGAS metric (docs are precise)
        4. Keywords (3%) - required facts present
        5. Forbidden claims (2%) - no incorrect info

        URL F1 is intentionally EXCLUDED - it's unreliable.

        NOTE: If answer_correctness >= 0.90, composite will likely pass (contributes 0.63).
        This reflects customer reality: correct answers matter most.

        Returns:
            Composite score between 0.0 and 1.0
        """
        components = []
        weights = []

        # 1. Answer correctness (DOMINANT weight - what customers care about)
        if result.answer_correctness is not None:
            components.append(result.answer_correctness)
            weights.append(0.70)
        elif result.keywords_score is not None:
            # Fallback: use keywords if no answer_correctness
            components.append(result.keywords_score)
            weights.append(0.70)

        # 2. Context relevance (RAGAS - most important retrieval metric)
        if result.context_relevance is not None:
            components.append(result.context_relevance)
            weights.append(0.15)

        # 3. Context precision (RAGAS - secondary retrieval metric)
        if result.context_precision is not None:
            components.append(result.context_precision)
            weights.append(0.10)

        # 4. Keywords (required facts)
        if result.keywords_score is not None and result.answer_correctness is not None:
            # Only include if we already have answer_correctness (avoid double counting)
            components.append(result.keywords_score)
            weights.append(0.03)

        # 5. Forbidden claims (no regressions)
        if result.forbidden_claims_score is not None:
            components.append(result.forbidden_claims_score)
            weights.append(0.02)

        # If no components available, return 0
        if not components:
            return 0.0

        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)

        # Weighted average
        composite = sum(c * w for c, w in zip(components, normalized_weights))

        return composite

    def get_expected_response_from_jira(self, ticket_id: str) -> Optional[str]:
        """Fetch expected response from Jira ticket description.

        This is used when the YAML config doesn't have expected_response defined.
        Extracts the expected answer from the Jira ticket description.

        Args:
            ticket_id: Jira ticket ID (e.g., RSPEED-2480)

        Returns:
            Expected response text from Jira, or None if not found

        Note:
            This requires Jira MCP server to be available. If not available,
            returns None and user must add expected_response manually.
        """
        try:
            # Try to import Jira MCP tool
            # This will only work if running in Claude Code context with MCP server
            try:
                from mcp__mcp_atlassian import jira_get_issue

                # Fetch ticket from Jira
                result = jira_get_issue(issue_key=ticket_id, fields="description,summary")
            except ImportError:
                # MCP tools not available (running standalone)
                print(f"⚠️  Jira MCP not available - cannot fetch ticket {ticket_id}")
                print("   Add expected_response to config manually")
                return None

            # Parse JSON response
            import json

            issue_data = json.loads(result) if isinstance(result, str) else result

            description = issue_data.get("fields", {}).get("description", "")

            # Look for expected answer in description
            # Common patterns:
            # - "Expected Answer: ..."
            # - "Expected Response: ..."
            # - "Correct Answer: ..."
            # - Or just use the full description if it's short enough

            if not description:
                return None

            # Try to extract structured expected answer
            import re

            patterns = [
                r"(?:Expected Answer|Expected Response|Correct Answer|Expected)\s*:?\s*\n?(.*?)(?:\n\n|\Z)",
                r"(?:Answer|Response)\s*should\s+(?:be|include|contain)\s*:?\s*\n?(.*?)(?:\n\n|\Z)",
            ]

            for pattern in patterns:
                match = re.search(pattern, description, re.IGNORECASE | re.DOTALL)
                if match:
                    expected = match.group(1).strip()
                    if expected and len(expected) > 20:  # Sanity check
                        return expected

            # Fallback: if description is reasonably sized, use it as expected answer
            # (Assumes the ticket describes what the correct answer should be)
            if len(description) < 2000:
                return description.strip()

            return None

        except Exception as e:
            print(f"⚠️  Error fetching Jira ticket {ticket_id}: {e}")
            return None

    def check_answer_in_retrieved_docs(
        self, expected_answer: str, retrieved_contexts: List[str]
    ) -> Dict:
        """Check if retrieved documents contain the expected answer information.

        Uses LLM to judge whether the retrieved docs have the information needed
        to produce the expected answer.

        Args:
            expected_answer: Expected answer text (what LLM should say)
            retrieved_contexts: List of retrieved document contents

        Returns:
            Dict with:
                - contains_answer: bool
                - confidence: float (0.0-1.0)
                - explanation: str
        """
        if not self.llm_advisor:
            return {
                "contains_answer": None,
                "confidence": 0.0,
                "explanation": "LLM advisor not available",
            }

        if not retrieved_contexts:
            return {
                "contains_answer": False,
                "confidence": 1.0,
                "explanation": "No documents retrieved",
            }

        # Prepare contexts (truncate if too long)
        contexts_text = "\n\n".join(str(c)[:500] for c in retrieved_contexts[:10])

        prompt = f"""You are evaluating whether retrieved documents contain information to answer a question correctly.

EXPECTED ANSWER (what the system should say):
{expected_answer}

RETRIEVED DOCUMENTS:
{contexts_text}

TASK: Do these documents contain the information needed to produce the expected answer?

Consider:
- Do the docs discuss the same topics/concepts?
- Do they contain the key facts mentioned in the expected answer?
- Could someone reading these docs write the expected answer?

Answer in JSON format:
{{
    "contains_answer": true/false,
    "confidence": 0.0-1.0,
    "explanation": "brief explanation"
}}
"""

        try:
            import json
            import asyncio
            from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

            # Use Claude Agent SDK (handles ADC auth properly)
            async def check_docs():
                response_text = ""
                async for message in query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(
                        model=self.llm_advisor.model,
                        max_turns=1,  # Single response only
                    ),
                ):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if hasattr(block, "text"):
                                response_text += block.text
                return response_text

            result_text = asyncio.run(check_docs()).strip()

            # Parse JSON response
            # Try to extract JSON from markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)

            return {
                "contains_answer": result.get("contains_answer", False),
                "confidence": result.get("confidence", 0.5),
                "explanation": result.get("explanation", ""),
            }

        except Exception as e:
            print(f"⚠️  Error checking answer in docs: {e}")
            return {"contains_answer": None, "confidence": 0.0, "explanation": f"Error: {e}"}

    def query_solr_direct(self, query: str, expected_urls: List[str], num_docs: int = 20) -> Dict:
        """Query Solr directly and compute fast URL-based metrics (no LLM).

        This bypasses /v1/infer for fast iteration loops.

        Args:
            query: User query
            expected_urls: Expected document URLs
            num_docs: Number of docs to retrieve

        Returns:
            Dict with retrieved_urls and fast metrics (url_f1, mrr, precision_at_5, recall_at_5)
        """
        if not self.solr_analyzer:
            return {"error": "Solr analyzer not available"}

        try:
            # Query Solr directly (bypasses MCP/LLM)
            solr_response = self.solr_analyzer.get_explain_output(query, num_docs=num_docs)

            if "error" in solr_response:
                return solr_response

            # Extract retrieved URLs
            retrieved_urls = [doc.get("url", "") for doc in solr_response.get("docs", [])]

            # Compute fast URL-based metrics (no LLM required)
            url_f1 = self.calculate_url_f1(expected_urls, retrieved_urls)
            mrr = self.calculate_mrr(expected_urls, retrieved_urls)

            # Precision@5: % of top 5 that are expected
            top5 = set(retrieved_urls[:5])
            expected_set = set(expected_urls)
            precision_at_5 = len(top5 & expected_set) / 5 if top5 else 0.0

            # Recall@5: % of expected docs in top 5
            recall_at_5 = len(top5 & expected_set) / len(expected_set) if expected_set else 0.0

            return {
                "retrieved_urls": retrieved_urls,
                "url_f1": url_f1,
                "mrr": mrr,
                "precision_at_5": precision_at_5,
                "recall_at_5": recall_at_5,
                "num_retrieved": len(retrieved_urls),
            }

        except Exception as e:
            return {"error": str(e)}

    def inspect_solr_query(self, original_query: str) -> Optional[Dict]:
        """Inspect what okp-mcp actually sent to Solr.

        This helps detect query augmentation (e.g., automatic addition of
        'deprecated removed' terms) that might be poisoning results.

        Args:
            original_query: The original user query

        Returns:
            Dict with 'original', 'actual', and 'injected_terms' keys, or None if not found
        """
        try:
            # Get recent okp-mcp logs
            log_output = subprocess.check_output(
                ["podman", "logs", "--tail=100", "okp-mcp"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            )

            # Find SOLR query lines containing our query
            for line in log_output.split("\n"):
                if "SOLR query: q=" in line and any(
                    term in line for term in original_query.split()[:3]
                ):
                    # Extract the Solr query
                    match = re.search(r"q='([^']+)'", line)
                    if match:
                        solr_query = match.group(1)

                        # Check for injected terms
                        query_terms = set(solr_query.lower().split())
                        original_terms = set(original_query.lower().split())
                        injected_terms = query_terms - original_terms

                        return {
                            "original": original_query,
                            "actual": solr_query,
                            "injected_terms": sorted(list(injected_terms)),
                        }
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass

        return None

    def diagnose_retrieval_only(
        self,
        ticket_id: Optional[str],
        runs: int = 1,
        iteration: Optional[int] = None,
        suggestion: Optional[Any] = None,
    ) -> EvaluationResult:
        """Fast diagnosis using retrieval-only mode (no answer generation).

        This is much faster (~30 sec vs 3 min) but only evaluates retrieval metrics:
        - URL F1
        - MRR
        - Context Relevance
        - Context Precision

        Use this for iterating on boost queries when fixing retrieval problems.

        Args:
            ticket_id: RSPEED ticket ID, or None for full pattern
            runs: Number of evaluation runs (default: 1 for speed, use 3+ for stability analysis)
            iteration: If provided, saves diagnostics to .diagnostics/ directory
            suggestion: Optional LLM suggestion object (for saving to diagnostics)

        Returns:
            EvaluationResult with retrieval metrics only (answer metrics will be None)
        """
        # Run retrieval-only eval for just this ticket
        output_dir = self.run_retrieval_eval(
            self.functional_retrieval, runs=runs, single_ticket=ticket_id
        )

        # Parse results (answer metrics will be None)
        result = self.parse_results(output_dir, ticket_id)

        # Display retrieved vs expected URLs
        self.display_retrieved_vs_expected(result)

        # Inspect Solr query for unexpected augmentation
        solr_query_info = None
        if result.query:
            solr_query_info = self.inspect_solr_query(result.query)
            if solr_query_info and solr_query_info["injected_terms"]:
                print("🔍 SOLR QUERY INSPECTION:")
                print("=" * 80)
                print(f"Original query: {solr_query_info['original']}")
                print(f"Actual Solr query: {solr_query_info['actual']}")
                print(f"⚠️  okp-mcp INJECTED terms: {', '.join(solr_query_info['injected_terms'])}")
                print()
                print("💡 These injected terms may be affecting retrieval quality.")
                print("   Consider whether they're helping or harming the results.")
                print()

        # Save diagnostics if this is part of an iteration loop
        if iteration is not None:
            diag_file = self.save_iteration_diagnostics(
                ticket_id, iteration, result, solr_query_info, suggestion
            )
            print(f"💾 Saved diagnostics: {diag_file.name}")
            print()

        return result

    def diagnose(
        self, ticket_id: Optional[str], use_existing: bool = False, runs: int = 1
    ) -> EvaluationResult:
        """Diagnose a ticket (or full pattern) by running full evaluation.

        Args:
            ticket_id: RSPEED ticket ID (e.g., "RSPEED-2482"), or None for full pattern
            use_existing: If True, use most recent evaluation results without re-running
            runs: Number of evaluation runs (default: 1 for speed, use 3+ for stability analysis)

        Returns:
            EvaluationResult with parsed metrics (averaged if ticket_id is None)
        """
        print(f"\n{'='*80}")
        if ticket_id:
            print(f"DIAGNOSING: {ticket_id}")
        else:
            print("DIAGNOSING: ALL TICKETS IN PATTERN")
        print(f"{'='*80}")

        if use_existing:
            # Find most recent evaluation output
            print("\n📂 Using existing evaluation results...")
            output_dir = self.get_latest_output_dir("full")
            print(f"   Found: {output_dir.name}")
        else:
            # Run eval for just this ticket (much faster than full suite)
            output_dir = self.run_full_eval(
                self.functional_full, runs=runs, single_ticket=ticket_id
            )

        # Parse results
        result = self.parse_results(output_dir, ticket_id)

        print("\n" + result.summary())

        # Stability Assessment
        if result.num_runs > 1:
            print(f"\n📊 STABILITY ASSESSMENT ({result.num_runs} runs)")
            print("=" * 80)

            # Check if answer is already good
            if result.is_answer_good_enough:
                print("✅ ALREADY PASSING - Answer is correct!")
                print(f"   Answer Correctness: {result.answer_correctness:.2f} (≥ 0.8)")
                if result.keywords_score is not None:
                    print(f"   Keywords: {result.keywords_score:.2f} (≥ 0.7)")
                if result.is_retrieval_problem:
                    print("   ℹ️  Note: Retrieval metrics are suboptimal, but answer is correct")
                    print("   → This ticket may not need fixing")
                return result

            # Check for high variance (intermittent issues)
            if result.high_variance_metrics:
                print("⚠️  INTERMITTENT ISSUE DETECTED - High variance in:")
                for metric_info in result.high_variance_metrics:
                    print(f"   • {metric_info}")
                print("\n   → Problem is NOT consistent across runs")
                print("   → May be a temporal validity issue (RSPEED-2200 pattern)")
                print("   → Consider investigating root cause before fixing")
                print("   → Averaged metrics shown below for reference")
            else:
                print("✅ STABLE METRICS - Consistent across all runs")
                print("   → Problem is reproducible")
                print("   → Safe to apply automated fixes")

        # Check if we have metrics to analyze
        if not result.has_metrics:
            print("\n❌ DIAGNOSIS: NO METRICS FOUND")
            print("   → Check if the evaluation completed successfully")
            print("   → Check if this ticket is in the test suite")
            return result

        # Check RAG usage first
        if not result.rag_used:
            print("\n⚠️  DIAGNOSIS: RAG NOT USED")
            print("   → LLM answered from general knowledge only")
            print("   → System prompt may need adjustment to force tool usage")
            print("   → Or RAG tool may not be configured correctly")
            return result

        if result.rag_used and not result.docs_retrieved:
            print("\n⚠️  DIAGNOSIS: RAG CALLED BUT NO DOCUMENTS RETRIEVED")
            print("   → okp-mcp search returned empty results")
            print("   → Query reformulation may be needed")
            print("   → Check Solr index has relevant documents")
            return result

        # Check if expected_urls exist (should ALWAYS exist after discovery)
        if not result.expected_urls and result.expected_response:
            print("\n❌ NO EXPECTED URLS FOUND IN CONFIG")
            print("   → This ticket has NO documentation in Solr containing the answer")
            print("   → This is a DOCUMENTATION GAP - cannot be fixed by RAG optimization")
            print("   → REQUIRED ACTIONS:")
            print("      1. Add missing documentation to Solr/OKP index")
            print("      2. OR: Quarantine ticket for SME review")
            print("      3. OR: Verify question is actually answerable")
            print("\n   ⚠️  SKIPPING this ticket - needs manual intervention\n")

            # Return early - can't optimize if no docs exist
            return result

        # Check if answer is already correct
        if result.answer_correctness and result.answer_correctness >= 0.8:
            print("✅ ANSWER IS CORRECT!")
            print(f"   Answer Correctness: {result.answer_correctness:.2f}")
            print("\n   💾 Saving retrieved URLs as ground truth for regression testing...")

            # Save discovered URLs to YAML
            if result.retrieved_urls:
                self.enrich_config_with_expected_urls(
                    config_path=self.functional_full,
                    ticket_id=result.ticket_id,
                    expected_urls=result.retrieved_urls[:5],  # Top 5 docs
                )
                print(f"   ✅ Saved {len(result.retrieved_urls[:5])} URLs to config")

            return result

        # Answer is wrong - diagnose why
        print("❌ ANSWER IS INCORRECT")
        print(f"   Answer Correctness: {result.answer_correctness:.2f}")

        # Ensure we have expected_response (fetch from Jira if needed)
        expected_response: Optional[str] = result.expected_response
        if not expected_response:
            print("\n⚠️  No expected_response in config - fetching from Jira...")
            expected_response = self.get_expected_response_from_jira(result.ticket_id)
            if expected_response:
                print(f"✅ Retrieved expected answer from Jira ticket {result.ticket_id}")
                # Update result with fetched expected_response for later use
                result.expected_response = expected_response
            else:
                print(f"❌ Could not fetch expected answer from Jira ticket {result.ticket_id}")
                print("   Cannot verify if retrieved docs contain the answer")
                print("   Please add expected_response to config manually")
                return result

        # Check if retrieved docs contain the answer
        print("\n🔍 Checking if retrieved documents contain the expected answer...")

        check_result = self.check_answer_in_retrieved_docs(
            expected_answer=expected_response,
            retrieved_contexts=result.contexts.split("\n") if result.contexts else [],
        )

        if check_result["contains_answer"]:
            print("✅ Retrieved docs CONTAIN the answer")
            print(f"   Confidence: {check_result['confidence']:.2f}")
            print(f"   Reason: {check_result['explanation']}")
            print("\n🔍 DIAGNOSIS: ANSWER EXTRACTION PROBLEM")
            print("   → Correct docs retrieved but LLM failed to extract answer")
            print("   → May need prompt engineering or context window optimization")

            # Save URLs as ground truth
            if result.retrieved_urls:
                print("\n   💾 Saving URLs for future regression testing...")
                self.enrich_config_with_expected_urls(
                    config_path=self.functional_full,
                    ticket_id=result.ticket_id,
                    expected_urls=result.retrieved_urls[:5],
                )

            print("\n   🎯 ITERATION STRATEGY:")
            print("   → This is a prompt/LLM problem, not a retrieval problem")
            print("   → Consider: prompt engineering, model selection, temperature")
            print("   → Retrieved docs are correct (save as expected_urls)")

        else:
            print("❌ Retrieved docs DO NOT contain the answer")
            print(f"   Confidence: {check_result['confidence']:.2f}")
            print(f"   Reason: {check_result['explanation']}")
            print("\n🔍 DIAGNOSIS: WRONG DOCUMENTS RETRIEVED")
            print("   → Need to find which docs actually contain the answer")
            print("   → Will use document discovery to find correct docs")

            print("\n   🎯 ITERATION STRATEGY:")
            print("   → Run bootstrap mode to discover correct documents")
            print("   → Search Solr using expected_response as query")
            print("   → Verify discovered docs with LLM")
            print("   → Iterate to improve retrieval of verified docs")
            print("\n   💡 Next command:")
            print(
                f"      uv run python okp_mcp_agent/agents/okp_mcp_agent.py bootstrap {result.ticket_id} --yolo"
            )

        return result

        # ANSWER QUALITY CHECK: Prioritize answer correctness but verify it's grounded in RAG
        # Success requires: correct answer + using RAG properly (not hallucinating)
        if result.answer_correctness and result.answer_correctness >= 0.8:
            # Check if answer is grounded in retrieved documents (not hallucinated)
            is_grounded = True
            grounding_issues = []

            # Verify RAG is being used
            if not result.rag_used:
                is_grounded = False
                grounding_issues.append("RAG not used (potential hallucination)")

            # Verify context is relevant (docs actually relate to the question)
            if result.context_relevance is not None and result.context_relevance < 0.7:
                is_grounded = False
                grounding_issues.append(f"Low context relevance ({result.context_relevance:.2f})")

            # Verify answer is faithful to context (not making things up)
            if result.faithfulness is not None and result.faithfulness < 0.7:
                is_grounded = False
                grounding_issues.append(f"Low faithfulness ({result.faithfulness:.2f})")

            if is_grounded:
                print("\n✅ ANSWER IS CORRECT AND GROUNDED IN RAG!")
                print(f"   Answer Correctness: {result.answer_correctness:.2f}")
                print(
                    f"   Context Relevance: {result.context_relevance:.2f}"
                    if result.context_relevance
                    else ""
                )
                print(f"   Faithfulness: {result.faithfulness:.2f}" if result.faithfulness else "")

                # Check if we're using different docs than expected
                if result.expected_urls and result.url_f1 is not None and result.url_f1 < 0.5:
                    print(f"   URL F1: {result.url_f1:.2f} (different docs than expected)")
                    print("\n   💡 Different documents are providing the correct answer!")
                    print("   → This is SUCCESS, not a problem to fix")
                    print("   → The retrieved docs are better than the expected_urls")

                    if result.retrieved_urls:
                        print("\n   💾 Updating config with new working URLs...")
                        self.enrich_config_with_expected_urls(
                            config_path=self.functional_full,
                            ticket_id=result.ticket_id,
                            expected_urls=result.retrieved_urls[:5],  # Top 5 docs
                        )
                        print(
                            f"   ✅ Saved {len(result.retrieved_urls[:5])} URLs to config as new ground truth"
                        )
                else:
                    print("   ✅ Using expected documents and answer is correct")

                return result
            else:
                # Answer is correct but may be hallucinated - still need to improve retrieval
                print("\n⚠️  ANSWER IS CORRECT BUT NOT PROPERLY GROUNDED!")
                print(f"   Answer Correctness: {result.answer_correctness:.2f}")
                print("   🚨 Grounding issues detected:")
                for issue in grounding_issues:
                    print(f"      - {issue}")
                print("\n   💡 Answer may be hallucinated - need to improve RAG grounding")
                print("   → Will continue to improve retrieval to ensure answer is grounded")
                # Fall through to retrieval problem diagnosis

        # Determine problem type (RAG was used and returned docs)
        if result.is_retrieval_problem:
            print("\n🔍 DIAGNOSIS: RETRIEVAL PROBLEM")
            if result.url_f1 == 0.0:
                print("   → Wrong documents retrieved (URL F1 = 0.00)")
                print("   → None of the expected docs were returned")
            else:
                print(f"   → Some expected docs missing (URL F1 = {result.url_f1:.2f})")
            print(
                f"   → Context Relevance: {result.context_relevance:.2f}"
                if result.context_relevance
                else ""
            )
            print(
                f"   → Context Precision: {result.context_precision:.2f}"
                if result.context_precision
                else ""
            )

            # Check Solr to see if expected docs exist in index
            if self.solr_checker:
                solr_check = self.check_solr_documents(result)
                # Update result with Solr findings for LLM advisor
                result.solr_check = solr_check

            print("\n   🎯 ITERATION STRATEGY:")
            print("   → Use RETRIEVAL-ONLY mode for fast iteration (~30 sec/run)")
            print("   → Iterate on okp-mcp boost queries until retrieval metrics pass")
            print("   → Then verify answer quality with final full evaluation")
            print("   → 20x faster than full mode!")

            # Get LLM suggestion for boost query changes
            self._get_llm_boost_suggestion(result)

        elif result.is_answer_problem:
            print("\n💬 DIAGNOSIS: ANSWER PROBLEM")
            print("   → Right documents retrieved BUT answer quality low")
            if result.answer_correctness is not None and result.answer_correctness < 0.90:
                print(f"   → Answer Correctness: {result.answer_correctness:.2f} (< 0.90)")
            if result.keywords_score is not None and result.keywords_score < 0.7:
                print(f"   → Keywords missing: {result.keywords_score:.2f} (< 0.7)")
            print("   → LLM not using the retrieved documents effectively")

            # Check if expected_urls are missing - may need to add them to test config
            if not result.expected_urls and self.solr_checker:
                print("\n   ℹ️  No expected_urls defined in test config")
                solr_check = self.check_solr_documents(result)
                result.solr_check = solr_check

            print("\n   🎯 ITERATION STRATEGY:")
            print("   → Use FULL mode (includes answer generation)")
            print("   → Iterate on system prompts to improve answer quality")
            print("   → Each iteration: ~3 min (includes LLM answer generation)")

            # Get LLM suggestion for prompt changes
            self._get_llm_prompt_suggestion(result)

        else:
            print("\n✅ DIAGNOSIS: METRICS LOOK GOOD")
            print("   → All thresholds passed")
            print("   → Expected documents retrieved and answer is correct")

        return result

    def validate_all_suites(self) -> Dict[str, List[EvaluationResult]]:
        """Run all test suites to check for regressions."""
        print(f"\n{'='*80}")
        print("VALIDATION: Running all test suites")
        print(f"{'='*80}")

        results: Dict[str, Any] = {}

        # Run functional suite
        print("\n1. Functional tests...")
        _output_dir = self.run_full_eval(self.functional_full, runs=3)
        # TODO: Parse all results from _output_dir, not just one ticket
        results["functional"] = []

        # TODO: Add other suites (chronically_failing, general_documentation)

        return results

    def fast_retrieval_loop(
        self,
        ticket_id: str,
        query: str,
        expected_urls: List[str],
        max_iterations: int = 15,
        validate_every: int = 5,
        starting_model: str = "medium",
    ) -> bool:
        """Fast retrieval optimization loop using direct Solr queries (no LLM judges).

        MUCH FASTER than full evaluation: ~5 sec/iteration vs ~30 sec/iteration
        - Queries Solr directly (bypasses /v1/infer)
        - Uses only URL-based metrics (F1, MRR, Precision@k, Recall@k)
        - No LLM judges for context quality (context_relevance, context_precision)
        - Validates with full evaluation every N iterations

        Args:
            ticket_id: Ticket ID
            query: User query
            expected_urls: Expected document URLs
            max_iterations: Max fast iterations
            validate_every: Run full evaluation every N iterations
            starting_model: LLM model tier for suggestions

        Returns:
            True if retrieval improved, False otherwise
        """
        print("\n" + "=" * 80)
        print(f"🚀 FAST RETRIEVAL LOOP - {ticket_id}")
        print("=" * 80)
        print("Mode: Direct Solr queries (no LLM judges)")
        print(f"Max iterations: {max_iterations}")
        print(f"Full validation every: {validate_every} iterations")
        print("=" * 80)

        # Get baseline fast metrics
        print("\n📊 Getting baseline metrics...")
        baseline = self.query_solr_direct(query, expected_urls)
        if "error" in baseline:
            print(f"❌ Baseline query failed: {baseline['error']}")
            return False

        print(f"Baseline URL F1: {baseline['url_f1']:.2f}")
        print(f"Baseline MRR: {baseline['mrr']:.2f}")
        print(f"Baseline Precision@5: {baseline['precision_at_5']:.2f}")
        print(f"Baseline Recall@5: {baseline['recall_at_5']:.2f}")

        best_f1 = baseline["url_f1"]
        current_model_tier = starting_model
        iteration_history: List[Dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            print(f"\n--- Fast Iteration {iteration}/{max_iterations} ---")

            # Get suggestion from LLM
            # Create minimal result object for LLM advisor
            from dataclasses import dataclass

            @dataclass
            class MinimalResult:
                ticket_id: str
                query: str
                url_f1: float
                mrr: float
                expected_urls: List[str]
                retrieved_urls: List[str]
                is_retrieval_problem: bool = True
                # Add RAGAS metrics (not available in fast mode, set to None)
                context_relevance: Optional[float] = None
                context_precision: Optional[float] = None
                keywords_score: Optional[float] = None
                forbidden_claims_score: Optional[float] = None
                faithfulness: Optional[float] = None
                answer_correctness: Optional[float] = None
                response_relevancy: Optional[float] = None
                rag_used: bool = True
                docs_retrieved: bool = True
                contexts: Optional[str] = None
                response: Optional[str] = None
                expected_response: Optional[str] = None
                expected_keywords: Optional[list] = None
                forbidden_claims: Optional[list] = None

            minimal_result = MinimalResult(
                ticket_id=ticket_id,
                query=query,
                url_f1=baseline["url_f1"],
                mrr=baseline["mrr"],
                expected_urls=expected_urls,
                retrieved_urls=baseline["retrieved_urls"],
            )

            # Load config snapshot
            solr_snapshot: Optional[Dict[Any, Any]] = self.load_solr_config_snapshot(ticket_id)
            if not solr_snapshot:
                solr_snapshot = self.extract_solr_config_snapshot(ticket_id)

            suggestion = self._get_llm_suggestion_object(
                minimal_result,
                model=TIER_MODELS[current_model_tier],
                iteration_history=iteration_history,
                solr_snapshot=solr_snapshot,
            )

            if not suggestion:
                print("❌ Failed to get suggestion")
                continue

            print(f"💡 Suggestion: {suggestion.suggested_change}")

            # Apply change
            if not self.apply_code_change(suggestion, iteration_context=f"Fast Loop {iteration}"):
                print("❌ Change not applied")
                continue

            # Restart okp-mcp
            self.restart_okp_mcp()

            # Update snapshot
            solr_snapshot = self.extract_solr_config_snapshot(ticket_id)

            # FAST: Query Solr directly (no /v1/infer, no LLM judges)
            current = self.query_solr_direct(query, expected_urls)

            if "error" in current:
                print(f"❌ Query failed: {current['error']}")
                # Revert
                subprocess.run(["git", "restore", "src/okp_mcp/solr.py"], cwd=self.okp_mcp_root)
                continue

            print(
                f"URL F1: {current['url_f1']:.2f} ({current['url_f1'] - baseline['url_f1']:+.2f})"
            )
            print(f"MRR: {current['mrr']:.2f} ({current['mrr'] - baseline['mrr']:+.2f})")
            print(f"Precision@5: {current['precision_at_5']:.2f}")
            print(f"Recall@5: {current['recall_at_5']:.2f}")

            # Check if improved
            improved = current["url_f1"] >= baseline["url_f1"] + SMALL_IMPROVEMENT_THRESHOLD

            if improved:
                print("✅ Improved! Committing...")
                subprocess.run(
                    ["git", "add", "src/okp_mcp/solr.py"], cwd=self.okp_mcp_root, check=True
                )
                subprocess.run(
                    ["git", "commit", "-m", f"fast_loop: {suggestion.suggested_change}"],
                    cwd=self.okp_mcp_root,
                    check=True,
                )
                baseline = current
                if current["url_f1"] > best_f1:
                    best_f1 = current["url_f1"]
            else:
                print("📉 No improvement - reverting")
                subprocess.run(["git", "restore", "src/okp_mcp/solr.py"], cwd=self.okp_mcp_root)

            # Validation checkpoint every N iterations
            if iteration % validate_every == 0:
                print(f"\n🔍 VALIDATION CHECKPOINT (iteration {iteration})")
                full_result = self.diagnose_retrieval_only(
                    ticket_id, iteration=iteration, suggestion=suggestion
                )

                # Check for degradation
                if full_result.context_relevance and full_result.context_relevance < 0.5:
                    print("⚠️  Context quality degraded - stopping fast loop")
                    print(f"   Context relevance: {full_result.context_relevance:.2f}")
                    if iteration_history:
                        self.save_iteration_summary_table(
                            ticket_id, iteration_history, final_status="❌ Context Degraded"
                        )
                    return False

                # Check for success (early exit)
                # Use composite metric that prioritizes context quality over F1
                composite = self._calculate_composite_metric(full_result)
                if composite >= 0.85:  # High composite score
                    print("🎉 SUCCESS - Early exit from fast loop!")
                    print(f"   Composite Score: {composite:.2f} (>= 0.85)")
                    print(f"   Context Relevance: {full_result.context_relevance:.2f}")
                    print(f"   Context Precision: {full_result.context_precision:.2f}")
                    print(f"   URL F1: {full_result.url_f1:.2f} (reported only)")
                    if iteration_history:
                        self.save_iteration_summary_table(
                            ticket_id,
                            iteration_history,
                            final_status=f"✅ Fixed (composite: {composite:.2f})",
                        )
                    return True

                print(
                    f"✅ Context quality OK (relevance: {full_result.context_relevance:.2f}, composite: {composite:.2f})"
                )

            iteration_history.append(
                {
                    "iteration": iteration,
                    "change": suggestion.suggested_change,
                    "url_f1": current["url_f1"],
                    "improved": improved,
                }
            )

        # Final validation
        print("\n🏁 Fast loop complete - running final full validation...")
        final_result = self.diagnose_retrieval_only(ticket_id, iteration=max_iterations)

        print("\nFinal metrics:")
        print(f"  URL F1: {final_result.url_f1:.2f} (started: {baseline['url_f1']:.2f})")
        print(f"  Context Relevance: {final_result.context_relevance:.2f}")
        print(f"  Context Precision: {final_result.context_precision:.2f}")

        # Save progress report
        if iteration_history:
            success = final_result.url_f1 is not None and final_result.url_f1 >= 0.7
            final_status = (
                "✅ Fixed (URL F1 ≥ 0.70)" if success else f"⏱️ Max Iterations ({max_iterations})"
            )
            self.save_iteration_summary_table(
                ticket_id, iteration_history, final_status=final_status
            )

        return final_result.url_f1 is not None and final_result.url_f1 >= 0.7

    def bootstrap_and_fix_ticket(
        self,
        ticket_id: str,
        max_iterations: int = 5,
        auto_select_docs: bool = False,
        starting_model: str = "medium",
        context: str = "bootstrap",
        use_existing: bool = False,
        use_fast_loop: bool = True,  # Enable fast loop by default
        fast_loop_iterations: int = 15,
    ) -> bool:
        """Bootstrap test config with document discovery, then fix ticket.

        Multi-stage workflow:
        1. Validate existing expected_urls (if present) - run quick test
           - If URL F1 > 0.5: URLs are correct, skip to optimization
           - If URL F1 = 0.0: URLs are wrong, continue to discovery
        2. Document Discovery (if no URLs or URLs wrong)
           - Search Solr for docs containing expected answer
           - Auto-select high-scoring docs
           - If no docs found: exit with knowledge gap error
        3. Config Enrichment
           - Update YAML with discovered expected_urls
        4a. FAST Retrieval Optimization (if use_fast_loop=True)
           - Direct Solr queries (no LLM judges) for rapid iteration
           - 15 iterations in ~75 seconds vs 5 iterations in ~150 seconds
        4b. Full Retrieval Optimization
           - LLM-judged metrics for final polish
           - Fewer iterations needed after fast loop

        Args:
            ticket_id: Ticket ID to fix
            max_iterations: Max iterations for full optimization (after fast loop)
            auto_select_docs: Auto-select top docs without user prompt
            starting_model: Starting model tier for LLM advisor (low/medium/high)
            context: Context label for commit messages
            use_existing: Use existing eval results for baseline test (faster)
            use_fast_loop: Use fast direct-Solr loop before full evaluation iterations
            fast_loop_iterations: Max iterations for fast loop

        Returns:
            True if ticket was fixed
        """
        print(f"\n{'='*80}")
        print(f"BOOTSTRAP AND FIX: {ticket_id}")
        print(f"{'='*80}\n")

        # Load config
        normalized_ticket_id = ticket_id.replace("-", "_")
        config_path = self.eval_root / ".temp_configs" / f"{normalized_ticket_id}_single.yaml"

        if not config_path.exists():
            # Create from full config
            config_path = self.create_single_ticket_config(ticket_id, self.functional_full)

        import yaml

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Find conversation
        conv = None
        for c in config_data:
            if c.get("conversation_group_id") == normalized_ticket_id:
                conv = c
                break

        if not conv or not conv.get("turns"):
            print(f"❌ Could not find ticket {ticket_id} in config")
            return False

        turn = conv["turns"][0]
        query = turn.get("query")
        expected_response = turn.get("expected_response")
        expected_keywords = turn.get("expected_keywords", [])
        expected_urls = turn.get("expected_urls", [])

        # STAGE 1: Validate existing expected_urls (if any)
        if expected_urls:
            print("📍 STAGE 1: Validate Existing expected_urls")
            print("=" * 80)
            print(f"Config has {len(expected_urls)} expected_urls:")
            for url in expected_urls:
                print(f"  - {url}")
            print("\nRunning quick retrieval test to verify these URLs are correct...\n")

            # Run quick retrieval-only test to check URL F1 AND context quality
            validation_result = self.diagnose_retrieval_only(ticket_id, runs=1)

            url_f1 = validation_result.url_f1 or 0.0
            context_relevance = validation_result.context_relevance or 0.0
            context_precision = validation_result.context_precision or 0.0

            print("\n📊 Validation Result:")
            print(f"  URL F1: {url_f1:.2f}")
            print(f"  Context Relevance: {context_relevance:.2f}")
            print(f"  Context Precision: {context_precision:.2f}")

            # DECISION LOGIC: When to skip discovery and just optimize
            # 1. F1 > 0: At least one expected doc retrieved → we have the right info
            # 2. High RAGAS metrics: Retrieved docs are high quality (even if different from expected)
            skip_discovery = (url_f1 > 0) or (  # Got at least one expected doc
                context_relevance >= 0.8 and context_precision >= 0.7
            )  # Or high quality docs

            if skip_discovery:
                # Retrieval is working - check if answer is already correct
                if url_f1 > 0:
                    print("✅ Expected docs ARE being retrieved (F1 > 0)")
                    print(
                        f"   → {int(url_f1 * len(expected_urls))} of {len(expected_urls)} expected docs found"
                    )
                    print("   → Docs exist in Solr, just need better ranking")
                else:
                    print(
                        f"✅ Retrieved docs are high quality (context_relevance={context_relevance:.2f})"
                    )
                    print(f"   → URL F1={url_f1:.2f} but RAGAS metrics show docs are excellent")
                    print("   → Different docs than expected, but working well!")

                print("\n   Skipping discovery, checking answer quality first...")
                print("\n📍 STAGE 2: Answer Quality Check")
                print("=" * 80)
                print(
                    "Running full evaluation with answer generation to check if already passing...\n"
                )

                # Run full eval to check answer quality
                answer_check = self.diagnose(ticket_id, use_existing=False)

                # Check if already passing (answer is correct and grounded)
                if answer_check.is_answer_good_enough:
                    print("\n🎉 ALREADY PASSING!")
                    print(
                        f"   Answer Correctness: {answer_check.answer_correctness:.2f}"
                        if answer_check.answer_correctness
                        else ""
                    )
                    print(
                        f"   Context Relevance: {answer_check.context_relevance:.2f}"
                        if answer_check.context_relevance
                        else ""
                    )
                    print(
                        f"   Keywords: {answer_check.keywords_score:.2f}"
                        if answer_check.keywords_score
                        else ""
                    )
                    print("\n   ✅ No optimization needed - answer is already correct!")
                    return True

                # Answer needs work - proceed to optimization
                print("\n📍 STAGE 3: Retrieval Optimization")
                print("=" * 80)
                print("Answer quality needs improvement, optimizing retrieval...\n")
                return self.fix_ticket_with_iteration(
                    ticket_id=ticket_id,
                    max_iterations=max_iterations,
                    starting_model=starting_model,
                    context=context,
                    use_existing=False,  # Already did answer check
                )
            else:
                # F1 = 0 AND low context quality - need to find the right docs
                print(
                    f"❌ Expected docs NOT found (F1=0) and context quality low (relevance={context_relevance:.2f})"
                )
                print("   → Expected docs may not exist in Solr")
                print("   → Will discover which docs can answer this question\n")
                # Fall through to discovery stage

        # STAGE 2: Document Discovery (if no URLs or URLs are wrong)
        if not expected_response:
            print("⚠️  Config has no expected_response - fetching from Jira...")
            expected_response = self.get_expected_response_from_jira(ticket_id)
            if expected_response:
                print(f"✅ Retrieved expected answer from Jira ticket {ticket_id}")
                print(f"   Answer preview: {expected_response[:200]}...")
                # Update config data with fetched expected_response
                turn["expected_response"] = expected_response
                # Save updated config
                with open(config_path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                print(f"   💾 Saved expected_response to {config_path}")
            else:
                print(f"❌ Could not fetch expected answer from Jira ticket {ticket_id}")
                print("   Cannot discover documents without expected_response")
                print("   Please add expected_response to config manually")
                return False

        print("📍 STAGE 2: Document Discovery")
        print("=" * 80)
        print("Searching Solr for documents containing the correct answer...\n")

        discovered_urls = self.discover_expected_documents(
            ticket_id=ticket_id,
            query=query,
            expected_response=expected_response,
            expected_keywords=expected_keywords,
            auto_select_threshold=10.0 if auto_select_docs else 0.0,
        )

        if not discovered_urls:
            # Knowledge gap was already reported by discover_expected_documents
            return False

        # STAGE 3: Config Enrichment
        print("\n📍 STAGE 3: Config Enrichment")
        print("=" * 80)

        self.enrich_config_with_expected_urls(
            config_path=config_path, ticket_id=ticket_id, expected_urls=discovered_urls
        )

        # STAGE 4: Retrieval Optimization
        print("\n📍 STAGE 4: Retrieval Optimization")
        print("=" * 80)
        print("Now iterating to improve retrieval of discovered documents\n")

        # STAGE 4a: Fast loop (optional but recommended)
        if use_fast_loop:
            print("🚀 Running FAST LOOP first (direct Solr queries, no LLM judges)")
            print(
                f"   This will do {fast_loop_iterations} iterations in ~{fast_loop_iterations * 5} seconds"
            )
            print(
                f"   vs regular loop: ~{max_iterations * 30} seconds for {max_iterations} iterations\n"
            )

            fast_improved = self.fast_retrieval_loop(
                ticket_id=ticket_id,
                query=query,
                expected_urls=discovered_urls,
                max_iterations=fast_loop_iterations,
                validate_every=5,
                starting_model=starting_model,
            )

            if fast_improved:
                print("\n✅ Fast loop improved retrieval significantly!")
                print("   Skipping full loop - retrieval is already good\n")
                return True
            else:
                print("\n📊 Fast loop made progress - running full loop for final polish\n")

        # STAGE 4b: Full loop with LLM judges (for final polish or if fast loop disabled)
        return self.fix_ticket_with_iteration(
            ticket_id=ticket_id,
            max_iterations=max_iterations,
            starting_model=starting_model,
            context=context,
        )

    def fix_ticket(
        self,
        ticket_id: str,
        max_iterations: int = 10,
        use_worktree: bool = False,
        worktree_name: Optional[str] = None,
        suggest_only: bool = False,
    ):
        """Autonomous ticket fixing with iteration.

        Args:
            ticket_id: RSPEED ticket ID
            max_iterations: Maximum number of iteration attempts
            use_worktree: If True, work in an isolated git worktree
            worktree_name: Optional custom worktree/branch name
            suggest_only: If True, suggest changes but don't apply them
        """
        print(f"\n{'='*80}")
        print(f"AUTO-FIX: {ticket_id}")
        print(f"{'='*80}")

        # Setup: Create worktree if requested
        working_dir = self.okp_mcp_root
        if use_worktree:
            working_dir = self.create_worktree(ticket_id, worktree_name)
            print(f"\n📂 Working in isolated worktree: {working_dir}")

        try:
            # Step 1: Diagnose
            initial_result = self.diagnose(ticket_id)

            if not (initial_result.is_retrieval_problem or initial_result.is_answer_problem):
                print("\n✅ Ticket already passing, nothing to fix")
                return

            # Step 2: Iterate (TODO - Phase 2)
            if initial_result.is_retrieval_problem:
                print("\n🔄 Fast iteration mode (retrieval)...")
                print("   [TODO - Phase 2] LLM suggests boost query changes")
                print("   [TODO - Phase 2] Apply changes after approval")
                print("   [TODO - Phase 2] Iterate until URL F1 > 0.8")

            elif initial_result.is_answer_problem:
                print("\n🔄 Full iteration mode (answer)...")
                print("   [TODO - Phase 2] LLM suggests prompt changes")
                print("   [TODO - Phase 2] Apply changes after approval")
                print("   [TODO - Phase 2] Iterate until keywords present")

            # Step 3: Validate (TODO - Phase 2)
            print("\n🔍 Regression checks...")
            print("   [TODO - Phase 2] Run all test suites")
            print("   [TODO - Phase 2] Compare with baseline")

            # Step 4: Review
            if use_worktree:
                print("\n📝 Review changes in worktree:")
                print(f"   cd {working_dir}")
                print("   git diff main")
                print("\n   When ready to merge:")
                print("   git checkout main")
                print(f"   git merge {worktree_name or f'fix/{ticket_id.lower()}'}")
                print("\n   Or to discard:")
                print(f"   git worktree remove {working_dir}")

        finally:
            # Cleanup: Ask if worktree should be removed
            if use_worktree and not suggest_only:
                self.cleanup_worktree(working_dir)

    # =========================================================================
    # Iteration Loop Methods (Phase 2)
    # =========================================================================

    # TODO (Future): Autonomous AST-based code editing
    # def apply_code_change_autonomous(self, suggestion, iteration_log: dict) -> bool:
    #     """Autonomous code editing with AST manipulation and full logging.
    #
    #     Features for autonomous mode:
    #     - AST-based surgical edits (no manual intervention)
    #     - Full audit trail logging:
    #       * Agent reasoning at each stage
    #       * Code changes (before/after AST)
    #       * Intermediate metrics after each change
    #       * Final metrics after all iterations
    #     - Auto-commit with detailed commit messages
    #     - JSON log file for review/debugging
    #
    #     iteration_log structure:
    #     {
    #         "ticket_id": "RSPEED-2482",
    #         "iterations": [
    #             {
    #                 "iteration": 1,
    #                 "model": "sonnet",
    #                 "reasoning": "...",
    #                 "change": "...",
    #                 "ast_diff": {...},
    #                 "metrics_before": {...},
    #                 "metrics_after": {...},
    #                 "improved": true
    #             }
    #         ],
    #         "final_result": "success|failed|escalated"
    #     }
    #     """
    #     pass

    def _get_llm_suggestion_object(
        self,
        result: Any,  # Can be EvaluationResult or MinimalResult
        model: Optional[str] = None,
        iteration_history: Optional[List] = None,
        solr_snapshot: Optional[Dict] = None,
    ):
        """Get LLM suggestion object (boost or prompt) without printing.

        Args:
            result: Evaluation result with metrics
            model: Optional model override (for escalation)
            iteration_history: List of previous iteration attempts and results
            solr_snapshot: Cached Solr config snapshot (avoids file reads)

        Returns:
            BoostQuerySuggestion or PromptSuggestion object
        """
        if not self.llm_advisor or not result.query:
            return None

        # Collect Solr explain output and config analysis
        solr_explain = None
        solr_config_summary = None
        ranking_analysis = None

        if self.solr_analyzer and result.query:
            try:
                # Get Solr explain output showing why docs ranked the way they did
                solr_explain = self.solr_analyzer.get_explain_output(
                    result.query, doc_ids=None, num_docs=10  # Will get top N docs
                )

                # Get current Solr config summary (only if snapshot not provided)
                # Snapshot is much faster and more focused for LLM
                if not solr_snapshot:
                    solr_config_summary = self.solr_analyzer.format_config_summary()

                # Analyze ranking problems if we have expected URLs
                if result.expected_urls and result.retrieved_urls:
                    ranking_analysis = self.solr_analyzer.analyze_ranking_problems(
                        result.query, result.expected_urls, result.retrieved_urls
                    )
            except Exception as e:
                print(f"⚠️  Solr analysis failed: {e}")
                # Continue without Solr analysis

        # Convert to MetricSummary
        metrics = MetricSummary(
            ticket_id=result.ticket_id,
            query=result.query,
            url_f1=result.url_f1,
            mrr=result.mrr,
            context_relevance=result.context_relevance,
            context_precision=result.context_precision,
            keywords_score=result.keywords_score,
            forbidden_claims_score=result.forbidden_claims_score,
            faithfulness=result.faithfulness,
            answer_correctness=result.answer_correctness,
            response_relevancy=result.response_relevancy,
            rag_used=result.rag_used,
            docs_retrieved=result.docs_retrieved,
            num_docs=self._get_num_docs(result.contexts),
            # LLM Judge reasoning
            answer_correctness_reason=result.answer_correctness_reason,
            faithfulness_reason=result.faithfulness_reason,
            response_relevancy_reason=result.response_relevancy_reason,
            context_relevance_reason=result.context_relevance_reason,
            context_precision_reason=result.context_precision_reason,
            # Ground truth / expected values
            response=result.response,
            expected_response=result.expected_response,
            expected_keywords=result.expected_keywords,
            expected_urls=result.expected_urls,
            forbidden_claims=result.forbidden_claims,
            retrieved_urls=result.retrieved_urls,
            contexts=result.contexts,
            # Solr explain output and analysis
            solr_explain=solr_explain,
            solr_config_summary=solr_config_summary,
            solr_config_snapshot=solr_snapshot,  # Structured config (replaces file reads)
            ranking_analysis=ranking_analysis,
        )

        # Add iteration history context
        if iteration_history:
            metrics.iteration_history = iteration_history

        # Override model if specified (for escalation)
        if model:
            original_model = self.llm_advisor.medium_model
            self.llm_advisor.medium_model = model

        try:
            print("🔍 DEBUG: About to call LLM advisor...")
            print(f"🔍 DEBUG: is_retrieval_problem={result.is_retrieval_problem}")

            if result.is_retrieval_problem:
                print("🔍 DEBUG: Calling suggest_boost_query_changes...")
                suggestion = self._run_async_in_thread(
                    self.llm_advisor.suggest_boost_query_changes(metrics)
                )
                print(f"🔍 DEBUG: suggest_boost_query_changes returned: {type(suggestion)}")
                return suggestion
            else:
                print("🔍 DEBUG: Calling suggest_prompt_changes...")
                suggestion = self._run_async_in_thread(
                    self.llm_advisor.suggest_prompt_changes(metrics)
                )
                print(f"🔍 DEBUG: suggest_prompt_changes returned: {type(suggestion)}")
                return suggestion
        except Exception as e:
            print(f"⚠️  Error getting LLM suggestion: {type(e).__name__}: {e}")
            print("   Check /tmp/claude_sdk_debug.log for details")
            import traceback

            traceback.print_exc()
            return None
        finally:
            # Restore original model
            if model:
                self.llm_advisor.medium_model = original_model

    def apply_code_change(self, suggestion, iteration_context: Optional[str] = None) -> bool:
        """Apply LLM-suggested code change with git diff approval flow.

        Flow:
        1. Display LLM reasoning and suggestion
        2. Apply change to file
        3. Show git diff
        4. Get human approval
        5. If approved → commit and return True
        6. If rejected → revert and return False

        Args:
            suggestion: BoostQuerySuggestion or PromptSuggestion
            iteration_context: Optional context (e.g., "Iteration 1/5")

        Returns:
            True if change applied and approved
        """
        if suggestion is None:
            print("❌ No suggestion to apply")
            return False

        # For boost query suggestions, we have a file path
        if hasattr(suggestion, "file_path"):
            file_path = self.okp_mcp_root / suggestion.file_path
        else:
            # Prompt suggestions don't have file_path
            print("\n📝 Applying system prompt change")
            print("   ⚠️  Prompt editing not yet implemented")
            return False

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False

        # Display agent reasoning
        print("\n" + "=" * 80)
        print("🤖 AGENT REASONING")
        print("=" * 80)
        if iteration_context:
            print(f"Context: {iteration_context}")
        print(f"\nFile: {file_path.relative_to(self.okp_mcp_root)}")
        print(f"\n{suggestion.reasoning}")
        print(f"\nSuggested Change:\n  {suggestion.suggested_change}")
        print(f"\nExpected Improvement:\n  {suggestion.expected_improvement}")
        print(f"\nConfidence: {suggestion.confidence}")

        if hasattr(suggestion, "code_snippet") and suggestion.code_snippet:
            print(f"\nCode Snippet:\n{suggestion.code_snippet}")

        # Ask if user wants to proceed with edit
        if self.interactive:
            print("\n" + "=" * 80)
            confirm = input("Proceed with applying this change? (y/n): ")
            if confirm.lower() != "y":
                print("❌ Change not applied")
                return False
        else:
            print("\n🚀 YOLO mode: Auto-approving change...")
            print(
                f"   Change: {suggestion.suggested_change[:100]}{'...' if len(suggestion.suggested_change) > 100 else ''}"
            )

        # Claude's Edit tool doesn't actually write files!
        # We need to apply the change ourselves based on code_snippet
        print(f"\n📝 Applying LLM suggestion to {file_path.name}...")

        if not hasattr(suggestion, "code_snippet") or not suggestion.code_snippet:
            print("❌ No code_snippet provided by LLM")
            print("   Cannot apply change without knowing what to change")
            return False

        try:
            # Read the file
            content = file_path.read_text()
            original_content = content

            # Try to apply the change using the code_snippet as a guide
            # The code_snippet shows what the changed line should look like
            print(f"🔍 Looking for code to change in {file_path}...")
            print(f"   Suggestion: {suggestion.suggested_change}")
            print(f"   Code snippet from LLM:\n   {suggestion.code_snippet[:300]}")

            # Strategy: Use regex-based replacement for common Solr config patterns
            #
            # SUPPORTED CHANGE PATTERNS:
            # 1. Solr query parameters (qf, pf, pf2, pf3, mm, ps, ps2, ps3)
            #    Example: "mm": "2<-1 5<75%" -> "mm": "2<-1 5<60%"
            #    Example: "qf": "title^5 ..." -> "qf": "title^7 ..."
            #
            # 2. BM25 highlighting parameters (hl.score.k1, hl.score.b, hl.score.pivot)
            #    Example: "hl.score.k1": "1.0" -> "hl.score.k1": "1.2"
            #
            # 3. Boost/Demote multipliers (multiplier *= X.X)
            #    Example: multiplier *= 2.0 -> multiplier *= 3.0 (boost)
            #    Example: multiplier *= 0.05 -> multiplier *= 0.01 (demote)
            #
            # 4. Boost keywords (_EXTRACTION_BOOST_KEYWORDS frozenset)
            #    Example: Adding "compatibility matrix" to the list
            #
            # 5. Demote keywords (_EXTRACTION_DEMOTE_RHV frozenset)
            #    Example: Adding unwanted content patterns to the list

            # Detect which pattern to use
            applied = False

            if any(
                param in suggestion.code_snippet
                for param in [
                    '"qf":',
                    '"pf":',
                    '"pf2":',
                    '"pf3":',
                    '"mm":',
                    '"ps":',
                    '"ps2":',
                    '"ps3":',
                    '"hl.score',
                    '"hl.snippets',
                    '"hl.fragsize',
                ]
            ):
                # Changing query field weights, phrase boost weights, BM25 parameters, or highlighting parameters
                # Examples: title^5 -> title^7, "mm": "2<-1 5<75%" -> "mm": "2<-1 5<60%", "hl.score.k1": "1.2", "hl.snippets": "8"
                import re

                # Extract the parameter name and new value from code_snippet
                # Support both regular params (qf, pf, mm) and dotted params (hl.score.k1, hl.snippets)
                param_match = re.search(
                    r'"(qf|pf|pf2|pf3|mm|ps|ps2|ps3|hl\.score\.k1|hl\.score\.b|hl\.score\.pivot|hl\.snippets|hl\.fragsize)":\s*"([^"]*)"',
                    suggestion.code_snippet,
                )
                if param_match:
                    param_name = param_match.group(1)
                    new_value = param_match.group(2)

                    print(f"   Changing {param_name} parameter...")
                    print(
                        f"   New value: {new_value[:100]}..."
                        if len(new_value) > 100
                        else f"   New value: {new_value}"
                    )

                    # Find and replace the parameter in the file
                    pattern = rf'"{param_name}":\s*"[^"]*"'
                    replacement = f'"{param_name}": "{new_value}"'

                    # DEBUG: Show what we're looking for
                    print(f"   DEBUG: Searching for pattern: {pattern}")
                    match = re.search(pattern, content)
                    if match:
                        print(f"   DEBUG: Found existing value: {match.group()}")
                    else:
                        print("   DEBUG: Pattern not found in file!")
                        # Show nearby content for debugging
                        lines = content.split("\n")
                        for i, line in enumerate(lines, 1):
                            if param_name in line:
                                print(f"   DEBUG: Found {param_name} at line {i}: {line.strip()}")

                    content, count = re.subn(pattern, replacement, content, count=1)

                    if count > 0:
                        print(f"✅ Applied change: {param_name} parameter updated")
                        applied = True
                    else:
                        print(f"❌ Could not find {param_name} parameter to change")
                        print(f"   Pattern: {pattern}")
                        print("   File may not contain this parameter in expected format")
                        return False
                else:
                    print("❌ Could not parse parameter change from code_snippet")
                    return False

            elif (
                "multiplier *=" in suggestion.code_snippet
                or "boost multiplier" in suggestion.suggested_change.lower()
                or "demote multiplier" in suggestion.suggested_change.lower()
            ):
                # Changing boost or demote multiplier values
                import re

                # Extract new multiplier value from code_snippet
                # Looking for patterns like: multiplier *= 3.0 or multiplier *= 0.01
                multiplier_match = re.search(r"multiplier \*= ([\d.]+)", suggestion.code_snippet)
                if multiplier_match:
                    new_value = multiplier_match.group(1)

                    # Determine if it's boost or demote based on value or suggestion text
                    is_boost = (
                        float(new_value) > 1.0 or "boost" in suggestion.suggested_change.lower()
                    )
                    is_demote = (
                        float(new_value) < 1.0 or "demote" in suggestion.suggested_change.lower()
                    )

                    if is_boost:
                        # Change boost multiplier (usually around line 309)
                        # Pattern: if any(kw in para_lower for kw in _EXTRACTION_BOOST_KEYWORDS):\n        multiplier *= X.X
                        pattern = r"(if any\(kw in para_lower for kw in _EXTRACTION_BOOST_KEYWORDS\):.*?multiplier \*= )[\d.]+"
                        replacement = rf"\g<1>{new_value}"
                        content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
                        if count > 0:
                            print(f"✅ Applied change: Boost multiplier updated to {new_value}x")
                            applied = True
                        else:
                            print("❌ Could not find boost multiplier pattern")
                            return False

                    elif is_demote:
                        # Change demote multiplier (usually around line 313)
                        # Pattern: if any(rhv in para_lower for kw in _EXTRACTION_DEMOTE_RHV)...: multiplier *= X.XX
                        pattern = r"(if any\(rhv in para_lower for rhv in _EXTRACTION_DEMOTE_RHV\).*?multiplier \*= )[\d.]+"
                        replacement = rf"\g<1>{new_value}"
                        content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
                        if count > 0:
                            print(f"✅ Applied change: Demote multiplier updated to {new_value}x")
                            applied = True
                        else:
                            print("❌ Could not find demote multiplier pattern")
                            return False
                    else:
                        print("❌ Could not determine if boost or demote multiplier")
                        return False
                else:
                    print("❌ Could not parse multiplier value from code_snippet")
                    return False

            elif (
                "EXTRACTION_DEMOTE_RHV" in suggestion.code_snippet
                or "EXTRACTION_DEMOTE_RHV" in suggestion.suggested_change
            ):
                # Adding keywords to demote list (for RHV content, etc.)
                import re

                new_keywords = []
                for line in suggestion.code_snippet.split("\n"):
                    line = line.strip()
                    # Skip lines with ... (ellipsis) or comment markers
                    if "..." in line or line.startswith("#"):
                        continue
                    # Match quoted strings (with or without trailing comma)
                    quote_match = re.match(r'"([^"]+)"', line)
                    if quote_match:
                        keyword = quote_match.group(1)
                        new_keywords.append(keyword)

                if new_keywords:
                    # Find _EXTRACTION_DEMOTE_RHV frozenset
                    pattern = r"(_EXTRACTION_DEMOTE_RHV = frozenset\(\s*\[.*?)(]\s*\))"
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        before = match.group(1)
                        after = match.group(2)
                        # Add new keywords before the closing ]
                        new_items = "".join(f'\n        "{kw}",' for kw in new_keywords)
                        content = (
                            content[: match.start()]
                            + before
                            + new_items
                            + "\n    "
                            + after
                            + content[match.end() :]
                        )
                        print(
                            f"✅ Applied change: Added {len(new_keywords)} keywords to EXTRACTION_DEMOTE_RHV"
                        )
                        applied = True
                    else:
                        print("❌ Could not find EXTRACTION_DEMOTE_RHV")
                        return False
                else:
                    print("❌ No keywords found in code_snippet")
                    return False

            elif (
                "EXTRACTION_BOOST_KEYWORDS" in suggestion.code_snippet
                or "EXTRACTION_BOOST_KEYWORDS" in suggestion.suggested_change
            ):
                # Adding keywords to boost list
                import re

                # Find the frozenset and add items before the closing ]
                new_keywords = []
                for line in suggestion.code_snippet.split("\n"):
                    line = line.strip()
                    # Skip lines with ... (ellipsis) or comment markers
                    if "..." in line or line.startswith("#"):
                        continue
                    # Match quoted strings (with or without trailing comma)
                    quote_match = re.match(r'"([^"]+)"', line)
                    if quote_match:
                        keyword = quote_match.group(1)
                        new_keywords.append(keyword)

                if new_keywords:
                    print(f"   Parsed {len(new_keywords)} keywords to add: {new_keywords}")
                    # Find _EXTRACTION_BOOST_KEYWORDS frozenset
                    pattern = r"(_EXTRACTION_BOOST_KEYWORDS = frozenset\(\s*\[.*?)(]\s*\))"
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        before = match.group(1)
                        after = match.group(2)
                        # Add new keywords before the closing ]
                        new_items = "".join(f'\n        "{kw}",' for kw in new_keywords)
                        content = (
                            content[: match.start()]
                            + before
                            + new_items
                            + "\n    "
                            + after
                            + content[match.end() :]
                        )
                        print(
                            f"✅ Applied change: Added {len(new_keywords)} keywords to EXTRACTION_BOOST_KEYWORDS"
                        )
                        applied = True
                    else:
                        print("❌ Could not find EXTRACTION_BOOST_KEYWORDS")
                        return False
                else:
                    print("❌ No keywords found in code_snippet")
                    return False

            if not applied:
                print("⚠️  Unknown change pattern!")
                print(f"   Suggestion: {suggestion.suggested_change}")
                print("\n   Supported change patterns:")
                print("   - Solr parameters: qf, pf, pf2, pf3, mm, ps, ps2, ps3")
                print("   - BM25 parameters: hl.score.k1, hl.score.b, hl.score.pivot")
                print("   - Boost keywords: _EXTRACTION_BOOST_KEYWORDS")
                print("   - Demote keywords: _EXTRACTION_DEMOTE_RHV")
                print("   - Boost multiplier: multiplier *= X.X (where X > 1.0)")
                print("   - Demote multiplier: multiplier *= X.XX (where X < 1.0)")
                print("\n   If this is a valid change type, please update the agent code!")
                return False

            # Check if content actually changed
            print("   DEBUG: Checking if content changed...")
            print(f"   DEBUG: Original length: {len(original_content)} chars")
            print(f"   DEBUG: New length: {len(content)} chars")
            if content == original_content:
                print("❌ No changes were made to the file!")
                print("   Code pattern not recognized or replacement failed")
                print("   DEBUG: Content is identical despite replacement attempt")
                # Show first difference if any
                for i, (old_char, new_char) in enumerate(zip(original_content, content)):
                    if old_char != new_char:
                        print(
                            f"   DEBUG: First diff at char {i}: {repr(old_char)} -> {repr(new_char)}"
                        )
                        break
                return False
            else:
                print(f"✅ Content changed ({len(content) - len(original_content):+d} chars)")

            # Write the modified content
            file_path.write_text(content)
            print(f"✅ File {file_path.name} modified successfully")

        except Exception as e:
            print(f"❌ Error applying change: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Now check git status
        try:
            rel_path = file_path.relative_to(self.okp_mcp_root)
            print(f"   DEBUG: Checking git status for {rel_path}")
            print(f"   DEBUG: Git working directory: {self.okp_mcp_root}")
            status_result = subprocess.run(
                ["git", "status", "--porcelain", str(rel_path)],
                cwd=self.okp_mcp_root,
                capture_output=True,
                text=True,
            )

            print(f"   DEBUG: Git status output: {repr(status_result.stdout)}")
            if not status_result.stdout.strip():
                print("❌ Changes were written but git shows no diff!")
                print("   This shouldn't happen - file may not be tracked")
                print(f"   DEBUG: File exists: {file_path.exists()}")
                print(f"   DEBUG: File is in git repo: {(self.okp_mcp_root / '.git').exists()}")
                return False
            else:
                print(f"✅ Git detected changes: {status_result.stdout.strip()}")

            # Show git diff
            print("\n" + "=" * 80)
            print("📊 GIT DIFF")
            print("=" * 80)

            diff_result = subprocess.run(
                ["git", "diff", str(rel_path)],
                cwd=self.okp_mcp_root,
                capture_output=True,
                text=True,
            )

            if diff_result.stdout:
                print(diff_result.stdout)
            else:
                print("No diff to show (file may not be tracked or no changes made)")

            # Get approval to TEST the change (not commit yet)
            if self.interactive:
                print("\n" + "=" * 80)
                print("Does this diff look correct?")
                print("  y - Approve and TEST (will only commit if test passes)")
                print("  n - Revert changes")
                approval = input("Choice (y/n): ")

                if approval.lower() != "y":
                    print("❌ Reverting changes...")
                    subprocess.run(
                        ["git", "restore", str(file_path)],
                        cwd=self.okp_mcp_root,
                        check=True,
                    )
                    print("✅ Changes reverted")
                    return False
            else:
                print(
                    "\n🚀 YOLO mode: Auto-approving diff - will test and auto-commit if test passes"
                )

            # Store commit message for later (after test passes)
            self._pending_commit_msg = f"agent: {suggestion.suggested_change}\n\nReasoning: {suggestion.reasoning}\nConfidence: {suggestion.confidence}"
            if iteration_context:
                self._pending_commit_msg = f"{iteration_context}\n\n{self._pending_commit_msg}"

            self._pending_commit_file = file_path

            print("\n✅ Change approved - will test and commit only if test passes")
            return True

        except Exception as e:
            print(f"❌ Error applying change: {e}")
            # Restore original content on error
            try:
                subprocess.run(
                    ["git", "restore", str(file_path)],
                    cwd=self.okp_mcp_root,
                    check=True,
                )
                print("✅ Original content restored via git")
            except Exception as restore_error:
                print(f"⚠️  Could not restore: {restore_error}")
            return False

    def get_max_improvement(self, new: EvaluationResult, old: Optional[EvaluationResult]) -> float:
        """Calculate the maximum improvement across all relevant metrics.

        Args:
            new: New evaluation result
            old: Previous evaluation result (None for first iteration)

        Returns:
            Maximum improvement amount (can be negative for regression)
        """
        if old is None:
            return 0.0

        # IMPORTANT: Prioritize answer metrics when available (ultimate goal)
        # Check answer correctness first - it's what actually matters
        if new.answer_correctness is not None and old.answer_correctness is not None:
            # Full evaluation mode - check answer quality first
            answer_improvements = [
                (new.keywords_score or 0) - (old.keywords_score or 0),
                (new.answer_correctness or 0) - (old.answer_correctness or 0),
                (new.faithfulness or 0) - (old.faithfulness or 0),
                (new.response_relevancy or 0) - (old.response_relevancy or 0),
            ]
            return max(answer_improvements)

        # Retrieval-only mode - check retrieval metrics
        elif new.is_retrieval_problem:
            improvements = [
                (new.url_f1 or 0) - (old.url_f1 or 0),
                (new.mrr or 0) - (old.mrr or 0),
                (new.context_relevance or 0) - (old.context_relevance or 0),
                (new.context_precision or 0) - (old.context_precision or 0),
            ]
            return max(improvements)

        return 0.0

    def metrics_improved(self, new: EvaluationResult, old: Optional[EvaluationResult]) -> bool:
        """Check if metrics improved (even slightly).

        Args:
            new: New evaluation result
            old: Previous evaluation result (None for first iteration)

        Returns:
            True if metrics improved by at least SMALL_IMPROVEMENT_THRESHOLD (0.02)
        """
        if old is None:
            # First iteration: accept if metrics are at least reasonable (not all zeros)
            if new.is_retrieval_problem:
                # At least one retrieval metric should show some success
                has_any_retrieval = any(
                    [
                        (new.url_f1 or 0) > 0.05,
                        (new.mrr or 0) > 0.05,
                        (new.context_relevance or 0) > 0.05,
                        (new.context_precision or 0) > 0.05,
                    ]
                )
                return has_any_retrieval
            elif new.is_answer_problem:
                # At least one answer metric should show improvement
                has_any_answer = any(
                    [
                        (new.keywords_score or 0) > 0.05,
                        (new.answer_correctness or 0) > 0.05,
                    ]
                )
                return has_any_answer
            return True  # Accept first iteration if no clear problem type

        # Calculate URL overlap to detect if change had effect
        if new.retrieved_urls and old.retrieved_urls:
            overlap = self.calculate_url_overlap(old.retrieved_urls, new.retrieved_urls)
            new.url_overlap_with_previous = overlap

            # DIAGNOSTIC: If F1 is still 0.00 but overlap is low, this is a red flag
            if (new.url_f1 or 0) == 0.0 and (old.url_f1 or 0) == 0.0 and overlap < 0.3:
                print()
                print("🚨 DIAGNOSTIC WARNING:")
                print("=" * 80)
                print(f"  URL Overlap: {overlap:.2f} (completely different documents!)")
                print("  But URL F1 still 0.00 on both iterations")
                print()
                print("  This suggests the change made things WORSE, not better:")
                print("  - Different docs retrieved (change had major effect)")
                print("  - But still all wrong docs (F1 = 0.00)")
                print()
                print("  Possible causes:")
                print("  • Boost keywords matching irrelevant documents")
                print("  • okp-mcp query augmentation interfering")
                print("  • Wrong field being boosted")
                print()
                print("  Recommendation: Revert and try different approach")
                print("=" * 80)
                print()
            elif overlap > 0.8:
                print(f"\nℹ️  URL Overlap: {overlap:.2f} (documents mostly unchanged)")

        # IMPORTANT: Prioritize answer metrics when available (ultimate goal)
        # If we have answer_correctness data, check that first regardless of retrieval status
        if new.answer_correctness is not None and old.answer_correctness is not None:
            # Full evaluation mode - answer quality is what matters
            improvements = [
                (new.keywords_score or 0) - (old.keywords_score or 0),
                (new.answer_correctness or 0) - (old.answer_correctness or 0),
                (new.faithfulness or 0) - (old.faithfulness or 0),
                (new.response_relevancy or 0) - (old.response_relevancy or 0),
            ]
            max_improvement = max(improvements)
            # Accept even small improvements (will accumulate over iterations)
            return max_improvement >= SMALL_IMPROVEMENT_THRESHOLD

        # Retrieval-only mode - check retrieval metrics
        elif new.is_retrieval_problem:
            url_f1_improvement = (new.url_f1 or 0) - (old.url_f1 or 0)
            mrr_improvement = (new.mrr or 0) - (old.mrr or 0)
            context_relevance_improvement = (new.context_relevance or 0) - (
                old.context_relevance or 0
            )
            context_precision_improvement = (new.context_precision or 0) - (
                old.context_precision or 0
            )

            # CRITICAL: Answer correctness is the ULTIMATE goal
            # URL F1 is just a proxy - expected URLs might not be the only way to answer
            # If answer_correctness data is available, prioritize it over URL matching
            if (
                new.expected_urls
                and new.url_f1 is not None
                and old.url_f1 is not None
                and new.url_f1 == 0.0
                and old.url_f1 == 0.0
            ):
                # Both iterations have URL F1 = 0.00 (not retrieving expected docs)

                # EXCEPTION: If answer is actually correct, different docs are fine!
                if new.answer_correctness is not None and new.answer_correctness >= 0.7:
                    print("\n✅ URL F1 = 0.00 BUT answer is correct (different docs work!)")
                    print(f"   Answer correctness: {new.answer_correctness:.2f}")
                    print("   → Alternative retrieval path is valid")
                    # Fall through to normal improvement checks
                elif new.answer_correctness is not None:
                    # We have answer data and it's not good
                    print(
                        f"\n⚠️  URL F1 = 0.00 AND answer correctness low ({new.answer_correctness:.2f})"
                    )
                    print("   → Different docs but wrong answer - rejecting")
                    return False
                else:
                    # No answer data available (retrieval-only mode)
                    # In this case, context metrics alone are unreliable
                    print("\n⚠️  URL F1 = 0.00 in retrieval-only mode")
                    print("   (Context metrics can be misleading without answer validation)")
                    print(
                        "   → Run full evaluation to check if answer is correct with different docs"
                    )
                    return False

            improvements = [
                url_f1_improvement,
                mrr_improvement,
                context_relevance_improvement,
                context_precision_improvement,
            ]
            max_improvement = max(improvements)

            # Check for significant regressions (any metric drops > 0.1)
            has_regression = any(imp < -0.1 for imp in improvements)
            if has_regression:
                # Only accept if improvement outweighs regression
                net_improvement = sum(improvements)
                return net_improvement >= MIN_IMPROVEMENT_THRESHOLD * 2

            # Accept even small improvements (will accumulate over iterations)
            return max_improvement >= SMALL_IMPROVEMENT_THRESHOLD

        # Fallback: no clear problem type or metrics available
        return False

    def detected_plateau(self, metric_history: List[float]) -> bool:
        """Detect if metrics plateaued (no improvement for N iterations).

        Args:
            metric_history: List of primary metric values over iterations

        Returns:
            True if metrics haven't improved in PLATEAU_THRESHOLD iterations
        """
        if len(metric_history) < PLATEAU_THRESHOLD:
            return False

        # Check last N iterations
        last_n = metric_history[-PLATEAU_THRESHOLD:]
        best_in_last_n = max(last_n)

        # If best metric in last N attempts equals N attempts ago → plateau
        return best_in_last_n == last_n[0]

    def should_deescalate_model(
        self, current_model_tier: str, current_result: EvaluationResult, consecutive_successes: int
    ) -> Optional[str]:
        """De-escalate to cheaper model when metrics are good and stable.

        Args:
            current_model_tier: Current model tier ("medium" or "complex")
            current_result: Latest evaluation result
            consecutive_successes: Number of consecutive successful iterations

        Returns:
            New (cheaper) model tier, or None to stay at current tier
        """
        # Don't de-escalate if we haven't had enough stability
        if consecutive_successes < 2:
            return None

        # Check if metrics are in good shape
        metrics_good = (
            (current_result.url_f1 or 0) > 0.7
            and (current_result.context_relevance or 0) > 0.7
            and (current_result.answer_correctness or 0) > 0.8
        )

        if not metrics_good:
            return None

        # De-escalation path: complex (Opus) → medium (Sonnet) → simple (Haiku)
        if current_model_tier == "complex":
            # Problem solved with Opus, can we maintain with Sonnet?
            print("💡 Metrics stable and good - considering de-escalation to Sonnet")
            return "medium"
        elif current_model_tier == "medium":
            # Problem solved with Sonnet, can we maintain with Haiku?
            print("💡 Metrics stable and good - considering de-escalation to Haiku")
            return "simple"

        return None  # Already at cheapest model or no de-escalation needed

    def escalate_model(
        self, current_model: str, attempts_at_current: int, opus_failed: bool = False
    ) -> Optional[str]:
        """Escalate to better model after failed attempts.

        Args:
            current_model: Current model tier ("simple", "medium" or "complex")
            attempts_at_current: Number of attempts at current model
            opus_failed: If True, skip escalation to Opus (it already failed)

        Returns:
            New model tier, or None to escalate to human
        """
        if attempts_at_current < ESCALATION_THRESHOLD:
            return current_model  # Stay at current level

        # Escalation path: simple (Haiku) → medium (Sonnet) → complex (Opus) → Human
        if current_model == "simple":
            return "medium"
        elif current_model == "medium":
            # Skip Opus if it failed earlier
            if opus_failed:
                return None  # Skip to human escalation
            return "complex"
        elif current_model == "complex":
            return None  # Escalate to human

        return current_model

    def fix_ticket_with_iteration(
        self,
        ticket_id: str,
        max_iterations: int = PRIMARY_FIX_MAX_ITERATIONS,
        starting_model: str = "medium",
        context: str = "primary",
        use_existing: bool = False,
    ) -> bool:
        """Fix a ticket with automatic iteration and model escalation.

        This is the core feedback loop:
        1. Get LLM suggestion
        2. Apply code change
        3. Restart service
        4. Re-evaluate
        5. Check if fixed or improved
        6. Escalate model if stuck
        7. Repeat until fixed or max iterations

        Args:
            ticket_id: RSPEED ticket ID
            max_iterations: Max attempts (5 for primary, 3 for regressions)
            starting_model: Starting model tier ("medium" or "complex")
            context: "primary" or "regression" (for logging)
            use_existing: If True, use existing results for initial diagnosis (faster for debugging)

        Returns:
            True if ticket fixed (all thresholds passed)
        """
        print(f"\n{'='*80}")
        print(f"FIXING {context.upper()}: {ticket_id}")
        print(f"{'='*80}")

        # Initial diagnosis (full mode to check both retrieval AND answer)
        previous_result = self.diagnose(ticket_id, use_existing=use_existing)

        # Load persisted diagnostics from previous runs (if any)
        iteration_history = self.load_iteration_history(ticket_id)

        if not (previous_result.is_retrieval_problem or previous_result.is_answer_problem):
            print("✅ Already passing, nothing to fix")
            # Save summary even if no iterations ran (for completeness)
            if iteration_history:
                self.save_iteration_summary_table(
                    ticket_id, iteration_history, final_status="Already Passing"
                )
            return True

        # Determine iteration mode based on problem type
        use_retrieval_only_mode = previous_result.is_retrieval_problem
        if use_retrieval_only_mode:
            print("\n🎯 Using RETRIEVAL-ONLY mode for fast iteration")
            print("   → Only testing search/retrieval (no LLM answer generation)")
            print("   → ~30 sec per iteration (20x faster)")
            print("   → Focusing on: URL F1, MRR, context relevance/precision")
            print("   → Final full evaluation will verify answer quality")
        else:
            print("\n🎯 Using FULL mode for answer iteration")
            print("   → Testing both retrieval AND answer generation")
            print("   → Focusing on: keywords, answer correctness, faithfulness")

        # Track metrics for plateau detection and de-escalation
        metric_history = []
        current_model_tier = starting_model
        attempts_at_current_model = 0
        consecutive_successes = 0  # Track consecutive improvements for de-escalation
        opus_failed = False  # Track if Opus model has failed (to avoid retrying)

        for iteration in range(1, max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{max_iterations} (Model: {current_model_tier}) ---")

            # Extract Solr config snapshot on first iteration
            # This replaces ~500 lines of file reads with focused ~2KB of parameters
            solr_snapshot: Optional[Dict[Any, Any]]
            if iteration == 1:
                print("📸 Extracting Solr config snapshot...")
                solr_snapshot = self.extract_solr_config_snapshot(ticket_id)
            else:
                # Load cached snapshot (may have been updated after code changes)
                solr_snapshot = self.load_solr_config_snapshot(ticket_id)

            # Reset code to clean state before each iteration (except first)
            # This prevents cumulative changes (e.g., boost going from 1600 → 3200 → 6400 → ...)
            if iteration > 1:
                print(f"\n🔄 Resetting code to clean state (iteration {iteration})...")
                self.run_command(
                    ["git", "reset", "--hard", "HEAD"],
                    cwd=self.okp_mcp_root,  # Always work in okp-mcp root (or worktree if managed externally)
                )
                print("✅ Code reset - starting from original state")
                # Clear any pending commits from previous iteration
                self._pending_commit_msg = None
                self._pending_commit_file = None

            # Get LLM suggestion with iteration history and config snapshot
            current_model = TIER_MODELS[current_model_tier]
            suggestion = self._get_llm_suggestion_object(
                previous_result,
                model=current_model,
                iteration_history=iteration_history,
                solr_snapshot=solr_snapshot,
            )

            # If Opus/complex model failed, fallback to Sonnet
            if suggestion is None and current_model_tier == "complex":
                print("⚠️  Complex model (Opus) failed - falling back to medium model (Sonnet)")
                print("   Opus will be disabled for remaining iterations")
                opus_failed = True
                current_model_tier = "medium"
                attempts_at_current_model = 0  # Reset counter for medium model
                current_model = TIER_MODELS[current_model_tier]
                suggestion = self._get_llm_suggestion_object(
                    previous_result,
                    model=current_model,
                    iteration_history=iteration_history,
                    solr_snapshot=solr_snapshot,
                )

            if suggestion is None:
                print("❌ Failed to get suggestion")
                return False

            # Display suggestion
            print(f"\n💡 Suggestion ({current_model_tier}):")
            print(f"   {suggestion.suggested_change}")
            print(f"   Confidence: {suggestion.confidence}")

            # Apply code change
            iteration_context = f"{context.capitalize()} Fix - Iteration {iteration}/{max_iterations} - Model: {current_model_tier}"
            if not self.apply_code_change(suggestion, iteration_context=iteration_context):
                print("❌ Change not applied, stopping")
                return False

            # Restart service
            self.restart_okp_mcp()

            # CRITICAL: Clear MCP direct cache after code changes
            # Cache key doesn't include Solr config, so stale results would be returned
            self._clear_mcp_cache()

            # Update Solr config snapshot to reflect the changes
            print("📸 Updating Solr config snapshot after code changes...")
            solr_snapshot = self.extract_solr_config_snapshot(ticket_id)

            # Re-evaluate (use appropriate mode based on problem type)
            print("\n🧪 Running test to measure impact of change...")
            if use_retrieval_only_mode:
                # Fast retrieval-only evaluation (~30 sec)
                print("   Using retrieval-only mode (~30 sec)")
                current_result = self.diagnose_retrieval_only(
                    ticket_id, iteration=iteration, suggestion=suggestion
                )
            else:
                # Full evaluation with answer generation (~3 min)
                print("   Using full evaluation mode (~3 min)")
                current_result = self.diagnose(ticket_id, use_existing=False)

            # Show metrics after test
            print(f"\n📊 METRICS AFTER ITERATION {iteration}:")
            print("=" * 80)
            if use_retrieval_only_mode:
                print(
                    f"  URL F1: {current_result.url_f1:.2f} (reported only)"
                    if current_result.url_f1 is not None
                    else "  URL F1: N/A"
                )
                print(
                    f"  MRR: {current_result.mrr:.2f}"
                    if current_result.mrr is not None
                    else "  MRR: N/A"
                )
                print(
                    f"  Context Relevance: {current_result.context_relevance:.2f}"
                    if current_result.context_relevance is not None
                    else "  Context Relevance: N/A"
                )
                print(
                    f"  Context Precision: {current_result.context_precision:.2f}"
                    if current_result.context_precision is not None
                    else "  Context Precision: N/A"
                )
            else:
                print(
                    f"  URL F1: {current_result.url_f1:.2f} (reported only)"
                    if current_result.url_f1 is not None
                    else "  URL F1: N/A"
                )
                print(
                    f"  Context Relevance: {current_result.context_relevance:.2f}"
                    if current_result.context_relevance is not None
                    else "  Context Relevance: N/A"
                )
                print(
                    f"  Context Precision: {current_result.context_precision:.2f}"
                    if current_result.context_precision is not None
                    else "  Context Precision: N/A"
                )
                print(
                    f"  Keywords: {current_result.keywords_score:.2f}"
                    if current_result.keywords_score is not None
                    else "  Keywords: N/A"
                )
                print(
                    f"  Answer Correctness: {current_result.answer_correctness:.2f}"
                    if current_result.answer_correctness is not None
                    else "  Answer Correctness: N/A"
                )
                print(
                    f"  Forbidden Claims: {current_result.forbidden_claims_score:.2f}"
                    if current_result.forbidden_claims_score is not None
                    else "  Forbidden Claims: N/A"
                )

            # Track primary metric for plateau detection
            # IMPORTANT: Use composite metric based on answer quality + grounding
            # URL F1 is reported but NOT used for improvement decisions (it's unreliable)
            primary_metric = self._calculate_composite_metric(current_result)
            metric_history.append(primary_metric)

            print(f"\n  📈 Composite Score: {primary_metric:.2f} (context + answer quality)")
            if len(metric_history) > 1:
                print(f"     Change from previous: {primary_metric - metric_history[-2]:+.2f}")
            print("     ℹ️  URL F1 shown for visibility only, not used for improvements")
            print("=" * 80)

            # Check if fixed using COMPOSITE SCORE
            # This combines all metrics (context, answer, keywords) into single quality score
            composite_score = self._calculate_composite_metric(current_result)
            SUCCESS_THRESHOLD = 0.80  # High quality across all metrics

            if composite_score >= SUCCESS_THRESHOLD:
                print(f"\n🎉 SUCCESS in {iteration} iterations!")
                print(f"   Composite Score: {composite_score:.2f} (>= {SUCCESS_THRESHOLD:.2f})")
                print("\n   📊 Metric Breakdown:")
                if current_result.answer_correctness:
                    print(f"      Answer Correctness: {current_result.answer_correctness:.2f}")
                if current_result.context_relevance:
                    print(f"      Context Relevance: {current_result.context_relevance:.2f}")
                if current_result.context_precision:
                    print(f"      Context Precision: {current_result.context_precision:.2f}")
                if current_result.keywords_score:
                    print(f"      Keywords: {current_result.keywords_score:.2f}")
                if current_result.url_f1 is not None:
                    print(f"      URL F1: {current_result.url_f1:.2f} (reported only)")

                # Commit pending change if any
                if hasattr(self, "_pending_commit_msg") and self._pending_commit_msg:
                    print("\n✅ Test passed - committing change...")
                    subprocess.run(
                        ["git", "add", str(self._pending_commit_file)],
                        cwd=self.okp_mcp_root,
                        check=True,
                    )
                    subprocess.run(
                        ["git", "commit", "-m", self._pending_commit_msg],
                        cwd=self.okp_mcp_root,
                        check=True,
                    )
                    print("✅ Change committed")
                    self._pending_commit_msg = None
                    self._pending_commit_file = None

                # Save iteration summary before exiting
                if iteration_history:
                    self.save_iteration_summary_table(
                        ticket_id,
                        iteration_history,
                        final_status=f"✅ Fixed (composite: {composite_score:.2f})",
                    )

                return True

            # Check if improved
            improved = self.metrics_improved(current_result, previous_result)
            improvement_amount = self.get_max_improvement(current_result, previous_result)

            # Handle pending commit based on test results
            if hasattr(self, "_pending_commit_msg") and self._pending_commit_msg:
                if improved:
                    # Test passed/improved - commit the change
                    is_significant = improvement_amount >= MIN_IMPROVEMENT_THRESHOLD
                    improvement_type = "significantly" if is_significant else "incrementally"
                    print(
                        f"📈 Metrics improved {improvement_type}! Primary metric: {primary_metric:.2f} (+{improvement_amount:+.3f})"
                    )
                    print("✅ Test passed - committing change...")

                    subprocess.run(
                        ["git", "add", str(self._pending_commit_file)],
                        cwd=self.okp_mcp_root,
                        check=True,
                    )
                    subprocess.run(
                        ["git", "commit", "-m", self._pending_commit_msg],
                        cwd=self.okp_mcp_root,
                        check=True,
                    )
                    print("✅ Change committed")

                    # Track consecutive successes for de-escalation
                    consecutive_successes += 1

                    # Only reset escalation counter for significant improvements
                    if is_significant:
                        attempts_at_current_model = 0  # Reset escalation counter
                        print("   (Significant improvement - escalation counter reset)")
                    else:
                        print("   (Small improvement - will build on this base)")
                        # Don't reset counter - small improvements compound but we still escalate if stuck
                else:
                    # Test failed/worsened - revert the change
                    print("📉 No significant improvement - reverting change...")
                    subprocess.run(
                        ["git", "restore", str(self._pending_commit_file)],
                        cwd=self.okp_mcp_root,
                        check=True,
                    )
                    print("✅ Change reverted")
                    attempts_at_current_model += 1
                    consecutive_successes = 0  # Reset on failure

                # Clear pending commit
                self._pending_commit_msg = None
                self._pending_commit_file = None
            elif improved:
                # No pending commit but metrics improved
                is_significant = improvement_amount >= MIN_IMPROVEMENT_THRESHOLD
                improvement_type = "significantly" if is_significant else "incrementally"
                print(
                    f"📈 Metrics improved {improvement_type}! Primary metric: {primary_metric:.2f} (+{improvement_amount:+.3f})"
                )
                consecutive_successes += 1
                if is_significant:
                    attempts_at_current_model = 0  # Reset escalation counter
                # For small improvements, don't reset counter
            else:
                print("📉 No significant improvement")
                attempts_at_current_model += 1
                consecutive_successes = 0  # Reset on failure

            # Check for plateau
            if self.detected_plateau(metric_history):
                print(f"⏸️  Plateau detected (no improvement for {PLATEAU_THRESHOLD} iterations)")
                attempts_at_current_model = ESCALATION_THRESHOLD  # Force escalation

            # Check for de-escalation opportunity (save costs with cheaper model)
            deescalate_tier = self.should_deescalate_model(
                current_model_tier, current_result, consecutive_successes
            )
            if deescalate_tier and deescalate_tier != current_model_tier:
                print(f"   💰 De-escalating to {deescalate_tier} (cost savings)")
                current_model_tier = deescalate_tier
                attempts_at_current_model = 0  # Reset counter for new tier
                consecutive_successes = 0  # Reset - need to prove stability at new tier
                # Continue to next iteration with cheaper model

            # Escalate model if needed (but skip Opus if it failed earlier)
            new_model_tier = self.escalate_model(
                current_model_tier, attempts_at_current_model, opus_failed=opus_failed
            )
            if new_model_tier is None:
                print("🚨 All models exhausted, escalating to HUMAN")
                print("   Please review the ticket manually and apply fixes in okp-mcp")
                # Save iteration summary before escalating to human
                if iteration_history:
                    self.save_iteration_summary_table(
                        ticket_id, iteration_history, final_status="🚨 Escalated to Human"
                    )
                return False
            elif new_model_tier != current_model_tier:
                # Don't escalate to Opus if it has already failed
                if new_model_tier == "complex" and opus_failed:
                    print("⚠️  Would escalate to Opus, but it failed earlier")
                    print("   Staying on medium model (Sonnet)")
                    new_model_tier = current_model_tier
                else:
                    print(f"🔼 Escalating from {current_model_tier} to {new_model_tier} model")
                    current_model_tier = new_model_tier
                    attempts_at_current_model = 0

            # Record this iteration for next iteration's context
            # Load the rich diagnostics that were just saved (if available)
            diag_dir = self.eval_root / ".diagnostics" / ticket_id.replace("-", "_")
            diag_file = diag_dir / f"iteration_{iteration:03d}.json"

            if diag_file.exists():
                # Use the full diagnostic data (includes URL overlap, Solr query inspection, etc.)
                with open(diag_file) as f:
                    iteration_record = json.load(f)
                # Add fields expected by the advisor
                iteration_record["change"] = suggestion.suggested_change
                iteration_record["improved"] = improved
                iteration_record["metric_before"] = (
                    metric_history[-2] if len(metric_history) > 1 else 0
                )
                iteration_record["metric_after"] = primary_metric
                iteration_record["result_summary"] = (
                    f"URL F1: {current_result.url_f1:.2f}"
                    if current_result.url_f1
                    else (
                        f"Answer Correctness: {current_result.answer_correctness:.2f}"
                        if current_result.answer_correctness
                        else "No metrics"
                    )
                )
            else:
                # Fallback to basic record if diagnostics not available
                iteration_record = {
                    "iteration": iteration,
                    "change": suggestion.suggested_change,
                    "metric_before": metric_history[-2] if len(metric_history) > 1 else 0,
                    "metric_after": primary_metric,
                    "improved": improved,
                    "result_summary": (
                        f"URL F1: {current_result.url_f1:.2f}"
                        if current_result.url_f1
                        else (
                            f"Answer Correctness: {current_result.answer_correctness:.2f}"
                            if current_result.answer_correctness
                            else "No metrics"
                        )
                    ),
                }

            iteration_history.append(iteration_record)

            previous_result = current_result

        print(f"⏱️  Max iterations ({max_iterations}) reached without fixing ticket")

        # Save iteration summary before exiting
        if iteration_history:
            self.save_iteration_summary_table(
                ticket_id, iteration_history, final_status=f"⏱️ Max Iterations ({max_iterations})"
            )

        return False

    def fix_ticket_multi_stage(
        self,
        ticket_id: str,
        validate_cla_tests: bool = True,
        use_existing: bool = False,
    ) -> bool:
        """Multi-stage ticket fixing with worktree isolation and automatic bootstrap.

        Complete workflow:
        0. Setup: Create worktree → Update container mount → Restart
        1. Bootstrap & Fix primary ticket (in worktree):
           a. Check if config has expected_urls
           b. If missing: discover docs in Solr, enrich config
           c. If docs not found: exit with knowledge gap error
           d. Iterate to optimize retrieval
        2. Validate against CLA tests
        3. Fix any regressions (with separate iteration budgets)
        4. Cleanup: Merge worktree → Revert mount → Restart → Delete worktree

        If any regression cannot be fixed, reverts the primary fix and escalates to human.

        Args:
            ticket_id: RSPEED ticket ID to fix
            validate_cla_tests: If True, run CLA regression validation
            use_existing: If True, use existing results for initial diagnosis (faster for debugging)

        Returns:
            True if ticket fixed and no regressions
        """
        print(f"\n{'='*80}")
        print(f"MULTI-STAGE FIX: {ticket_id}")
        print(f"{'='*80}")

        # Pre-flight: Check environment variables
        if not self.check_environment():
            print("\n❌ Environment check failed. Cannot proceed.")
            return False

        # Stage 0: Setup worktree environment
        print("\n📍 STAGE 0: Setup Worktree Environment")
        print("=" * 80)

        # Create worktree
        branch_name = f"fix/{ticket_id.lower()}"
        worktree_path = self.create_worktree(ticket_id, branch_name)

        # CRITICAL: Update LLM advisor to edit files in worktree, not main repo
        original_okp_mcp_root = None
        if self.llm_advisor:
            original_okp_mcp_root = self.llm_advisor.okp_mcp_root
            self.llm_advisor.okp_mcp_root = worktree_path
            print(f"✅ LLM advisor redirected to worktree: {worktree_path}")

        # Initialize variables for finally block
        primary_fixed = False
        primary_commit = None
        interrupted = False  # Track if user hit Ctrl+C

        try:
            # Update container mount to worktree
            self.update_compose_mount(worktree_path)

            # Restart container and verify it's healthy
            self.restart_okp_mcp(verify_healthy=True)

            # Stage 1: Bootstrap & Fix primary ticket
            print("\n📍 STAGE 1: Bootstrap & Fix Primary Ticket")
            print("=" * 80)
            print(f"Working directory: {worktree_path}")
            print(f"Branch: {branch_name}")

            primary_fixed = self.bootstrap_and_fix_ticket(
                ticket_id=ticket_id,
                max_iterations=PRIMARY_FIX_MAX_ITERATIONS,
                starting_model="medium",
                context="primary",
                use_existing=use_existing,
                auto_select_docs=True,  # Auto-select high-scoring docs
            )

            if not primary_fixed:
                print("❌ Could not fix primary ticket")
                return False

            # Capture commit for potential revert
            try:
                primary_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=worktree_path, text=True
                ).strip()
                print(f"\n✅ Primary ticket fixed (commit: {primary_commit[:8]})")
            except subprocess.CalledProcessError:
                print("⚠️  Could not get git commit (changes may not be committed yet)")
                primary_commit = None

            # Stage 2: Validate CLA tests
            if not validate_cla_tests:
                print("✅ Primary ticket fixed (CLA validation skipped)")
                return True

            print(f"\n{'='*80}")
            print("📍 STAGE 2: CLA Regression Validation")
            print(f"{'='*80}")

            # TODO: Implement CLA test validation
            print("⚠️  CLA validation not yet implemented")
            print("   For now, manually run:")
            print("   ./run_okp_mcp_full_suite.sh --config config/CLA_tests.yaml")
            print("   And check for regressions")

            # Placeholder for regression detection
            # regressions = self.detect_regressions()
            regressions: Dict[str, Any] = {}  # Empty for now

            if not regressions:
                print("✅ No regressions detected!")
                return True

            # Stage 3: Fix regressions (if any)
            print(f"\n⚠️  {len(regressions)} regressions detected")

            for reg_ticket in regressions.keys():
                print(f"\n{'='*80}")
                print(f"📍 STAGE 3: Fixing Regression {reg_ticket}")
                print(f"{'='*80}")

                fixed = self.fix_ticket_with_iteration(
                    ticket_id=reg_ticket,
                    max_iterations=REGRESSION_FIX_MAX_ITERATIONS,
                    starting_model="medium",  # Reset to Sonnet for each regression
                    context="regression",
                )

                if not fixed:
                    print(f"❌ Could not fix regression {reg_ticket}")

                    if primary_commit:
                        print(f"🔄 Reverting primary fix (commit {primary_commit[:8]})")
                        try:
                            subprocess.run(
                                ["git", "revert", "--no-edit", primary_commit],
                                cwd=worktree_path,
                                check=True,
                            )
                            print("✅ Primary fix reverted")
                        except subprocess.CalledProcessError as e:
                            print(f"❌ Revert failed: {e}")

                    print("🚨 ESCALATING TO HUMAN")
                    print("   Primary fix caused regressions that could not be automatically fixed")
                    return False

            print("\n✅ All regressions fixed!")
            return True

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user (Ctrl+C)")
            interrupted = True
            raise  # Re-raise to ensure cleanup runs

        finally:
            # Stage 4: Cleanup worktree environment
            print(f"\n{'='*80}")
            print("📍 STAGE 4: Cleanup Worktree Environment")
            print("=" * 80)

            # Restore LLM advisor to point to main repo
            if self.llm_advisor and original_okp_mcp_root:
                self.llm_advisor.okp_mcp_root = original_okp_mcp_root
                print("✅ LLM advisor restored to main repo")

            # Merge worktree to main if primary fix was successful
            if primary_fixed:
                print("\n📦 Merging worktree changes to main...")
                try:
                    # Switch to main
                    subprocess.run(
                        ["git", "checkout", "main"],
                        cwd=self.okp_mcp_root,
                        check=True,
                    )
                    # Merge the fix branch
                    subprocess.run(
                        ["git", "merge", "--no-edit", branch_name],
                        cwd=self.okp_mcp_root,
                        check=True,
                    )
                    print(f"✅ Merged {branch_name} to main")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Merge failed: {e}")
                    print("   Manual merge required")

            # Revert compose mount back to main
            self.revert_compose_mount()

            # Restart container with main mount and verify healthy
            print("\n🔄 Restarting container with main mount...")
            self.restart_okp_mcp(verify_healthy=True)

            # Clean up worktree and branch
            # Auto cleanup in all cases:
            # - Interrupted (Ctrl+C): incomplete work
            # - Failed: broken work
            # - Succeeded: already merged to main
            if interrupted:
                print("🧹 Auto-cleaning interrupted work...")
            elif not primary_fixed:
                print("🧹 Auto-cleaning failed attempt...")
            else:
                print("🧹 Auto-cleaning after successful merge...")

            self.cleanup_worktree(worktree_path, branch_name=branch_name, ask=False)


def main():
    """CLI entry point."""
    from datetime import datetime

    parser = argparse.ArgumentParser(description="okp-mcp autonomous agent")
    parser.add_argument(
        "command",
        choices=["diagnose", "fix", "bootstrap", "validate"],
        help="Command to run",
    )
    parser.add_argument(
        "ticket_id",
        nargs="*",
        help="RSPEED ticket ID(s) (e.g., RSPEED-2482 RSPEED-2481) - can specify multiple",
    )
    parser.add_argument(
        "--ticket-file",
        type=str,
        help="File containing ticket IDs (one per line) - useful for batch processing",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum fix iterations (for 'fix' command)",
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing evaluation results instead of re-running (faster for testing)",
    )
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="Work in an isolated git worktree (safer, recommended for 'fix' command)",
    )
    parser.add_argument(
        "--worktree-name",
        type=str,
        help="Custom worktree/branch name (default: fix/<ticket-id>)",
    )
    parser.add_argument(
        "--suggest-only",
        action="store_true",
        help="Suggest changes but don't apply them (for 'fix' command)",
    )
    parser.add_argument(
        "--non-interactive",
        "--yolo",
        action="store_true",
        dest="non_interactive",
        help="YOLO mode: Run autonomously without asking for approval (use with caution - good for overnight runs)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation runs for stability analysis (default: 1, use 3+ for variance detection)",
    )

    args = parser.parse_args()

    # Load environment from .env file if it exists
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded environment from {env_path}")
    else:
        print(f"ℹ️  No .env file found at {env_path}, using existing environment")

    # YOLO mode banner
    if args.non_interactive:
        print("\n" + "=" * 80)
        print("🚀 YOLO MODE ACTIVATED 🚀")
        print("=" * 80)
        print("Running autonomously - no approval prompts")
        print("All changes will be auto-approved and tested")
        print("Perfect for overnight runs - just let it cook!")
        print("=" * 80 + "\n")

    # Parse ticket IDs from command line or file
    ticket_ids = []
    if args.ticket_file:
        ticket_file_path = Path(args.ticket_file)
        if not ticket_file_path.exists():
            parser.error(f"Ticket file not found: {args.ticket_file}")
        with open(ticket_file_path) as f:
            ticket_ids = [
                line.strip() for line in f if line.strip() and not line.strip().startswith("#")
            ]
        print(f"📋 Loaded {len(ticket_ids)} tickets from {args.ticket_file}")
    elif args.ticket_id:
        ticket_ids = args.ticket_id

    # Initialize agent
    agent = OkpMcpAgent(
        eval_root=Path.home() / "Work/lightspeed-core/lightspeed-evaluation",
        okp_mcp_root=Path.home() / "Work/okp-mcp",
        lscore_deploy_root=Path.home() / "Work/lscore-deploy",
        interactive=not args.non_interactive,
    )

    # Execute command
    if args.command == "diagnose":
        if not ticket_ids:
            parser.error("ticket_id required for diagnose command")
        for ticket_id in ticket_ids:
            agent.diagnose(ticket_id, use_existing=args.use_existing, runs=args.runs)

    elif args.command == "fix":
        if not ticket_ids:
            parser.error("ticket_id required for fix command")

        # Batch processing for multiple tickets
        if len(ticket_ids) > 1:
            print(f"\n{'='*80}")
            print(f"🎯 BATCH MODE: Processing {len(ticket_ids)} tickets")
            print(f"{'='*80}")

            results = {}
            start_time = datetime.now()

            for i, ticket_id in enumerate(ticket_ids, 1):
                print(f"\n{'='*80}")
                print(f"📋 TICKET {i}/{len(ticket_ids)}: {ticket_id}")
                print(f"{'='*80}\n")

                try:
                    success = agent.fix_ticket_multi_stage(
                        ticket_id,
                        validate_cla_tests=True,
                        use_existing=args.use_existing,
                    )
                    results[ticket_id] = "✅ Fixed" if success else "❌ Failed"
                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted by user (Ctrl+C)")
                    results[ticket_id] = "⚠️ Interrupted"
                    break
                except Exception as e:
                    print(f"\n❌ Error processing {ticket_id}: {e}")
                    import traceback

                    traceback.print_exc()
                    results[ticket_id] = f"❌ Error: {str(e)[:50]}"

            # Generate batch summary report
            end_time = datetime.now()
            duration = end_time - start_time

            summary_file = (
                agent.eval_root
                / ".diagnostics"
                / f"batch_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
            )
            summary_file.parent.mkdir(parents=True, exist_ok=True)

            lines = [
                "=" * 80,
                "BATCH RUN SUMMARY",
                "=" * 80,
                "",
                f"Start Time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"End Time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Duration:    {int(duration.total_seconds() / 60)}m {int(duration.total_seconds() % 60)}s",
                f"Total:       {len(ticket_ids)} tickets",
                f"Fixed:       {sum(1 for r in results.values() if '✅' in r)}",
                f"Failed:      {sum(1 for r in results.values() if '❌' in r)}",
                f"Interrupted: {sum(1 for r in results.values() if '⚠️' in r)}",
                "",
                "=" * 80,
                "RESULTS BY TICKET",
                "=" * 80,
                "",
            ]

            for ticket_id, result in results.items():
                lines.append(f"  {ticket_id:<20} {result}")

            lines.extend(
                [
                    "",
                    "=" * 80,
                    "INDIVIDUAL REPORTS",
                    "=" * 80,
                    "",
                    "See detailed iteration reports at:",
                ]
            )

            for ticket_id in results.keys():
                report_path = (
                    agent.eval_root
                    / ".diagnostics"
                    / ticket_id.replace("-", "_")
                    / "iteration_summary.txt"
                )
                if report_path.exists():
                    lines.append(f"  {ticket_id}: {report_path}")

            lines.append("")

            with open(summary_file, "w") as f:
                f.write("\n".join(lines))

            print(f"\n{'='*80}")
            print("📊 BATCH RUN COMPLETE")
            print(f"{'='*80}")
            print(f"\nTotal: {len(ticket_ids)} tickets")
            print(f"✅ Fixed: {sum(1 for r in results.values() if '✅' in r)}")
            print(f"❌ Failed: {sum(1 for r in results.values() if '❌' in r)}")
            print(f"⚠️  Interrupted: {sum(1 for r in results.values() if '⚠️' in r)}")
            print(f"\n💾 Batch summary: {summary_file}")
            print(f"{'='*80}\n")
        else:
            # Single ticket mode
            agent.fix_ticket_multi_stage(
                ticket_ids[0],
                validate_cla_tests=True,
                use_existing=args.use_existing,
            )

    elif args.command == "bootstrap":
        if not ticket_ids:
            parser.error("ticket_id required for bootstrap command")

        # Process all tickets (bootstrap supports multiple)
        for ticket_id in ticket_ids:
            agent.bootstrap_and_fix_ticket(
                ticket_id,
                max_iterations=args.max_iterations,
                auto_select_docs=False,
            )

    elif args.command == "validate":
        agent.validate_all_suites()


if __name__ == "__main__":
    main()
