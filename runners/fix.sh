#!/usr/bin/env bash
# Fix Patterns - Run the pattern fix loop for evaluation and improvement
#
# This script runs the agentic fix loop for patterns:
# 1. Run evaluation on pattern tickets
# 2. Diagnose failures with OKP MCP Agent
# 3. Suggest fixes to okp-mcp codebase
# 4. Re-run evaluation to validate
# 5. Repeat until stable or max iterations
#
# Usage:
#   ./runners/fix.sh                      # Run all patterns (batch mode)
#   ./runners/fix.sh PATTERN_ID           # Run single pattern
#   ./runners/fix.sh --quick              # Quick test (2 iterations, single ticket)

set -e  # Exit on error

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default settings
MAX_ITERATIONS=10
STABILITY_RUNS=3
VALIDATION_CYCLES=3  # Outer loop cycles (answer validations) - default 3 for correlation data
MODE="full"  # full = test all tickets, single = test one ticket per pattern

# Integration flags (both default to OFF for safety)
CREATE_PR=false
NO_JIRA_UPDATES=true  # Jira updates DISABLED by default - use --enable-jira to enable
DRY_RUN_INTEGRATIONS=false
YOLO_MODE=false  # Interactive review ENABLED by default - use --yolo to disable

# Parse arguments
PATTERN_ID=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MAX_ITERATIONS=2
            STABILITY_RUNS=2
            MODE="single"
            shift
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --stability-runs)
            STABILITY_RUNS="$2"
            shift 2
            ;;
        --validation-cycles)
            VALIDATION_CYCLES="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            if [[ "$MODE" != "single" && "$MODE" != "full" ]]; then
                echo "Error: --mode must be 'single' or 'full'"
                exit 1
            fi
            shift 2
            ;;
        --create-pr)
            CREATE_PR=true
            shift
            ;;
        --enable-jira)
            NO_JIRA_UPDATES=false  # Enable Jira updates
            shift
            ;;
        --no-jira-updates)
            NO_JIRA_UPDATES=true  # Kept for backwards compatibility
            shift
            ;;
        --dry-run-integrations)
            DRY_RUN_INTEGRATIONS=true
            shift
            ;;
        --yolo)
            YOLO_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [PATTERN_ID] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  PATTERN_ID            Run fix loop on specific pattern (default: all patterns)"
            echo ""
            echo "Options:"
            echo "  --quick                    Quick test mode (2 iterations, single ticket)"
            echo "  --max-iterations N         Max Solr iterations per cycle (default: 10)"
            echo "  --validation-cycles N      Outer loop cycles with full answer validation (default: 3)"
            echo "  --stability-runs N         Evaluations to confirm stability (default: 3)"
            echo "  --mode single|full         Test mode - single ticket or full pattern (default: full)"
            echo ""
            echo "Integration Options (both DEFAULT: OFF):"
            echo "  --enable-jira              Enable Jira comment updates (default: disabled)"
            echo "  --create-pr                Create GitHub PR after successful fix (requires gh CLI)"
            echo "  --dry-run-integrations     Preview integration actions without executing"
            echo ""
            echo "Review Options:"
            echo "  --yolo                     YOLO mode: auto-approve changes (default: interactive review)"
            echo ""
            echo "  --help                     Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all patterns (full mode)"
            echo "  $0 BOOTLOADER_GRUB_ISSUES             # Run single pattern"
            echo "  $0 --quick                            # Quick test all patterns"
            echo "  $0 BOOTLOADER_GRUB_ISSUES --quick     # Quick test single pattern"
            echo "  $0 --max-iterations 5 --mode single   # Custom params"
            exit 0
            ;;
        *)
            # Assume it's a pattern ID
            if [[ -z "$PATTERN_ID" ]]; then
                PATTERN_ID="$1"
                shift
            else
                echo "Unknown option: $1"
                echo "Run with --help for usage"
                exit 1
            fi
            ;;
    esac
done

# Print banner
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}   HEAL - Pattern Fix Loop${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check patterns directory exists
if [ ! -d "config/patterns" ]; then
    echo -e "${YELLOW}⚠️  Patterns directory not found: config/patterns${NC}"
    echo ""
    echo "Run pattern discovery and split first:"
    echo "  ./runners/pattern.sh"
    echo "  ./runners/split.sh"
    exit 1
fi

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    uv sync --group dev
fi

# Run single pattern or batch mode
if [ -n "$PATTERN_ID" ]; then
    # Single pattern mode
    echo -e "${BLUE}Mode:${NC} Single pattern"
    echo -e "${BLUE}Pattern:${NC} $PATTERN_ID"
    echo -e "${BLUE}Test mode:${NC} $MODE"
    echo -e "${BLUE}Max Solr iterations:${NC} $MAX_ITERATIONS (per cycle)"
    echo -e "${BLUE}Validation cycles:${NC} $VALIDATION_CYCLES (outer loop with full answer eval)"
    echo -e "${BLUE}Stability runs:${NC} $STABILITY_RUNS"
    echo -e "${BLUE}Date:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Check pattern file exists
    if [ ! -f "config/patterns/${PATTERN_ID}.yaml" ]; then
        echo -e "${YELLOW}⚠️  Pattern file not found: config/patterns/${PATTERN_ID}.yaml${NC}"
        echo ""
        echo "Available patterns:"
        ls config/patterns/*.yaml | xargs -n1 basename | sed 's/.yaml$//' | sed 's/^/  - /'
        exit 1
    fi

    echo -e "${BOLD}${GREEN}Running fix loop on $PATTERN_ID...${NC}"
    echo ""

    # Run single pattern
    PYTHONUNBUFFERED=1 uv run python src/heal/runners/run_pattern_fix_poc.py \
        "$PATTERN_ID" \
        --mode "$MODE" \
        --max-iterations "$MAX_ITERATIONS" \
        --validation-cycles "$VALIDATION_CYCLES" \
        --stability-runs "$STABILITY_RUNS" \
        $([ "$CREATE_PR" = true ] && echo "--create-pr") \
        $([ "$NO_JIRA_UPDATES" = true ] && echo "--no-jira-updates") \
        $([ "$DRY_RUN_INTEGRATIONS" = true ] && echo "--dry-run-integrations") \
        $([ "$YOLO_MODE" = true ] && echo "--yolo")

else
    # Batch mode - all patterns
    echo -e "${BLUE}Mode:${NC} Batch (all patterns)"
    echo -e "${BLUE}Test mode:${NC} $MODE"
    echo -e "${BLUE}Max Solr iterations:${NC} $MAX_ITERATIONS (per cycle)"
    echo -e "${BLUE}Validation cycles:${NC} $VALIDATION_CYCLES (outer loop with full answer eval)"
    echo -e "${BLUE}Stability runs:${NC} $STABILITY_RUNS"
    echo -e "${BLUE}Date:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    echo -e "${BOLD}${GREEN}Running fix loop on all patterns...${NC}"
    echo ""

    # Run batch script
    exec scripts/run_all_pattern_fixes.sh \
        --max-iterations "$MAX_ITERATIONS" \
        --validation-cycles "$VALIDATION_CYCLES" \
        --stability-runs "$STABILITY_RUNS" \
        --mode "$MODE"
fi
