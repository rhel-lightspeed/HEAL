"""Test fixtures for evaluation logic testing."""

import json
from pathlib import Path
from typing import Dict, List


FIXTURES_DIR = Path(__file__).parent


def load_fixture(fixture_name: str) -> dict:
    """Load a test fixture.

    Args:
        fixture_name: Relative path to fixture JSON (e.g., "bootloader_grub_pattern/run_001_results.json")

    Returns:
        Dict with fixture data
    """
    fixture_path = FIXTURES_DIR / fixture_name

    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    with open(fixture_path) as f:
        return json.load(f)


def get_mock_per_ticket_results(fixture_name: str) -> Dict[str, Dict]:
    """Get per_ticket_data from fixture for mocking (NEW format).

    Args:
        fixture_name: Fixture to load

    Returns:
        Dict[ticket_id, {"runs": [...], "metadata": {...}}]
        suitable for mocking parse_results_per_ticket() (NEW format)
    """
    fixture = load_fixture(fixture_name)
    # Return NEW format with runs + metadata
    return fixture.get("per_ticket_data", fixture.get("per_ticket_results", {}))
