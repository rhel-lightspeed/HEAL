#!/usr/bin/env python3
"""Apply multi-agent suggestion to okp-mcp codebase.

Reads suggestion JSON and applies the code change automatically.

Usage:
    python src/heal/runners/apply_suggestion.py \\
        --suggestion .diagnostics/PATTERN_ID/suggestion.json \\
        --okp-mcp-root /path/to/okp-mcp
"""

import argparse
import json
import sys
from pathlib import Path


def apply_suggestion(suggestion_path: Path, okp_mcp_root: Path) -> bool:
    """Apply suggested code change.

    Args:
        suggestion_path: Path to suggestion JSON file
        okp_mcp_root: Root of okp-mcp repository

    Returns:
        True if successful, False otherwise
    """
    # Load suggestion
    with open(suggestion_path) as f:
        suggestion = json.load(f)

    file_path = suggestion.get("file_path")
    old_code = suggestion.get("old_code")
    new_code = suggestion.get("new_code")

    if not all([file_path, old_code, new_code]):
        print("❌ Suggestion missing required fields: file_path, old_code, new_code")
        return False

    # Resolve file path
    target_file = okp_mcp_root / file_path
    if not target_file.exists():
        print(f"❌ Target file not found: {target_file}")
        return False

    print(f"📝 Applying suggestion to: {target_file}")
    print(f"   File: {file_path}")
    print()

    # Read current content
    content = target_file.read_text()

    # Check if old_code exists
    if old_code not in content:
        print("❌ Old code not found in file!")
        print()
        print("Expected to find:")
        print("-" * 80)
        print(old_code[:500])
        if len(old_code) > 500:
            print(f"... ({len(old_code) - 500} more chars)")
        print("-" * 80)
        print()
        print("This might mean:")
        print("  - The suggestion is outdated (code already changed)")
        print("  - Whitespace/formatting differences")
        print("  - The suggestion was for a different version of the code")
        print()
        print("You can manually apply the fix by editing:")
        print(f"  {target_file}")
        return False

    # Apply replacement
    new_content = content.replace(old_code, new_code, 1)

    # Verify it actually changed
    if new_content == content:
        print("⚠️  Warning: Replacement made no changes to file content")
        return False

    # Write back
    target_file.write_text(new_content)

    print("✅ Code change applied successfully!")
    print()
    print("Changed:")
    print("-" * 80)
    print(f"  {old_code[:200]}...")
    print()
    print("To:")
    print("-" * 80)
    print(f"  {new_code[:200]}...")
    print("-" * 80)
    print()

    return True


def main():
    parser = argparse.ArgumentParser(description="Apply multi-agent suggestion to okp-mcp codebase")
    parser.add_argument(
        "--suggestion",
        type=Path,
        required=True,
        help="Path to suggestion JSON file",
    )
    parser.add_argument(
        "--okp-mcp-root",
        type=Path,
        required=True,
        help="Root directory of okp-mcp repository",
    )

    args = parser.parse_args()

    if not args.suggestion.exists():
        print(f"❌ Suggestion file not found: {args.suggestion}")
        sys.exit(1)

    if not args.okp_mcp_root.exists():
        print(f"❌ OKP-MCP root not found: {args.okp_mcp_root}")
        sys.exit(1)

    success = apply_suggestion(args.suggestion, args.okp_mcp_root)

    if not success:
        sys.exit(1)

    print("💡 Next step: Test the fix")
    print(f"   cd {args.okp_mcp_root}")
    print("   git diff  # Review changes")
    print()


if __name__ == "__main__":
    main()
