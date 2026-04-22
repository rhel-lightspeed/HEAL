#!/usr/bin/env python3
"""Inspect what's in the mcp_direct_cache."""

import sys
import sqlite3
import pickle
from pathlib import Path

# Add lightspeed-evaluation to path for unpickling
EVAL_ROOT = Path("/home/emackey/Work/lightspeed-core/lightspeed-evaluation")
sys.path.insert(0, str(EVAL_ROOT / "src"))

cache_db = Path("/home/emackey/Work/lightspeed-core/lightspeed-evaluation/.caches/mcp_direct_cache/cache.db")

conn = sqlite3.connect(cache_db)
cursor = conn.cursor()

# Get all cache entries
cursor.execute("SELECT key, value FROM Cache")

for i, (key, value) in enumerate(cursor.fetchall(), 1):
    print(f"\n{'='*80}")
    print(f"Cache Entry #{i}")
    print(f"{'='*80}")
    print(f"Key (hash): {str(key)[:16]}...")

    try:
        # Try to unpickle the value
        data = pickle.loads(value)

        print(f"Type: {type(data).__name__}")

        # It's an APIResponse object
        if hasattr(data, 'tool_calls') and data.tool_calls:
            # Extract query from first tool call
            first_tool = data.tool_calls[0][0] if isinstance(data.tool_calls[0], list) else data.tool_calls[0]
            query = first_tool.get('arguments', {}).get('query', 'N/A')
            print(f"Query: {query}")

        if hasattr(data, 'retrieved_contexts'):
            contexts = data.retrieved_contexts
            print(f"Retrieved contexts: {len(contexts) if contexts else 0} docs")
            if contexts:
                first_doc = str(contexts[0])[:200]
                print(f"  First doc: {first_doc}...")

        if hasattr(data, 'contexts'):
            contexts = data.contexts
            print(f"Contexts: {len(contexts) if contexts else 0} items")
            if contexts:
                first_ctx = str(contexts[0])[:200]
                print(f"  First context: {first_ctx}...")

        if hasattr(data, 'response'):
            resp_preview = data.response[:150] if data.response else "(empty)"
            print(f"Response: {resp_preview}...")

        # Show all attributes
        attrs = [a for a in dir(data) if not a.startswith('_')]
        print(f"All attributes: {', '.join(attrs[:10])}")

    except Exception as e:
        print(f"Error unpickling: {e}")
        print(f"Raw value size: {len(value)} bytes")

conn.close()
