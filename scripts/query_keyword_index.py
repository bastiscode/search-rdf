#!/usr/bin/env python3
"""Query a keyword search index."""

import argparse
import sys
from pathlib import Path

import search_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query a keyword search index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "index_dir",
        type=str,
        help="Directory containing the keyword index",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the text data (default: <index_dir>/data)",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query string",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to return",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        print(f"Error: Index directory does not exist: {index_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine data directory location
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = index_dir / "data"

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading text data from: {data_dir}")
    try:
        text_data = search_index.TextData.load(str(data_dir))
    except Exception as e:
        print(f"Error loading text data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading keyword index from: {index_dir}")
    try:
        index = search_index.KeywordIndex.load(text_data, str(index_dir))
    except Exception as e:
        print(f"Error loading keyword index: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nSearching for: '{args.query}' (k={args.k})")
    print("=" * 80)

    try:
        results = index.search(args.query, args.k)
    except Exception as e:
        print(f"Error searching: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} results:\n")

    for i, (doc_id, field_idx, score) in enumerate(results, 1):
        # Get the matched field content
        field_content = text_data.field(doc_id, field_idx)
        # Get the identifier
        identifier = text_data.identifier(doc_id) or "N/A"

        print(f"{i}. [ID: {identifier}] Score: {score:.4f}")
        print(f"   Field {field_idx}: {field_content}")
        print()


if __name__ == "__main__":
    main()
