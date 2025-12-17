#!/usr/bin/env python3
"""Build a keyword search index from a TSV file."""

import argparse
import sys
import tempfile
from pathlib import Path

import search_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a keyword search index from a TSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "tsv_file",
        type=str,
        help="Input TSV file with text data (first column is identifier)",
    )
    parser.add_argument(
        "index_dir",
        type=str,
        help="Output directory for the keyword index",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Save text data to this directory (default: <index_dir>/data)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    tsv_file = Path(args.tsv_file)
    if not tsv_file.exists():
        print(f"Error: TSV file does not exist: {tsv_file}", file=sys.stderr)
        sys.exit(1)

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Determine data directory location
    if args.data_dir:
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        use_temp_dir = False
    else:
        data_dir = index_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        use_temp_dir = False

    print("=" * 80)
    print("Step 1: Building text data")
    print("=" * 80)

    print(f"Building text data from: {tsv_file}")
    print(f"Output directory: {data_dir}")

    try:
        search_index.TextData.build(str(tsv_file), str(data_dir))
    except Exception as e:
        print(f"Error building text data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading text data from: {data_dir}")
    try:
        text_data = search_index.TextData.load(str(data_dir))
    except Exception as e:
        print(f"Error loading text data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(text_data)} documents")

    print()
    print("=" * 80)
    print("Step 2: Building keyword index")
    print("=" * 80)

    print(f"Building keyword index to: {index_dir}")
    try:
        search_index.KeywordIndex.build(text_data, str(index_dir))
    except Exception as e:
        print(f"Error building keyword index: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 80)
    print("Verification")
    print("=" * 80)

    print("Loading keyword index to verify...")
    try:
        index = search_index.KeywordIndex.load(text_data, str(index_dir))
        print("Successfully built keyword index:")
        print(f"  Documents: {len(index)}")
    except Exception as e:
        print(f"Warning: Could not load built index: {e}", file=sys.stderr)

    print()
    print("=" * 80)
    print("Success!")
    print("=" * 80)
    print(f"Keyword index saved to: {index_dir}")
    print(f"Text data saved to: {data_dir}")


if __name__ == "__main__":
    main()
