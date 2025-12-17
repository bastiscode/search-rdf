#!/usr/bin/env python3
"""Build a text embedding search index from embeddings data."""

import argparse
import sys
from pathlib import Path

import search_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a text embedding search index from embeddings data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing the text embeddings data (must be built first)",
    )
    parser.add_argument(
        "index_dir",
        type=str,
        help="Output directory for the embedding index",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "inner_product", "ip", "l2", "hamming"],
        help="Distance metric to use for similarity search",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading text embeddings from: {data_dir}")
    try:
        embeddings = search_index.TextEmbeddings.load(str(data_dir))
    except Exception as e:
        print(f"Error loading text embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Dimensions: {embeddings.num_dimensions()}")
    print(f"Precision: {embeddings.precision()}")

    print(f"Building embedding index to: {index_dir} (metric: {args.metric})")
    try:
        search_index.TextEmbeddingIndex.build(embeddings, str(index_dir), args.metric)
    except Exception as e:
        print(f"Error building embedding index: {e}", file=sys.stderr)
        sys.exit(1)

    # Make the default query path work by linking data alongside the index
    link_target = index_dir / "data"
    try:
        if link_target.is_symlink() or not link_target.exists():
            # Replace broken symlink with a fresh absolute link
            if link_target.is_symlink():
                link_target.unlink(missing_ok=True)
            link_target.symlink_to(data_dir, target_is_directory=True)
    except Exception as e:
        print(
            f"Warning: Could not create data symlink at {link_target}: {e}",
            file=sys.stderr,
        )

    print("Embedding index built successfully!")


if __name__ == "__main__":
    main()
