#!/usr/bin/env python3
"""Build text embeddings from a TSV file by generating embeddings and creating the data structure."""

import argparse
import sys
from pathlib import Path

import numpy as np
import search_index
from safetensors.numpy import save_file
from search_index.model.embedding import TextEmbeddingModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build text embeddings from a TSV file by generating embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "tsv_file",
        type=str,
        help="Input TSV file with text data (first column is identifier)",
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Output directory for the text embeddings data",
    )
    parser.add_argument(
        "--embeddings-file",
        type=str,
        default=None,
        help="Save embeddings to this file (default: <data_dir>/embeddings.safetensors)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Sentence transformer model to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "ubinary"],
        help="Embedding precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (cuda, cpu, or None for auto)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Target embedding dimension (must be <= model dimension)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    tsv_file = Path(args.tsv_file)
    if not tsv_file.exists():
        print(f"Error: TSV file does not exist: {tsv_file}", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine embeddings file location
    # Default: save directly into data_dir (no symlink needed)
    if args.embeddings_file:
        embeddings_file = Path(args.embeddings_file)
        embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        embeddings_file = data_dir / "embeddings.safetensors"

    print("=" * 80)
    print("Step 1: Building text data")
    print("=" * 80)

    print(f"Building text data to: {data_dir}")
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
    print("Step 2: Generating embeddings for all fields")
    print("=" * 80)

    # Collect all fields to embed
    print("Collecting fields to embed...")
    all_fields = []
    num_fields = []

    for fields in text_data:
        if fields:
            all_fields.extend(fields)
            num_fields.append(len(fields))

    print(f"Total fields to embed: {len(all_fields)}")

    if not all_fields:
        print("Error: No fields found to embed", file=sys.stderr)
        sys.exit(1)

    # Load embedding model
    print(f"Loading embedding model: {args.model}")
    try:
        embedding_model = TextEmbeddingModel(args.model, device=args.device)
    except Exception as e:
        print(f"Error loading embedding model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Model dimension: {embedding_model.dim}")
    if args.embedding_dim:
        print(f"Target dimension: {args.embedding_dim}")

    # Generate embeddings
    print("Generating embeddings...")
    try:
        embeddings = embedding_model.embed(
            all_fields,
            precision=args.precision,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            show_progress=True,
        )
    except Exception as e:
        print(f"Error generating embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Save to safetensors
    print(f"Saving embeddings to: {embeddings_file}")

    try:
        save_file(
            {"embeddings": embeddings, "split": np.array(num_fields)},
            str(embeddings_file),
            metadata={
                "precision": args.precision,
                "model": args.model,
            },
        )
    except Exception as e:
        print(f"Error saving embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 80)
    print("Step 3: Building text embeddings data structure")
    print("=" * 80)

    print("Building text embeddings from:")
    print(f"  TSV file: {tsv_file}")
    print(f"  Embeddings file: {embeddings_file}")
    print(f"  Output directory: {data_dir}")

    try:
        search_index.TextEmbeddings.build(
            str(tsv_file), str(embeddings_file), str(data_dir)
        )
    except Exception as e:
        print(f"Error building text embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 80)
    print("Verification")
    print("=" * 80)

    print("Loading built text embeddings to verify...")
    try:
        embeddings_data = search_index.TextEmbeddings.load(str(data_dir))
        print("Successfully built text embeddings:")
        print(f"  Documents: {len(embeddings_data)}")
        print(f"  Dimensions: {embeddings_data.num_dimensions()}")
        print(f"  Precision: {embeddings_data.precision()}")
    except Exception as e:
        print(f"Warning: Could not load built embeddings: {e}", file=sys.stderr)

    print()
    print("=" * 80)
    print("Success!")
    print("=" * 80)
    print(f"Text embeddings data saved to: {data_dir}")
    print(f"Embeddings file: {embeddings_file}")


if __name__ == "__main__":
    main()
