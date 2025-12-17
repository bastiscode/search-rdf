#!/usr/bin/env python3
"""Query a text embedding search index."""

import argparse
import sys
from pathlib import Path

import search_index
from search_index.model.embedding import TextEmbeddingModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query a text embedding search index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "index_dir",
        type=str,
        help="Directory containing the embedding index",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the text embeddings data (default: <index_dir>/data)",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query string (will be embedded using the model)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to return",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum score threshold",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use exact search instead of approximate",
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
        print(
            f"Error: Data directory does not exist: {data_dir}\n"
            "Pass --data-dir with the path used to build TextEmbeddings, or ensure "
            "a data symlink exists at the index directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading text embeddings from: {data_dir}")
    try:
        embeddings = search_index.TextEmbeddings.load(str(data_dir))
    except Exception as e:
        print(f"Error loading text embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    data_precision = embeddings.precision()
    data_dim = embeddings.num_dimensions()
    model = embeddings.model()

    print(f"Embeddings precision: {data_precision}")
    print(f"Embeddings dimensions: {data_dim}")

    print(f"Loading embedding index from: {index_dir}")
    try:
        index = search_index.TextEmbeddingIndex.load(embeddings, str(index_dir))
    except Exception as e:
        print(f"Error loading embedding index: {e}", file=sys.stderr)
        sys.exit(1)

    # Load the embedding model (handles precision/dimension conversion)
    print(f"Loading model: {model}")
    try:
        model = TextEmbeddingModel(model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # Embed the query
    print(f"Embedding query: '{args.query}'")
    # Match stored dimensions; data_dim is already in bits for ubinary
    target_dim = data_dim
    try:
        query_embedding = model.embed(
            [args.query],
            precision=data_precision,
            embedding_dim=target_dim,
            batch_size=1,
            show_progress=False,
        )[0]
        search_vector: bytes | list[float]
        if data_precision == "ubinary":
            search_vector = query_embedding.tobytes()
        else:
            search_vector = query_embedding.tolist()
    except Exception as e:
        print(f"Error embedding query: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Search vector type and length: {type(search_vector)}, {len(search_vector)}")
    print(f"\nSearching (k={args.k}, exact={args.exact})")
    if args.min_score is not None:
        print(f"Min score: {args.min_score}")
    print("=" * 80)

    try:
        results = index.search_embedding(
            search_vector,
            k=args.k,
            min_score=args.min_score,
            exact=args.exact,
        )
    except Exception as e:
        print(f"Error searching: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} results:\n")

    # Get the underlying text data
    text_data = embeddings.text_data()

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
