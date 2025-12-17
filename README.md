# Search Index V2

Rust library for building and querying search indices with clean trait-based interfaces.

## Features

- **Multiple search strategies**: Keyword search and embedding-based similarity search
- **Flexible data sources**: Trait-based abstraction for different data types
- **Efficient filtering**: Filter search results by ID predicates
- **Type-safe**: Strong typing with Rust's trait system

## Architecture

- **DataSource trait**: Abstraction for indexed data
- **SearchIndex trait**: Common interface for all index types
- **KeywordSearch**: Fast prefix-based keyword matching
- **EmbeddingSearch**: Semantic similarity search using embeddings

## Status

Currently in development. This is a rewrite of the original search-index with improved architecture.
