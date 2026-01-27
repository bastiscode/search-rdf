# Search RDF

Rust library with restricted Python interface for building and querying
search indices, primarily intended to be used with RDF query engines.

## Getting Started

### Installation

Build from source using Cargo:

```bash
cargo build --release
```

The binary will be available at `target/release/search-rdf`.

### CLI Overview

The `search-rdf` CLI provides commands to build and serve search indices. All commands require a YAML configuration file.

```
search-rdf [OPTIONS] [CONFIG] [COMMAND]

Commands:
  data    Download and prepare data
  embed   Generate embeddings for data
  index   Build search indices
  serve   Serve indices via HTTP

Options:
      --force    Force rebuild even if output exists
  -v, --verbose  Enable verbose/debug logging
  -q, --quiet    Suppress info messages (errors and warnings only)
  -h, --help     Print help
  -V, --version  Print version
```

#### Running All Steps

To run the complete pipeline (data → embed → index → serve):

```bash
search-rdf config.yaml
```

#### Running Individual Steps

```bash
# Step 1: Download/prepare data
search-rdf data config.yaml

# Step 2: Generate embeddings
search-rdf embed config.yaml

# Step 3: Build indices
search-rdf index config.yaml

# Step 4: Start HTTP server
search-rdf serve config.yaml
```

Use `--force` to rebuild outputs even if they already exist:

```bash
search-rdf index config.yaml --force
```

### Configuration File Format

The configuration file is written in YAML and has five main sections: `datasets`, `models`, `embeddings`, `indices`, and `server`.

#### Datasets

Defines data sources to be indexed. Each dataset produces a data directory used by indices.

```yaml
datasets:
  - name: my-dataset           # Unique identifier
    output: data/              # Output directory for processed data
    source:
      # Option 1: SPARQL query against an endpoint
      type: sparql-query
      endpoint: https://query.wikidata.org/sparql
      query: |
        SELECT ?item ?label WHERE {
          ?item rdfs:label ?label .
        }
        LIMIT 1000
      format: json             # json, xml, or tsv
      default_field_type: text # text, image, or image-inline
      headers:                 # Optional HTTP headers
        User-Agent: MyApp/1.0

      # Option 2: Local SPARQL results file
      type: sparql
      path: results.json
      format: json
      default_field_type: text

      # Option 3: JSONL file
      type: jsonl
      path: data.jsonl
```

SPARQL queries must return exactly 2 columns: an identifier (first column) and a field value (second column). Multiple rows with the same identifier create multiple fields for that item.

#### Models

Defines embedding models used to generate vector representations.

```yaml
models:
  # vLLM server (recommended for large-scale embedding)
  - name: my-vllm-model
    type: vllm
    endpoint: http://localhost:8000
    model_name: mixedbread-ai/mxbai-embed-large-v1

  # Sentence Transformers (local inference)
  - name: my-local-model
    type: sentence-transformer
    model_name: sentence-transformers/all-MiniLM-L6-v2
    device: cuda                # cpu, cuda, or mps (default: cpu)
    batch_size: 16              # Inference batch size (default: 16)

  # HuggingFace image models
  - name: my-image-model
    type: huggingface-image
    model_name: openai/clip-vit-base-patch32
    device: cuda
    batch_size: 16
```

Optional embedding parameters can be added to any model:

```yaml
models:
  - name: my-model
    type: vllm
    endpoint: http://localhost:8000
    model_name: mixedbread-ai/mxbai-embed-large-v1
    params:
      num_dimensions: 512      # Truncate embeddings (for MRL models)
      normalize: true          # L2 normalize embeddings (default: true)
```

#### Embeddings

Defines embedding generation jobs that use models to embed dataset fields.

```yaml
embeddings:
  - name: my-embeddings
    model: my-vllm-model       # Reference to model name
    data: data/                # Input data directory
    output: data/embeddings.safetensors
    batch_size: 64             # Processing batch size (default: 64)
```

#### Indices

Defines search indices to build from data and embeddings.

```yaml
indices:
  # Keyword index (exact token matching with BM25 scoring)
  - name: keyword-index
    type: keyword
    data: data/
    output: index/keyword/

  # Full-text index (Tantivy-based with stemming/tokenization)
  - name: fulltext-index
    type: full-text
    data: data/
    output: index/fulltext/

  # Embedding index with data (semantic search)
  - name: embedding-index
    type: embedding-with-data
    data: data/
    embedding_data: data/embeddings.safetensors
    output: index/embedding/
    model: my-vllm-model       # For query embedding at search time

  # Embedding-only index (no associated text data)
  - name: embedding-only
    type: embedding
    embedding_data: data/embeddings.safetensors
    output: index/embedding-only/
```

Embedding index parameters:

```yaml
indices:
  - name: embedding-index
    type: embedding-with-data
    data: data/
    embedding_data: data/embeddings.safetensors
    output: index/embedding/
    model: my-model
    params:
      metric: cosine-normalized  # cosine-normalized, cosine, inner-product, l2, hamming
      precision: bfloat16        # float32, float16, bfloat16, int8, binary
      connectivity: 16           # HNSW M parameter (default: 16)
      expansion_add: 128         # HNSW efConstruction (default: 128)
      expansion_search: 64       # HNSW ef (default: 64)
```

#### Server

Configures the HTTP server for serving indices.

```yaml
server:
  host: 0.0.0.0                 # Bind address (default: 127.0.0.1)
  port: 8080                    # Port (default: 8080)
  cors: true                    # Enable CORS (default: false)
  max_input_size: 100MB         # Max request size in bytes (default: 100MB)
  indices:                      # Indices to serve
    - keyword-index
    - embedding-index
  sparql:                       # Optional: Enable SPARQL service endpoints
    prefix: "http://example.org/"
```

### HTTP API

When the server is running, the following endpoints are available:

#### Health Check

```
GET /health
```

Returns `200 OK` if the server is running.

#### List Indices

```
GET /indices
```

Returns a list of available index names.

#### Search

```
POST /search/{index_name}
Content-Type: application/json
```

The request body contains a `queries` array and search parameters. Query format depends on the index type:

**Text queries** (for keyword, full-text, and text embedding indices):

```json
{
  "queries": [{"type": "text", "value": "search query"}],
  "k": 10
}
```

**Image URL queries** (for image embedding indices):

```json
{
  "queries": [{"type": "url", "value": "https://example.com/image.jpg"}],
  "k": 10
}
```

**Pre-computed embedding queries**:

```json
{
  "queries": [{"type": "embedding", "value": [0.1, 0.2, 0.3, ...]}],
  "k": 10
}
```

Search parameters vary by index type:

**Keyword/Full-text indices:**

- `k` - Number of results (default: 10)

**Embedding indices:**

- `k` - Number of results (default: 10)
- `min-score` - Minimum similarity score filter
- `exact` - Use exact search instead of approximate (default: false)
- `rerank` - Reranking factor (retrieves k*rerank candidates, then reranks)

Response format:

```json
{
  "matches": [
    [
      {"id": 42, "score": 0.95},
      {"id": 17, "score": 0.87}
    ]
  ]
}
```

#### SPARQL Service (optional)

When `sparql` is configured in the server section:

```
POST /service/{index_name}
POST /qlproxy/{index_name}
```

These endpoints enable integration with SPARQL engines that support federated queries.

### Example Configuration

Here's a complete example that sets up keyword and semantic search over Wikidata human labels:

```yaml
datasets:
  - name: wikidata-humans
    output: data/
    source:
      type: sparql-query
      endpoint: https://query.wikidata.org/sparql
      query: |
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT ?item ?label WHERE {
          ?item wdt:P31 wd:Q5 .
          ?item rdfs:label ?label .
          FILTER(LANG(?label) = "en")
        }
        LIMIT 10000
      format: json
      default_field_type: text

models:
  - name: text-embedding
    type: vllm
    endpoint: http://localhost:8000
    model_name: mixedbread-ai/mxbai-embed-xsmall-v1

embeddings:
  - name: wikidata-embeddings
    model: text-embedding
    data: data/
    output: data/embeddings.safetensors
    batch_size: 128

indices:
  - name: keyword
    type: keyword
    data: data/
    output: index/keyword/

  - name: semantic
    type: embedding-with-data
    data: data/
    embedding_data: data/embeddings.safetensors
    output: index/semantic/
    model: text-embedding
    params:
      metric: cosine-normalized
      precision: bfloat16

server:
  host: 0.0.0.0
  port: 8080
  cors: true
  indices:
    - keyword
    - semantic
```

Run with:

```bash
# Build everything and start serving
search-rdf config.yaml

# Or run steps individually
search-rdf data config.yaml
search-rdf embed config.yaml
search-rdf index config.yaml
search-rdf serve config.yaml
```

Test with curl:

```bash
# Keyword search
curl -X POST http://localhost:8080/search/keyword \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"type": "text", "value": "Albert Einstein"}], "k": 5}'

# Semantic search
curl -X POST http://localhost:8080/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"type": "text", "value": "famous physicist"}], "k": 5}'
```
