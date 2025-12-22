# Wikidata Humans

Build a keyword and embedding search index for English labels and
aliases of all humans in Wikidata.

## Usage

```bash
# Start a vLLM embedding server
vllm serve mixedbread-ai/mxbai-embed-xsmall-v1 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 65536

# Do everything in one command:
search-rdf config.yaml

# Or do it step by step:
# Fetch and index data from Wikidata
search-rdf data config.yaml

# Build indices
search-rdf index config.yaml

# Start server on http://localhost:8080
search-rdf serve config.yaml
```

## Search Examples

```bash
# Simple keyword search
curl -X POST http://localhost:8080/search/wikidata-humans-keyword \
  -H "Content-Type: application/json" \
  -d '{"query": ["Einstein"], "k": 10}'

# Batched keyword search
curl -X POST http://localhost:8080/search/wikidata-humans-keyword \
  -H "Content-Type: application/json" \
  -d '{"query": ["Einstein", "Peter Parker"], "k": 10}'

# Embedding search
curl -X POST http://localhost:8080/search/wikidata-humans-embedding \
  -H "Content-Type: application/json" \
  -d '{"query": ["Einstein"], "k": 10}'
```

## Notes

- Query fetches instances of human (Q5) with English labels and aliases
- Limited to 100,000 results (adjust `LIMIT` in config.yaml for more)
- Puts data in `data/` and indices in `indices/`
