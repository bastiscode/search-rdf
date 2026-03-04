# Wikidata Movie Descriptions

Build a full-text and embedding search index for English Wikipedia
abstracts of movies in Wikidata.

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
# Simple full text search
curl -X POST http://localhost:8080/search/wikidata-movies-full-text \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"type": "value", "value": "chastain nolan space"}], "k": 10}'

# Simple embedding search
curl -X POST http://localhost:8080/search/wikidata-movies-embedding \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"type": "value", "value": "space movie by nolan"}], "k": 10}'
```

## Notes

- Query fetches instances of and subclasses of movies (Q11424) with
their English Wikipedia abstracts
- Limited to 10,000 results (adjust `LIMIT` in config.yaml for more)
- Puts data in `data/` and indices in `index/`
