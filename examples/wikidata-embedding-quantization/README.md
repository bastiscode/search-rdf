# Wikidata Humans Embedding Quantization

Build a embedding indices with different precision
for English labels of all humans in Wikidata.

## Usage

```bash
# Start a vLLM embedding server
vllm serve mixedbread-ai/mxbai-embed-xsmall-v1 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 65536

# Do everything in one command:
search-rdf config.yaml

# Or do it step by step:
# Fetch data from Wikidata
search-rdf data config.yaml

# Build keyword index
search-rdf index config.yaml

# Start server on http://localhost:8080
search-rdf serve config.yaml
```

## Search Examples

```bash
# Simple embedding search with FP32 index
curl -X POST http://localhost:8080/search/wikidata-movies-embedding-fp32 \
  -H "Content-Type: application/json" \
  -d '{"query": ["space movie by nolan"], "k": 10}'

# Simple embedding search with binary index
curl -X POST http://localhost:8080/search/wikidata-movies-embedding-binary \
  -H "Content-Type: application/json" \
  -d '{"query": ["space movie by nolan"], "k": 10}'
```

## Notes

- Query fetches instances of human (Q5) with English labels and aliases
- Limited to 10,000 results (adjust `LIMIT` in config.yaml for more)
- Puts data in `data/` and indices in `indices/`
- Feel free to add more indices with other precision types, like fp16,
bf16, or int8
