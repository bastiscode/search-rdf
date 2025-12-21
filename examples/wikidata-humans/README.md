# Wikidata Humans Keyword Index

Build a keyword search index for English labels of all humans in Wikidata.

## Usage

```bash
# Do everything below
search-rdf config.yaml

# Or do it step by step:
# Fetch data from Wikidata (may take several minutes)
search-rdf data config.yaml

# Build keyword index
search-rdf index config.yaml

# Start server on http://localhost:8080
search-rdf serve config.yaml
```

## Search Examples

```bash
curl -X POST http://localhost:8080/search/wikidata-humans-keyword \
  -H "Content-Type: application/json" \
  -d '{"query": ["Einstein"], "k": 10}'

curl -X POST http://localhost:8080/search/wikidata-humans-keyword \
  -H "Content-Type: application/json" \
  -d '{"query": ["Marie Curie"], "k": 5}'

curl -X POST http://localhost:8080/search/wikidata-humans-keyword \
  -H "Content-Type: application/json" \
  -d '{"query": ["Ada Lovelace"], "exact": true}'
```

## Notes

- Query fetches instances of human (Q5) with English labels
- Limited to 100,000 results (adjust `LIMIT` in config.yaml if needed)
- Generated files: `data/text/wikidata-humans/`, `indices/wikidata-humans/keyword/`
