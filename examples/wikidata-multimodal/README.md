# Wikidata Multimodal (Image-to-Text Search)

Build a CLIP-based embedding index over Wikidata animals and plants,
enabling image-to-text search. Text fields (labels, synonyms, descriptions)
are embedded using the text encoder of a SigLIP2 model. At query time,
images can be embedded using the vision encoder and matched against the
text embeddings in the shared CLIP space.

## Usage

```bash
# Do everything in one command:
search-rdf config.yaml

# Or do it step by step:
# Fetch data from Wikidata
search-rdf data config.yaml

# Embed text fields with CLIP text encoder
search-rdf embed config.yaml

# Build index
search-rdf index config.yaml

# Start server on http://localhost:8080
search-rdf serve config.yaml
```

## Search Examples

```bash
# Text search (uses CLIP text encoder)
curl -X POST http://localhost:8080/search/nature \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"type": "value", "value": "large feline predator"}], "k": 10}'

# Image search via URL (uses CLIP vision encoder)
curl -X POST http://localhost:8080/search/nature \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"type": "value", "value": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Standing_jaguar.jpg/373px-Standing_jaguar.jpg", "modality": "image"}], "k": 10}'

# Image search via base64
curl -X POST http://localhost:8080/search/nature \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"type": "value", "value": "<base64-encoded-image>", "modality": "image-base64"}], "k": 10}'

# Modality is auto-detected when not specified:
# - URLs (http/https/file) are treated as images for CLIP models
# - Plain text is embedded with the text encoder
curl -X POST http://localhost:8080/search/nature \
  -H "Content-Type: application/json" \
  -d '{"queries": [{"type": "value", "value": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Standing_jaguar.jpg/373px-Standing_jaguar.jpg"}], "k": 10}'
```

## Notes

- Fetches animals (Q729) and plants (Q756) with English labels, synonyms, and descriptions
- Uses SigLIP2 (ViT-B/16) via OpenCLIP for shared text/image embedding space
- Text and images are embedded into the same vector space, enabling cross-modal search
- Requires a CUDA GPU for embedding
