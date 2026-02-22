# `src/models/inference`

Inference wrappers built on trained architecture weights.

## Main Files

- `full_graph.py`
  - full-graph predictor API
  - precomputes embeddings and serves pair/top-k predictions
  - supports path-based explanation calls (`mode=path_attention`)

- `cached_embeddings.py`
  - predictor API that reads precomputed embedding caches from disk
  - optimized for low-memory serving
  - supports path-based explanation with graph input (no model-attention extraction)

- `_shared.py`
  - shared scoring/ranking/mapping helpers

## Related Scripts

- `scripts/predict.py` (full-graph mode)
- `scripts/cache_embeddings_chunked.py` + `scripts/predict_cached.py` (cached mode)
- `scripts/explain.py` (path_attention explainability)

## Important Constraints

- Checkpoint graph schema must match runtime graph schema (`extended` vs `--no-extended`).
- Cached inference requires `chemical_embeddings.npy`, `disease_embeddings.npy`, and `W_cd.pt`.
