# `src/explainability`

Pair-level explanation utilities for model predictions.

## Main Files

- `schema.py`: request/context/result contracts
- `service.py`: explainer router (current mode: `path_attention`)
- `engines/path_attention.py`: metapath + scoring engine
- `paths.py`: metapath/path extraction over the heterogeneous graph
- `scoring.py`: path and attention-based scoring/aggregation logic
- `templates.py`: template registry + validation
- `index.py`: forward/reverse adjacency indexing helpers

## What It Produces

- Human-readable rationale artifacts for a `(chemical, disease)` pair
- Scored path evidence grouped by metapath template type
- Runtime/debug metadata (path counts, timings, score weights)

## Entry Point

CLI wrapper:

```bash
python -m scripts.explain --mode path_attention ...
```

## Operational Pattern

For OOM-safe usage:
1. Run a fast no-attention pass first (`--no-attention` + tight path limits)
2. Re-run with attention only for a small number of selected pairs
