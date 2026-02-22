# Models Structure

This package is organized by responsibility so architecture code and inference wrappers are clearly separated.

## Namespaces

- `src/models/architectures/`
  - production trainable model definitions
  - main model: HGAT (`HGATPredictor`)
  - graph-agnostic schema-driven variant: `generic_hgat.py` (`GenericLinkPredictor`)

- `src/models/inference/`
  - user-facing predictor wrappers around trained model outputs
  - `full_graph.py`: full-graph predictor (`ChemDiseasePredictor` / `FullGraphPredictor`)
  - `cached_embeddings.py`: cache-based predictor (`EmbeddingCachePredictor` / `CachedEmbeddingPredictor`)

- `src/models/baselines/`
  - baseline models and comparison helpers
  - includes heuristic and trainable baselines (degree, MF, MLP, LightGCN CD, RGCN CD, HeteroSAGE, HGAT-no-edge-attr, GenericHGAT baseline)

## Current Source-of-Truth Files

- Core HGAT implementation: `src/models/architectures/hgat.py`
- Full-graph predictor implementation: `src/models/inference/full_graph.py`
- Cache predictor implementation: `src/models/inference/cached_embeddings.py`

Compatibility shims in `src/models/hgt.py`, `src/models/predictor.py`, and `src/models/predictor_efficient.py` re-export these namespaced modules for backward compatibility only.

## Recommended Imports

- Architecture (training/eval):
  - `from src.models.architectures.hgat import HGATPredictor`
- Generic architecture:
  - `from src.models.architectures.generic_hgat import GraphSchema, GenericLinkPredictor`
- Full-graph inference:
  - `from src.models.inference.full_graph import FullGraphPredictor`
- Memory-efficient inference:
  - `from src.models.inference.cached_embeddings import CachedEmbeddingPredictor`

## Generic Entry Point

`src/models/architectures/generic_hgat.py` provides:
- `GraphSchema` (node/edge contract)
- `GenericHGATEncoder` (schema-driven heterogeneous encoder)
- `BilinearLinkHead` (relation-aware decoder)
- `GenericLinkPredictor` (encoder + head)

For fast prototyping, start from `infer_schema_from_data(data)`.
For production, prefer explicit `GraphSchema` definitions.
