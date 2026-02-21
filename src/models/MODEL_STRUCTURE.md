# Models Structure

This package is organized by **responsibility** so architecture code and inference wrappers are easy to distinguish.

## Namespaces

- `src/models/architectures/`
  - production trainable model definitions
  - current main model: HGT (`HGTPredictor`)
  - graph-agnostic schema-driven variant: `generic_hgt.py` (`GenericLinkPredictor`)

- `src/models/inference/`
  - user-facing predictor wrappers around trained model outputs
  - `full_graph.py`: full-graph predictor (`ChemDiseasePredictor` / `FullGraphPredictor`)
  - `cached_embeddings.py`: cache-based predictor (`EmbeddingCachePredictor` / `CachedEmbeddingPredictor`)

- `src/models/baselines/`
  - reserved for baseline models (e.g., logistic matrix factorization, heuristics)

## Current Source-of-Truth Files

- Core HGT implementation remains in `src/models/hgt.py`.
- Full-graph predictor implementation remains in `src/models/predictor.py`.
- Cache predictor implementation remains in `src/models/predictor_efficient.py`.

The new namespaced modules currently re-export these implementations to keep backward compatibility while making usage clearer.

## Recommended Imports Going Forward

- Architecture (training/eval):
  - `from src.models.architectures.hgt import HGTPredictor`
  - Graph-agnostic (schema-driven):
    - `from src.models.architectures.generic_hgt import GraphSchema, GenericLinkPredictor`

- Full-graph inference:
  - `from src.models.inference.full_graph import FullGraphPredictor`

- Memory-efficient inference:
  - `from src.models.inference.cached_embeddings import CachedEmbeddingPredictor`

## Naming Convention for Future Additions

- Architecture modules: `src/models/architectures/<family>_<version>.py`
  - Example: `hgt_v2.py`, `rgcn_baseline.py`

- Inference wrappers: `src/models/inference/<mode>.py`
  - Example: `cached_embeddings.py`, `full_graph.py`, `ann_index.py`

- Baselines: `src/models/baselines/<method>.py`
  - Example: `matrix_factorization.py`, `common_neighbors.py`

## Graph-Agnostic Entry Point

`src/models/architectures/generic_hgt.py` introduces:
- `GraphSchema` (node/edge feature contract)
- `GenericHGTEncoder` (schema-driven heterogeneous encoder)
- `BilinearLinkHead` (relation-aware generic link decoder)
- `GenericLinkPredictor` (encoder + head)

For fast prototyping, you can start from:
- `infer_schema_from_data(data)`

For production, prefer explicit `GraphSchema` and strict validation.
