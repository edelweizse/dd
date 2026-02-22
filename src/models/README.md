# `src/models`

Model namespace split by responsibility.

## Submodules

- `architectures/`
  - trainable model definitions (`HGATPredictor`, `GenericLinkPredictor`)

- `inference/`
  - predictor wrappers for full-graph and cached-embedding serving

- `baselines/`
  - baseline model implementations and comparison utilities

## Compatibility Layers

Legacy wrappers are kept for backward compatibility:
- `hgat.py`
- `predictor.py`
- `predictor_efficient.py`

Use namespaced imports for new code.

## Import Guidance

- Main architecture:
  - `from src.models.architectures.hgat import HGATPredictor`
- Generic architecture:
  - `from src.models.architectures.generic_hgat import GenericLinkPredictor`
- Full-graph inference:
  - `from src.models.inference.full_graph import FullGraphPredictor`
- Cached inference:
  - `from src.models.inference.cached_embeddings import CachedEmbeddingPredictor`
- Baseline utilities:
  - `from src.models.baselines import compare_main_and_baselines`
