# `src` Package

Core implementation package for the project.

## Subpackages

- `src/data`
  - Raw-data processing
  - Heterogeneous graph construction (`HeteroData`)
  - Split artifacts, loaders, negative sampling

- `src/models`
  - Trainable architectures (`HGTPredictor`, `GenericLinkPredictor`)
  - Inference wrappers (full-graph and cached)
  - Baselines and model-comparison utilities

- `src/training`
  - Main training loops and tuning loop support
  - Losses, ranking metrics, eval helpers

- `src/explainability`
  - Pair explanation service (`path_attention`)
  - Path extraction, indexing, and scoring

- `src/cli_config.py`
  - Shared `--config` YAML loading/merge logic used by scripts

## Docs

- `src/data/README.md`
- `src/models/README.md`
- `src/training/README.md`
- `src/explainability/README.md`
