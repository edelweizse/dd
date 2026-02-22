# `src/data`

Data pipeline from raw CTD/PPI tables to train/eval-ready graph artifacts.

## Main Files

- `processing.py`
  - raw ingestion and table normalization
  - entity/edge construction
  - parquet persistence and reload

- `graph.py`
  - builds PyG `HeteroData`
  - optional reverse-edge creation
  - graph summary helpers

- `splits.py`
  - train/val/test edge splits
  - split artifact save/load/compatibility validation
  - split-aware loaders and negative sampling

- `feature_encoders.py`
  - feature encoding helpers

## Core APIs

- `process_and_save(...)`
- `load_processed_data(...)`
- `build_graph_from_processed(...)`
- `prepare_splits_and_loaders(...)`
- `negative_sample_cd_batch_local(...)`

## Key Semantics

- Split artifacts are first-class and validated against graph compatibility.
- Training graph includes only training CD edges; non-CD structure remains available for message passing.
- Known-positive filters are phase-aware (`train`, `val`, `test`) to avoid split leakage.

For a fuller walkthrough, see `src/data/DATA_MODULE_SUMMARY.md`.
