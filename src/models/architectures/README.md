# `src/models/architectures`

Trainable architecture definitions.

## Main Files

- `hgat.py`
  - `HGATPredictor` for heterogeneous CD link prediction
  - edge-attribute-aware message passing + bilinear CD decoder
  - checkpoint-parameter inference helper (`infer_hgat_hparams_from_state`)

- `generic_hgat.py`
  - schema-driven `GenericLinkPredictor`
  - reusable across heterogeneous schemas via `GraphSchema`

- `_shared.py`
  - shared encoder/convolution utility functions

## Primary Usage

Used directly by:
- `scripts/train.py` (main HGAT path)
- `scripts/tune.py` (HGAT hyperparameter tuning)
- `scripts/smoke_generic_hgat.py` (generic architecture path)
- baseline workflows (`scripts/compare_baselines.py`, smoke scripts)
