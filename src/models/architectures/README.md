# `src/models/architectures`

Trainable architecture definitions.

## Main Files

- `hgt.py`
  - `HGTPredictor` for heterogeneous CD link prediction
  - edge-attribute-aware message passing + bilinear CD decoder
  - checkpoint-parameter inference helper (`infer_hgt_hparams_from_state`)

- `generic_hgt.py`
  - schema-driven `GenericLinkPredictor`
  - reusable across heterogeneous schemas via `GraphSchema`

- `_shared.py`
  - shared encoder/convolution utility functions

## Primary Usage

Used directly by:
- `scripts/train.py` (main HGT path)
- `scripts/tune.py` (HGT hyperparameter tuning)
- `scripts/smoke_generic_hgt.py` (generic architecture path)
- baseline workflows (`scripts/compare_baselines.py`, smoke scripts)
