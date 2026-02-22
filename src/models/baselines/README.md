# `src/models/baselines`

Baseline models and unified comparison utilities.

## Main Files

- `core.py`
  - baseline model definitions
  - build/train/evaluate helpers
  - main-vs-baselines comparison entrypoint

- `__init__.py`
  - exported API surface

## Available Baselines

- `degree`
- `mf`
- `mlp`
- `lightgcn_cd`
- `rgcn_cd`
- `heterosage`
- `hgat_no_edge_attr`
- `generic_hgat`

## Main APIs

- `build_baseline(...)`
- `train_baseline(...)`
- `evaluate_baseline(...)`
- `compare_main_and_baselines(...)`

## Related Scripts

- `scripts/compare_baselines.py`
- `scripts/smoke_e2e.py`
- `scripts/smoke_generic_hgat.py`

## Evaluation Notes

- Comparisons should run on the same split artifact as the main checkpoint.
- Reported metrics depend on sampled negatives/eval settings; compare models under identical settings.
