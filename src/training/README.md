# `src/training`

Training and evaluation utilities shared across scripts.

## Main Files

- `trainer.py`
  - main HGT training loop with MLflow logging
  - scheduler + early stopping
  - checkpoint save/load helpers
  - Optuna-oriented tuning loop (`train_for_tuning`)

- `utils.py`
  - BCE/focal loss helpers
  - sampled ranking metrics (`MRR`, `Hits@K`)
  - eval-epoch routine with split-aware negative sampling

## Typical Usage

Used by:
- `scripts/train.py`
- `scripts/tune.py`
- baseline comparison flows via shared metric/sampling utilities

## Notes

- Training/eval flows are designed for split-artifact reuse.
- Negative sampling checks known-positive filters to preserve split semantics.
- `scripts/train.py` stores run-scoped checkpoints under `./checkpoints/`.
