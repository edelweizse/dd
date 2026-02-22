# Example Configs

This folder contains configs for active CLI entrypoints.

- `process_data.yaml` -> `python -m scripts.process_data --config ...`
- `create_split.yaml` -> `python -m scripts.create_split --config ...`
- `train.yaml` -> `python -m scripts.train --config ...`
- `evaluate.yaml` -> `python -m scripts.evaluate --config ...`
- `eval_protocol.yaml` -> protocol contract consumed by `evaluate`/`compare_baselines`
- `predict_pair.yaml` -> `python -m scripts.predict --config ...`
- `predict_cached.yaml` -> `python -m scripts.predict_cached --config ...`
- `cache_embeddings_chunked.yaml` -> `python -m scripts.cache_embeddings_chunked --config ...`
- `explain.yaml` -> `python -m scripts.explain --config ...`
- `tune.yaml` -> `python -m scripts.tune --config ...`
- `compare_baselines.yaml` -> `python -m scripts.compare_baselines --config ...`
- `smoke_e2e.yaml` -> `python -m scripts.smoke_e2e --config ...`
- `smoke_generic_hgat.yaml` -> `python -m scripts.smoke_generic_hgat --config ...`

## Usage Notes

- Config keys map directly to CLI arguments (nested keys are flattened by leaf name).
- CLI flags override YAML values.
- For reproducibility, keep one explicit split artifact and pass it to train/eval/compare scripts.
- `evaluate` and `compare_baselines` enforce `eval_protocol.yaml` by default; use `--allow-noncomparable` only for exploratory/smoke runs.
- `explain.yaml` intentionally uses `mode: path_attention` (the only supported mode).
