# Chemical-Disease Link Prediction (HGT)

This repository implements an end-to-end pipeline for chemical-disease link prediction on a heterogeneous biomedical graph.

It includes:
- Data processing from raw CTD/PPI tables to parquet artifacts
- Heterogeneous graph construction with optional extended node/edge types
- Reproducible split artifact creation and strict split reuse across train/eval/test
- Main model training (`HGTPredictor`) and schema-driven variant (`GenericLinkPredictor`)
- Baseline training/evaluation and checkpoint-vs-baseline comparison
- Full-graph and cached-embedding inference
- Path-based explainability (`path_attention`)
- Streamlit UI for cached inference workflows

## Pipeline Overview

```mermaid
flowchart LR
    raw["Raw CTD + PPI<br/>`data/raw`"]
    process["`scripts.process_data`<br/>Parquet Artifacts"]
    split["`scripts.create_split`<br/>Reusable Split Artifact"]
    train["`scripts.train`<br/>HGT / Generic HGT"]
    eval["`scripts.evaluate`<br/>Val / Test Metrics"]
    compare["`scripts.compare_baselines`<br/>Baseline Comparison"]
    ckpt["Checkpoint<br/>`checkpoints/&lt;run_id&gt;/best.pt`"]
    explain["`scripts.explain`<br/>`path_attention`"]
    cache["`scripts.cache_embeddings_chunked`<br/>`embeddings/`"]
    predict["`scripts.predict`<br/>Full-Graph Inference"]
    predict_cached["`scripts.predict_cached`<br/>Cached Inference"]
    app["`streamlit run app.py`<br/>UI"]

    raw --> process --> split --> train
    split --> eval
    train --> ckpt
    ckpt --> eval
    ckpt --> compare
    split --> compare
    ckpt --> explain
    ckpt --> cache
    ckpt --> predict
    cache --> predict_cached --> app

    classDef data fill:#E7F5FF,stroke:#1C7ED6,color:#0B3A66;
    classDef run fill:#FFF4E6,stroke:#F08C00,color:#663C00;
    classDef out fill:#E6FCF5,stroke:#0CA678,color:#0B4F3A;

    class raw,cache data;
    class process,split,train,eval,compare,explain,predict,predict_cached,app run;
    class ckpt out;
```

## Quick Start

Run commands from repository root.

1. Process raw data:
```bash
python -m scripts.process_data \
  --raw-dir ./data/raw \
  --processed-dir ./data/processed
```

2. Create a reusable split artifact (recommended):
```bash
python -m scripts.create_split \
  --processed-dir ./data/processed \
  --output-path ./splits/cd_split_seed42.pt \
  --include-extended \
  --seed 42 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --split-strategy stratified
```

3. Train the main HGT model using that exact split:
```bash
python -m scripts.train \
  --processed-dir ./data/processed \
  --split-artifact-path ./splits/cd_split_seed42.pt \
  --epochs 50 \
  --batch-size 4096
```

4. Evaluate on val/test with the same split artifact:
```bash
python -m scripts.evaluate \
  --checkpoint ./checkpoints/<run_id>/best.pt \
  --processed-dir ./data/processed \
  --split-artifact-path ./splits/cd_split_seed42.pt \
  --split test
```

Evaluation/compare runs enforce `configs/examples/eval_protocol.yaml` by default (currently fixed `num_neg_eval=20`, `eval_hard_negative_ratio=0.0`, split-artifact-required). For exploratory runs, pass `--allow-noncomparable`.

## Inference

### Inference Modes

```mermaid
flowchart TD
    start["Need a prediction for Chemical + Disease"]
    mem{"Memory constrained?"}
    full["Use full-graph path<br/>`scripts.predict`"]
    build["Build embedding cache once<br/>`scripts.cache_embeddings_chunked`"]
    cached["Query fast path<br/>`scripts.predict_cached` / Streamlit"]

    start --> mem
    mem -- "No" --> full
    mem -- "Yes" --> build --> cached

    classDef decision fill:#FFF9DB,stroke:#E67700,color:#5F3B00;
    classDef route fill:#EDF2FF,stroke:#4263EB,color:#1E2E7A;

    class mem decision;
    class full,build,cached,start route;
```

### Full-Graph Predictor

```bash
python -m scripts.predict \
  --checkpoint ./checkpoints/<run_id>/best.pt \
  --processed-dir ./data/processed \
  --disease MESH:D014202 \
  --chemical C006901
```

### Cached-Embedding Predictor (Memory-Efficient)

1. Build cache tensors:
```bash
python -m scripts.cache_embeddings_chunked \
  --processed-dir ./data/processed \
  --checkpoint ./checkpoints/<run_id>/best.pt \
  --output-dir ./embeddings
```

2. Query with cached embeddings:
```bash
python -m scripts.predict_cached \
  --processed-dir ./data/processed \
  --embeddings-dir ./embeddings \
  --disease MESH:D014202 \
  --chemical C006901
```

## Explainability

Current production mode is `path_attention` only.

```bash
python -m scripts.explain \
  --processed-dir ./data/processed \
  --checkpoint ./checkpoints/<run_id>/best.pt \
  --disease MESH:D014202 \
  --chemical C006901 \
  --mode path_attention
```

Fast/OOM-safe pattern:
1. First pass without attention (`--no-attention`, tighter path limits)
2. Re-run with attention only for a small set of pairs
3. In attention runs, keep path limits and graph scope conservative

## Baseline Comparison

Compare a trained main checkpoint against baselines on the same split artifact:

```bash
python -m scripts.compare_baselines \
  --checkpoint ./checkpoints/<run_id>/best.pt \
  --processed-dir ./data/processed \
  --split-artifact-path ./splits/cd_split_seed42.pt \
  --baselines degree,mf,generic_hgt \
  --output-dir ./baseline_comparison
```

Outputs:
- `comparison_results.json`
- `comparison_results.csv`

`comparison_results.json` now also contains an `evaluation_protocol` section describing comparability checks and any violations.

## Smoke Tests

Main HGT end-to-end smoke:
```bash
python -m scripts.smoke_e2e \
  --epochs 1 \
  --batch-size 1024 \
  --baseline-models degree,mf,generic_hgt
```

GenericHGT end-to-end smoke:
```bash
python -m scripts.smoke_generic_hgt \
  --processed-dir ./data/processed \
  --skip-process \
  --epochs 1 \
  --baseline-models degree,mf,generic_hgt
```

## Streamlit App

The UI runs on cached embeddings (to avoid loading full-graph model inference into memory):

```bash
streamlit run app.py
```

Before first run, generate `./embeddings` with `scripts.cache_embeddings_chunked`.

## YAML Configs

Most scripts support `--config`.

See:
- `YAML_CONFIGS.md`
- `configs/examples/README.md`

Example:
```bash
python -m scripts.train --config configs/examples/train.yaml
```

## Project Layout

```mermaid
flowchart TB
    root["dd/"]
    app["app.py<br/>Streamlit UI (cached inference)"]
    scripts["scripts/<br/>CLI entrypoints"]
    src["src/<br/>Core package"]
    src_data["src/data/<br/>processing + graph + splits + negatives"]
    src_models["src/models/<br/>architectures + inference + baselines"]
    src_training["src/training/<br/>train loops + losses/metrics"]
    src_explain["src/explainability/<br/>path-based explanation service"]
    configs["configs/examples/<br/>example YAML configs"]
    raw["data/raw/<br/>raw CTD + PPI inputs"]
    processed["data/processed/<br/>processed parquet artifacts"]
    checkpoints["checkpoints/<br/>training checkpoints"]
    tests["tests/<br/>unit + integration tests"]

    root --> app
    root --> scripts
    root --> src
    root --> configs
    root --> raw
    root --> processed
    root --> checkpoints
    root --> tests
    src --> src_data
    src --> src_models
    src --> src_training
    src --> src_explain

    classDef rootc fill:#F8F0FC,stroke:#9C36B5,color:#4A154B;
    classDef leaf fill:#F1F3F5,stroke:#868E96,color:#343A40;
    classDef core fill:#E3FAFC,stroke:#1098AD,color:#0B4F5C;

    class root rootc;
    class src,src_data,src_models,src_training,src_explain core;
    class app,scripts,configs,raw,processed,checkpoints,tests leaf;
```

## Module Docs

- `src/README.md`
- `src/data/README.md`
- `src/models/README.md`
- `src/models/architectures/README.md`
- `src/models/inference/README.md`
- `src/models/baselines/README.md`
- `src/training/README.md`
- `src/explainability/README.md`

## Testing

Run all tests:

```bash
pytest -q
```

Run selected suites:

```bash
pytest -q tests/test_data_splits.py tests/test_models.py tests/test_inference.py
```

## Operational Notes

- For reproducibility, prefer explicit split artifacts (`scripts.create_split`) and pass them to train/eval/compare.
- `scripts.train.py` logs to MLflow and writes run-scoped checkpoints under `./checkpoints/`.
- Checkpoint graph schema must match runtime graph schema (`--no-extended` vs default extended graph).
- Cached inference/explainability is the default path for memory-constrained environments.
