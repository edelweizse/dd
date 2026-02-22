#!/usr/bin/env python
"""
Run an end-to-end smoke test for the CD pipeline with strict split reuse.

Pipeline:
1. Process raw data
2. Create one split artifact
3. Train (loading that split artifact)
4. Evaluate on val (loading same split artifact)
5. Evaluate on test (loading same split artifact)
6. Run one pair prediction
7. Compare main model against selected baselines on the same split

Usage:
    PYTHONPATH=. python scripts/smoke_e2e.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import polars as pl

from src.cli_config import parse_args_with_config
from src.data.splits import load_split_artifact
from src.models.baselines import normalize_baseline_name


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, env: Dict[str, str], cwd: Path = REPO_ROOT) -> str:
    """Run a command, print output, and fail fast on non-zero exit."""
    pretty = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n$ {pretty}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {pretty}")
    return (proc.stdout or "") + (proc.stderr or "")


def _extract(pattern: str, text: str, label: str) -> str:
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"Could not parse {label} from output with pattern: {pattern}")
    return m.group(1)


def _split_hashes(split_path: Path) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, object]]:
    split, meta = load_split_artifact(split_path)

    hashes = {
        "train": hashlib.sha256(split.train_pos.cpu().numpy().tobytes()).hexdigest()[:16],
        "val": hashlib.sha256(split.val_pos.cpu().numpy().tobytes()).hexdigest()[:16],
        "test": hashlib.sha256(split.test_pos.cpu().numpy().tobytes()).hexdigest()[:16],
    }
    sizes = {
        "train": int(split.train_pos.size(1)),
        "val": int(split.val_pos.size(1)),
        "test": int(split.test_pos.size(1)),
    }
    return hashes, sizes, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end smoke test with shared split artifact")
    parser.add_argument("--raw-dir", type=str, default="./data/raw", help="Raw data directory")
    parser.add_argument("--work-dir", type=str, default=None, help="Working directory for smoke outputs")
    parser.add_argument("--seed", type=int, default=123, help="Seed for split reproducibility")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="stratified",
        choices=["stratified", "random"],
        help="Split strategy",
    )
    parser.add_argument("--stratify-bins", type=int, default=8, help="Stratification bins")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs for smoke test")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Model hidden dim")
    parser.add_argument("--num-layers", type=int, default=1, help="Model layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument(
        "--baseline-models",
        type=str,
        default="degree,mf,generic_hgat",
        help="Comma-separated baselines for comparison stage",
    )
    parser.add_argument("--baseline-epochs", type=int, default=1, help="Train epochs for baseline comparison")
    parser.add_argument(
        "--num-neighbours",
        type=int,
        nargs=2,
        default=[3, 2],
        metavar=("HOP1", "HOP2"),
        help="Neighbor counts for 2-hop sampling",
    )
    args, _ = parse_args_with_config(parser)

    ts = int(time.time())
    work_dir = Path(args.work_dir) if args.work_dir else Path(f"/tmp/dd_smoke_e2e_{ts}")
    processed_dir = work_dir / "processed"
    artifacts_dir = work_dir / "artifacts"
    eval_val_dir = work_dir / "eval_val"
    eval_test_dir = work_dir / "eval_test"
    compare_dir = work_dir / "baseline_comparison"
    split_path = artifacts_dir / "cd_split.pt"
    ckpt_parent = REPO_ROOT / "checkpoints" / "smoke_e2e"

    for path in [work_dir, processed_dir, artifacts_dir, eval_val_dir, eval_test_dir, compare_dir]:
        path.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "." if not py_path else f".:{py_path}"

    print("Smoke test configuration")
    print(f"  work_dir: {work_dir}")
    print(f"  processed_dir: {processed_dir}")
    print(f"  split_artifact: {split_path}")
    print(f"  checkpoint_parent: {ckpt_parent}")

    _run(
        [
            sys.executable,
            "scripts/process_data.py",
            "--raw-dir",
            args.raw_dir,
            "--processed-dir",
            str(processed_dir),
        ],
        env=env,
    )

    _run(
        [
            sys.executable,
            "scripts/create_split.py",
            "--processed-dir",
            str(processed_dir),
            "--output-path",
            str(split_path),
            "--include-extended",
            "--seed",
            str(args.seed),
            "--val-ratio",
            str(args.val_ratio),
            "--test-ratio",
            str(args.test_ratio),
            "--split-strategy",
            args.split_strategy,
            "--stratify-bins",
            str(args.stratify_bins),
        ],
        env=env,
    )

    train_out = _run(
        [
            sys.executable,
            "scripts/train.py",
            "--processed-dir",
            str(processed_dir),
            "--split-artifact-path",
            str(split_path),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--hidden-dim",
            str(args.hidden_dim),
            "--num-layers",
            str(args.num_layers),
            "--num-heads",
            str(args.num_heads),
            "--dropout",
            str(args.dropout),
            "--num-neighbours",
            str(args.num_neighbours[0]),
            str(args.num_neighbours[1]),
            "--num-neg-train",
            "2",
            "--num-neg-eval",
            "2",
            "--hard-negative-ratio",
            "0.0",
            "--eval-hard-negative-ratio",
            "0.0",
            "--seed",
            str(args.seed),
            "--ckpt-dir",
            "./checkpoints/smoke_e2e",
            "--run-name",
            "smoke_e2e",
            "--experiment-name",
            "smoke_e2e",
            "--no-amp",
        ],
        env=env,
    )

    if "Loaded split metadata:" not in train_out:
        raise RuntimeError("Train stage did not report loading split metadata.")
    ckpt_dir = _extract(r"Checkpoint directory:\s*(.+)", train_out, "checkpoint directory").strip()
    checkpoint = str((REPO_ROOT / ckpt_dir / "best.pt").resolve())
    if not Path(checkpoint).exists():
        raise RuntimeError(f"Expected checkpoint not found: {checkpoint}")

    eval_val_out = _run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            checkpoint,
            "--processed-dir",
            str(processed_dir),
            "--split-artifact-path",
            str(split_path),
            "--split",
            "val",
            "--batch-size",
            str(args.batch_size),
            "--num-neg-eval",
            "2",
            "--seed",
            str(args.seed),
            "--allow-noncomparable",
            "--no-tsne",
            "--output-dir",
            str(eval_val_dir),
        ],
        env=env,
    )
    if f"Using split artifact: {split_path}" not in eval_val_out:
        raise RuntimeError("Eval(val) stage did not report using the expected split artifact.")

    eval_test_out = _run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            checkpoint,
            "--processed-dir",
            str(processed_dir),
            "--split-artifact-path",
            str(split_path),
            "--split",
            "test",
            "--batch-size",
            str(args.batch_size),
            "--num-neg-eval",
            "2",
            "--seed",
            str(args.seed),
            "--allow-noncomparable",
            "--no-tsne",
            "--output-dir",
            str(eval_test_dir),
        ],
        env=env,
    )
    if f"Using split artifact: {split_path}" not in eval_test_out:
        raise RuntimeError("Eval(test) stage did not report using the expected split artifact.")

    hashes, sizes, meta = _split_hashes(split_path)
    val_pos_seen = int(_extract(r"Positive samples:\s*([0-9,]+)", eval_val_out, "val positive samples").replace(",", ""))
    test_pos_seen = int(_extract(r"Positive samples:\s*([0-9,]+)", eval_test_out, "test positive samples").replace(",", ""))
    if val_pos_seen != sizes["val"]:
        raise RuntimeError(f"Val positive samples mismatch: eval={val_pos_seen}, split={sizes['val']}")
    if test_pos_seen != sizes["test"]:
        raise RuntimeError(f"Test positive samples mismatch: eval={test_pos_seen}, split={sizes['test']}")

    diseases = pl.read_parquet(processed_dir / "diseases_nodes.parquet").select(["DS_OMIM_MESH_ID"]).head(1)
    chemicals = pl.read_parquet(processed_dir / "chemicals_nodes.parquet").select(["CHEM_MESH_ID"]).head(1)
    disease_id = str(diseases["DS_OMIM_MESH_ID"][0])
    chemical_id = str(chemicals["CHEM_MESH_ID"][0])

    _run(
        [
            sys.executable,
            "scripts/predict.py",
            "--processed-dir",
            str(processed_dir),
            "--checkpoint",
            checkpoint,
            "--disease",
            disease_id,
            "--chemical",
            chemical_id,
        ],
        env=env,
    )

    compare_out = _run(
        [
            sys.executable,
            "-m",
            "scripts.compare_baselines",
            "--checkpoint",
            checkpoint,
            "--processed-dir",
            str(processed_dir),
            "--output-dir",
            str(compare_dir),
            "--baselines",
            args.baseline_models,
            "--split-artifact-path",
            str(split_path),
            "--batch-size",
            str(args.batch_size),
            "--num-neighbours",
            str(args.num_neighbours[0]),
            str(args.num_neighbours[1]),
            "--epochs",
            str(args.baseline_epochs),
            "--num-neg-train",
            "2",
            "--num-neg-eval",
            "2",
            "--allow-noncomparable",
            "--cpu",
        ],
        env=env,
    )
    if "Loaded split metadata:" not in compare_out:
        raise RuntimeError("Baseline comparison stage did not report loading split metadata.")
    compare_json_path = compare_dir / "comparison_results.json"
    compare_csv_path = compare_dir / "comparison_results.csv"
    if not compare_json_path.exists() or not compare_csv_path.exists():
        raise RuntimeError("Baseline comparison outputs are missing.")
    compare_results = json.loads(compare_json_path.read_text())
    requested_baselines = [normalize_baseline_name(x) for x in args.baseline_models.split(",") if x.strip()]
    requested_baselines = list(dict.fromkeys(requested_baselines))
    expected_models = {"main_hgat", *requested_baselines}
    missing = sorted(expected_models.difference(compare_results.keys()))
    if missing:
        raise RuntimeError(f"Baseline comparison results missing models: {missing}")

    val_metrics = json.loads((eval_val_dir / "metrics.json").read_text())
    test_metrics = json.loads((eval_test_dir / "metrics.json").read_text())

    print("\nSmoke test passed")
    print(f"  split_path: {split_path}")
    print(f"  split_metadata: seed={meta.get('seed')}, val_ratio={meta.get('val_ratio')}, test_ratio={meta.get('test_ratio')}, strategy={meta.get('split_strategy')}")
    print(f"  split_sizes: train={sizes['train']}, val={sizes['val']}, test={sizes['test']}")
    print(f"  split_hashes: train={hashes['train']}, val={hashes['val']}, test={hashes['test']}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  eval_val_auprc: {val_metrics.get('auprc'):.6f}")
    print(f"  eval_test_auprc: {test_metrics.get('auprc'):.6f}")
    print(f"  baseline_compare_models: {sorted(compare_results.keys())}")
    print(f"  baseline_compare_json: {compare_json_path}")
    print(f"  outputs_root: {work_dir}")


if __name__ == "__main__":
    main()
