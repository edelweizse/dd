#!/usr/bin/env python
"""Compare a trained main HGAT checkpoint against baseline models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import torch

from src.cli_config import parse_args_with_config
from src.data.graph import build_graph_from_processed, print_graph_summary
from src.data.splits import prepare_splits_and_loaders
from src.evaluation.protocol import (
    format_protocol_violations,
    load_evaluation_protocol,
    validate_evaluation_protocol,
)
from src.models.baselines import (
    BASELINE_NAMES,
    ComparisonConfig,
    compare_main_and_baselines,
    normalize_baseline_name,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare main HGAT checkpoint to baselines")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained main HGAT checkpoint")
    parser.add_argument("--processed-dir", type=str, default="./data/processed", help="Processed data directory")
    parser.add_argument("--output-dir", type=str, default="./baseline_comparison", help="Output directory")
    parser.add_argument(
        "--baselines",
        type=str,
        default=",".join(BASELINE_NAMES),
        help=f"Comma-separated baselines to run. Available: {', '.join(BASELINE_NAMES)}",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Loader batch size")
    parser.add_argument(
        "--num-neighbours",
        type=int,
        nargs=2,
        default=[5, 3],
        metavar=("HOP1", "HOP2"),
        help="Neighbor counts for loaders",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="stratified",
        choices=["stratified", "random"],
        help="Split strategy",
    )
    parser.add_argument("--stratify-bins", type=int, default=8, help="Stratify bins")
    parser.add_argument("--split-artifact-path", type=str, default=None, help="Load split artifact path")
    parser.add_argument("--save-split-artifact", type=str, default=None, help="Save split artifact path")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Baseline hidden dim")
    parser.add_argument("--num-layers", type=int, default=2, help="Baseline num layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Baseline num heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Baseline dropout")
    parser.add_argument("--epochs", type=int, default=1, help="Baseline train epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Baseline LR")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Baseline weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Baseline grad clip")
    parser.add_argument("--num-neg-train", type=int, default=2, help="Train negatives per positive")
    parser.add_argument("--num-neg-eval", type=int, default=20, help="Eval negatives per positive")
    parser.add_argument(
        "--protocol-config",
        type=str,
        default="./configs/examples/eval_protocol.yaml",
        help="Evaluation protocol config (YAML)",
    )
    parser.add_argument(
        "--allow-noncomparable",
        action="store_true",
        help="Allow protocol violations (writes non-comparable marker instead of failing)",
    )
    parser.add_argument(
        "--protocol-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail on protocol violations unless --allow-noncomparable is set",
    )
    parser.add_argument(
        "--protocol-report-path",
        type=str,
        default=None,
        help="Optional JSON path to save protocol validation report",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N batches (0 disables progress logs)",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Cap baseline train batches per epoch (0 = all; useful for quick debugging)",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Cap eval batches per split (0 = all; useful for quick debugging)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    args, _ = parse_args_with_config(parser)

    baseline_names = [normalize_baseline_name(x) for x in args.baselines.split(",") if x.strip()]
    # De-duplicate while preserving order.
    baseline_names = list(dict.fromkeys(baseline_names))
    invalid = [x for x in baseline_names if x not in BASELINE_NAMES]
    if invalid:
        raise ValueError(f"Invalid baseline names: {invalid}. Available: {BASELINE_NAMES}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building graph...")
    data, vocabs = build_graph_from_processed(
        processed_data_dir=args.processed_dir,
        add_reverse_edges=True,
        save_vocabs=False,
        include_extended=True,
    )
    print_graph_summary(data)

    print("Preparing splits/loaders...")
    arts = prepare_splits_and_loaders(
        data_full=data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_strategy=args.split_strategy,
        stratify_bins=args.stratify_bins,
        enforce_train_node_coverage=True,
        batch_size=args.batch_size,
        num_neighbours=[int(args.num_neighbours[0]), int(args.num_neighbours[1])],
        split_artifact_load_path=args.split_artifact_path,
        split_artifact_save_path=args.save_split_artifact,
    )
    if args.split_artifact_path and arts.split_metadata:
        print(
            "Loaded split metadata: "
            f"seed={arts.split_metadata.get('seed')}, "
            f"val_ratio={arts.split_metadata.get('val_ratio')}, "
            f"test_ratio={arts.split_metadata.get('test_ratio')}, "
            f"strategy={arts.split_metadata.get('split_strategy')}"
        )
    train_batches = len(arts.train_loader) if hasattr(arts.train_loader, "__len__") else "?"
    val_batches = len(arts.val_loader) if hasattr(arts.val_loader, "__len__") else "?"
    test_batches = len(arts.test_loader) if hasattr(arts.test_loader, "__len__") else "?"
    print(f"Loader batches: train={train_batches}, val={val_batches}, test={test_batches}")

    protocol = load_evaluation_protocol(args.protocol_config)
    protocol_result = validate_evaluation_protocol(
        protocol,
        split_artifact_path=args.split_artifact_path,
        split_metadata=arts.split_metadata,
        num_neg_eval=int(args.num_neg_eval),
        eval_hard_negative_ratio=0.0,
        runtime_seed=int(args.seed),
        runtime_val_ratio=float(args.val_ratio),
        runtime_test_ratio=float(args.test_ratio),
        runtime_split_strategy=str(args.split_strategy),
        runtime_stratify_bins=int(args.stratify_bins),
        allow_noncomparable=bool(args.allow_noncomparable),
    )
    strict_mode = bool(args.protocol_strict) and not bool(args.allow_noncomparable)
    if protocol_result.violations:
        msg = format_protocol_violations(protocol_result.violations, prefix="Evaluation protocol violation")
        if strict_mode:
            raise ValueError(msg)
        print(msg)
        print("Proceeding in non-comparable mode due to --allow-noncomparable.")
    else:
        print("Evaluation protocol check passed.")
    protocol_payload = {
        "version": int(protocol.version),
        "config_path": str(Path(args.protocol_config).expanduser()),
        "strict_mode": bool(strict_mode),
        "allow_noncomparable": bool(args.allow_noncomparable),
        **protocol_result.to_dict(),
    }
    if args.protocol_report_path:
        report_path = Path(args.protocol_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(protocol_payload, indent=2))
        print(f"Saved protocol report: {report_path}")

    cfg = ComparisonConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_neg_train=args.num_neg_train,
        num_neg_eval=args.num_neg_eval,
        progress_every=args.progress_every,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    if device.type == "cpu" and "generic_hgat" in baseline_names and int(args.epochs) > 1:
        print(
            "Warning: generic_hgat on CPU with multiple epochs can be very slow. "
            "Use --epochs 1 for smoke runs, or reduce loader work via "
            "--num-neighbours, --num-neg-eval, --max-*-batches.",
            flush=True,
        )

    print(f"Running comparison for baselines: {baseline_names}")
    results = compare_main_and_baselines(
        checkpoint_path=args.checkpoint,
        data_full=data,
        vocabs=vocabs,
        arts=arts,
        baseline_names=baseline_names,
        device=device,
        config=cfg,
    )

    results_path = output_dir / "comparison_results.json"
    payload = {
        **results,
        "evaluation_protocol": protocol_payload,
    }
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {results_path}")

    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pl.DataFrame(rows).sort("val_auprc", descending=True, nulls_last=True)
    csv_path = output_dir / "comparison_results.csv"
    df.write_csv(csv_path)
    print(f"Saved: {csv_path}")
    print("\nTop models by val_auprc:")
    print(df.select(["model", "val_auprc", "test_auprc", "val_auroc", "test_auroc"]).head(10))


if __name__ == "__main__":
    main()
