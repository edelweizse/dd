#!/usr/bin/env python
"""
Run an end-to-end smoke test for GenericHGT with strict split reuse.

Pipeline:
1. Process raw data
2. Build graph and create one split artifact
3. Reload and verify exact split reuse
4. Train GenericLinkPredictor
5. Evaluate val/test on the same split
6. Run one sample prediction
7. Compare GenericHGT against selected baselines on the same split

Usage:
    PYTHONPATH=. python scripts/smoke_generic_hgt.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.cli_config import parse_args_with_config
from src.data.graph import build_graph_from_processed
from src.data.processing import process_and_save
from src.data.splits import (
    load_split_artifact,
    negative_sample_cd_batch_local,
    prepare_splits_and_loaders,
)
from src.models.architectures.generic_hgt import GenericLinkPredictor, infer_schema_from_data
from src.models.baselines import BASELINE_NAMES, build_baseline, evaluate_baseline, train_baseline
from src.training.utils import bce_with_logits, sampled_ranking_metrics


CD_REL = ("chemical", "associated_with", "disease")


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _edge_hash(edge_idx: torch.Tensor) -> str:
    return hashlib.sha256(edge_idx.cpu().numpy().tobytes()).hexdigest()[:16]


@torch.no_grad()
def _eval_generic(
    model: GenericLinkPredictor,
    loader,
    known_pos,
    device: torch.device,
    *,
    num_neg_per_pos: int,
    degree_alpha: float = 0.75,
) -> Dict[str, float]:
    model.eval()
    all_scores = []
    all_labels = []
    mrr_sum = 0.0
    hits10_sum = 0.0
    n_pos_total = 0

    for batch in loader:
        batch = batch.to(device)
        pos_edge = batch[CD_REL].edge_label_index
        neg_edge = negative_sample_cd_batch_local(
            batch_data=batch,
            pos_edge_index_local=pos_edge,
            known_pos=known_pos,
            num_neg_per_pos=num_neg_per_pos,
            hard_negative_ratio=0.0,
            degree_alpha=degree_alpha,
            global_chem_degree=None,
            global_dis_degree=None,
            generator=None,
        )
        pos_logits, neg_logits = model(
            batch_data=batch,
            pos_edge_idx=pos_edge,
            neg_edge_idx=neg_edge,
            target_edge_type=CD_REL,
        )

        scores = torch.sigmoid(torch.cat([pos_logits, neg_logits], dim=0)).cpu().numpy()
        labels = torch.cat(
            [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)],
            dim=0,
        ).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels)

        r = sampled_ranking_metrics(pos_logits, neg_logits, num_neg_per_pos=num_neg_per_pos, ks=(10,))
        bsz = pos_logits.numel()
        n_pos_total += bsz
        mrr_sum += float(r["mrr"]) * bsz
        hits10_sum += float(r["hits_10"]) * bsz

    y_score = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return {
        "auroc": _safe_auc(y_true, y_score),
        "auprc": _safe_ap(y_true, y_score),
        "mrr": mrr_sum / max(n_pos_total, 1),
        "hits_10": hits10_sum / max(n_pos_total, 1),
        "n_pos": int(n_pos_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end smoke test for GenericHGT")
    parser.add_argument("--raw-dir", type=str, default="./data/raw", help="Raw data directory")
    parser.add_argument("--processed-dir", type=str, default=None, help="Processed data directory (optional)")
    parser.add_argument("--work-dir", type=str, default=None, help="Working directory for artifacts/results")
    parser.add_argument("--skip-process", action="store_true", help="Skip raw->processed step")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
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
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of message passing layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip")
    parser.add_argument("--num-neg-train", type=int, default=2, help="Negatives per positive for train")
    parser.add_argument("--num-neg-eval", type=int, default=2, help="Negatives per positive for eval")
    parser.add_argument(
        "--baseline-models",
        type=str,
        default="degree,mf,generic_hgt",
        help=f"Comma-separated baselines to compare against GenericHGT. Available: {', '.join(BASELINE_NAMES)}",
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

    torch.manual_seed(int(args.seed))

    ts = int(time.time())
    work_dir = Path(args.work_dir) if args.work_dir else Path(f"/tmp/dd_generic_hgt_e2e_{ts}")
    work_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(args.processed_dir) if args.processed_dir else work_dir / "processed"
    artifacts_dir = work_dir / "artifacts"
    ckpt_dir = work_dir / "checkpoints"
    compare_dir = work_dir / "baseline_comparison"
    split_path = artifacts_dir / "cd_split.pt"
    summary_path = work_dir / "summary.json"
    for p in [processed_dir, artifacts_dir, ckpt_dir, compare_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("GenericHGT smoke configuration")
    print(f"  work_dir: {work_dir}")
    print(f"  processed_dir: {processed_dir}")
    print(f"  split_artifact: {split_path}")

    if not args.skip_process:
        print("Processing raw data...")
        process_and_save(args.raw_dir, str(processed_dir))
    else:
        print("Skipping processing step.")

    print("Building graph...")
    data, _ = build_graph_from_processed(
        processed_data_dir=str(processed_dir),
        add_reverse_edges=True,
        save_vocabs=False,
        include_extended=True,
    )
    requested_baselines = [x.strip().lower() for x in args.baseline_models.split(",") if x.strip()]
    invalid_baselines = [x for x in requested_baselines if x not in BASELINE_NAMES]
    if invalid_baselines:
        raise ValueError(f"Invalid baseline names: {invalid_baselines}. Available: {BASELINE_NAMES}")

    print("Creating split artifact...")
    _ = prepare_splits_and_loaders(
        data_full=data,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        split_strategy=args.split_strategy,
        stratify_bins=int(args.stratify_bins),
        enforce_train_node_coverage=True,
        batch_size=int(args.batch_size),
        num_neighbours=[int(args.num_neighbours[0]), int(args.num_neighbours[1])],
        split_artifact_save_path=str(split_path),
    )

    print("Reloading split artifact for train/val/test...")
    arts = prepare_splits_and_loaders(
        data_full=data,
        batch_size=int(args.batch_size),
        num_neighbours=[int(args.num_neighbours[0]), int(args.num_neighbours[1])],
        split_artifact_load_path=str(split_path),
    )
    split_loaded, meta = load_split_artifact(split_path)

    if not torch.equal(split_loaded.train_pos, arts.split.train_pos):
        raise RuntimeError("Loaded split train_pos differs from active split.")
    if not torch.equal(split_loaded.val_pos, arts.split.val_pos):
        raise RuntimeError("Loaded split val_pos differs from active split.")
    if not torch.equal(split_loaded.test_pos, arts.split.test_pos):
        raise RuntimeError("Loaded split test_pos differs from active split.")

    print("Building GenericHGT model...")
    schema = infer_schema_from_data(arts.data_train)
    model = GenericLinkPredictor(
        schema=schema,
        metadata=arts.data_train.metadata(),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        relation_types=[CD_REL],
    )
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    print("Training...")
    train_loss = float("nan")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        n_pos = 0
        for batch in arts.train_loader:
            batch = batch.to(device)
            pos_edge = batch[CD_REL].edge_label_index
            neg_edge = negative_sample_cd_batch_local(
                batch_data=batch,
                pos_edge_index_local=pos_edge,
                known_pos=arts.known_pos,
                num_neg_per_pos=int(args.num_neg_train),
                hard_negative_ratio=0.0,
                degree_alpha=0.75,
                global_chem_degree=None,
                global_dis_degree=None,
                generator=None,
            )

            optimizer.zero_grad()
            pos_logits, neg_logits = model(
                batch_data=batch,
                pos_edge_idx=pos_edge,
                neg_edge_idx=neg_edge,
                target_edge_type=CD_REL,
            )
            loss = bce_with_logits(
                pos_logits,
                neg_logits,
                pos_weight=float(args.num_neg_train),
                focal_gamma=0.0,
            )
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            bsz = pos_edge.size(1)
            loss_sum += float(loss.item()) * bsz
            n_pos += bsz

        train_loss = loss_sum / max(n_pos, 1)
        print(f"  epoch={epoch} train_loss={train_loss:.4f}")

    ckpt_path = ckpt_dir / "generic_last.pt"
    torch.save({"model_state": model.state_dict(), "schema": schema}, ckpt_path)

    print("Evaluating val/test...")
    val_metrics = _eval_generic(
        model,
        arts.val_loader,
        arts.known_pos_val,
        device,
        num_neg_per_pos=int(args.num_neg_eval),
    )
    test_metrics = _eval_generic(
        model,
        arts.test_loader,
        arts.known_pos_test,
        device,
        num_neg_per_pos=int(args.num_neg_eval),
    )

    print("Running baseline comparison...")
    comparison_results: Dict[str, Dict[str, float]] = {
        "main_generic_hgt": {
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            **{f"test_{k}": float(v) for k, v in test_metrics.items()},
            "train_loss": float(train_loss),
            "train_seconds": float("nan"),
            "eval_seconds": float("nan"),
        }
    }
    for baseline_name in requested_baselines:
        t0 = time.time()
        baseline = build_baseline(
            baseline_name,
            data_train=arts.data_train,
            split_train_pos=arts.split.train_pos,
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            num_heads=int(args.num_heads),
            dropout=float(args.dropout),
            device=device,
        )
        train_stats = train_baseline(
            baseline,
            arts=arts,
            device=device,
            epochs=int(args.baseline_epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            grad_clip=float(args.grad_clip),
            num_neg_train=int(args.num_neg_train),
        )
        train_seconds = float(time.time() - t0)

        et0 = time.time()
        b_val = evaluate_baseline(
            baseline,
            loader=arts.val_loader,
            known_pos=arts.known_pos_val,
            device=device,
            num_neg_eval=int(args.num_neg_eval),
        )
        b_test = evaluate_baseline(
            baseline,
            loader=arts.test_loader,
            known_pos=arts.known_pos_test,
            device=device,
            num_neg_eval=int(args.num_neg_eval),
        )
        eval_seconds = float(time.time() - et0)
        comparison_results[baseline_name] = {
            **{f"val_{k}": float(v) for k, v in b_val.items()},
            **{f"test_{k}": float(v) for k, v in b_test.items()},
            "train_loss": float(train_stats.get("train_loss", float("nan"))),
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
        }

    expected_models = {"main_generic_hgt", *requested_baselines}
    missing_models = sorted(expected_models.difference(comparison_results.keys()))
    if missing_models:
        raise RuntimeError(f"Baseline comparison missing expected models: {missing_models}")

    compare_json_path = compare_dir / "comparison_results.json"
    compare_csv_path = compare_dir / "comparison_results.csv"
    compare_json_path.write_text(json.dumps(comparison_results, indent=2))
    compare_rows = []
    for model_name, metrics in comparison_results.items():
        row = {"model": model_name}
        row.update(metrics)
        compare_rows.append(row)
    pl.DataFrame(compare_rows).sort("val_auprc", descending=True, nulls_last=True).write_csv(compare_csv_path)

    expected_val = int(arts.split.val_pos.size(1))
    expected_test = int(arts.split.test_pos.size(1))
    if int(val_metrics["n_pos"]) != expected_val:
        raise RuntimeError(f"Val positive samples mismatch: eval={val_metrics['n_pos']}, split={expected_val}")
    if int(test_metrics["n_pos"]) != expected_test:
        raise RuntimeError(f"Test positive samples mismatch: eval={test_metrics['n_pos']}, split={expected_test}")

    print("Running one sample prediction...")
    edge_attr_dict = {
        et: data[et].edge_attr.to(device)
        for et in data.edge_types
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
    }
    with torch.no_grad():
        z = model.encode(
            x_dict={k: v.to(device) for k, v in data.x_dict.items()},
            edge_index_dict={k: v.to(device) for k, v in data.edge_index_dict.items()},
            edge_attr_dict=edge_attr_dict,
            return_attention=False,
        )

    chem_idx = int(arts.split.test_pos[0, 0].item())
    dis_idx = int(arts.split.test_pos[1, 0].item())
    one_edge = torch.tensor([[chem_idx], [dis_idx]], dtype=torch.long, device=device)
    with torch.no_grad():
        logit = model.head.score(z, CD_REL, one_edge).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

    chem_df = pl.read_parquet(processed_dir / "chemicals_nodes.parquet").select(
        ["CHEM_ID", "CHEM_MESH_ID", "CHEM_NAME"]
    )
    dis_df = pl.read_parquet(processed_dir / "diseases_nodes.parquet").select(
        ["DS_ID", "DS_OMIM_MESH_ID", "DS_NAME"]
    )
    chem_row = chem_df.filter(pl.col("CHEM_ID") == chem_idx).row(0)
    dis_row = dis_df.filter(pl.col("DS_ID") == dis_idx).row(0)

    summary = {
        "work_dir": str(work_dir),
        "processed_dir": str(processed_dir),
        "split_artifact": str(split_path),
        "checkpoint": str(ckpt_path),
        "split_metadata": {
            "seed": meta.get("seed"),
            "val_ratio": meta.get("val_ratio"),
            "test_ratio": meta.get("test_ratio"),
            "split_strategy": meta.get("split_strategy"),
        },
        "split_sizes": {
            "train": int(arts.split.train_pos.size(1)),
            "val": int(arts.split.val_pos.size(1)),
            "test": int(arts.split.test_pos.size(1)),
        },
        "split_hashes": {
            "train": _edge_hash(arts.split.train_pos),
            "val": _edge_hash(arts.split.val_pos),
            "test": _edge_hash(arts.split.test_pos),
        },
        "train_loss_last_epoch": float(train_loss),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_comparison": {
            "requested_models": requested_baselines,
            "results_json": str(compare_json_path),
            "results_csv": str(compare_csv_path),
            "models": sorted(comparison_results.keys()),
        },
        "sample_prediction": {
            "chem_idx": chem_idx,
            "dis_idx": dis_idx,
            "chemical_id": chem_row[1],
            "chemical_name": chem_row[2],
            "disease_id": dis_row[1],
            "disease_name": dis_row[2],
            "logit": float(logit),
            "probability": float(prob),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("GenericHGT smoke test passed")
    print(f"  summary: {summary_path}")
    print(f"  split_hashes: {summary['split_hashes']}")
    print(f"  val_auprc: {summary['val_metrics']['auprc']:.6f}")
    print(f"  test_auprc: {summary['test_metrics']['auprc']:.6f}")
    print(f"  baseline_compare_models: {summary['baseline_comparison']['models']}")


if __name__ == "__main__":
    main()
