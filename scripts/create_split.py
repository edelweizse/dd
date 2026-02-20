#!/usr/bin/env python
"""
Create and persist train/val/test split artifacts for CD link prediction.

Usage:
    python scripts/create_split.py --output-path ./artifacts/splits/cd_split.pt
    python scripts/create_split.py --config configs/examples/create_split.yaml
"""

import argparse
from pathlib import Path

import torch

from src.cli_config import parse_args_with_config
from src.data.graph import build_graph_from_processed, print_graph_summary
from src.data.splits import (
    LinkSplit,
    load_split_artifact,
    save_split_artifact,
    split_cd,
    validate_split_artifact_compatibility,
)


def _coverage_stats(split: LinkSplit, num_chem: int, num_dis: int) -> dict:
    train_chem = int(split.train_pos[0].unique().numel())
    train_dis = int(split.train_pos[1].unique().numel())
    all_chem = int(torch.cat([split.train_pos[0], split.val_pos[0], split.test_pos[0]], dim=0).unique().numel())
    all_dis = int(torch.cat([split.train_pos[1], split.val_pos[1], split.test_pos[1]], dim=0).unique().numel())
    return {
        "train_chem": train_chem,
        "train_dis": train_dis,
        "all_chem": all_chem,
        "all_dis": all_dis,
        "num_chem": int(num_chem),
        "num_dis": int(num_dis),
    }


def _fmt_pct(part: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * float(part) / float(total):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create split artifact for chemical-disease link prediction"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="./data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save split artifact (.pt)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="stratified",
        choices=["stratified", "random"],
        help="CD edge split strategy",
    )
    parser.add_argument(
        "--stratify-bins",
        type=int,
        default=8,
        help="Number of log-degree bins for stratified split",
    )
    parser.add_argument(
        "--no-enforce-train-node-coverage",
        action="store_true",
        help="Disable best-effort train node coverage rebalancing",
    )
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="Include pathway/GO nodes and edges when building graph",
    )
    parser.add_argument(
        "--print-graph-summary",
        action="store_true",
        help="Print graph summary before creating split",
    )
    parser.add_argument(
        "--cd-rel",
        nargs=3,
        default=["chemical", "associated_with", "disease"],
        metavar=("SRC", "REL", "DST"),
        help="Relation tuple to split (default: chemical associated_with disease)",
    )

    args, _ = parse_args_with_config(parser)

    if not args.output_path:
        parser.error("--output-path is required (via CLI or --config)")

    cd_rel = (args.cd_rel[0], args.cd_rel[1], args.cd_rel[2])

    print("Building graph...")
    data, _ = build_graph_from_processed(
        processed_data_dir=args.processed_dir,
        add_reverse_edges=False,
        save_vocabs=False,
        include_extended=bool(args.include_extended),
    )

    if args.print_graph_summary:
        print_graph_summary(data)

    if cd_rel not in data.edge_types:
        raise ValueError(
            f"Relation {cd_rel} not found in graph edge types: {sorted(data.edge_types)}"
        )

    cd_idx = data[cd_rel].edge_index.long().cpu()
    use_stratify = args.split_strategy == "stratified"
    split, used_strategy = split_cd(
        cd_idx=cd_idx,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        stratify=use_stratify,
        stratify_bins=int(args.stratify_bins),
        enforce_train_node_coverage=not args.no_enforce_train_node_coverage,
        return_strategy=True,
    )

    out_path = save_split_artifact(
        artifact_path=Path(args.output_path),
        split=split,
        data_full=data,
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        split_strategy=used_strategy,
        stratify_bins=int(args.stratify_bins),
        enforce_train_node_coverage=not args.no_enforce_train_node_coverage,
        cd_rel=cd_rel,
    )

    # Verify on-disk artifact is reusable for this graph.
    loaded_split, metadata = load_split_artifact(out_path)
    validate_split_artifact_compatibility(
        split=loaded_split,
        metadata=metadata,
        data_full=data,
        cd_rel=cd_rel,
    )

    e_train = int(split.train_pos.size(1))
    e_val = int(split.val_pos.size(1))
    e_test = int(split.test_pos.size(1))
    e_total = e_train + e_val + e_test

    cov = _coverage_stats(
        split=split,
        num_chem=int(data["chemical"].num_nodes),
        num_dis=int(data["disease"].num_nodes),
    )

    print("\nSplit artifact created")
    print(f"  path: {out_path}")
    print(f"  relation: {cd_rel}")
    print(f"  strategy_requested: {args.split_strategy}")
    print(f"  strategy_used: {used_strategy}")
    print(
        "  enforce_train_node_coverage: "
        f"{not args.no_enforce_train_node_coverage}"
    )
    print(f"  seed: {int(args.seed)}")
    print(f"  edges_train: {e_train} ({_fmt_pct(e_train, e_total)})")
    print(f"  edges_val: {e_val} ({_fmt_pct(e_val, e_total)})")
    print(f"  edges_test: {e_test} ({_fmt_pct(e_test, e_total)})")
    print(
        f"  train_chem_coverage: {cov['train_chem']}/{cov['num_chem']} "
        f"({_fmt_pct(cov['train_chem'], cov['num_chem'])})"
    )
    print(
        f"  train_dis_coverage: {cov['train_dis']}/{cov['num_dis']} "
        f"({_fmt_pct(cov['train_dis'], cov['num_dis'])})"
    )
    print(
        f"  full_chem_coverage: {cov['all_chem']}/{cov['num_chem']} "
        f"({_fmt_pct(cov['all_chem'], cov['num_chem'])})"
    )
    print(
        f"  full_dis_coverage: {cov['all_dis']}/{cov['num_dis']} "
        f"({_fmt_pct(cov['all_dis'], cov['num_dis'])})"
    )


if __name__ == "__main__":
    main()
