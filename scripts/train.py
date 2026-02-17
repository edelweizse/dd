#!/usr/bin/env python
"""
Train the HGT model for chemical-disease link prediction.

Usage:
    python scripts/train.py --epochs 50 --batch-size 4096 --hidden-dim 128

Supports both the original 3-node-type graph (chemical, disease, gene) and
the enriched 5-node-type graph (+ pathway, go_term).
"""

import argparse
from pathlib import Path
import torch
from datetime import datetime

from src.cli_config import parse_args_with_config
from src.data.graph import build_graph_from_processed, print_graph_summary
from src.data.splits import prepare_splits_and_loaders
from src.models.hgt import HGTPredictor
from src.training.trainer import train


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def resolve_checkpoint_dir(ckpt_dir_arg: str, run_id: str) -> str:
    """
    Resolve checkpoint directory under /checkpoints.

    Rules:
    - Base directory is always /checkpoints
    - If user passes an absolute path outside /checkpoints, only its name is used
    - If user passes a relative path, it becomes a subdirectory under /checkpoints
    - run_id is always appended as the final directory
    """
    root = Path('/checkpoints')
    raw = (ckpt_dir_arg or '').strip()
    if not raw:
        return str(root / run_id)

    requested = Path(raw)
    if requested.is_absolute():
        try:
            rel = requested.relative_to(root)
        except ValueError:
            rel = Path(requested.name)
    else:
        rel = requested

    rel_str = str(rel).strip()
    if rel_str in {'', '.'}:
        return str(root / run_id)

    return str(root / rel / run_id)


def main():
    parser = argparse.ArgumentParser(description='Train HGT model for CD link prediction')
    
    # Data arguments
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--use-node-features', action='store_true',
                        help='Use precomputed node feature tables for inductive training')
    parser.add_argument('--node-features-dir', type=str, default=None,
                        help='Directory with *node_features.parquet files (default: <processed-dir>/features)')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for embeddings')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of message passing layers')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--num-neg-train', type=int, default=5,
                        help='Number of negatives per positive during training')
    parser.add_argument('--num-neg-eval', type=int, default=20,
                        help='Number of negatives per positive during evaluation')
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='Positive-class weight for BCE loss (default: num-neg-train)')
    parser.add_argument('--focal-gamma', type=float, default=0.0,
                        help='Focal loss gamma (0 disables focal modulation)')
    parser.add_argument('--hard-negative-ratio', type=float, default=0.5,
                        help='Fraction [0,1] of degree-biased negatives during training')
    parser.add_argument('--eval-hard-negative-ratio', type=float, default=0.0,
                        help='Fraction [0,1] of degree-biased negatives during eval sampling')
    parser.add_argument('--degree-alpha', type=float, default=0.75,
                        help='Exponent for degree-biased negative sampling weights')
    
    # Split arguments
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Scheduler arguments
    parser.add_argument('--patience', type=int, default=5,
                        help='LR scheduler patience')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='LR scheduler reduction factor')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--monitor', type=str, default='auprc',
                        help='Metric to monitor for model selection')
    
    # Neighbor sampling arguments
    parser.add_argument('--num-neighbours', type=int, nargs='+', default=[10, 5],
                        help='Neighbor samples per layer (e.g., --num-neighbours 8 4)')
    
    # Other arguments
    parser.add_argument('--ckpt-dir', type=str, default='/checkpoints',
                        help='Checkpoint subdirectory under /checkpoints (run ID appended)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='MLflow run name (auto-generated if not provided)')
    parser.add_argument('--experiment-name', type=str, default='HGT_linkpred',
                        help='MLflow experiment name')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    
    args, _ = parse_args_with_config(parser)
    
    # Generate unique run ID for this experiment
    run_id = generate_run_id()
    run_name = args.run_name if args.run_name else f'hgt_cd_{run_id}'
    ckpt_dir = resolve_checkpoint_dir(args.ckpt_dir, run_id)
    
    print(f'Run ID: {run_id}')
    print(f'Run name: {run_name}')
    print(f'Checkpoint directory: {ckpt_dir}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Build graph from processed data
    print('Building heterogeneous graph...')
    data, vocabs = build_graph_from_processed(
        processed_data_dir=args.processed_dir,
        add_reverse_edges=True,
        save_vocabs=True,
        include_extended=True,
        use_node_features=args.use_node_features,
        node_features_dir=args.node_features_dir
    )
    
    print_graph_summary(data)
    
    print('Preparing data splits and loaders...')
    print(f'Neighbor sampling: {args.num_neighbours}')
    arts = prepare_splits_and_loaders(
        data_full = data,
        val_ratio = args.val_ratio,
        test_ratio = args.test_ratio,
        seed = args.seed,
        batch_size = args.batch_size,
        num_neighbours = args.num_neighbours
    )
    
    num_action_types = vocabs['action_type'].height
    num_action_subjects = vocabs['action_subject'].height
    print(f'Action types: {num_action_types}, Action subjects: {num_action_subjects}')
    
    print('Creating model...')
    num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
    node_input_dims = {
        ntype: int(data[ntype].x.size(1))
        for ntype in data.node_types
        if isinstance(data[ntype].x, torch.Tensor)
        and data[ntype].x.dim() == 2
        and data[ntype].x.is_floating_point()
    }
    if node_input_dims:
        print(f'Using inductive node features for: {sorted(node_input_dims.keys())}')
    
    model = HGTPredictor(
        num_nodes_dict = num_nodes_dict,
        metadata = arts.data_train.metadata(),
        node_input_dims = node_input_dims,
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        dropout = args.dropout,
        num_action_types = num_action_types,
        num_action_subjects = num_action_subjects
    )
    
    print('Starting training...')
    model = train(
        arts = arts,
        model = model,
        device = device,
        run_name = run_name,
        experiment_name = args.experiment_name,
        epochs = args.epochs,
        lr = args.lr,
        weight_decay = args.weight_decay,
        grad_clip = args.grad_clip,
        num_neg_train = args.num_neg_train,
        num_neg_eval = args.num_neg_eval,
        amp = not args.no_amp,
        ckpt_dir = ckpt_dir,
        monitor = args.monitor,
        patience = args.patience,
        factor = args.factor,
        early_stopping_patience = args.early_stopping,
        pos_weight = args.pos_weight,
        focal_gamma = args.focal_gamma,
        hard_negative_ratio = args.hard_negative_ratio,
        degree_alpha = args.degree_alpha,
        eval_hard_negative_ratio = args.eval_hard_negative_ratio
    )
    
    print(f'Training complete! Checkpoints saved to: {ckpt_dir}')


if __name__ == '__main__':
    main()
