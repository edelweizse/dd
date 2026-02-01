#!/usr/bin/env python
"""
Train the HGT model for chemical-disease link prediction.

Usage:
    python scripts/train.py --epochs 50 --batch-size 4096 --hidden-dim 128

Supports both the original 3-node-type graph (chemical, disease, gene) and
the enriched 5-node-type graph (+ pathway, go_term).
"""

import argparse
import torch
import polars as pl

from src.data.graph import build_graph_from_processed, load_vocabs, print_graph_summary
from src.data.splits import prepare_splits_and_loaders
from src.models.hgt import HGTPredictor
from src.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description='Train HGT model for CD link prediction')
    
    # Data arguments
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')
    
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
    
    # Other arguments
    parser.add_argument('--ckpt-dir', type=str, default='./checkpoints/',
                        help='Checkpoint directory')
    parser.add_argument('--run-name', type=str, default='hgt_cd_lp',
                        help='MLflow run name')
    parser.add_argument('--experiment-name', type=str, default='HGT_linkpred',
                        help='MLflow experiment name')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Build graph from processed data (includes extended nodes/edges automatically)
    print('Building heterogeneous graph...')
    data, vocabs = build_graph_from_processed(
        processed_data_dir=args.processed_dir,
        add_reverse_edges=True,
        save_vocabs=True,
        include_extended=True  # Include pathway and GO term nodes/edges
    )
    
    # Print graph summary
    print_graph_summary(data)
    
    # Prepare splits and loaders
    print('Preparing data splits and loaders...')
    arts = prepare_splits_and_loaders(
        data_full = data,
        val_ratio = args.val_ratio,
        test_ratio = args.test_ratio,
        seed = args.seed,
        batch_size = args.batch_size
    )
    
    # Get vocabulary sizes for edge attributes
    num_action_types = vocabs['action_type'].height
    num_action_subjects = vocabs['action_subject'].height
    print(f'Action types: {num_action_types}, Action subjects: {num_action_subjects}')
    
    # Create model with all node types from the graph
    print('Creating model...')
    num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
    
    model = HGTPredictor(
        num_nodes_dict = num_nodes_dict,
        metadata = arts.data_train.metadata(),
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        dropout = args.dropout,
        num_action_types = num_action_types,
        num_action_subjects = num_action_subjects
    )
    
    # Train
    print('Starting training...')
    model = train(
        arts = arts,
        model = model,
        device = device,
        run_name = args.run_name,
        experiment_name = args.experiment_name,
        epochs = args.epochs,
        lr = args.lr,
        weight_decay = args.weight_decay,
        grad_clip = args.grad_clip,
        num_neg_train = args.num_neg_train,
        num_neg_eval = args.num_neg_eval,
        amp = not args.no_amp,
        ckpt_dir = args.ckpt_dir,
        monitor = args.monitor,
        patience = args.patience,
        factor = args.factor,
        early_stopping_patience = args.early_stopping
    )
    
    print('Training complete!')


if __name__ == '__main__':
    main()
