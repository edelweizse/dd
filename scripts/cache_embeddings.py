#!/usr/bin/env python
"""
Pre-compute and cache node embeddings for memory-efficient inference.

This script loads the full graph once, computes all node embeddings,
and saves them to disk. After this, inference can be done with minimal
memory using EmbeddingCachePredictor.

Usage:
    python scripts/cache_embeddings.py --checkpoint ./checkpoints/best.pt --output-dir ./embeddings
"""

import argparse
import torch
from pathlib import Path

from src.data.processing import load_processed_data
from src.data.graph import build_graph_from_processed
from src.models.hgt import HGTPredictor
from src.models.predictor_efficient import EmbeddingCachePredictor
from src.training.trainer import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Cache node embeddings for efficient inference')
    
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./embeddings',
                        help='Directory to save embeddings')
    
    # Model arguments (should match training)
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for embeddings')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of message passing layers')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data and build graph
    print('Loading data...')
    data_dict = load_processed_data(args.processed_dir)
    
    print('Building graph (with extended entities)...')
    data, vocabs = build_graph_from_processed(
        processed_data_dir=args.processed_dir,
        add_reverse_edges=True,
        save_vocabs=False,
        include_extended=True
    )
    
    # Create model
    print('Creating model...')
    num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
    print(f'Node types: {list(num_nodes_dict.keys())}')
    for ntype, count in num_nodes_dict.items():
        print(f'  {ntype}: {count:,}')
    
    model = HGTPredictor(
        num_nodes_dict=num_nodes_dict,
        metadata=data.metadata(),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_action_types=vocabs['action_type'].height,
        num_action_subjects=vocabs['action_subject'].height
    )
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Compute and save embeddings
    print(f'\nComputing and saving embeddings to {args.output_dir}...')
    EmbeddingCachePredictor.compute_and_save_embeddings(
        model=model,
        data=data,
        output_dir=args.output_dir,
        device=device
    )
    
    print('\nDone! You can now use EmbeddingCachePredictor for memory-efficient inference:')
    print('''
    from src.models.predictor_efficient import EmbeddingCachePredictor
    from src.data.processing import load_processed_data
    
    data_dict = load_processed_data('./data/processed')
    predictor = EmbeddingCachePredictor.from_cache(
        cache_dir='./embeddings',
        disease_df=data_dict['diseases'],
        chemical_df=data_dict['chemicals']
    )
    result = predictor.predict_pair('MESH:D003920', 'D008687')
    ''')


if __name__ == '__main__':
    main()
