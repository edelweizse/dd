#!/usr/bin/env python
"""
Run inference with a trained model.

Usage:
    # Predict single pair
    python scripts/predict.py --disease MESH:D003920 --chemical D008687
    
    # Get top chemicals for a disease
    python scripts/predict.py --disease MESH:D003920 --top-k 10
    
    # Get top diseases for a chemical
    python scripts/predict.py --chemical D008687 --top-k 10
"""

import argparse
import torch
import polars as pl
from pathlib import Path

from src.data.processing import load_processed_data
from src.data.graph import build_hetero_data
from src.models.hgt import HGTPredictor
from src.models.predictor import ChemDiseasePredictor
from src.training.trainer import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Run inference for CD link prediction')
    
    # Data arguments
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pt',
                        help='Path to model checkpoint')
    
    # Model arguments (should match training)
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for embeddings')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of message passing layers')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    
    # Prediction arguments
    parser.add_argument('--disease', type=str, default=None,
                        help='Disease ID (e.g., MESH:D003920)')
    parser.add_argument('--chemical', type=str, default=None,
                        help='Chemical ID (e.g., D008687)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top predictions to return')
    parser.add_argument('--include-known', action='store_true',
                        help='Include known associations in results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.disease is None and args.chemical is None:
        parser.error('At least one of --disease or --chemical must be provided')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    data_dict = load_processed_data(args.processed_dir)
    
    # Build graph (full graph, not training subset)
    print('Building graph...')
    data, vocabs = build_hetero_data(
        cnodes=data_dict['chemicals'],
        dnodes=data_dict['diseases'],
        gnodes=data_dict['genes'],
        cd=data_dict['chem_disease'],
        cg=data_dict['chem_gene'],
        dg=data_dict['disease_gene'],
        ppi=data_dict['ppi'],
        add_reverse_edges=True
    )
    
    # Create model
    print('Creating model...')
    num_nodes_dict = {
        'chemical': data_dict['chemicals'].height,
        'disease': data_dict['diseases'].height,
        'gene': data_dict['genes'].height
    }
    
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
    
    # Create predictor
    print('Creating predictor...')
    predictor = ChemDiseasePredictor(
        model=model,
        data=data,
        disease_df=data_dict['diseases'],
        chemical_df=data_dict['chemicals'],
        device=device,
        threshold=args.threshold
    )
    
    # Run predictions
    if args.disease and args.chemical:
        # Single pair prediction
        print(f'\nPredicting association between:')
        print(f'  Disease: {args.disease}')
        print(f'  Chemical: {args.chemical}')
        
        try:
            result = predictor.predict_pair(args.disease, args.chemical)
            print(f'\nResult:')
            print(f'  Disease: {result["disease_name"]} ({result["disease_id"]})')
            print(f'  Chemical: {result["chemical_name"]} ({result["chemical_id"]})')
            print(f'  Probability: {result["probability"]:.4f}')
            print(f'  Prediction: {"Associated" if result["label"] == 1 else "Not Associated"}')
            print(f'  Logit: {result["logit"]:.4f}')
        except ValueError as e:
            print(f'Error: {e}')
    
    elif args.disease:
        # Top chemicals for disease
        print(f'\nTop {args.top_k} chemicals for disease: {args.disease}')
        
        try:
            results = predictor.predict_chemicals_for_disease(
                args.disease,
                top_k=args.top_k,
                exclude_known=not args.include_known
            )
            print(results)
        except ValueError as e:
            print(f'Error: {e}')
    
    else:
        # Top diseases for chemical
        print(f'\nTop {args.top_k} diseases for chemical: {args.chemical}')
        
        try:
            results = predictor.predict_diseases_for_chemical(
                args.chemical,
                top_k=args.top_k,
                exclude_known=not args.include_known
            )
            print(results)
        except ValueError as e:
            print(f'Error: {e}')


if __name__ == '__main__':
    main()
