#!/usr/bin/env python
"""
Memory-efficient inference using cached embeddings.

This script uses pre-computed embeddings for inference, requiring minimal memory.
Run scripts/cache_embeddings_chunked.py first to generate the embedding cache.

Usage:
    # Predict single pair
    python scripts/predict_cached.py --disease MESH:D003920 --chemical D008687
    
    # Get top chemicals for a disease
    python scripts/predict_cached.py --disease MESH:D003920 --top-k 10
    
    # Get top diseases for a chemical
    python scripts/predict_cached.py --chemical D008687 --top-k 10
    
    # Exclude known associations from results
    python scripts/predict_cached.py --disease MESH:D003920 --top-k 10 --exclude-known
"""

import argparse
import torch
from pathlib import Path

from src.cli_config import parse_args_with_config
from src.data.processing import load_processed_data
from src.models.predictor_efficient import EmbeddingCachePredictor


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient CD link prediction')
    
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--embeddings-dir', type=str, default='./embeddings',
                        help='Path to cached embeddings directory')
    
    parser.add_argument('--disease', type=str, default=None,
                        help='Disease ID (e.g., MESH:D003920)')
    parser.add_argument('--chemical', type=str, default=None,
                        help='Chemical ID (e.g., D008687)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top predictions to return')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--exclude-known', action='store_true',
                        help='Exclude known associations from top-k results')
    
    args, _ = parse_args_with_config(parser)
    
    if args.disease is None and args.chemical is None:
        parser.error('At least one of --disease or --chemical must be provided')
    
    embeddings_path = Path(args.embeddings_dir)
    if not embeddings_path.exists():
        print(f'Error: Embeddings directory not found: {args.embeddings_dir}')
        print('Run scripts/cache_embeddings_chunked.py first to generate embeddings.')
        return
    
    required_files = ['chemical_embeddings.npy', 'disease_embeddings.npy', 'W_cd.pt']
    missing = [f for f in required_files if not (embeddings_path / f).exists()]
    if missing:
        print(f'Error: Missing embedding files: {missing}')
        print('Run scripts/cache_embeddings_chunked.py first to generate embeddings.')
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading metadata...')
    data_dict = load_processed_data(args.processed_dir)
    
    print('Loading cached embeddings...')
    predictor = EmbeddingCachePredictor.from_cache(
        cache_dir=args.embeddings_dir,
        disease_df=data_dict['diseases'],
        chemical_df=data_dict['chemicals'],
        chem_disease_df=data_dict['chem_disease'],
        device=device,
        threshold=args.threshold
    )
    
    if args.disease and args.chemical:
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
            print(f'  Known link: {"Yes" if result["known"] else "No"}')
            print(f'  Logit: {result["logit"]:.4f}')
        except ValueError as e:
            print(f'Error: {e}')
    
    elif args.disease:
        print(f'\nTop {args.top_k} chemicals for disease: {args.disease}')
        if args.exclude_known:
            print('(excluding known associations)')
        
        try:
            results = predictor.predict_chemicals_for_disease(
                args.disease,
                top_k=args.top_k,
                exclude_known=args.exclude_known
            )
            print(results)
        except ValueError as e:
            print(f'Error: {e}')
    
    else:
        print(f'\nTop {args.top_k} diseases for chemical: {args.chemical}')
        if args.exclude_known:
            print('(excluding known associations)')
        
        try:
            results = predictor.predict_diseases_for_chemical(
                args.chemical,
                top_k=args.top_k,
                exclude_known=args.exclude_known
            )
            print(results)
        except ValueError as e:
            print(f'Error: {e}')


if __name__ == '__main__':
    main()
