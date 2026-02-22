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

from src.cli_config import parse_args_with_config
from src.data.processing import load_processed_data
from src.data.graph import build_graph_from_processed
from src.models.architectures.hgt import HGTPredictor, infer_hgt_hparams_from_state
from src.models.inference.full_graph import FullGraphPredictor


def _checkpoint_has_extended_types(model_state: dict) -> bool:
    """Best-effort detection of pathway/GO parameters in checkpoint state."""
    markers = ("pathway", "go_term")
    for key in model_state.keys():
        if any(marker in key for marker in markers):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Run inference for CD link prediction')
    
    # Data arguments
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pt',
                        help='Path to model checkpoint')
    
    # Optional model overrides (checkpoint values are used by default)
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help='Override hidden dimension from checkpoint')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Override number of message passing layers from checkpoint')
    parser.add_argument('--num-heads', type=int, default=None,
                        help='Override number of attention heads from checkpoint')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override dropout (inference uses 0.0 by default)')
    parser.add_argument('--no-extended', action='store_true',
                        help='Disable extended graph (pathways, GO terms). '
                             'Use if checkpoint was trained without extended entities.')
    
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
    
    args, _ = parse_args_with_config(parser)
    
    # Validate arguments
    if args.disease is None and args.chemical is None:
        parser.error('At least one of --disease or --chemical must be provided')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data and build graph (with extended entities: pathways, GO terms)
    print('Loading data...')
    data_dict = load_processed_data(args.processed_dir)
    
    # Build graph - include extended entities (pathways, GO terms) by default
    # to match models trained with full graph
    include_extended = not args.no_extended
    print(f'Building graph (extended={include_extended})...')
    data, vocabs = build_graph_from_processed(
        processed_data_dir=args.processed_dir,
        add_reverse_edges=True,
        save_vocabs=False,
        include_extended=include_extended
    )
    
    # Create model with all node types
    print('Creating model...')
    num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
    node_input_dims = {
        ntype: int(data[ntype].x.size(1))
        for ntype in data.node_types
        if isinstance(data[ntype].x, torch.Tensor)
        and data[ntype].x.dim() == 2
        and data[ntype].x.is_floating_point()
    }
    print(f'Node types: {list(num_nodes_dict.keys())}')
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = infer_hgt_hparams_from_state(ckpt['model_state'])
    ckpt_has_extended = _checkpoint_has_extended_types(ckpt['model_state'])

    model = HGTPredictor(
        num_nodes_dict=num_nodes_dict,
        metadata=data.metadata(),
        node_input_dims=model_cfg['node_input_dims'] or node_input_dims,
        hidden_dim=args.hidden_dim or model_cfg['hidden_dim'],
        num_layers=args.num_layers or model_cfg['num_layers'],
        num_heads=args.num_heads or model_cfg['num_heads'],
        dropout=args.dropout if args.dropout is not None else 0.0,
        num_action_types=model_cfg['num_action_types'] or vocabs['action_type'].height,
        num_action_subjects=model_cfg['num_action_subjects'] or vocabs['action_subject'].height,
        num_pheno_action_types=model_cfg['num_pheno_action_types'],
    )
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    try:
        model.load_state_dict(ckpt['model_state'])
    except RuntimeError:
        if ckpt_has_extended and args.no_extended:
            raise RuntimeError(
                'Checkpoint appears to be trained with extended graph types '
                '(pathway/go_term), but --no-extended was passed. '
                'Remove --no-extended or use a non-extended checkpoint.'
            ) from None
        if (not ckpt_has_extended) and (not args.no_extended):
            raise RuntimeError(
                'Checkpoint appears to be trained without extended graph types, '
                'but current run builds the extended graph. Pass --no-extended '
                'or use an extended checkpoint.'
            ) from None
        raise
    model = model.to(device)
    model.eval()
    
    # Create predictor
    print('Creating predictor...')
    predictor = FullGraphPredictor(
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
            raise SystemExit(2) from e
    
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
            raise SystemExit(2) from e
    
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
            raise SystemExit(2) from e


if __name__ == '__main__':
    main()
