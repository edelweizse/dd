#!/usr/bin/env python
"""
Explain a chemical-disease prediction with ranked graph paths.

Supports both full-model and cached-embedding modes:

    # Full model — Tier 1 + Tier 2 (attention)
    python scripts/explain.py --disease MESH:D014202 --chemical C006901

    # Cached embeddings — Tier 1 only
    python scripts/explain.py --disease MESH:D014202 --chemical C006901 --cached

    # Skip attention (faster, Tier 1 only even with full model)
    python scripts/explain.py --disease MESH:D014202 --chemical C006901 --no-attention

    # Show more paths
    python scripts/explain.py --disease MESH:D014202 --chemical C006901 --max-paths 20
"""

import argparse
import torch
from pathlib import Path

from src.cli_config import parse_args_with_config
from src.data.processing import load_processed_data
from src.data.graph import build_graph_from_processed
from src.models.architectures.hgt import HGTPredictor
from src.models.inference.full_graph import FullGraphPredictor
from src.models.inference.cached_embeddings import CachedEmbeddingPredictor
from src.explainability.explain import build_node_names
from src.training.trainer import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Explain a chemical-disease prediction with ranked graph paths'
    )

    # Required
    parser.add_argument('--disease', type=str, required=True,
                        help='Disease ID (e.g., MESH:D014202)')
    parser.add_argument('--chemical', type=str, required=True,
                        help='Chemical ID (e.g., C006901)')

    # Data
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')

    # Mode selection
    parser.add_argument('--cached', action='store_true',
                        help='Use cached embeddings (Tier 1 only, no attention)')
    parser.add_argument('--embeddings-dir', type=str, default='./embeddings',
                        help='Path to cached embeddings directory')
    parser.add_argument('--checkpoint', type=str, default='/checkpoints/best.pt',
                        help='Path to model checkpoint (full model mode)')

    # Model args (full model mode only)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--no-extended', action='store_true',
                        help='Disable extended graph (pathways, GO terms)')

    # Explainability options
    parser.add_argument('--no-attention', action='store_true',
                        help='Skip attention extraction (faster, Tier 1 only)')
    parser.add_argument('--max-paths', type=int, default=10,
                        help='Max paths to display in output')
    parser.add_argument('--max-paths-per-template', type=int, default=100,
                        help='Max paths per metapath template during enumeration')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')

    args, _ = parse_args_with_config(parser)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Load data ---
    print('Loading metadata...')
    data_dict = load_processed_data(args.processed_dir)
    node_names = build_node_names(data_dict)

    include_extended = not args.no_extended

    if args.cached:
        # ---- Cached embeddings mode (Tier 1 only) ----
        embeddings_path = Path(args.embeddings_dir)
        required = ['chemical_embeddings.npy', 'disease_embeddings.npy', 'W_cd.pt']
        missing = [f for f in required if not (embeddings_path / f).exists()]
        if missing:
            print(f'Error: Missing embedding files: {missing}')
            print('Run scripts/cache_embeddings_chunked.py first.')
            return

        print('Loading cached embeddings...')
        predictor = CachedEmbeddingPredictor.from_cache(
            cache_dir=args.embeddings_dir,
            disease_df=data_dict['diseases'],
            chemical_df=data_dict['chemicals'],
            chem_disease_df=data_dict.get('chem_disease'),
            device=device,
            threshold=args.threshold,
        )

        print('Building graph for path enumeration...')
        data, _vocabs = build_graph_from_processed(
            processed_data_dir=args.processed_dir,
            add_reverse_edges=True,
            save_vocabs=False,
            include_extended=include_extended,
        )

        print('Generating explanation (Tier 1 — metapath + embedding similarity)...')
        explanation = predictor.explain_prediction(
            args.disease, args.chemical,
            data=data,
            node_names=node_names,
            max_paths_per_template=args.max_paths_per_template,
        )
    else:
        # ---- Full model mode (Tier 1 + optional Tier 2) ----
        print(f'Building graph (extended={include_extended})...')
        data, vocabs = build_graph_from_processed(
            processed_data_dir=args.processed_dir,
            add_reverse_edges=True,
            save_vocabs=False,
            include_extended=include_extended,
        )

        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        node_input_dims = {
            ntype: int(data[ntype].x.size(1))
            for ntype in data.node_types
            if isinstance(data[ntype].x, torch.Tensor)
            and data[ntype].x.dim() == 2
            and data[ntype].x.is_floating_point()
        }

        model = HGTPredictor(
            num_nodes_dict=num_nodes_dict,
            metadata=data.metadata(),
            node_input_dims=node_input_dims,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_action_types=vocabs['action_type'].height,
            num_action_subjects=vocabs['action_subject'].height,
        )

        print(f'Loading checkpoint from {args.checkpoint}...')
        load_checkpoint(args.checkpoint, model, device=device)

        predictor = FullGraphPredictor(
            model=model,
            data=data,
            disease_df=data_dict['diseases'],
            chemical_df=data_dict['chemicals'],
            device=device,
            threshold=args.threshold,
        )

        use_attention = not args.no_attention
        tier_label = 'Tier 1 + Tier 2 (attention)' if use_attention else 'Tier 1 only'
        print(f'Generating explanation ({tier_label})...')
        explanation = predictor.explain_prediction(
            args.disease, args.chemical,
            use_attention=use_attention,
            node_names=node_names,
            max_paths_per_template=args.max_paths_per_template,
        )

    # --- Print results ---
    print()
    print('=' * 72)
    print(explanation.summary_text(max_paths=args.max_paths))
    print('=' * 72)

    if explanation.metapath_summary:
        print('\nMetapath type breakdown:')
        for mp_type, count in sorted(
            explanation.metapath_summary.items(), key=lambda x: -x[1]
        ):
            print(f'  {mp_type}: {count}')

    print(f'\nTotal paths found: {len(explanation.paths)}')
    print(f'Displayed top {min(args.max_paths, len(explanation.paths))} paths above.')


if __name__ == '__main__':
    main()
