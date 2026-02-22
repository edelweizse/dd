#!/usr/bin/env python
"""
Explain a chemical-disease prediction with ranked graph evidence.

Mode:
- path_attention (default): fast metapath + optional attention scoring
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from src.cli_config import parse_args_with_config
from src.data.processing import load_processed_data
from src.data.graph import build_graph_from_processed
from src.models.architectures.hgt import HGTPredictor, infer_hgt_hparams_from_state
from src.models.inference.full_graph import FullGraphPredictor
from src.models.inference.cached_embeddings import CachedEmbeddingPredictor
from src.explainability.explain import build_node_names


def _checkpoint_has_extended_types(model_state: dict) -> bool:
    """Best-effort detection of pathway/GO parameters in checkpoint state."""
    markers = ("pathway", "go_term")
    for key in model_state.keys():
        if any(marker in key for marker in markers):
            return True
    return False


def _result_to_dict(result) -> Dict[str, Any]:
    paths = []
    for sp in result.paths:
        paths.append(
            {
                'evidence_type': sp.evidence_type,
                'combined_score': float(sp.combined_score),
                'attention_score': float(sp.attention_score),
                'embedding_score': float(sp.embedding_score),
                'description': sp.description,
                'edge_attentions': [float(x) for x in sp.edge_attentions],
                'template_name': sp.path.template_name,
                'node_indices': [int(x) for x in sp.path.node_indices],
                'node_types': list(sp.path.node_types),
                'edge_types': [list(et) for et in sp.path.edge_types],
                'edge_positions': [int(x) for x in sp.path.edge_positions],
            }
        )
    return {
        'chemical_id': result.chemical_id,
        'disease_id': result.disease_id,
        'chemical_name': result.chemical_name,
        'disease_name': result.disease_name,
        'probability': float(result.probability),
        'label': int(result.label),
        'logit': float(result.logit),
        'known': bool(result.known),
        'engine': result.engine,
        'runtime_profile': result.runtime_profile,
        'attention_available': bool(result.attention_available),
        'metapath_summary': dict(result.metapath_summary),
        'debug_metrics': result.debug_metrics,
        'paths': paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Explain a chemical-disease prediction with ranked graph evidence'
    )

    parser.add_argument('--disease', type=str, required=True, help='Disease ID (e.g., MESH:D014202)')
    parser.add_argument('--chemical', type=str, required=True, help='Chemical ID (e.g., C006901)')

    parser.add_argument('--processed-dir', type=str, default='./data/processed', help='Processed data directory')

    parser.add_argument('--cached', action='store_true', help='Use cached embeddings mode (path_attention only)')
    parser.add_argument('--embeddings-dir', type=str, default='./embeddings', help='Cached embeddings directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pt', help='Model checkpoint path')

    parser.add_argument('--hidden-dim', type=int, default=None)
    parser.add_argument('--num-layers', type=int, default=None)
    parser.add_argument('--num-heads', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--no-extended', action='store_true', help='Disable extended graph (pathways/GO)')

    parser.add_argument('--mode', type=str, default='path_attention', choices=['path_attention'])
    parser.add_argument('--runtime-profile', type=str, default='fast', choices=['fast', 'balanced', 'deep'])
    parser.add_argument('--template-set', type=str, default='default', help='Template set name')
    parser.add_argument('--no-attention', action='store_true', help='Skip attention extraction in path_attention mode')
    parser.add_argument('--max-paths', type=int, default=10, help='Number of top paths to print')
    parser.add_argument('--max-paths-total', type=int, default=500, help='Total paths retained after scoring')
    parser.add_argument('--max-paths-per-template', type=int, default=100, help='Max raw paths per template')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--output-format', type=str, default='text', choices=['text', 'json'])

    args, _ = parse_args_with_config(parser)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading metadata...')
    data_dict = load_processed_data(args.processed_dir)
    node_names = build_node_names(data_dict)

    include_extended = not args.no_extended

    if args.cached:
        if args.mode != 'path_attention':
            raise ValueError('Cached mode supports only --mode path_attention.')

        embeddings_path = Path(args.embeddings_dir)
        required = ['chemical_embeddings.npy', 'disease_embeddings.npy', 'W_cd.pt']
        missing = [f for f in required if not (embeddings_path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f'Missing embedding files: {missing}. Run scripts/cache_embeddings_chunked.py first.'
            )

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

        explanation = predictor.explain_prediction(
            args.disease,
            args.chemical,
            mode=args.mode,
            runtime_profile=args.runtime_profile,
            template_set=args.template_set,
            data=data,
            node_names=node_names,
            max_paths_total=args.max_paths_total,
            max_paths_per_template=args.max_paths_per_template,
        )
    else:
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

        predictor = FullGraphPredictor(
            model=model,
            data=data,
            disease_df=data_dict['diseases'],
            chemical_df=data_dict['chemicals'],
            device=device,
            threshold=args.threshold,
        )

        explanation = predictor.explain_prediction(
            args.disease,
            args.chemical,
            mode=args.mode,
            runtime_profile=args.runtime_profile,
            template_set=args.template_set,
            use_attention=(not args.no_attention),
            node_names=node_names,
            max_paths_total=args.max_paths_total,
            max_paths_per_template=args.max_paths_per_template,
        )

    if args.output_format == 'json':
        print(json.dumps(_result_to_dict(explanation), indent=2, sort_keys=False))
        return

    print()
    print('=' * 72)
    print(explanation.summary_text(max_paths=args.max_paths))
    print('=' * 72)

    if explanation.metapath_summary:
        print('\nMetapath type breakdown:')
        for mp_type, count in sorted(explanation.metapath_summary.items(), key=lambda x: -x[1]):
            print(f'  {mp_type}: {count}')

    print(f'\nTotal paths found: {len(explanation.paths)}')
    print(f'Displayed top {min(args.max_paths, len(explanation.paths))} paths above.')


if __name__ == '__main__':
    main()
