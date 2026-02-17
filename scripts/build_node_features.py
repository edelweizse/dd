#!/usr/bin/env python
"""
Build inductive node features for all graph node types.

Usage:
  PYTHONPATH=. python scripts/build_node_features.py \
      --processed-dir ./data/processed \
      --raw-dir ./data/raw \
      --output-dir ./data/processed/features
"""

import argparse
from pathlib import Path

from src.cli_config import parse_args_with_config
from src.data.node_features import NodeFeatureConfig, build_node_feature_tables


def main() -> None:
    parser = argparse.ArgumentParser(description='Build inductive node feature tables')
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed parquet directory')
    parser.add_argument('--raw-dir', type=str, default='./data/raw',
                        help='Path to raw CTD directory')
    parser.add_argument('--output-dir', type=str, default='./data/processed/features',
                        help='Output directory for node feature parquet files')

    parser.add_argument('--text-dim', type=int, default=128,
                        help='Text embedding dimension per node type')
    parser.add_argument('--chem-fp-bits', type=int, default=1024,
                        help='Morgan fingerprint bits for chemical structure')

    parser.add_argument('--no-pubchem', action='store_true',
                        help='Disable PubChem SMILES enrichment')
    parser.add_argument('--no-uniprot', action='store_true',
                        help='Disable UniProt sequence enrichment')
    parser.add_argument('--max-pubchem-fetch', type=int, default=None,
                        help='Maximum new PubChem CID fetches (for quick dry runs)')
    parser.add_argument('--max-uniprot-fetch', type=int, default=None,
                        help='Maximum new UniProt fetches (for quick dry runs)')
    parser.add_argument('--use-umls', action='store_true',
                        help='Augment chemical/disease text with UMLS definitions')
    parser.add_argument('--umls-api-key', type=str, default=None,
                        help='UMLS API key (or set UMLS_API_KEY env var)')
    parser.add_argument('--max-umls-fetch', type=int, default=None,
                        help='Maximum new UMLS term lookups (for quick dry runs)')
    parser.add_argument('--disgenet-file', type=str, default=None,
                        help='Optional DisGeNET file (CSV/TSV) for disease/gene evidence features')
    parser.add_argument('--request-timeout-s', type=int, default=20,
                        help='HTTP timeout for external API calls')
    parser.add_argument('--sleep-s', type=float, default=0.02,
                        help='Sleep between external API requests')

    args, _ = parse_args_with_config(parser)

    cfg = NodeFeatureConfig(
        text_dim=args.text_dim,
        chem_fp_bits=args.chem_fp_bits,
        include_pubchem=not args.no_pubchem,
        include_uniprot=not args.no_uniprot,
        include_umls=args.use_umls,
        umls_api_key=args.umls_api_key,
        disgenet_file=args.disgenet_file,
        request_timeout_s=args.request_timeout_s,
        sleep_s=args.sleep_s,
        max_pubchem_fetch=args.max_pubchem_fetch,
        max_uniprot_fetch=args.max_uniprot_fetch,
        max_umls_fetch=args.max_umls_fetch,
    )

    print('Building node features...')
    print(f'  processed: {args.processed_dir}')
    print(f'  raw:       {args.raw_dir}')
    print(f'  output:    {args.output_dir}')
    print(f'  text_dim:  {cfg.text_dim}')
    print(f'  fp_bits:   {cfg.chem_fp_bits}')
    print(f'  pubchem:   {cfg.include_pubchem}')
    print(f'  uniprot:   {cfg.include_uniprot}')
    print(f'  umls:      {cfg.include_umls}')
    print(f'  disgenet:  {bool(cfg.disgenet_file)}')

    tables = build_node_feature_tables(
        processed_data_dir=args.processed_dir,
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        config=cfg,
    )

    print('\nFeature tables written:')
    for node_type, df in tables.items():
        print(f'  {node_type:8s} nodes={df.height:6d} dim={max(df.width - 1, 0):4d}')

    metadata_file = Path(args.output_dir) / 'node_feature_metadata.json'
    if metadata_file.exists():
        print(f'\nMetadata: {metadata_file}')


if __name__ == '__main__':
    main()
