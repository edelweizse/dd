"""
Process raw CTD data and save as parquet files.

Usage:
    python scripts/process_data.py --raw-dir ./data/raw --processed-dir ./data/processed
"""

import argparse
from src.data.processing import process_and_save
from src.cli_config import parse_args_with_config


def main():
    parser = argparse.ArgumentParser(description='Process raw CTD data')
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='./data/raw',
        help='Path to raw data directory'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='./data/processed',
        help='Path to processed data directory'
    )
    
    args, _ = parse_args_with_config(parser)
    
    print(f'Processing data from {args.raw_dir} to {args.processed_dir}')
    process_and_save(args.raw_dir, args.processed_dir)
    print('Done!')


if __name__ == '__main__':
    main()
