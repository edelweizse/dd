#!/usr/bin/env python
"""
Memory-efficient embedding computation using chunked/streaming processing.

This script computes node embeddings WITHOUT loading the full graph into memory.
It processes edges in chunks and accumulates messages incrementally.

Usage:
    python scripts/cache_embeddings_chunked.py \
        --checkpoint ./checkpoints/best.pt \
        --output-dir ./embeddings \
        --chunk-size 100000
"""

import argparse
import gc
import torch
import torch.nn as nn
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def load_node_counts(processed_dir: str) -> Dict[str, int]:
    """Load node counts without loading full dataframes."""
    processed_path = Path(processed_dir)
    counts = {}
    
    node_files = {
        'chemical': 'chemicals_nodes.parquet',
        'disease': 'diseases_nodes.parquet', 
        'gene': 'genes_nodes.parquet',
        'pathway': 'pathways_nodes.parquet',
        'go_term': 'go_terms_nodes.parquet'
    }
    
    for node_type, filename in node_files.items():
        filepath = processed_path / filename
        if filepath.exists():
            # Just count rows without loading all data
            df = pl.scan_parquet(filepath).select(pl.len()).collect()
            counts[node_type] = df.item()
    
    return counts


def load_edge_chunk(
    filepath: Path,
    src_col: str,
    dst_col: str,
    attr_cols: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100000
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load a chunk of edges from parquet file."""
    cols = [src_col, dst_col]
    if attr_cols:
        cols.extend(attr_cols)
    
    df = pl.scan_parquet(filepath).select(cols).slice(offset, limit).collect()
    
    if df.height == 0:
        return None, None, None
    
    src = df[src_col].to_numpy()
    dst = df[dst_col].to_numpy()
    
    attr = None
    if attr_cols and len(attr_cols) > 0:
        attr = df.select(attr_cols).to_numpy()
    
    return src, dst, attr


def count_edges(filepath: Path) -> int:
    """Count edges in a parquet file."""
    if not filepath.exists():
        return 0
    return pl.scan_parquet(filepath).select(pl.len()).collect().item()


def extract_model_components(checkpoint_path: str, device: torch.device) -> Dict:
    """Extract model weights from checkpoint without instantiating full model."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model_state']
    
    # Extract node embeddings
    node_embeddings = {}
    for key, value in state.items():
        if key.startswith('node_emb.') and key.endswith('.weight'):
            node_type = key.split('.')[1]
            node_embeddings[node_type] = value
    
    # Extract W_cd decoder
    W_cd = state['W_cd']
    
    # Extract layer weights
    num_layers = 0
    while f'convs.{num_layers}.lin_src.chemical__associated_with__disease.weight' in state:
        num_layers += 1
    
    # Extract hidden dim
    hidden_dim = node_embeddings['chemical'].shape[1]
    
    print(f"  Found {len(node_embeddings)} node types: {list(node_embeddings.keys())}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    
    return {
        'node_embeddings': node_embeddings,
        'W_cd': W_cd,
        'state_dict': state,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers
    }


def chunked_message_passing(
    h: Dict[str, torch.Tensor],
    state_dict: Dict,
    layer_idx: int,
    processed_dir: str,
    chunk_size: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Perform one layer of message passing by processing edges in chunks.
    
    This avoids loading all edges into memory at once.
    """
    processed_path = Path(processed_dir)
    hidden_dim = h['chemical'].shape[1]
    
    # Define edge types and their files
    edge_configs = [
        # (file, src_col, dst_col, src_type, dst_type, rel, attr_cols)
        ('chem_disease_edges.parquet', 'CHEM_ID', 'DS_ID', 'chemical', 'disease', 'associated_with', None),
        ('chem_gene_edges.parquet', 'CHEM_ID', 'GENE_ID', 'chemical', 'gene', 'affects', ['ACTION_TYPE', 'ACTION_SUBJECT']),
        ('disease_gene_edges.parquet', 'DS_ID', 'GENE_ID', 'disease', 'gene', 'targets', None),
        ('ppi_edges.parquet', 'GENE_ID_1', 'GENE_ID_2', 'gene', 'gene', 'interacts_with', None),
    ]
    
    # Add extended edges if they exist
    extended_edges = [
        ('gene_pathway_edges.parquet', 'GENE_ID', 'PATHWAY_ID', 'gene', 'pathway', 'participates_in', None),
        ('disease_pathway_edges.parquet', 'DS_ID', 'PATHWAY_ID', 'disease', 'pathway', 'disrupts', ['INFERENCE_GENE_COUNT']),
        ('chem_pathway_edges.parquet', 'CHEM_ID', 'PATHWAY_ID', 'chemical', 'pathway', 'enriched_in', ['NEG_LOG_PVALUE', 'TARGET_RATIO', 'FOLD_ENRICHMENT']),
        ('chem_go_edges.parquet', 'CHEM_ID', 'GO_TERM_ID', 'chemical', 'go_term', 'enriched_in', ['NEG_LOG_PVALUE', 'TARGET_RATIO', 'FOLD_ENRICHMENT', 'ONTOLOGY_TYPE', 'GO_LEVEL_NORM']),
        ('chem_pheno_edges.parquet', 'CHEM_ID', 'GO_TERM_ID', 'chemical', 'go_term', 'affects_phenotype', ['PHENO_ACTION_TYPE']),
        ('go_disease_edges.parquet', 'GO_TERM_ID', 'DS_ID', 'go_term', 'disease', 'associated_with', ['ONTOLOGY_TYPE', 'LOG_INFERENCE_CHEM', 'LOG_INFERENCE_GENE']),
    ]
    
    for cfg in extended_edges:
        if (processed_path / cfg[0]).exists():
            edge_configs.append(cfg)
    
    # Initialize output accumulators
    h_out = {ntype: torch.zeros_like(emb) for ntype, emb in h.items()}
    h_count = {ntype: torch.zeros(emb.shape[0], device=emb.device) for ntype, emb in h.items()}
    
    # Process each edge type
    for config in tqdm(edge_configs, desc=f"Layer {layer_idx} edges"):
        filename, src_col, dst_col, src_type, dst_type, rel, attr_cols = config
        filepath = processed_path / filename
        
        if not filepath.exists():
            continue
        
        if src_type not in h or dst_type not in h:
            continue
        
        # Get linear transform weights
        edge_key = f'{src_type}__{rel}__{dst_type}'
        lin_src_key = f'convs.{layer_idx}.lin_src.{edge_key}.weight'
        lin_dst_key = f'convs.{layer_idx}.lin_dst.{edge_key}.weight'
        
        if lin_src_key not in state_dict:
            continue
        
        W_src = state_dict[lin_src_key].to(device)
        b_src = state_dict[f'convs.{layer_idx}.lin_src.{edge_key}.bias'].to(device)
        W_dst = state_dict[lin_dst_key].to(device)
        b_dst = state_dict[f'convs.{layer_idx}.lin_dst.{edge_key}.bias'].to(device)
        
        # Count total edges
        total_edges = count_edges(filepath)
        
        # Process in chunks
        offset = 0
        while offset < total_edges:
            src_ids, dst_ids, attrs = load_edge_chunk(
                filepath, src_col, dst_col, attr_cols, offset, chunk_size
            )
            
            if src_ids is None:
                break
            
            src_ids = torch.from_numpy(src_ids).long().to(device)
            dst_ids = torch.from_numpy(dst_ids).long().to(device)
            
            # Compute messages: linear(src) + linear(dst)
            src_feat = h[src_type][src_ids]  # [E, hidden]
            dst_feat = h[dst_type][dst_ids]  # [E, hidden]
            
            msg = (src_feat @ W_src.T + b_src) + (dst_feat @ W_dst.T + b_dst)
            
            # Aggregate to destination (scatter add)
            h_out[dst_type].scatter_add_(0, dst_ids.unsqueeze(-1).expand_as(msg), msg)
            h_count[dst_type].scatter_add_(0, dst_ids, torch.ones(dst_ids.shape[0], device=device))
            
            offset += chunk_size
            
            # Free memory
            del src_ids, dst_ids, src_feat, dst_feat, msg
            if attrs is not None:
                del attrs
        
        # Also process reverse edges
        rev_edge_key = f'{dst_type}__rev_{rel}__{src_type}'
        rev_lin_src_key = f'convs.{layer_idx}.lin_src.{rev_edge_key}.weight'
        
        if rev_lin_src_key in state_dict:
            W_src_rev = state_dict[rev_lin_src_key].to(device)
            b_src_rev = state_dict[f'convs.{layer_idx}.lin_src.{rev_edge_key}.bias'].to(device)
            W_dst_rev = state_dict[f'convs.{layer_idx}.lin_dst.{rev_edge_key}.weight'].to(device)
            b_dst_rev = state_dict[f'convs.{layer_idx}.lin_dst.{rev_edge_key}.bias'].to(device)
            
            offset = 0
            while offset < total_edges:
                src_ids, dst_ids, attrs = load_edge_chunk(
                    filepath, src_col, dst_col, attr_cols, offset, chunk_size
                )
                
                if src_ids is None:
                    break
                
                # Reverse the direction
                src_ids_rev = torch.from_numpy(dst_ids).long().to(device)
                dst_ids_rev = torch.from_numpy(src_ids).long().to(device)
                
                src_feat = h[dst_type][src_ids_rev]
                dst_feat = h[src_type][dst_ids_rev]
                
                msg = (src_feat @ W_src_rev.T + b_src_rev) + (dst_feat @ W_dst_rev.T + b_dst_rev)
                
                h_out[src_type].scatter_add_(0, dst_ids_rev.unsqueeze(-1).expand_as(msg), msg)
                h_count[src_type].scatter_add_(0, dst_ids_rev, torch.ones(dst_ids_rev.shape[0], device=device))
                
                offset += chunk_size
                
                del src_ids_rev, dst_ids_rev, src_feat, dst_feat, msg
        
        # Clean up layer weights
        del W_src, b_src, W_dst, b_dst
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Average messages and apply output projection + residual
    for ntype in h_out:
        count = h_count[ntype].clamp(min=1).unsqueeze(-1)
        h_out[ntype] = h_out[ntype] / count
        
        # Output projection
        out_key = f'convs.{layer_idx}.lin_out.{ntype}.weight'
        if out_key in state_dict:
            W_out = state_dict[out_key].to(device)
            b_out = state_dict[f'convs.{layer_idx}.lin_out.{ntype}.bias'].to(device)
            h_out[ntype] = h_out[ntype] @ W_out.T + b_out
            del W_out, b_out
        
        # Residual connection
        h_out[ntype] = h_out[ntype] + h[ntype]
        
        # LayerNorm
        norm_w_key = f'norms.{layer_idx}.{ntype}.weight'
        if norm_w_key in state_dict:
            norm_w = state_dict[norm_w_key].to(device)
            norm_b = state_dict[f'norms.{layer_idx}.{ntype}.bias'].to(device)
            
            mean = h_out[ntype].mean(dim=-1, keepdim=True)
            std = h_out[ntype].std(dim=-1, keepdim=True)
            h_out[ntype] = (h_out[ntype] - mean) / (std + 1e-5) * norm_w + norm_b
            del norm_w, norm_b
        
        # GELU activation
        h_out[ntype] = torch.nn.functional.gelu(h_out[ntype])
    
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return h_out


def compute_embeddings_chunked(
    checkpoint_path: str,
    processed_dir: str,
    output_dir: str,
    chunk_size: int = 100000,
    device: Optional[torch.device] = None
):
    """
    Compute all node embeddings using chunked processing.
    
    This never loads the full graph into memory.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model components
    components = extract_model_components(checkpoint_path, device)
    node_embeddings = components['node_embeddings']
    state_dict = components['state_dict']
    num_layers = components['num_layers']
    
    # Initialize node features from embeddings
    print("\nInitializing node embeddings...")
    h = {}
    for ntype, emb_weight in node_embeddings.items():
        num_nodes = emb_weight.shape[0]
        # Lookup embeddings for all node IDs (0 to num_nodes-1)
        h[ntype] = emb_weight.to(device)
        print(f"  {ntype}: {h[ntype].shape}")
    
    # Run message passing layers
    for layer_idx in range(num_layers):
        print(f"\nProcessing layer {layer_idx + 1}/{num_layers}...")
        h = chunked_message_passing(
            h, state_dict, layer_idx, processed_dir, chunk_size, device
        )
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save embeddings
    print(f"\nSaving embeddings to {output_dir}...")
    for ntype, emb in h.items():
        emb_np = emb.cpu().numpy()
        np.save(output_path / f'{ntype}_embeddings.npy', emb_np)
        print(f"  {ntype}: {emb_np.shape}")
    
    # Save decoder weights
    torch.save(components['W_cd'], output_path / 'W_cd.pt')
    print("  W_cd saved")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Compute embeddings with chunked processing')
    
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./embeddings',
                        help='Directory to save embeddings')
    parser.add_argument('--chunk-size', type=int, default=100000,
                        help='Number of edges to process at once')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')
    
    args = parser.parse_args()
    
    device = torch.device('cpu') if args.cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    compute_embeddings_chunked(
        checkpoint_path=args.checkpoint,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        device=device
    )


if __name__ == '__main__':
    main()
