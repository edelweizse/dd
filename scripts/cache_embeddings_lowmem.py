#!/usr/bin/env python
"""
Ultra memory-efficient embedding computation.

This script computes node embeddings using minimal memory by:
1. Only loading node embeddings (small)
2. Processing edges in small chunks
3. Using simplified message passing (no edge attributes)

The results may differ slightly from full model inference due to
simplified edge attribute handling, but will be close.

Usage:
    python scripts/cache_embeddings_lowmem.py \
        --checkpoint ./checkpoints/best.pt \
        --output-dir ./embeddings \
        --chunk-size 50000
"""

import argparse
import gc
import torch
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm


def compute_embeddings_lowmem(
    checkpoint_path: str,
    processed_dir: str,
    output_dir: str,
    chunk_size: int = 50000,
    device: Optional[torch.device] = None
):
    """
    Compute embeddings with minimal memory using chunked edge processing.
    """
    device = device or torch.device('cpu')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    processed_path = Path(processed_dir)
    
    print(f"Using device: {device}")
    print(f"Chunk size: {chunk_size:,}")
    
    # =========================================================================
    # STEP 1: Load checkpoint (only what we need)
    # =========================================================================
    print("\n[1/4] Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model_state']
    
    # Extract node embeddings
    node_emb = {}
    for key, value in state.items():
        if key.startswith('node_emb.') and key.endswith('.weight'):
            ntype = key.split('.')[1]
            node_emb[ntype] = value
            print(f"  {ntype}: {value.shape}")
    
    hidden_dim = node_emb['chemical'].shape[1]
    print(f"  Hidden dim: {hidden_dim}")
    
    # Count layers
    num_layers = 0
    while f'convs.{num_layers}.lin_out.chemical.weight' in state:
        num_layers += 1
    print(f"  Num layers: {num_layers}")
    
    # Save W_cd immediately and free memory
    W_cd = state['W_cd'].clone()
    torch.save(W_cd, output_path / 'W_cd.pt')
    print("  W_cd saved")
    
    # =========================================================================
    # STEP 2: Initialize embeddings
    # =========================================================================
    print("\n[2/4] Initializing embeddings...")
    h = {ntype: emb.clone().to(device) for ntype, emb in node_emb.items()}
    
    # Free original embeddings
    del node_emb
    gc.collect()
    
    # =========================================================================
    # STEP 3: Define edge configurations
    # =========================================================================
    # (parquet_file, src_col, dst_col, src_type, dst_type, relation)
    edge_configs = []
    
    # Core edges
    core_edges = [
        ('chem_disease_edges.parquet', 'CHEM_ID', 'DS_ID', 'chemical', 'disease', 'associated_with'),
        ('chem_gene_edges.parquet', 'CHEM_ID', 'GENE_ID', 'chemical', 'gene', 'affects'),
        ('disease_gene_edges.parquet', 'DS_ID', 'GENE_ID', 'disease', 'gene', 'targets'),
        ('ppi_edges.parquet', 'GENE_ID_1', 'GENE_ID_2', 'gene', 'gene', 'interacts_with'),
    ]
    
    # Extended edges (if files exist)
    extended_edges = [
        ('gene_pathway_edges.parquet', 'GENE_ID', 'PATHWAY_ID', 'gene', 'pathway', 'participates_in'),
        ('disease_pathway_edges.parquet', 'DS_ID', 'PATHWAY_ID', 'disease', 'pathway', 'disrupts'),
        ('chem_pathway_edges.parquet', 'CHEM_ID', 'PATHWAY_ID', 'chemical', 'pathway', 'enriched_in'),
        ('chem_go_edges.parquet', 'CHEM_ID', 'GO_TERM_ID', 'chemical', 'go_term', 'enriched_in'),
        ('chem_pheno_edges.parquet', 'CHEM_ID', 'GO_TERM_ID', 'chemical', 'go_term', 'affects_phenotype'),
        ('go_disease_edges.parquet', 'GO_TERM_ID', 'DS_ID', 'go_term', 'disease', 'associated_with'),
    ]
    
    for cfg in core_edges + extended_edges:
        if (processed_path / cfg[0]).exists():
            edge_configs.append(cfg)
    
    print(f"  Found {len(edge_configs)} edge types")
    
    # =========================================================================
    # STEP 4: Message passing layers
    # =========================================================================
    print("\n[3/4] Running message passing...")
    
    for layer_idx in range(num_layers):
        print(f"\n  Layer {layer_idx + 1}/{num_layers}")
        
        # Accumulators for this layer
        h_acc = {ntype: torch.zeros_like(emb) for ntype, emb in h.items()}
        h_cnt = {ntype: torch.zeros(emb.shape[0], device=device) for ntype, emb in h.items()}
        
        # Process each edge type
        for filename, src_col, dst_col, src_type, dst_type, rel in tqdm(edge_configs, desc="    Edge types"):
            if src_type not in h or dst_type not in h:
                continue
            
            filepath = processed_path / filename
            
            # Get transform weights
            edge_key = f'{src_type}__{rel}__{dst_type}'
            w_src_key = f'convs.{layer_idx}.lin_src.{edge_key}.weight'
            
            if w_src_key not in state:
                continue
            
            W_src = state[w_src_key].to(device)
            b_src = state[f'convs.{layer_idx}.lin_src.{edge_key}.bias'].to(device)
            W_dst = state[f'convs.{layer_idx}.lin_dst.{edge_key}.weight'].to(device)
            b_dst = state[f'convs.{layer_idx}.lin_dst.{edge_key}.bias'].to(device)
            
            # Process forward edges in chunks
            total_edges = pl.scan_parquet(filepath).select(pl.len()).collect().item()
            
            for offset in range(0, total_edges, chunk_size):
                chunk = pl.scan_parquet(filepath).slice(offset, chunk_size).collect()
                if chunk.height == 0:
                    break
                
                src_ids = torch.from_numpy(chunk[src_col].to_numpy()).long().to(device)
                dst_ids = torch.from_numpy(chunk[dst_col].to_numpy()).long().to(device)
                
                # Message: linear(src) + linear(dst)
                msg = (h[src_type][src_ids] @ W_src.T + b_src) + \
                      (h[dst_type][dst_ids] @ W_dst.T + b_dst)
                
                # Scatter to destination
                h_acc[dst_type].scatter_add_(0, dst_ids.unsqueeze(-1).expand_as(msg), msg)
                h_cnt[dst_type].scatter_add_(0, dst_ids, torch.ones_like(dst_ids, dtype=torch.float))
                
                del chunk, src_ids, dst_ids, msg
            
            # Process reverse edges
            rev_edge_key = f'{dst_type}__rev_{rel}__{src_type}'
            w_src_rev_key = f'convs.{layer_idx}.lin_src.{rev_edge_key}.weight'
            
            if w_src_rev_key in state:
                W_src_r = state[w_src_rev_key].to(device)
                b_src_r = state[f'convs.{layer_idx}.lin_src.{rev_edge_key}.bias'].to(device)
                W_dst_r = state[f'convs.{layer_idx}.lin_dst.{rev_edge_key}.weight'].to(device)
                b_dst_r = state[f'convs.{layer_idx}.lin_dst.{rev_edge_key}.bias'].to(device)
                
                for offset in range(0, total_edges, chunk_size):
                    chunk = pl.scan_parquet(filepath).slice(offset, chunk_size).collect()
                    if chunk.height == 0:
                        break
                    
                    # Reverse: dst -> src
                    src_ids = torch.from_numpy(chunk[dst_col].to_numpy()).long().to(device)
                    dst_ids = torch.from_numpy(chunk[src_col].to_numpy()).long().to(device)
                    
                    msg = (h[dst_type][src_ids] @ W_src_r.T + b_src_r) + \
                          (h[src_type][dst_ids] @ W_dst_r.T + b_dst_r)
                    
                    h_acc[src_type].scatter_add_(0, dst_ids.unsqueeze(-1).expand_as(msg), msg)
                    h_cnt[src_type].scatter_add_(0, dst_ids, torch.ones_like(dst_ids, dtype=torch.float))
                    
                    del chunk, src_ids, dst_ids, msg
                
                del W_src_r, b_src_r, W_dst_r, b_dst_r
            
            del W_src, b_src, W_dst, b_dst
            gc.collect()
        
        # Apply aggregation, output projection, residual, norm, activation
        for ntype in h:
            # Average messages
            cnt = h_cnt[ntype].clamp(min=1).unsqueeze(-1)
            h_acc[ntype] = h_acc[ntype] / cnt
            
            # Output projection
            out_w_key = f'convs.{layer_idx}.lin_out.{ntype}.weight'
            if out_w_key in state:
                W_out = state[out_w_key].to(device)
                b_out = state[f'convs.{layer_idx}.lin_out.{ntype}.bias'].to(device)
                h_acc[ntype] = h_acc[ntype] @ W_out.T + b_out
                del W_out, b_out
            
            # Residual
            h_acc[ntype] = h_acc[ntype] + h[ntype]
            
            # LayerNorm
            norm_w_key = f'norms.{layer_idx}.{ntype}.weight'
            if norm_w_key in state:
                gamma = state[norm_w_key].to(device)
                beta = state[f'norms.{layer_idx}.{ntype}.bias'].to(device)
                mean = h_acc[ntype].mean(dim=-1, keepdim=True)
                var = h_acc[ntype].var(dim=-1, keepdim=True, unbiased=False)
                h_acc[ntype] = (h_acc[ntype] - mean) / (var + 1e-5).sqrt() * gamma + beta
                del gamma, beta
            
            # GELU
            h_acc[ntype] = torch.nn.functional.gelu(h_acc[ntype])
        
        # Update h for next layer
        h = h_acc
        del h_cnt
        gc.collect()
    
    # =========================================================================
    # STEP 5: Save embeddings
    # =========================================================================
    print(f"\n[4/4] Saving embeddings to {output_dir}...")
    for ntype, emb in h.items():
        np.save(output_path / f'{ntype}_embeddings.npy', emb.cpu().numpy())
        print(f"  {ntype}: {emb.shape}")
    
    print("\nDone! Use predict_cached.py for inference.")


def main():
    parser = argparse.ArgumentParser(description='Low-memory embedding computation')
    
    parser.add_argument('--processed-dir', type=str, default='./data/processed')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pt')
    parser.add_argument('--output-dir', type=str, default='./embeddings')
    parser.add_argument('--chunk-size', type=int, default=50000,
                        help='Edges per chunk (lower = less memory)')
    
    args = parser.parse_args()
    
    compute_embeddings_lowmem(
        checkpoint_path=args.checkpoint,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        device=torch.device('cpu')  # Force CPU for low memory
    )


if __name__ == '__main__':
    main()
