#!/usr/bin/env python
"""
Memory-efficient embedding computation using chunked/streaming processing.

This script computes node embeddings WITHOUT loading the full graph into memory.
It processes edges in chunks and accumulates messages incrementally.
It saves only the prediction cache tensors used by `predict_cached.py`.

Usage:
    python scripts/cache_embeddings_chunked.py \
        --checkpoint ./checkpoints/best.pt \
        --output-dir ./embeddings \
        --chunk-size 100000
"""

import argparse
import gc
import torch
import torch.nn.functional as F
import numpy as np
import polars as pl
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm

from src.cli_config import parse_args_with_config


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
    
    node_embeddings = {}
    for key, value in state.items():
        if key.startswith('node_emb.') and key.endswith('.weight'):
            node_type = key.split('.')[1]
            node_embeddings[node_type] = value

    if not node_embeddings:
        raise RuntimeError(
            'No node_emb weights found in checkpoint. This chunked cache script currently '
            'supports embedding-table checkpoints only. For feature-based inductive models, '
            'use scripts/cache_embeddings.py.'
        )
    
    W_cd = state['W_cd']
    
    num_layers = 0
    while f'convs.{num_layers}.lin_src.chemical__associated_with__disease.weight' in state:
        num_layers += 1
    
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


def _build_chem_gene_action_maps(processed_dir: str) -> Dict[str, Dict[Any, int]]:
    """Recreate categorical IDs exactly like graph._cat_ids_from_col (sorted unique)."""
    cg_path = Path(processed_dir) / 'chem_gene_edges.parquet'
    if not cg_path.exists():
        return {'action_type': {}, 'action_subject': {}}

    type_values = (
        pl.scan_parquet(cg_path)
        .select(pl.col('ACTION_TYPE').drop_nulls().unique().sort())
        .collect()['ACTION_TYPE']
        .to_list()
    )
    subject_values = (
        pl.scan_parquet(cg_path)
        .select(pl.col('ACTION_SUBJECT').drop_nulls().unique().sort())
        .collect()['ACTION_SUBJECT']
        .to_list()
    )
    return {
        'action_type': {v: i for i, v in enumerate(type_values)},
        'action_subject': {v: i for i, v in enumerate(subject_values)},
    }


def _transform_edge_attrs(
    attrs: Optional[np.ndarray],
    attr_cols: Optional[List[str]],
    action_maps: Dict[str, Dict[Any, int]],
) -> Optional[torch.Tensor]:
    if attrs is None:
        return None

    cols = attr_cols or []
    if cols == ['ACTION_TYPE', 'ACTION_SUBJECT']:
        type_map = action_maps.get('action_type', {})
        subj_map = action_maps.get('action_subject', {})
        type_ids = np.asarray([type_map.get(v, 0) for v in attrs[:, 0]], dtype=np.int64)
        subj_ids = np.asarray([subj_map.get(v, 0) for v in attrs[:, 1]], dtype=np.int64)
        return torch.from_numpy(np.stack([type_ids, subj_ids], axis=1))
    if cols == ['PHENO_ACTION_TYPE']:
        return torch.from_numpy(attrs.astype(np.int64))
    if cols == ['INFERENCE_GENE_COUNT']:
        return torch.from_numpy(np.log1p(attrs.astype(np.float32)))
    return torch.from_numpy(attrs.astype(np.float32))


def _compute_edge_gate(
    *,
    state_dict: Dict,
    layer_idx: int,
    edge_key: str,
    edge_attr: Optional[torch.Tensor],
    get_param,
) -> Optional[torch.Tensor]:
    if edge_attr is None:
        return None
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.view(-1, 1)

    def _first_param(*keys: str) -> Optional[torch.Tensor]:
        for key in keys:
            if key in state_dict:
                return get_param(key)
        return None

    # Categorical gate (chem-gene, pheno action)
    cat_w_key = f'convs.{layer_idx}.lin_edge_cat.{edge_key}.0.weight'
    if cat_w_key in state_dict:
        W = get_param(cat_w_key)
        b = get_param(f'convs.{layer_idx}.lin_edge_cat.{edge_key}.0.bias')
        in_dim = int(W.size(1))
        gate_in = None

        action_type_emb = _first_param(
            f'convs.{layer_idx}.action_type_emb.weight',
            'action_type_emb.weight',
        )
        action_subject_emb = _first_param(
            f'convs.{layer_idx}.action_subject_emb.weight',
            'action_subject_emb.weight',
        )
        pheno_action_emb = _first_param(
            f'convs.{layer_idx}.pheno_action_emb.weight',
            'pheno_action_emb.weight',
        )

        # Chem-gene: ACTION_TYPE + ACTION_SUBJECT
        if (
            edge_attr.size(1) >= 2
            and action_type_emb is not None
            and action_subject_emb is not None
        ):
            type_emb = F.embedding(edge_attr[:, 0].long(), action_type_emb)
            subj_emb = F.embedding(edge_attr[:, 1].long(), action_subject_emb)
            pair_in = torch.cat([type_emb, subj_emb], dim=-1)
            if pair_in.size(1) == in_dim:
                gate_in = pair_in

        # Pheno action: one categorical column
        if gate_in is None and edge_attr.size(1) >= 1 and pheno_action_emb is not None:
            pheno_in = F.embedding(edge_attr[:, 0].long(), pheno_action_emb)
            if pheno_in.size(1) == in_dim:
                gate_in = pheno_in

        if gate_in is None:
            gate_in = edge_attr.float()
            if gate_in.size(1) != in_dim:
                raise RuntimeError(
                    f'Categorical gate input mismatch for {edge_key}: '
                    f'got {gate_in.size(1)}, expected {in_dim}'
                )
        return torch.sigmoid(gate_in @ W.T + b)

    # Continuous gate
    cont_w0_key = f'convs.{layer_idx}.lin_edge_cont.{edge_key}.0.weight'
    if cont_w0_key in state_dict:
        W0 = get_param(cont_w0_key)
        b0 = get_param(f'convs.{layer_idx}.lin_edge_cont.{edge_key}.0.bias')
        W2 = get_param(f'convs.{layer_idx}.lin_edge_cont.{edge_key}.2.weight')
        b2 = get_param(f'convs.{layer_idx}.lin_edge_cont.{edge_key}.2.bias')
        cont = edge_attr.float()
        if cont.size(1) != int(W0.size(1)):
            raise RuntimeError(
                f'Continuous gate input mismatch for {edge_key}: '
                f'got {cont.size(1)}, expected {int(W0.size(1))}'
            )
        return torch.sigmoid(F.gelu(cont @ W0.T + b0) @ W2.T + b2)

    return None


def _process_single_edge_type(
    *,
    filepath: Path,
    src_col: str,
    dst_col: str,
    src_type: str,
    dst_type: str,
    rel: str,
    attr_cols: Optional[List[str]],
    reverse: bool,
    h: Dict[str, torch.Tensor],
    state_dict: Dict,
    layer_idx: int,
    chunk_size: int,
    device: torch.device,
    action_maps: Dict[str, Dict[Any, int]],
    get_param,
) -> Optional[torch.Tensor]:
    edge_key = f'{src_type}__{rel}__{dst_type}'
    lin_src_key = f'convs.{layer_idx}.lin_src.{edge_key}.weight'
    if lin_src_key not in state_dict:
        return None

    num_dst = int(h[dst_type].size(0))
    total_edges = count_edges(filepath)
    if total_edges == 0:
        return None

    W_src = get_param(lin_src_key)
    b_src = get_param(f'convs.{layer_idx}.lin_src.{edge_key}.bias')
    W_dst = get_param(f'convs.{layer_idx}.lin_dst.{edge_key}.weight')
    b_dst = get_param(f'convs.{layer_idx}.lin_dst.{edge_key}.bias')
    attn = get_param(f'convs.{layer_idx}.attn_{edge_key}')
    heads = int(attn.size(1))
    head_dim = int(attn.size(2))
    hidden_dim = heads * head_dim

    max_logits = torch.full((num_dst, heads), -torch.inf, device=device)

    # Pass 1: per-destination max logit for stable softmax.
    for offset in range(0, total_edges, chunk_size):
        src_np, dst_np, attrs_np = load_edge_chunk(
            filepath, src_col, dst_col, attr_cols, offset, chunk_size
        )
        if src_np is None:
            break

        if reverse:
            src_ids = torch.from_numpy(dst_np).long().to(device)
            dst_ids = torch.from_numpy(src_np).long().to(device)
        else:
            src_ids = torch.from_numpy(src_np).long().to(device)
            dst_ids = torch.from_numpy(dst_np).long().to(device)

        edge_attr = _transform_edge_attrs(attrs_np, attr_cols, action_maps)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

        src_x = h[src_type][src_ids]
        dst_x = h[dst_type][dst_ids]
        msg_src = src_x @ W_src.T + b_src
        msg_dst = dst_x @ W_dst.T + b_dst
        msg = msg_src + msg_dst

        gate = _compute_edge_gate(
            state_dict=state_dict,
            layer_idx=layer_idx,
            edge_key=edge_key,
            edge_attr=edge_attr,
            get_param=get_param,
        )
        if gate is not None:
            msg = msg * gate

        msg_heads = msg.view(-1, heads, head_dim)
        src_heads = msg_src.view(-1, heads, head_dim)
        dst_heads = msg_dst.view(-1, heads, head_dim)
        logits = (src_heads * dst_heads).sum(dim=-1) / (head_dim ** 0.5)
        logits = logits + (msg_heads * attn).sum(dim=-1)

        idx = dst_ids.unsqueeze(-1).expand(-1, heads)
        max_logits.scatter_reduce_(0, idx, logits, reduce='amax', include_self=True)

    # Pass 2: aggregate normalized attention-weighted messages.
    denom = torch.zeros((num_dst, heads), device=device)
    numer = torch.zeros((num_dst, heads, head_dim), device=device)
    for offset in range(0, total_edges, chunk_size):
        src_np, dst_np, attrs_np = load_edge_chunk(
            filepath, src_col, dst_col, attr_cols, offset, chunk_size
        )
        if src_np is None:
            break

        if reverse:
            src_ids = torch.from_numpy(dst_np).long().to(device)
            dst_ids = torch.from_numpy(src_np).long().to(device)
        else:
            src_ids = torch.from_numpy(src_np).long().to(device)
            dst_ids = torch.from_numpy(dst_np).long().to(device)

        edge_attr = _transform_edge_attrs(attrs_np, attr_cols, action_maps)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

        src_x = h[src_type][src_ids]
        dst_x = h[dst_type][dst_ids]
        msg_src = src_x @ W_src.T + b_src
        msg_dst = dst_x @ W_dst.T + b_dst
        msg = msg_src + msg_dst

        gate = _compute_edge_gate(
            state_dict=state_dict,
            layer_idx=layer_idx,
            edge_key=edge_key,
            edge_attr=edge_attr,
            get_param=get_param,
        )
        if gate is not None:
            msg = msg * gate

        msg_heads = msg.view(-1, heads, head_dim)
        src_heads = msg_src.view(-1, heads, head_dim)
        dst_heads = msg_dst.view(-1, heads, head_dim)
        logits = (src_heads * dst_heads).sum(dim=-1) / (head_dim ** 0.5)
        logits = logits + (msg_heads * attn).sum(dim=-1)

        weights_unnorm = torch.exp(logits - max_logits[dst_ids])
        idx = dst_ids.unsqueeze(-1).expand(-1, heads)
        denom.scatter_add_(0, idx, weights_unnorm)

        weighted = msg_heads * weights_unnorm.unsqueeze(-1)
        idx3 = dst_ids.view(-1, 1, 1).expand(-1, heads, head_dim)
        numer.scatter_add_(0, idx3, weighted)

    aggr_heads = numer / denom.clamp_min(1e-12).unsqueeze(-1)
    aggr = aggr_heads.reshape(num_dst, hidden_dim)
    return aggr


def chunked_message_passing(
    h: Dict[str, torch.Tensor],
    state_dict: Dict,
    layer_idx: int,
    processed_dir: str,
    chunk_size: int,
    device: torch.device,
    action_maps: Dict[str, Dict[Any, int]],
) -> Dict[str, torch.Tensor]:
    """
    Perform one exact HGT layer with chunked edge streaming.

    This matches EdgeAttrHeteroConv semantics (edge gates + attention softmax)
    while keeping memory bounded by chunk size.
    """
    processed_path = Path(processed_dir)

    edge_configs = [
        # (file, src_col, dst_col, src_type, dst_type, rel, attr_cols)
        ('chem_disease_edges.parquet', 'CHEM_ID', 'DS_ID', 'chemical', 'disease', 'associated_with', None),
        ('chem_gene_edges.parquet', 'CHEM_ID', 'GENE_ID', 'chemical', 'gene', 'affects', ['ACTION_TYPE', 'ACTION_SUBJECT']),
        ('disease_gene_edges.parquet', 'DS_ID', 'GENE_ID', 'disease', 'gene', 'targets', ['DIRECT_EVIDENCE_TYPE', 'LOG_PUBMED_COUNT']),
        ('ppi_edges.parquet', 'GENE_ID_1', 'GENE_ID_2', 'gene', 'gene', 'interacts_with', None),
    ]

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

    param_cache: Dict[str, torch.Tensor] = {}

    def get_param(key: str) -> torch.Tensor:
        if key not in param_cache:
            param_cache[key] = state_dict[key].to(device)
        return param_cache[key]

    out_dict: Dict[str, List[torch.Tensor]] = {ntype: [] for ntype in h.keys()}

    for config in tqdm(edge_configs, desc=f'Layer {layer_idx} edges'):
        filename, src_col, dst_col, src_type, dst_type, rel, attr_cols = config
        filepath = processed_path / filename
        if not filepath.exists():
            continue
        if src_type not in h or dst_type not in h:
            continue

        # Forward relation
        aggr_fw = _process_single_edge_type(
            filepath=filepath,
            src_col=src_col,
            dst_col=dst_col,
            src_type=src_type,
            dst_type=dst_type,
            rel=rel,
            attr_cols=attr_cols,
            reverse=False,
            h=h,
            state_dict=state_dict,
            layer_idx=layer_idx,
            chunk_size=chunk_size,
            device=device,
            action_maps=action_maps,
            get_param=get_param,
        )
        if aggr_fw is not None:
            out_dict[dst_type].append(aggr_fw)

        # Reverse relation if present in checkpoint.
        rev_rel = f'rev_{rel}'
        rev_key = f'convs.{layer_idx}.lin_src.{dst_type}__{rev_rel}__{src_type}.weight'
        if rev_key in state_dict:
            aggr_rev = _process_single_edge_type(
                filepath=filepath,
                src_col=src_col,
                dst_col=dst_col,
                src_type=dst_type,
                dst_type=src_type,
                rel=rev_rel,
                attr_cols=attr_cols,
                reverse=True,
                h=h,
                state_dict=state_dict,
                layer_idx=layer_idx,
                chunk_size=chunk_size,
                device=device,
                action_maps=action_maps,
                get_param=get_param,
            )
            if aggr_rev is not None:
                out_dict[src_type].append(aggr_rev)

    # Combine per-edge-type outputs and apply lin_out.
    h_new: Dict[str, torch.Tensor] = {}
    for ntype in h.keys():
        msgs = out_dict[ntype]
        if not msgs:
            h_new[ntype] = h[ntype]
            continue
        combined = torch.stack(msgs, dim=0).mean(dim=0)
        out_key = f'convs.{layer_idx}.lin_out.{ntype}.weight'
        if out_key in state_dict:
            W_out = get_param(out_key)
            b_out = get_param(f'convs.{layer_idx}.lin_out.{ntype}.bias')
            combined = combined @ W_out.T + b_out
        h_new[ntype] = combined

    # Residual + LayerNorm + GELU (dropout is disabled in eval mode).
    result: Dict[str, torch.Tensor] = {}
    for ntype in h.keys():
        x = h_new[ntype] + h[ntype]
        norm_w_key = f'norms.{layer_idx}.{ntype}.weight'
        if norm_w_key in state_dict:
            x = F.layer_norm(
                x,
                (x.size(-1),),
                weight=get_param(norm_w_key),
                bias=get_param(f'norms.{layer_idx}.{ntype}.bias'),
                eps=1e-5,
            )
        result[ntype] = F.gelu(x)

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


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
    action_maps = _build_chem_gene_action_maps(processed_dir)
    
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
            h=h,
            state_dict=state_dict,
            layer_idx=layer_idx,
            processed_dir=processed_dir,
            chunk_size=chunk_size,
            device=device,
            action_maps=action_maps,
        )
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save only embeddings required for prediction cache
    print(f"\nSaving prediction cache tensors to {output_dir}...")
    required_node_types = ('chemical', 'disease')
    missing_node_types = [ntype for ntype in required_node_types if ntype not in h]
    if missing_node_types:
        raise RuntimeError(
            f'Missing required node embeddings for cache prediction: {missing_node_types}'
        )

    for ntype in required_node_types:
        emb_np = h[ntype].cpu().numpy()
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
    
    args, _ = parse_args_with_config(parser)
    
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
