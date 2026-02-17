"""
Data splitting and loading module for link prediction.

This module provides functionality for:
- Splitting chemical-disease edges into train/val/test sets
- Creating data loaders for training
- Negative sampling utilities
"""

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class LinkSplit:
    """Container for train/val/test edge splits."""
    train_pos: torch.Tensor  # [2, E_train]
    val_pos: torch.Tensor    # [2, E_val]
    test_pos: torch.Tensor   # [2, E_test]


@dataclass
class SplitArtifacts:
    """Container for all split-related artifacts."""
    data_train: HeteroData
    split: LinkSplit
    known_pos: 'PackedPairFilter'
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


class PackedPairFilter:
    """
    Membership test for (chem, dis) pairs using packed int64 keys:
        key = chem * num_dis + dis
    
    Stores all known positives (train+val+test) for efficient collision checking.
    """
    
    def __init__(self, pos_idx: torch.Tensor, num_dis: int):
        """
        Args:
            pos_idx: [2, E] tensor of (chem_id, dis_id) pairs.
            num_dis: Total number of diseases.
        """
        assert pos_idx.dtype == torch.long and pos_idx.size(0) == 2
        self.num_dis = int(num_dis)
        
        chem = pos_idx[0].to(torch.int64)
        dis = pos_idx[1].to(torch.int64)
        keys = (chem * self.num_dis + dis).tolist()
        self._set = set(keys)
    
    def contains_mask_cpu(
        self,
        chem: torch.Tensor,
        dis: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns bool mask on CPU: True where (chem, dis) is a known positive.
        
        Args:
            chem: Tensor of chemical IDs.
            dis: Tensor of disease IDs.
            
        Returns:
            Boolean tensor indicating which pairs are known positives.
        """
        chem = chem.to(torch.int64).cpu()
        dis = dis.to(torch.int64).cpu()
        keys = (chem * self.num_dis + dis).tolist()
        return torch.tensor([k in self._set for k in keys], dtype=torch.bool)


def _get_global_node_ids(batch_data: HeteroData, node_type: str) -> torch.Tensor:
    """Return global node IDs for a sampled local node store."""
    store = batch_data[node_type]

    if hasattr(store, 'node_id') and store.node_id is not None:
        return store.node_id.view(-1).long()

    if hasattr(store, 'n_id') and store.n_id is not None:
        return store.n_id.view(-1).long()

    # Backward compatibility: x may contain global IDs in legacy graphs.
    return store.x.view(-1).long()


def split_cd(
    cd_idx: torch.Tensor,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> LinkSplit:
    """
    Split chemical-disease edges into train/val/test sets.
    
    Args:
        cd_idx: [2, E] tensor of chemical-disease edge indices.
        val_ratio: Fraction of edges for validation.
        test_ratio: Fraction of edges for testing.
        seed: Random seed for reproducibility.
        
    Returns:
        LinkSplit containing train/val/test positive edges.
    """
    assert cd_idx.dtype == torch.long
    assert cd_idx.dim() == 2 and cd_idx.size(0) == 2
    E = cd_idx.size(1)
    
    n_test = int(E * test_ratio)
    n_val = int(E * val_ratio)
    n_train = E - n_val - n_test
    assert n_train > 0 and n_val > 0 and n_test > 0, "Not enough edges to split."
    
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(E, generator=g)
    
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    
    return LinkSplit(
        train_pos=cd_idx[:, train_idx].contiguous(),
        val_pos=cd_idx[:, val_idx].contiguous(),
        test_pos=cd_idx[:, test_idx].contiguous(),
    )


def make_split_graph(
    data: HeteroData,
    train_cd_idx: torch.Tensor,
    cd_rel: Tuple[str, str, str] = ('chemical', 'associated_with', 'disease'),
    cd_rev_rel: Tuple[str, str, str] = ('disease', 'rev_associated_with', 'chemical')
) -> HeteroData:
    """
    Create a training graph with only training CD edges.
    
    Args:
        data: Full HeteroData graph.
        train_cd_idx: Training chemical-disease edge indices.
        cd_rel: Chemical-disease relation tuple.
        cd_rev_rel: Reverse chemical-disease relation tuple.
        
    Returns:
        HeteroData with CD edges replaced by training edges only.
    """
    dat = data.clone()
    dat[cd_rel].edge_index = train_cd_idx
    if cd_rev_rel in dat.edge_types:
        dat[cd_rev_rel].edge_index = torch.flip(train_cd_idx, dims=[0])
    dat.validate()
    return dat


@torch.no_grad()
def negative_sample_cd_batch_local(
    batch_data: HeteroData,
    pos_edge_index_local: torch.Tensor,
    known_pos: PackedPairFilter,
    num_neg_per_pos: int = 5,
    max_tries: int = 20,
    hard_negative_ratio: float = 0.0,
    degree_alpha: float = 0.75
) -> torch.Tensor:
    """
    Generate negative samples for chemical-disease link prediction.
    
    For each positive (chem, disease) pair, creates num_neg_per_pos negatives
    by corrupting either the chemical OR the disease (not both initially).
    
    On collision (if randomly generated pair is a known positive),
    maintains the same corruption strategy for consistency.
    
    Args:
        batch_data: Batch HeteroData from LinkNeighborLoader.
        pos_edge_index_local: [2, B] positive edge indices (local).
        known_pos: PackedPairFilter for collision checking.
        num_neg_per_pos: Number of negatives per positive.
        max_tries: Maximum resampling attempts for collisions.
        hard_negative_ratio: Fraction [0, 1] of corruptions sampled from
            degree-biased distributions (harder negatives). Remaining fraction
            is sampled uniformly.
        degree_alpha: Exponent for degree-biased sampling weights
            w = (degree + 1) ** degree_alpha.
        
    Returns:
        [2, B * num_neg_per_pos] negative edge indices (local).
    """
    if not (0.0 <= hard_negative_ratio <= 1.0):
        raise ValueError(f'hard_negative_ratio must be in [0, 1], got {hard_negative_ratio}.')

    device = pos_edge_index_local.device
    B = pos_edge_index_local.size(1)
    N = B * num_neg_per_pos
    
    nchem = batch_data['chemical'].num_nodes
    ndis = batch_data['disease'].num_nodes
    
    # Map local to global IDs for collision checking
    chem_gid = _get_global_node_ids(batch_data, 'chemical')
    dis_gid = _get_global_node_ids(batch_data, 'disease')

    chem_weights = None
    dis_weights = None
    cd_etype = ('chemical', 'associated_with', 'disease')
    if hard_negative_ratio > 0.0 and cd_etype in batch_data.edge_types:
        cd_edge_index = batch_data[cd_etype].edge_index
        if cd_edge_index.numel() > 0:
            chem_deg = torch.bincount(cd_edge_index[0], minlength=nchem).float()
            dis_deg = torch.bincount(cd_edge_index[1], minlength=ndis).float()
            chem_weights = (chem_deg + 1.0).pow(degree_alpha)
            dis_weights = (dis_deg + 1.0).pow(degree_alpha)

    def _sample_ids(num_samples: int, num_nodes: int, weights: torch.Tensor | None) -> torch.Tensor:
        if num_samples <= 0:
            return torch.empty(0, dtype=torch.long, device=device)

        if weights is None or hard_negative_ratio <= 0.0:
            return torch.randint(0, num_nodes, (num_samples,), device=device)

        num_hard = int(round(num_samples * hard_negative_ratio))
        num_hard = max(0, min(num_hard, num_samples))
        num_uniform = num_samples - num_hard

        parts = []
        if num_hard > 0:
            hard_ids = torch.multinomial(weights, num_hard, replacement=True).to(device)
            parts.append(hard_ids)
        if num_uniform > 0:
            uniform_ids = torch.randint(0, num_nodes, (num_uniform,), device=device)
            parts.append(uniform_ids)

        sampled = torch.cat(parts, dim=0)
        if sampled.numel() > 1:
            sampled = sampled[torch.randperm(sampled.numel(), device=device)]
        return sampled
    
    # Start from positive pairs
    chem_l = pos_edge_index_local[0].repeat_interleave(num_neg_per_pos).clone()
    dis_l = pos_edge_index_local[1].repeat_interleave(num_neg_per_pos).clone()
    
    # Decide corruption strategy: True = corrupt disease, False = corrupt chemical
    corrupt_disease = torch.rand(N, device=device) < 0.5
    
    # Apply initial corruption
    nd = int(corrupt_disease.sum().item())
    if nd > 0:
        dis_l[corrupt_disease] = _sample_ids(nd, ndis, dis_weights)
    nc = N - nd
    if nc > 0:
        chem_l[~corrupt_disease] = _sample_ids(nc, nchem, chem_weights)
    
    # Handle collisions while maintaining corruption strategy
    for _ in range(max_tries):
        chem_g = chem_gid[chem_l].cpu()
        dis_g = dis_gid[dis_l].cpu()
        coll = known_pos.contains_mask_cpu(chem_g, dis_g).to(device)
        if not bool(coll.any()):
            break
        
        # Re-sample collisions using SAME corruption strategy
        coll_corrupt_dis = coll & corrupt_disease
        coll_corrupt_chem = coll & (~corrupt_disease)
        
        nd_coll = int(coll_corrupt_dis.sum().item())
        nc_coll = int(coll_corrupt_chem.sum().item())
        
        if nd_coll > 0:
            dis_l[coll_corrupt_dis] = _sample_ids(nd_coll, ndis, dis_weights)
        if nc_coll > 0:
            chem_l[coll_corrupt_chem] = _sample_ids(nc_coll, nchem, chem_weights)
    
    return torch.stack([chem_l, dis_l], dim=0).long()


def make_link_loaders(
    data: HeteroData,
    split: LinkSplit,
    batch_size: int = 1024,
    num_neighbours: List[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create LinkNeighborLoader instances for train/val/test.
    
    Args:
        data: HeteroData (training graph with only train CD edges).
        split: LinkSplit containing edge splits.
        batch_size: Batch size for loaders.
        num_neighbours: List of neighbor counts per hop (default [10, 5]).
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if num_neighbours is None:
        num_neighbours = [10, 5]
    
    cd_etype = ('chemical', 'associated_with', 'disease')
    
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbours,
        edge_label_index=(cd_etype, split.train_pos),
        edge_label=torch.ones(split.train_pos.size(1)),
        neg_sampling_ratio=0,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbours,
        edge_label_index=(cd_etype, split.val_pos),
        edge_label=torch.ones(split.val_pos.size(1)),
        neg_sampling_ratio=0,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbours,
        edge_label_index=(cd_etype, split.test_pos),
        edge_label=torch.ones(split.test_pos.size(1)),
        neg_sampling_ratio=0,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def prepare_splits_and_loaders(
    data_full: HeteroData,
    cd_rel: Tuple[str, str, str] = ('chemical', 'associated_with', 'disease'),
    rev_cd_rel: Tuple[str, str, str] = ('disease', 'rev_associated_with', 'chemical'),
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 4096,
    num_neighbours: List[int] = None
) -> SplitArtifacts:
    """
    Prepare all split artifacts for training.
    
    This function:
    1. Splits CD edges into train/val/test
    2. Creates training graph (without val/test CD edges)
    3. Builds PackedPairFilter for negative sampling
    4. Creates data loaders
    
    Args:
        data_full: Full HeteroData graph.
        cd_rel: Chemical-disease relation tuple.
        rev_cd_rel: Reverse chemical-disease relation tuple.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed.
        batch_size: Batch size for loaders.
        num_neighbours: Neighbor counts per hop.
        
    Returns:
        SplitArtifacts containing all prepared artifacts.
    """
    if num_neighbours is None:
        num_neighbours = [10, 5]
    
    # Split positives for CD
    cd_all = data_full[cd_rel].edge_index
    split = split_cd(
        cd_all,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Make training graph
    data_train = make_split_graph(
        data_full,
        split.train_pos,
        cd_rel=cd_rel,
        cd_rev_rel=rev_cd_rel
    )
    
    # Prepare known positives filter
    pos_all = torch.cat([split.test_pos, split.val_pos, split.train_pos], dim=1)
    num_dis = data_full['disease'].num_nodes
    known_pos = PackedPairFilter(pos_all.cpu(), num_dis)
    
    # Create loaders
    train_loader, val_loader, test_loader = make_link_loaders(
        data=data_train,
        split=split,
        batch_size=batch_size,
        num_neighbours=num_neighbours
    )
    
    return SplitArtifacts(
        data_train=data_train,
        split=split,
        known_pos=known_pos,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
