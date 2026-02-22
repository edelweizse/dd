"""
Data splitting and loading module for link prediction.

This module provides functionality for:
- Splitting chemical-disease edges into train/val/test sets
- Creating data loaders for training
- Negative sampling utilities
"""

import torch
import numpy as np
import hashlib
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
    known_pos_train: 'PackedPairFilter'
    known_pos_val: 'PackedPairFilter'
    known_pos_test: 'PackedPairFilter'
    # Backward-compatible alias to test-time filtered positives.
    known_pos: 'PackedPairFilter'
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    chem_train_degree: torch.Tensor | None = None
    dis_train_degree: torch.Tensor | None = None
    split_metadata: Dict[str, Any] | None = None


SPLIT_ARTIFACT_VERSION = 1


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
        max_key_i64 = torch.iinfo(torch.int64).max
        max_chem = int(chem.max().item()) if chem.numel() > 0 else 0
        max_dis = int(dis.max().item()) if dis.numel() > 0 else 0
        can_pack = (
            self.num_dis >= 0
            and max_chem >= 0
            and max_dis >= 0
            and (self.num_dis == 0 or max_chem <= (max_key_i64 - max_dis) // max(self.num_dis, 1))
        )

        self._packed_mode = bool(can_pack)
        if self._packed_mode:
            keys = (chem * self.num_dis + dis).tolist()
            self._set = set(keys)
            self._keys = torch.tensor(sorted(self._set), dtype=torch.int64)
            self._pair_set: set[tuple[int, int]] | None = None
        else:
            self._pair_set = set(zip(chem.tolist(), dis.tolist()))
            self._set = set(self._pair_set)
            self._keys = torch.empty(0, dtype=torch.int64)
    
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
        if self._packed_mode:
            keys = chem * self.num_dis + dis
            if self._keys.numel() == 0:
                return torch.zeros_like(keys, dtype=torch.bool)
            return torch.isin(keys, self._keys)

        if chem.numel() == 0:
            return torch.zeros_like(chem, dtype=torch.bool)
        assert self._pair_set is not None
        mask = [((int(c), int(d)) in self._pair_set) for c, d in zip(chem.tolist(), dis.tolist())]
        return torch.tensor(mask, dtype=torch.bool)


def _get_global_node_ids(batch_data: HeteroData, node_type: str) -> torch.Tensor:
    """Return global node IDs for a sampled local node store."""
    store = batch_data[node_type]

    if hasattr(store, 'node_id') and store.node_id is not None:
        return store.node_id.view(-1).long()

    if hasattr(store, 'n_id') and store.n_id is not None:
        return store.n_id.view(-1).long()

    # Backward compatibility: x may contain global IDs in legacy graphs.
    x = getattr(store, 'x', None)
    if isinstance(x, torch.Tensor) and not x.is_floating_point():
        if x.dim() == 1:
            return x.view(-1).long()
        if x.dim() == 2 and x.size(1) == 1:
            return x.view(-1).long()
    raise ValueError(
        f'Cannot infer global IDs for node type "{node_type}". '
        'Expected node_id/n_id, or integral x with shape [N] or [N,1].'
    )


def _compute_split_counts(E: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """Compute train/val/test sizes with stable rounding and non-empty splits."""
    if E < 3:
        raise ValueError(f'Need at least 3 edges for train/val/test split, got {E}.')
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f'val_ratio must be in (0, 1), got {val_ratio}.')
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f'test_ratio must be in (0, 1), got {test_ratio}.')
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f'val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio:.4f}.'
        )

    raw = torch.tensor(
        [E * (1.0 - val_ratio - test_ratio), E * val_ratio, E * test_ratio],
        dtype=torch.float64
    )
    counts = torch.floor(raw).long()
    remainder = int(E - counts.sum().item())
    if remainder > 0:
        frac = raw - counts.to(raw.dtype)
        for idx in torch.argsort(frac, descending=True)[:remainder]:
            counts[int(idx.item())] += 1

    # Ensure all splits are non-empty by borrowing from the largest split.
    for i in range(3):
        if counts[i] > 0:
            continue
        donor = int(torch.argmax(counts).item())
        if counts[donor] <= 1:
            raise ValueError(
                f'Not enough edges to create non-empty train/val/test splits for E={E}.'
            )
        counts[donor] -= 1
        counts[i] += 1

    n_train, n_val, n_test = (int(counts[0].item()), int(counts[1].item()), int(counts[2].item()))
    if n_train + n_val + n_test != E:
        raise RuntimeError('Internal split-size error: counts do not sum to E.')
    return n_train, n_val, n_test


def _degree_strata_labels(
    cd_idx: torch.Tensor,
    num_bins: int = 8,
    min_class_size: int = 3
) -> torch.Tensor:
    """
    Build per-edge stratification labels from joint (chemical degree, disease degree) bins.
    Rare classes are merged into a single fallback class.
    """
    if num_bins < 2:
        raise ValueError(f'num_bins must be >= 2, got {num_bins}.')

    chem = cd_idx[0]
    dis = cd_idx[1]

    chem_deg = torch.bincount(chem, minlength=int(chem.max().item()) + 1).float()
    dis_deg = torch.bincount(dis, minlength=int(dis.max().item()) + 1).float()

    chem_deg_e = chem_deg[chem]
    dis_deg_e = dis_deg[dis]

    chem_bin = torch.clamp(torch.log2(chem_deg_e + 1.0).floor().long(), 0, num_bins - 1)
    dis_bin = torch.clamp(torch.log2(dis_deg_e + 1.0).floor().long(), 0, num_bins - 1)
    labels = chem_bin * num_bins + dis_bin

    counts = torch.bincount(labels)
    rare = counts[labels] < min_class_size
    if bool(rare.any()):
        labels = labels.clone()
        labels[rare] = int(labels.max().item()) + 1
    return labels


def _rebalance_split_for_train_node_coverage(
    cd_idx: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Best-effort train node coverage rebalance with bounded, near-linear work.
    Uses swap operations (source<->train) so split sizes stay unchanged.
    """
    if train_idx.numel() == 0 or val_idx.numel() == 0 or test_idx.numel() == 0:
        return train_idx, val_idx, test_idx

    train_ids = train_idx.cpu().tolist()
    val_ids = val_idx.cpu().tolist()
    test_ids = test_idx.cpu().tolist()

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    chem_all = cd_idx[0].cpu()
    dis_all = cd_idx[1].cpu()
    num_chem = int(chem_all.max().item()) + 1
    num_dis = int(dis_all.max().item()) + 1

    chem_train_counts = torch.bincount(
        chem_all[torch.tensor(train_ids, dtype=torch.long)],
        minlength=num_chem,
    ).long()
    dis_train_counts = torch.bincount(
        dis_all[torch.tensor(train_ids, dtype=torch.long)],
        minlength=num_dis,
    ).long()
    chem_present = torch.bincount(chem_all, minlength=num_chem) > 0
    dis_present = torch.bincount(dis_all, minlength=num_dis) > 0
    if not bool((chem_present & (chem_train_counts == 0)).any()) and not bool(
        (dis_present & (dis_train_counts == 0)).any()
    ):
        return train_idx, val_idx, test_idx

    val_by_chem: Dict[int, List[int]] = {}
    val_by_dis: Dict[int, List[int]] = {}
    test_by_chem: Dict[int, List[int]] = {}
    test_by_dis: Dict[int, List[int]] = {}

    for eid in val_ids:
        chem = int(chem_all[eid].item())
        dis = int(dis_all[eid].item())
        val_by_chem.setdefault(chem, []).append(eid)
        val_by_dis.setdefault(dis, []).append(eid)
    for eid in test_ids:
        chem = int(chem_all[eid].item())
        dis = int(dis_all[eid].item())
        test_by_chem.setdefault(chem, []).append(eid)
        test_by_dis.setdefault(dis, []).append(eid)

    g = torch.Generator()
    g.manual_seed(int(seed) + 1729)

    val_chem_ptr: Dict[int, int] = {}
    val_dis_ptr: Dict[int, int] = {}
    test_chem_ptr: Dict[int, int] = {}
    test_dis_ptr: Dict[int, int] = {}

    def _peek_available(
        node_id: int,
        by_node: Dict[int, List[int]],
        ptrs: Dict[int, int],
        source_set: set[int],
    ) -> Tuple[int | None, int]:
        candidates = by_node.get(node_id)
        if not candidates:
            return None, 0
        ptr = ptrs.get(node_id, 0)
        while ptr < len(candidates) and candidates[ptr] not in source_set:
            ptr += 1
        ptrs[node_id] = ptr
        if ptr >= len(candidates):
            return None, ptr
        return int(candidates[ptr]), ptr

    def _choose_cover_edge(node_id: int, is_chem: bool) -> Tuple[str, int] | None:
        if is_chem:
            val_eid, val_ptr = _peek_available(node_id, val_by_chem, val_chem_ptr, val_set)
            test_eid, test_ptr = _peek_available(node_id, test_by_chem, test_chem_ptr, test_set)
        else:
            val_eid, val_ptr = _peek_available(node_id, val_by_dis, val_dis_ptr, val_set)
            test_eid, test_ptr = _peek_available(node_id, test_by_dis, test_dis_ptr, test_set)

        if val_eid is None and test_eid is None:
            return None
        if val_eid is None:
            if is_chem:
                test_chem_ptr[node_id] = test_ptr + 1
            else:
                test_dis_ptr[node_id] = test_ptr + 1
            return 'test', int(test_eid)
        if test_eid is None:
            if is_chem:
                val_chem_ptr[node_id] = val_ptr + 1
            else:
                val_dis_ptr[node_id] = val_ptr + 1
            return 'val', int(val_eid)

        # Keep deterministic randomness for tie cases.
        choose_val = bool(torch.randint(0, 2, (1,), generator=g).item() == 0)
        if choose_val:
            if is_chem:
                val_chem_ptr[node_id] = val_ptr + 1
            else:
                val_dis_ptr[node_id] = val_ptr + 1
            return 'val', int(val_eid)
        if is_chem:
            test_chem_ptr[node_id] = test_ptr + 1
        else:
            test_dis_ptr[node_id] = test_ptr + 1
        return 'test', int(test_eid)

    train_order = train_ids.copy()
    if len(train_order) > 1:
        perm = torch.randperm(len(train_order), generator=g).tolist()
        train_order = [train_order[i] for i in perm]
    train_cursor = 0

    def _next_safe_train_edge() -> int | None:
        nonlocal train_cursor
        while train_cursor < len(train_order):
            eid = int(train_order[train_cursor])
            train_cursor += 1
            if eid not in train_set:
                continue
            chem = int(chem_all[eid].item())
            dis = int(dis_all[eid].item())
            if chem_train_counts[chem] > 1 and dis_train_counts[dis] > 1:
                return eid
        return None

    missing_chem = torch.where(chem_present & (chem_train_counts == 0))[0]
    missing_dis = torch.where(dis_present & (dis_train_counts == 0))[0]
    if missing_chem.numel() > 1:
        missing_chem = missing_chem[torch.randperm(missing_chem.numel(), generator=g)]
    if missing_dis.numel() > 1:
        missing_dis = missing_dis[torch.randperm(missing_dis.numel(), generator=g)]

    safe_exhausted = False

    def _apply_swap_for_node(node_id: int, is_chem: bool) -> None:
        nonlocal safe_exhausted
        if safe_exhausted:
            return
        if is_chem and chem_train_counts[node_id] > 0:
            return
        if (not is_chem) and dis_train_counts[node_id] > 0:
            return

        choice = _choose_cover_edge(node_id, is_chem=is_chem)
        if choice is None:
            return
        src, in_eid = choice
        if src == 'val' and in_eid not in val_set:
            return
        if src == 'test' and in_eid not in test_set:
            return

        out_eid = _next_safe_train_edge()
        if out_eid is None:
            safe_exhausted = True
            return

        if src == 'val':
            val_set.remove(in_eid)
            val_set.add(out_eid)
        else:
            test_set.remove(in_eid)
            test_set.add(out_eid)

        train_set.remove(out_eid)
        train_set.add(in_eid)

        in_chem = int(chem_all[in_eid].item())
        in_dis = int(dis_all[in_eid].item())
        out_chem = int(chem_all[out_eid].item())
        out_dis = int(dis_all[out_eid].item())
        chem_train_counts[in_chem] += 1
        dis_train_counts[in_dis] += 1
        chem_train_counts[out_chem] -= 1
        dis_train_counts[out_dis] -= 1

        # Promoted edges can be swapped out later if still safe.
        train_order.append(in_eid)

    for node in missing_chem.tolist():
        _apply_swap_for_node(int(node), is_chem=True)
        if safe_exhausted:
            break
    for node in missing_dis.tolist():
        _apply_swap_for_node(int(node), is_chem=False)
        if safe_exhausted:
            break

    train_out = torch.tensor(sorted(train_set), dtype=torch.long)
    val_out = torch.tensor(sorted(val_set), dtype=torch.long)
    test_out = torch.tensor(sorted(test_set), dtype=torch.long)
    return train_out, val_out, test_out


def split_cd(
    cd_idx: torch.Tensor,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
    stratify_bins: int = 8,
    enforce_train_node_coverage: bool = True,
    return_strategy: bool = False,
) -> LinkSplit | Tuple[LinkSplit, str]:
    """
    Split chemical-disease edges into train/val/test sets.
    
    Args:
        cd_idx: [2, E] tensor of chemical-disease edge indices.
        val_ratio: Fraction of edges for validation.
        test_ratio: Fraction of edges for testing.
        seed: Random seed for reproducibility.
        stratify: If True, perform degree-stratified split.
        stratify_bins: Number of log-degree bins for stratification labels.
        enforce_train_node_coverage: If True, perform best-effort rebalancing
            so train split covers more chemical/disease nodes.
        return_strategy: If True, also return the realized strategy
            ('stratified' or 'random').
        
    Returns:
        LinkSplit containing train/val/test positive edges.
    """
    assert cd_idx.dtype == torch.long
    assert cd_idx.dim() == 2 and cd_idx.size(0) == 2
    E = cd_idx.size(1)
    n_train, n_val, n_test = _compute_split_counts(E, val_ratio, test_ratio)

    g = torch.Generator()
    g.manual_seed(int(seed))

    train_idx = None
    val_idx = None
    test_idx = None

    used_strategy = 'random'
    if stratify:
        labels = _degree_strata_labels(cd_idx, num_bins=stratify_bins).cpu().numpy()
        edge_ids = np.arange(E, dtype=np.int64)

        # Sklearn is already a project dependency via training metrics.
        from sklearn.model_selection import StratifiedShuffleSplit

        try:
            split_test = StratifiedShuffleSplit(
                n_splits=1, test_size=n_test, random_state=int(seed)
            )
            train_val_pos, test_pos = next(split_test.split(edge_ids, labels))
            train_val_ids = edge_ids[train_val_pos]
            test_ids = edge_ids[test_pos]

            labels_train_val = labels[train_val_ids]
            split_val = StratifiedShuffleSplit(
                n_splits=1, test_size=n_val, random_state=int(seed) + 1
            )
            train_pos, val_pos = next(split_val.split(train_val_ids, labels_train_val))

            train_idx = torch.from_numpy(train_val_ids[train_pos]).long()
            val_idx = torch.from_numpy(train_val_ids[val_pos]).long()
            test_idx = torch.from_numpy(test_ids).long()
            used_strategy = 'stratified'
        except ValueError:
            # Fallback for degenerate strata distributions.
            stratify = False

    if not stratify:
        perm = torch.randperm(E, generator=g)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

    if enforce_train_node_coverage:
        train_idx, val_idx, test_idx = _rebalance_split_for_train_node_coverage(
            cd_idx=cd_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=int(seed),
        )
    
    split = LinkSplit(
        train_pos=cd_idx[:, train_idx].contiguous(),
        val_pos=cd_idx[:, val_idx].contiguous(),
        test_pos=cd_idx[:, test_idx].contiguous(),
    )
    if return_strategy:
        return split, used_strategy
    return split


def _normalize_split_edge_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Validate a split edge tensor and return it as contiguous CPU int64."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f'Split artifact "{name}" must be a tensor, got {type(tensor).__name__}.')
    if tensor.dim() != 2 or tensor.size(0) != 2:
        raise ValueError(
            f'Split artifact "{name}" must have shape [2, E], got {tuple(tensor.shape)}.'
        )
    return tensor.contiguous().cpu().long()


def _build_split_metadata(
    data_full: HeteroData,
    split: LinkSplit,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    split_strategy: str,
    stratify_bins: int,
    enforce_train_node_coverage: bool,
    cd_rel: Tuple[str, str, str] = ('chemical', 'associated_with', 'disease'),
) -> Dict[str, Any]:
    """Build metadata that describes split generation inputs and graph sizes."""
    cd_edge_index = data_full[cd_rel].edge_index.detach().cpu().long()
    cd_sorted = np.asarray(cd_edge_index.t().numpy(), dtype=np.int64)
    if cd_sorted.shape[0] > 1:
        cd_sorted = cd_sorted[np.lexsort((cd_sorted[:, 1], cd_sorted[:, 0]))]
    cd_edge_set_sha256 = hashlib.sha256(cd_sorted.tobytes()).hexdigest()
    return {
        'version': SPLIT_ARTIFACT_VERSION,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'seed': int(seed),
        'val_ratio': float(val_ratio),
        'test_ratio': float(test_ratio),
        'split_strategy': str(split_strategy),
        'stratify_bins': int(stratify_bins),
        'enforce_train_node_coverage': bool(enforce_train_node_coverage),
        'num_chemical_nodes': int(data_full['chemical'].num_nodes),
        'num_disease_nodes': int(data_full['disease'].num_nodes),
        'num_cd_edges': int(data_full[cd_rel].edge_index.size(1)),
        'cd_edge_set_sha256': cd_edge_set_sha256,
        'cd_relation': list(cd_rel),
        'num_train_edges': int(split.train_pos.size(1)),
        'num_val_edges': int(split.val_pos.size(1)),
        'num_test_edges': int(split.test_pos.size(1)),
    }


def save_split_artifact(
    artifact_path: str | Path,
    split: LinkSplit,
    data_full: HeteroData,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    split_strategy: str = 'unknown',
    stratify_bins: int = 8,
    enforce_train_node_coverage: bool = True,
    cd_rel: Tuple[str, str, str] = ('chemical', 'associated_with', 'disease'),
) -> Path:
    """
    Persist split tensors and metadata so future runs can reuse the exact split.

    Returns:
        Resolved path to the saved artifact.
    """
    out_path = Path(artifact_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_norm = LinkSplit(
        train_pos=_normalize_split_edge_tensor('train_pos', split.train_pos),
        val_pos=_normalize_split_edge_tensor('val_pos', split.val_pos),
        test_pos=_normalize_split_edge_tensor('test_pos', split.test_pos),
    )
    metadata = _build_split_metadata(
        data_full=data_full,
        split=split_norm,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_strategy=split_strategy,
        stratify_bins=stratify_bins,
        enforce_train_node_coverage=enforce_train_node_coverage,
        cd_rel=cd_rel,
    )
    payload = {
        'artifact_type': 'cd_link_split',
        'version': SPLIT_ARTIFACT_VERSION,
        'metadata': metadata,
        'split': {
            'train_pos': split_norm.train_pos,
            'val_pos': split_norm.val_pos,
            'test_pos': split_norm.test_pos,
        },
    }
    torch.save(payload, out_path)
    return out_path


def load_split_artifact(
    artifact_path: str | Path,
) -> Tuple[LinkSplit, Dict[str, Any]]:
    """
    Load a persisted split artifact from disk.

    Returns:
        Tuple of (LinkSplit, metadata dict).
    """
    path = Path(artifact_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'Split artifact not found: {path}')

    payload = torch.load(path, map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError(f'Invalid split artifact format in {path}: expected dict payload.')

    split_data = payload.get('split')
    if not isinstance(split_data, dict):
        raise ValueError(f'Invalid split artifact format in {path}: missing "split" mapping.')

    split = LinkSplit(
        train_pos=_normalize_split_edge_tensor('train_pos', split_data.get('train_pos')),
        val_pos=_normalize_split_edge_tensor('val_pos', split_data.get('val_pos')),
        test_pos=_normalize_split_edge_tensor('test_pos', split_data.get('test_pos')),
    )
    metadata = payload.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError(f'Invalid split artifact format in {path}: "metadata" must be a mapping.')

    return split, metadata


def validate_split_artifact_compatibility(
    split: LinkSplit,
    metadata: Dict[str, Any],
    data_full: HeteroData,
    cd_rel: Tuple[str, str, str] = ('chemical', 'associated_with', 'disease'),
) -> None:
    """
    Validate a loaded split artifact against the current graph dimensions.

    Raises:
        ValueError if graph sizes are incompatible or split tensors are invalid.
    """
    errors: List[str] = []

    num_chem = int(data_full['chemical'].num_nodes)
    num_dis = int(data_full['disease'].num_nodes)
    num_cd_edges = int(data_full[cd_rel].edge_index.size(1))

    artifact_num_chem = metadata.get('num_chemical_nodes')
    artifact_num_dis = metadata.get('num_disease_nodes')
    artifact_num_edges = metadata.get('num_cd_edges')
    artifact_cd_edge_hash = metadata.get('cd_edge_set_sha256')
    artifact_cd_relation = metadata.get('cd_relation')

    if not isinstance(artifact_num_chem, int):
        errors.append('metadata.num_chemical_nodes is missing or not an integer.')
    elif artifact_num_chem != num_chem:
        errors.append(
            f'chemical node count mismatch (artifact={artifact_num_chem}, current={num_chem}).'
        )

    if not isinstance(artifact_num_dis, int):
        errors.append('metadata.num_disease_nodes is missing or not an integer.')
    elif artifact_num_dis != num_dis:
        errors.append(
            f'disease node count mismatch (artifact={artifact_num_dis}, current={num_dis}).'
        )

    if not isinstance(artifact_num_edges, int):
        errors.append('metadata.num_cd_edges is missing or not an integer.')
    elif artifact_num_edges != num_cd_edges:
        errors.append(
            f'CD edge count mismatch (artifact={artifact_num_edges}, current={num_cd_edges}).'
        )

    if artifact_cd_relation is None:
        errors.append('metadata.cd_relation is missing.')
    elif not isinstance(artifact_cd_relation, (list, tuple)) or len(artifact_cd_relation) != 3:
        errors.append('metadata.cd_relation must be a 3-item list/tuple.')
    elif tuple(artifact_cd_relation) != tuple(cd_rel):
        errors.append(
            'CD relation mismatch '
            f'(artifact={tuple(artifact_cd_relation)}, current={tuple(cd_rel)}).'
        )

    split_total = int(split.train_pos.size(1) + split.val_pos.size(1) + split.test_pos.size(1))
    if split_total != num_cd_edges:
        errors.append(
            f'split edge total mismatch (split_total={split_total}, current={num_cd_edges}).'
        )

    current_cd = data_full[cd_rel].edge_index.detach().cpu().long()
    split_all = torch.cat([split.train_pos, split.val_pos, split.test_pos], dim=1).contiguous().cpu().long()
    current_pairs = np.asarray(current_cd.t().numpy(), dtype=np.int64)
    split_pairs = np.asarray(split_all.t().numpy(), dtype=np.int64)
    if current_pairs.shape[0] > 1:
        current_pairs = current_pairs[np.lexsort((current_pairs[:, 1], current_pairs[:, 0]))]
    if split_pairs.shape[0] > 1:
        split_pairs = split_pairs[np.lexsort((split_pairs[:, 1], split_pairs[:, 0]))]
    if current_pairs.shape != split_pairs.shape or not np.array_equal(current_pairs, split_pairs):
        errors.append(
            'split edge multiset does not match the current CD edge multiset; '
            'artifact likely belongs to a different processed graph.'
        )

    current_cd_edge_hash = hashlib.sha256(current_pairs.tobytes()).hexdigest()
    if artifact_cd_edge_hash is not None:
        if not isinstance(artifact_cd_edge_hash, str):
            errors.append('metadata.cd_edge_set_sha256 must be a string when present.')
        elif artifact_cd_edge_hash != current_cd_edge_hash:
            errors.append(
                'CD edge-set hash mismatch '
                f'(artifact={artifact_cd_edge_hash}, current={current_cd_edge_hash}).'
            )

    for split_name, edge_idx in (
        ('train_pos', split.train_pos),
        ('val_pos', split.val_pos),
        ('test_pos', split.test_pos),
    ):
        if edge_idx.numel() == 0:
            continue

        min_chem = int(edge_idx[0].min().item())
        max_chem = int(edge_idx[0].max().item())
        min_dis = int(edge_idx[1].min().item())
        max_dis = int(edge_idx[1].max().item())

        if min_chem < 0 or max_chem >= num_chem:
            errors.append(
                f'{split_name} has chemical IDs outside [0, {num_chem - 1}] '
                f'(min={min_chem}, max={max_chem}).'
            )
        if min_dis < 0 or max_dis >= num_dis:
            errors.append(
                f'{split_name} has disease IDs outside [0, {num_dis - 1}] '
                f'(min={min_dis}, max={max_dis}).'
            )

    if errors:
        joined = '\n'.join(f'- {msg}' for msg in errors)
        raise ValueError(
            'Split artifact is incompatible with the current graph sizes:\n'
            f'{joined}'
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
    degree_alpha: float = 0.75,
    global_chem_degree: torch.Tensor | None = None,
    global_dis_degree: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate negative samples for chemical-disease link prediction.
    
    For each positive (chem, disease) pair, creates num_neg_per_pos negatives
    by corrupting either the chemical OR the disease (not both initially).
    
    On collision (if randomly generated pair is a known positive),
    re-samples while maintaining the same corruption strategy. If collisions
    remain after retries, a deterministic repair pass is used that may switch
    corruption side for unresolved samples.
    
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
        global_chem_degree: Optional full-train chemical degree vector used
            for degree-biased sampling.
        global_dis_degree: Optional full-train disease degree vector used
            for degree-biased sampling.
        generator: Optional torch.Generator for deterministic sampling.
        
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
    if hard_negative_ratio > 0.0:
        if global_chem_degree is not None and global_dis_degree is not None:
            chem_deg_local = global_chem_degree[chem_gid.cpu()].float()
            dis_deg_local = global_dis_degree[dis_gid.cpu()].float()
            chem_weights = (chem_deg_local + 1.0).pow(degree_alpha).to(device)
            dis_weights = (dis_deg_local + 1.0).pow(degree_alpha).to(device)
        elif cd_etype in batch_data.edge_types:
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
            return torch.randint(0, num_nodes, (num_samples,), device=device, generator=generator)

        num_hard = int(round(num_samples * hard_negative_ratio))
        num_hard = max(0, min(num_hard, num_samples))
        num_uniform = num_samples - num_hard

        parts = []
        if num_hard > 0:
            hard_ids = torch.multinomial(
                weights,
                num_hard,
                replacement=True,
                generator=generator
            ).to(device)
            parts.append(hard_ids)
        if num_uniform > 0:
            uniform_ids = torch.randint(
                0,
                num_nodes,
                (num_uniform,),
                device=device,
                generator=generator
            )
            parts.append(uniform_ids)

        sampled = torch.cat(parts, dim=0)
        if sampled.numel() > 1:
            sampled = sampled[
                torch.randperm(sampled.numel(), device=device, generator=generator)
            ]
        return sampled
    
    # Start from positive pairs
    chem_l = pos_edge_index_local[0].repeat_interleave(num_neg_per_pos).clone()
    dis_l = pos_edge_index_local[1].repeat_interleave(num_neg_per_pos).clone()
    
    # Decide corruption strategy: True = corrupt disease, False = corrupt chemical
    corrupt_disease = torch.rand(N, device=device, generator=generator) < 0.5
    
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

    # Final repair pass for unresolved collisions. This handles edge cases where
    # the initially chosen corruption side has no feasible negatives in the
    # sampled local node set.
    chem_g = chem_gid[chem_l].cpu()
    dis_g = dis_gid[dis_l].cpu()
    coll = known_pos.contains_mask_cpu(chem_g, dis_g).to(device)
    if bool(coll.any()):
        orig_chem_l = pos_edge_index_local[0].repeat_interleave(num_neg_per_pos).clone()
        orig_dis_l = pos_edge_index_local[1].repeat_interleave(num_neg_per_pos).clone()
        unresolved = torch.nonzero(coll, as_tuple=False).view(-1)

        for idx_t in unresolved:
            idx = int(idx_t.item())
            base_chem = int(orig_chem_l[idx].item())
            base_dis = int(orig_dis_l[idx].item())

            # Prefer keeping the original corruption side, then try the opposite.
            prefer_dis = bool(corrupt_disease[idx].item())
            try_orders = (True, False) if prefer_dis else (False, True)
            placed = False

            for try_dis in try_orders:
                if try_dis:
                    cand_chem_l = torch.full((ndis,), base_chem, dtype=torch.long, device=device)
                    cand_dis_l = torch.arange(ndis, dtype=torch.long, device=device)
                else:
                    cand_chem_l = torch.arange(nchem, dtype=torch.long, device=device)
                    cand_dis_l = torch.full((nchem,), base_dis, dtype=torch.long, device=device)

                cand_chem_g = chem_gid[cand_chem_l].cpu()
                cand_dis_g = dis_gid[cand_dis_l].cpu()
                valid = ~known_pos.contains_mask_cpu(cand_chem_g, cand_dis_g)
                if not bool(valid.any()):
                    continue

                valid_idx = torch.nonzero(valid, as_tuple=False).view(-1)
                pick_offset = int(
                    torch.randint(
                        low=0,
                        high=int(valid_idx.numel()),
                        size=(1,),
                        generator=generator,
                    ).item()
                )
                pick = int(valid_idx[pick_offset].item())
                chem_l[idx] = cand_chem_l[pick]
                dis_l[idx] = cand_dis_l[pick]
                corrupt_disease[idx] = bool(try_dis)
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    'Negative sampling failed: no valid local negative exists for '
                    f'pos sample index {idx} (batch has saturated neighborhoods).'
                )

        chem_g = chem_gid[chem_l].cpu()
        dis_g = dis_gid[dis_l].cpu()
        coll = known_pos.contains_mask_cpu(chem_g, dis_g).to(device)
    # Safety net: never emit known positives as negatives.
    if bool(coll.any()):
        raise RuntimeError(
            'Negative sampling failed to resolve collisions after '
            f'{max_tries} retries ({int(coll.sum().item())} collisions remain).'
        )
    
    return torch.stack([chem_l, dis_l], dim=0).long()


def make_link_loaders(
    data: HeteroData,
    split: LinkSplit,
    batch_size: int = 1024,
    num_neighbours: List[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create LinkNeighborLoader instances for train/val/test.
    
    Args:
        data: HeteroData (training graph with only train CD edges).
        split: LinkSplit containing edge splits.
        batch_size: Batch size for loaders.
        num_neighbours: List of neighbor counts per hop (default [10, 5]).
        seed: Random seed used by loader generators.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if num_neighbours is None:
        num_neighbours = [10, 5]
    
    cd_etype = ('chemical', 'associated_with', 'disease')
    g_train = torch.Generator().manual_seed(int(seed))
    g_val = torch.Generator().manual_seed(int(seed) + 1)
    g_test = torch.Generator().manual_seed(int(seed) + 2)
    
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbours,
        edge_label_index=(cd_etype, split.train_pos),
        edge_label=torch.ones(split.train_pos.size(1)),
        neg_sampling_ratio=0,
        batch_size=batch_size,
        shuffle=True,
        generator=g_train,
    )
    
    val_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbours,
        edge_label_index=(cd_etype, split.val_pos),
        edge_label=torch.ones(split.val_pos.size(1)),
        neg_sampling_ratio=0,
        batch_size=batch_size,
        shuffle=False,
        generator=g_val,
    )
    
    test_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbours,
        edge_label_index=(cd_etype, split.test_pos),
        edge_label=torch.ones(split.test_pos.size(1)),
        neg_sampling_ratio=0,
        batch_size=batch_size,
        shuffle=False,
        generator=g_test,
    )
    
    return train_loader, val_loader, test_loader


def prepare_splits_and_loaders(
    data_full: HeteroData,
    cd_rel: Tuple[str, str, str] = ('chemical', 'associated_with', 'disease'),
    rev_cd_rel: Tuple[str, str, str] = ('disease', 'rev_associated_with', 'chemical'),
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_strategy: str = 'stratified',
    stratify_bins: int = 8,
    enforce_train_node_coverage: bool = True,
    batch_size: int = 4096,
    num_neighbours: List[int] = None,
    split_artifact_load_path: str | Path | None = None,
    split_artifact_save_path: str | Path | None = None,
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
        split_strategy: Split strategy, one of {'stratified', 'random'}.
        stratify_bins: Number of log-degree bins for stratified splitting.
        enforce_train_node_coverage: Best-effort rebalancing to improve
            training split node-space coverage.
        batch_size: Batch size for loaders.
        num_neighbours: Neighbor counts per hop.
        split_artifact_load_path: Optional artifact path to reuse a saved split.
        split_artifact_save_path: Optional path to save the generated/reused split.
        
    Returns:
        SplitArtifacts containing all prepared artifacts.
    """
    if num_neighbours is None:
        num_neighbours = [10, 5]

    split_strategy_norm = split_strategy.strip().lower()
    if split_strategy_norm not in {'stratified', 'random'}:
        raise ValueError(
            f'split_strategy must be one of {{"stratified", "random"}}, got "{split_strategy}".'
        )
    
    split_metadata: Dict[str, Any] | None = None
    if split_artifact_load_path:
        split, split_metadata = load_split_artifact(split_artifact_load_path)
        validate_split_artifact_compatibility(
            split=split,
            metadata=split_metadata,
            data_full=data_full,
            cd_rel=cd_rel,
        )
    else:
        cd_all = data_full[cd_rel].edge_index
        use_stratify = split_strategy_norm == 'stratified'
        split, used_strategy = split_cd(
            cd_all,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratify=use_stratify,
            stratify_bins=stratify_bins,
            enforce_train_node_coverage=enforce_train_node_coverage,
            return_strategy=True,
        )
        split_metadata = _build_split_metadata(
            data_full=data_full,
            split=split,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_strategy=used_strategy,
            stratify_bins=stratify_bins,
            enforce_train_node_coverage=enforce_train_node_coverage,
            cd_rel=cd_rel,
        )

    if split_artifact_save_path:
        save_seed = int(split_metadata.get('seed', seed)) if split_metadata else int(seed)
        save_val_ratio = float(split_metadata.get('val_ratio', val_ratio)) if split_metadata else float(val_ratio)
        save_test_ratio = float(split_metadata.get('test_ratio', test_ratio)) if split_metadata else float(test_ratio)
        save_split_artifact(
            artifact_path=split_artifact_save_path,
            split=split,
            data_full=data_full,
            seed=save_seed,
            val_ratio=save_val_ratio,
            test_ratio=save_test_ratio,
            split_strategy=str(split_metadata.get('split_strategy', split_strategy)),
            stratify_bins=int(split_metadata.get('stratify_bins', stratify_bins)),
            enforce_train_node_coverage=bool(
                split_metadata.get('enforce_train_node_coverage', enforce_train_node_coverage)
            ),
            cd_rel=cd_rel,
        )
    
    # Make training graph
    data_train = make_split_graph(
        data_full,
        split.train_pos,
        cd_rel=cd_rel,
        cd_rev_rel=rev_cd_rel
    )
    
    # Prepare phase-aware positive filters to avoid split leakage.
    pos_train = split.train_pos
    pos_train_val = torch.cat([split.train_pos, split.val_pos], dim=1)
    pos_all = torch.cat([split.train_pos, split.val_pos, split.test_pos], dim=1)
    num_dis = data_full['disease'].num_nodes
    known_pos_train = PackedPairFilter(pos_train.cpu(), num_dis)
    known_pos_val = PackedPairFilter(pos_train_val.cpu(), num_dis)
    known_pos_test = PackedPairFilter(pos_all.cpu(), num_dis)
    
    # Create loaders
    loader_seed = int((split_metadata or {}).get('seed', seed))
    train_loader, val_loader, test_loader = make_link_loaders(
        data=data_train,
        split=split,
        batch_size=batch_size,
        num_neighbours=num_neighbours,
        seed=loader_seed,
    )

    num_chem = data_full['chemical'].num_nodes
    chem_train_degree = torch.bincount(split.train_pos[0].cpu(), minlength=num_chem).float()
    dis_train_degree = torch.bincount(split.train_pos[1].cpu(), minlength=num_dis).float()
    
    return SplitArtifacts(
        data_train=data_train,
        split=split,
        known_pos_train=known_pos_train,
        known_pos_val=known_pos_val,
        known_pos_test=known_pos_test,
        known_pos=known_pos_test,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        chem_train_degree=chem_train_degree,
        dis_train_degree=dis_train_degree,
        split_metadata=split_metadata,
    )
