"""
Training utilities: losses, metrics, and evaluation functions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Tuple
import time


def bce_with_logits(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    pos_weight: float | None = None,
    focal_gamma: float = 0.0
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss for link prediction.
    
    Args:
        pos_logits: Logits for positive samples.
        neg_logits: Logits for negative samples.
        pos_weight: Optional positive-class reweighting factor.
        focal_gamma: If > 0, applies focal modulation factor
            (1 - p_t)^gamma on top of BCE.
        
    Returns:
        BCE loss value.
    """
    y = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    logits = torch.cat([pos_logits, neg_logits], dim=0)

    pos_weight_t = None
    if pos_weight is not None:
        pos_weight_t = torch.tensor([float(pos_weight)], device=logits.device, dtype=logits.dtype)

    if focal_gamma <= 0.0:
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight_t)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        y,
        pos_weight=pos_weight_t,
        reduction='none'
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(y > 0.5, probs, 1.0 - probs)
    focal = (1.0 - pt).pow(focal_gamma)
    return (focal * bce).mean()


@torch.no_grad()
def sampled_ranking_metrics(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    num_neg_per_pos: int,
    ks: Tuple[int, ...] = (5, 10, 50),
) -> Dict[str, float]:
    """
    Compute Hits@K and MRR under sampled negatives.
    
    Assumes:
        pos_logits shape [B]
        neg_logits shape [B * num_neg_per_pos]
        negatives are grouped per positive in order.
        
    Args:
        pos_logits: Logits for positive samples.
        neg_logits: Logits for negative samples.
        num_neg_per_pos: Number of negatives per positive.
        ks: Tuple of K values for Hits@K.
        
    Returns:
        Dictionary with 'mrr' and 'hits_K' for each K.
    """
    B = pos_logits.numel()
    assert neg_logits.numel() == B * num_neg_per_pos
    
    pos = pos_logits.view(B, 1)  # [B, 1]
    neg = neg_logits.view(B, num_neg_per_pos)  # [B, k]
    
    # rank of positive among (1 + k) items: 1 = best
    # Higher logits = better.
    # rank = 1 + count(neg > pos)
    rank = 1 + (neg > pos).sum(dim=1)  # [B]
    
    mrr = (1.0 / rank.float()).mean().item()
    
    out = {'mrr': mrr}
    for K in ks:
        out[f'hits_{K}'] = (rank <= K).float().mean().item()
    return out


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC safely, returning NaN if only one class present."""
    if np.unique(y_true).size < 2:
        return float('nan')
    return roc_auc_score(y_true, y_score)


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Average Precision safely, returning NaN if only one class present."""
    if np.unique(y_true).size < 2:
        return float('nan')
    return average_precision_score(y_true, y_score)


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    known_pos,
    device: torch.device,
    num_neg_per_pos: int = 20,
    ks: Tuple[int, ...] = (5, 10, 50),
    amp: bool = True,
    hard_negative_ratio: float = 0.0,
    degree_alpha: float = 0.75,
    sampling_seed: int | None = None,
    global_chem_degree: torch.Tensor | None = None,
    global_dis_degree: torch.Tensor | None = None,
    progress_every: int = 0,
    max_batches: int = 0,
    progress_prefix: str | None = None,
) -> Dict[str, float]:
    """
    Evaluate model on a data loader.
    
    Args:
        model: HGATPredictor model.
        loader: DataLoader (val or test).
        known_pos: PackedPairFilter for negative sampling.
        device: Torch device.
        num_neg_per_pos: Number of negatives per positive for evaluation.
        ks: Tuple of K values for Hits@K.
        amp: Whether to use automatic mixed precision.
        hard_negative_ratio: Fraction of degree-biased negatives.
        degree_alpha: Exponent for degree-biased sampling.
        sampling_seed: Optional deterministic seed for negative sampling.
        global_chem_degree: Optional train-split chemical degree vector.
        global_dis_degree: Optional train-split disease degree vector.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    from src.data.splits import negative_sample_cd_batch_local
    
    model.eval()
    
    if sampling_seed is not None:
        if device.type == 'cuda':
            neg_generator = torch.Generator(device=device)
        else:
            neg_generator = torch.Generator()
        neg_generator.manual_seed(int(sampling_seed))
    else:
        neg_generator = None
    
    all_scores = []
    all_labels = []
    
    mrr_sum = 0.0
    hits_sums = {k: 0.0 for k in ks}
    n_pos_total = 0
    
    total_batches = len(loader) if hasattr(loader, '__len__') else None
    t0 = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        if int(max_batches) > 0 and batch_idx > int(max_batches):
            break
        batch = batch.to(device)
        cd_edge_store = batch[('chemical', 'associated_with', 'disease')]
        pos_edge = cd_edge_store.edge_label_index  # [2, B] local
        
        neg_edge = negative_sample_cd_batch_local(
            batch_data=batch,
            pos_edge_index_local=pos_edge,
            known_pos=known_pos,
            num_neg_per_pos=num_neg_per_pos,
            hard_negative_ratio=hard_negative_ratio,
            degree_alpha=degree_alpha,
            global_chem_degree=global_chem_degree,
            global_dis_degree=global_dis_degree,
            generator=neg_generator,
        )
        
        if amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pos_logits, neg_logits = model(batch, pos_edge, neg_edge)
        else:
            pos_logits, neg_logits = model(batch, pos_edge, neg_edge)
        
        scores = torch.sigmoid(torch.cat([pos_logits, neg_logits], dim=0)).detach().cpu().numpy()
        labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits)
        ], dim=0).detach().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels)
        
        r = sampled_ranking_metrics(pos_logits, neg_logits, num_neg_per_pos, ks=ks)
        B = pos_logits.numel()
        n_pos_total += B
        mrr_sum += r['mrr'] * B
        for k in ks:
            hits_sums[k] += r[f'hits_{k}'] * B

        if int(progress_every) > 0 and (batch_idx % int(progress_every) == 0):
            done = batch_idx
            total = (
                min(int(total_batches), int(max_batches))
                if total_batches is not None and int(max_batches) > 0
                else total_batches
            )
            prefix = f"{progress_prefix} " if progress_prefix else ""
            total_str = str(total) if total is not None else "?"
            print(
                f"{prefix}batch {done}/{total_str} "
                f"(elapsed={time.time() - t0:.1f}s)",
                flush=True,
            )
    
    y_score = np.concatenate(all_scores, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    out = {
        'auroc': _safe_auc(y_true, y_score),
        'auprc': _safe_ap(y_true, y_score),
        'mrr': mrr_sum / max(n_pos_total, 1)
    }
    for k in ks:
        out[f'hits_{k}'] = hits_sums[k] / max(n_pos_total, 1)
    return out
