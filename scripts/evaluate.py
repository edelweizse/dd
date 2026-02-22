#!/usr/bin/env python
"""
Comprehensive model evaluation for HGT chemical-disease link prediction.

This script generates:
- Overall performance metrics (AUROC, AUPRC, MRR, Hits@K)
- ROC and Precision-Recall curves
- Score distribution analysis
- Per-disease and per-chemical performance breakdown
- Calibration plots
- Top predictions and error analysis
- Embedding visualizations (t-SNE)
- Publication-ready tables and figures

Usage:
    python scripts/evaluate.py --checkpoint ./checkpoints/best.pt
    python scripts/evaluate.py --checkpoint ./checkpoints/best.pt --output-dir evaluation_results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

from src.cli_config import parse_args_with_config
from src.data.graph import build_graph_from_processed, print_graph_summary
from src.data.splits import prepare_splits_and_loaders, negative_sample_cd_batch_local
from src.data.processing import load_processed_data
from src.evaluation.protocol import (
    format_protocol_violations,
    load_evaluation_protocol,
    validate_evaluation_protocol,
)
from src.models.architectures.hgt import HGTPredictor, infer_hgt_hparams_from_state


def _batch_global_ids(batch, node_type: str) -> np.ndarray:
    store = batch[node_type]
    if hasattr(store, 'node_id') and store.node_id is not None:
        return store.node_id.view(-1).cpu().numpy()
    if hasattr(store, 'n_id') and store.n_id is not None:
        return store.n_id.view(-1).cpu().numpy()
    return store.x.view(-1).cpu().numpy()


# ============================================================================
# STYLING
# ============================================================================

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

COLORS = {
    'primary': '#2ecc71',
    'secondary': '#3498db', 
    'accent': '#e74c3c',
    'neutral': '#95a5a6',
    'dark': '#2c3e50'
}

FIG_DPI = 150
TABLE_FLOAT_FORMAT = '{:.4f}'


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

@torch.no_grad()
def collect_all_predictions(
    model: HGTPredictor,
    loader,
    known_pos,
    device: torch.device,
    num_neg_per_pos: int = 50,
    amp: bool = True,
    sampling_seed: int | None = None,
    hard_negative_ratio: float = 0.0,
    degree_alpha: float = 0.75,
    global_chem_degree: torch.Tensor | None = None,
    global_dis_degree: torch.Tensor | None = None,
) -> Dict[str, np.ndarray]:
    """
    Collect all predictions from the model.
    
    Returns dict with:
        - pos_scores: scores for positive edges
        - neg_scores: scores for negative edges  
        - pos_chem_ids: chemical IDs for positive edges (global)
        - pos_dis_ids: disease IDs for positive edges (global)
        - neg_chem_ids: chemical IDs for negative edges (global)
        - neg_dis_ids: disease IDs for negative edges (global)
    """
    model.eval()

    if sampling_seed is not None:
        if device.type == 'cuda':
            neg_generator = torch.Generator(device=device)
        else:
            neg_generator = torch.Generator()
        neg_generator.manual_seed(int(sampling_seed))
    else:
        neg_generator = None
    
    all_pos_scores = []
    all_neg_scores = []
    all_pos_chem_ids = []
    all_pos_dis_ids = []
    all_neg_chem_ids = []
    all_neg_dis_ids = []
    
    for batch in loader:
        batch = batch.to(device)
        
        cd_edge_store = batch[('chemical', 'associated_with', 'disease')]
        pos_edge = cd_edge_store.edge_label_index
        
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
        
        pos_scores = torch.sigmoid(pos_logits).cpu().numpy()
        neg_scores = torch.sigmoid(neg_logits).cpu().numpy()
        
        # Map local IDs to global IDs
        chem_gid = _batch_global_ids(batch, 'chemical')
        dis_gid = _batch_global_ids(batch, 'disease')
        
        pos_chem = chem_gid[pos_edge[0].cpu().numpy()]
        pos_dis = dis_gid[pos_edge[1].cpu().numpy()]
        neg_chem = chem_gid[neg_edge[0].cpu().numpy()]
        neg_dis = dis_gid[neg_edge[1].cpu().numpy()]
        
        all_pos_scores.append(pos_scores)
        all_neg_scores.append(neg_scores)
        all_pos_chem_ids.append(pos_chem)
        all_pos_dis_ids.append(pos_dis)
        all_neg_chem_ids.append(neg_chem)
        all_neg_dis_ids.append(neg_dis)
    
    return {
        'pos_scores': np.concatenate(all_pos_scores),
        'neg_scores': np.concatenate(all_neg_scores),
        'pos_chem_ids': np.concatenate(all_pos_chem_ids),
        'pos_dis_ids': np.concatenate(all_pos_dis_ids),
        'neg_chem_ids': np.concatenate(all_neg_chem_ids),
        'neg_dis_ids': np.concatenate(all_neg_dis_ids),
    }


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, ks: List[int] = [5, 10, 20, 50]) -> Dict:
    """Compute comprehensive metrics."""
    
    metrics = {
        'auroc': roc_auc_score(y_true, y_score),
        'auprc': average_precision_score(y_true, y_score),
    }
    
    # Threshold-based metrics at various points
    for threshold in [0.3, 0.5, 0.7]:
        y_pred = (y_score >= threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'precision@{threshold}'] = precision
        metrics[f'recall@{threshold}'] = recall
        metrics[f'f1@{threshold}'] = f1
    
    return metrics


def compute_ranking_metrics(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray, 
    num_neg_per_pos: int,
    ks: List[int] = [5, 10, 20, 50]
) -> Dict:
    """Compute ranking metrics (MRR, Hits@K)."""
    
    B = len(pos_scores)
    pos = pos_scores.reshape(B, 1)
    neg = neg_scores.reshape(B, num_neg_per_pos)
    
    # Rank: 1 + count of negatives with higher score
    ranks = 1 + (neg > pos).sum(axis=1)
    
    metrics = {
        'mrr': (1.0 / ranks).mean(),
        'mean_rank': ranks.mean(),
        'median_rank': np.median(ranks),
    }
    
    for k in ks:
        metrics[f'hits@{k}'] = (ranks <= k).mean()
    
    return metrics


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path):
    """Plot ROC curve with AUC."""
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color=COLORS['primary'], lw=2, 
            label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color=COLORS['neutral'], lw=2, linestyle='--',
            label='Random classifier')
    
    ax.fill_between(fpr, tpr, alpha=0.3, color=COLORS['primary'])
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add optimal threshold point
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', s=100, 
               color=COLORS['accent'], zorder=5,
               label=f'Optimal threshold = {optimal_threshold:.3f}')
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    
    return auc, optimal_threshold


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path):
    """Plot Precision-Recall curve with AUPRC."""
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    baseline = y_true.sum() / len(y_true)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(recall, precision, color=COLORS['secondary'], lw=2,
            label=f'PR curve (AUPRC = {auprc:.4f})')
    ax.axhline(y=baseline, color=COLORS['neutral'], lw=2, linestyle='--',
               label=f'Random baseline = {baseline:.4f}')
    
    ax.fill_between(recall, precision, alpha=0.3, color=COLORS['secondary'])
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    
    return auprc


def plot_score_distribution(pos_scores: np.ndarray, neg_scores: np.ndarray, output_path: Path):
    """Plot score distributions for positive and negative samples."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(neg_scores, bins=50, alpha=0.7, label=f'Negative (n={len(neg_scores):,})',
             color=COLORS['accent'], density=True)
    ax1.hist(pos_scores, bins=50, alpha=0.7, label=f'Positive (n={len(pos_scores):,})',
             color=COLORS['primary'], density=True)
    ax1.set_xlabel('Prediction Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    data = [neg_scores, pos_scores]
    bp = ax2.boxplot(data, tick_labels=['Negative', 'Positive'], patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['accent'])
    bp['boxes'][1].set_facecolor(COLORS['primary'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax2.set_ylabel('Prediction Score', fontsize=12)
    ax2.set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = (
        f"Positive: μ={pos_scores.mean():.3f}, σ={pos_scores.std():.3f}\n"
        f"Negative: μ={neg_scores.mean():.3f}, σ={neg_scores.std():.3f}\n"
        f"Separation: {pos_scores.mean() - neg_scores.mean():.3f}"
    )
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_calibration_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path):
    """Plot calibration curve."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curve
    ax1 = axes[0]
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy='uniform')
    
    ax1.plot(prob_pred, prob_true, marker='o', color=COLORS['primary'], lw=2, 
             label='Model calibration')
    ax1.plot([0, 1], [0, 1], color=COLORS['neutral'], lw=2, linestyle='--',
             label='Perfect calibration')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Score histogram by bin
    ax2 = axes[1]
    bins = np.linspace(0, 1, 11)
    ax2.hist(y_score[y_true == 0], bins=bins, alpha=0.7, label='Negative',
             color=COLORS['accent'], density=True)
    ax2.hist(y_score[y_true == 1], bins=bins, alpha=0.7, label='Positive',
             color=COLORS['primary'], density=True)
    ax2.set_xlabel('Prediction Score', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Score Distribution by Class', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_metrics_at_k(ranking_metrics: Dict, output_path: Path):
    """Plot Hits@K and other ranking metrics."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hits@K
    ax1 = axes[0]
    ks = [k for k in [5, 10, 20, 50] if f'hits@{k}' in ranking_metrics]
    hits = [ranking_metrics[f'hits@{k}'] for k in ks]
    
    bars = ax1.bar([str(k) for k in ks], hits, color=COLORS['secondary'], alpha=0.8)
    ax1.set_xlabel('K', fontsize=12)
    ax1.set_ylabel('Hits@K', fontsize=12)
    ax1.set_title('Hits@K (Ranking Accuracy)', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, hits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=11)
    
    # MRR and rank statistics
    ax2 = axes[1]
    metrics_to_show = ['mrr', 'mean_rank', 'median_rank']
    labels = ['MRR', 'Mean Rank', 'Median Rank']
    values = [ranking_metrics.get(m, 0) for m in metrics_to_show]
    
    # Normalize for visualization (MRR is 0-1, ranks need different scale)
    ax2_twin = ax2.twinx()
    
    ax2.bar(['MRR'], [ranking_metrics['mrr']], color=COLORS['primary'], alpha=0.8, width=0.4)
    ax2.set_ylabel('MRR', fontsize=12, color=COLORS['primary'])
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='y', labelcolor=COLORS['primary'])
    
    ax2_twin.bar(['Mean Rank', 'Median Rank'], 
                 [ranking_metrics['mean_rank'], ranking_metrics['median_rank']], 
                 color=COLORS['accent'], alpha=0.8, width=0.4)
    ax2_twin.set_ylabel('Rank', fontsize=12, color=COLORS['accent'])
    ax2_twin.tick_params(axis='y', labelcolor=COLORS['accent'])
    
    ax2.set_title('Ranking Metrics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def plot_per_entity_performance(
    predictions: Dict,
    entity_type: str,  # 'disease' or 'chemical'
    entity_names: Dict[int, str],
    output_path: Path,
    top_n: int = 20
):
    """Plot per-entity (disease or chemical) performance."""
    
    # Group predictions by entity
    if entity_type == 'disease':
        pos_ids = predictions['pos_dis_ids']
        neg_ids = predictions['neg_dis_ids']
    else:
        pos_ids = predictions['pos_chem_ids']
        neg_ids = predictions['neg_chem_ids']
    
    pos_scores = predictions['pos_scores']
    neg_scores = predictions['neg_scores']
    
    # Compute per-entity metrics
    entity_metrics = {}
    
    for eid in np.unique(pos_ids):
        pos_mask = pos_ids == eid
        if pos_mask.sum() < 5:  # Skip entities with too few samples
            continue
            
        entity_pos_scores = pos_scores[pos_mask]
        
        # For negatives, we need to match the same entity
        neg_mask = neg_ids == eid
        if neg_mask.sum() == 0:
            continue
        entity_neg_scores = neg_scores[neg_mask]
        
        # Combine for AUC calculation
        y_true = np.concatenate([np.ones(len(entity_pos_scores)), np.zeros(len(entity_neg_scores))])
        y_score = np.concatenate([entity_pos_scores, entity_neg_scores])
        
        if len(np.unique(y_true)) < 2:
            continue
            
        try:
            auc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            entity_metrics[eid] = {
                'auroc': auc,
                'auprc': ap,
                'n_pos': len(entity_pos_scores),
                'mean_pos_score': entity_pos_scores.mean(),
                'name': entity_names.get(eid, f'ID:{eid}')
            }
        except ValueError:
            continue
    
    if not entity_metrics:
        print(f"No valid {entity_type} metrics to plot")
        return None
    
    # Sort by AUROC
    sorted_entities = sorted(entity_metrics.items(), key=lambda x: x[1]['auroc'], reverse=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top performers
    ax1 = axes[0, 0]
    top_entities = sorted_entities[:top_n]
    names = [e[1]['name'][:30] for e in top_entities]
    aurocs = [e[1]['auroc'] for e in top_entities]
    
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, aurocs, color=COLORS['primary'], alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('AUROC', fontsize=12)
    ax1.set_title(f'Top {top_n} {entity_type.title()}s by AUROC', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Bottom performers
    ax2 = axes[0, 1]
    bottom_entities = sorted_entities[-top_n:]
    names = [e[1]['name'][:30] for e in bottom_entities]
    aurocs = [e[1]['auroc'] for e in bottom_entities]
    
    y_pos = np.arange(len(names))
    ax2.barh(y_pos, aurocs, color=COLORS['accent'], alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('AUROC', fontsize=12)
    ax2.set_title(f'Bottom {top_n} {entity_type.title()}s by AUROC', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # AUROC distribution
    ax3 = axes[1, 0]
    all_aurocs = [e[1]['auroc'] for e in sorted_entities]
    ax3.hist(all_aurocs, bins=30, color=COLORS['secondary'], alpha=0.8, edgecolor='white')
    ax3.axvline(np.mean(all_aurocs), color=COLORS['accent'], linestyle='--', lw=2,
                label=f'Mean = {np.mean(all_aurocs):.3f}')
    ax3.axvline(np.median(all_aurocs), color=COLORS['primary'], linestyle='--', lw=2,
                label=f'Median = {np.median(all_aurocs):.3f}')
    ax3.set_xlabel('AUROC', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title(f'AUROC Distribution Across {entity_type.title()}s', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # AUROC vs number of positive samples
    ax4 = axes[1, 1]
    n_pos_list = [e[1]['n_pos'] for e in sorted_entities]
    ax4.scatter(n_pos_list, all_aurocs, alpha=0.5, color=COLORS['dark'], s=30)
    ax4.set_xlabel(f'Number of Positive Associations', fontsize=12)
    ax4.set_ylabel('AUROC', fontsize=12)
    ax4.set_title(f'AUROC vs Sample Size per {entity_type.title()}', fontsize=14, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    
    return entity_metrics


def plot_embedding_tsne(
    model: HGTPredictor,
    data,
    node_type: str,
    entity_names: Dict[int, str],
    output_path: Path,
    n_samples: int = 2000,
    perplexity: int = 30
):
    """Create t-SNE visualization of node embeddings."""
    
    model.eval()
    
    # Get embeddings (supports both embedding-table and feature-projection modes)
    with torch.no_grad():
        if hasattr(model, 'node_emb') and node_type in model.node_emb:
            embeddings = model.node_emb[node_type].weight.cpu().numpy()
        elif hasattr(model, 'initial_node_states'):
            x_dict = {k: v.to(next(model.parameters()).device) for k, v in data.x_dict.items()}
            init_states = model.initial_node_states(x_dict)
            if node_type not in init_states:
                raise ValueError(f'Node type {node_type} is not available for embedding visualization.')
            embeddings = init_states[node_type].cpu().numpy()
        else:
            raise ValueError(f'Model does not expose embeddings for node type: {node_type}')
    
    n_nodes = embeddings.shape[0]
    if n_nodes > n_samples:
        indices = np.random.choice(n_nodes, n_samples, replace=False)
        embeddings = embeddings[indices]
    else:
        indices = np.arange(n_nodes)
    
    # Run t-SNE
    print(f"Running t-SNE for {node_type} embeddings ({len(indices)} samples)...")
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(indices)-1), 
                random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        alpha=0.6, s=20, c=indices, cmap='viridis')
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(f'{node_type.title()} Embedding Visualization (t-SNE)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()


def generate_top_predictions_table(
    predictions: Dict,
    chem_names: Dict[int, str],
    dis_names: Dict[int, str],
    known_pos_set: set,
    n_top: int = 100
) -> pd.DataFrame:
    """Generate table of top predictions (potential novel associations)."""
    
    # Combine all predictions
    all_chem_ids = np.concatenate([predictions['pos_chem_ids'], predictions['neg_chem_ids']])
    all_dis_ids = np.concatenate([predictions['pos_dis_ids'], predictions['neg_dis_ids']])
    all_scores = np.concatenate([predictions['pos_scores'], predictions['neg_scores']])
    
    # Sort by score
    sorted_idx = np.argsort(all_scores)[::-1]
    
    rows = []
    seen = set()
    
    for idx in sorted_idx:
        if len(rows) >= n_top:
            break
            
        chem_id = int(all_chem_ids[idx])
        dis_id = int(all_dis_ids[idx])
        pair = (chem_id, dis_id)
        
        if pair in seen:
            continue
        seen.add(pair)
        
        is_known = pair in known_pos_set
        
        rows.append({
            'Rank': len(rows) + 1,
            'Chemical': chem_names.get(chem_id, f'ID:{chem_id}'),
            'Chemical_ID': chem_id,
            'Disease': dis_names.get(dis_id, f'ID:{dis_id}'),
            'Disease_ID': dis_id,
            'Score': all_scores[idx],
            'Known': 'Yes' if is_known else 'No'
        })
    
    return pd.DataFrame(rows)


def generate_error_analysis(
    predictions: Dict,
    chem_names: Dict[int, str],
    dis_names: Dict[int, str],
    threshold: float = 0.5
) -> Dict[str, pd.DataFrame]:
    """Generate error analysis tables (FP, FN)."""
    
    pos_scores = predictions['pos_scores']
    neg_scores = predictions['neg_scores']
    
    # False negatives (positive samples with low scores)
    fn_mask = pos_scores < threshold
    fn_indices = np.where(fn_mask)[0]
    fn_sorted = fn_indices[np.argsort(pos_scores[fn_indices])][:50]  # Top 50 worst FN
    
    fn_rows = []
    for idx in fn_sorted:
        fn_rows.append({
            'Chemical': chem_names.get(int(predictions['pos_chem_ids'][idx]), 'Unknown'),
            'Disease': dis_names.get(int(predictions['pos_dis_ids'][idx]), 'Unknown'),
            'Score': pos_scores[idx],
            'Error': 'False Negative'
        })
    
    # False positives (negative samples with high scores)
    fp_mask = neg_scores >= threshold
    fp_indices = np.where(fp_mask)[0]
    fp_sorted = fp_indices[np.argsort(neg_scores[fp_indices])[::-1]][:50]  # Top 50 worst FP
    
    fp_rows = []
    for idx in fp_sorted:
        fp_rows.append({
            'Chemical': chem_names.get(int(predictions['neg_chem_ids'][idx]), 'Unknown'),
            'Disease': dis_names.get(int(predictions['neg_dis_ids'][idx]), 'Unknown'),
            'Score': neg_scores[idx],
            'Error': 'False Positive'
        })
    
    return {
        'false_negatives': pd.DataFrame(fn_rows),
        'false_positives': pd.DataFrame(fp_rows)
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_metrics_table(metrics: Dict, ranking_metrics: Dict) -> str:
    """Generate a formatted metrics table."""
    
    lines = []
    lines.append("=" * 60)
    lines.append("OVERALL PERFORMANCE METRICS")
    lines.append("=" * 60)
    lines.append("")
    
    # Classification metrics
    lines.append("Classification Metrics:")
    lines.append("-" * 40)
    lines.append(f"  AUROC:                {metrics['auroc']:.4f}")
    lines.append(f"  AUPRC:                {metrics['auprc']:.4f}")
    lines.append("")
    
    # Threshold-based metrics
    lines.append("Threshold-Based Metrics:")
    lines.append("-" * 40)
    for threshold in [0.3, 0.5, 0.7]:
        lines.append(f"  Threshold = {threshold}:")
        lines.append(f"    Precision:          {metrics.get(f'precision@{threshold}', 0):.4f}")
        lines.append(f"    Recall:             {metrics.get(f'recall@{threshold}', 0):.4f}")
        lines.append(f"    F1:                 {metrics.get(f'f1@{threshold}', 0):.4f}")
    lines.append("")
    
    # Ranking metrics
    lines.append("Ranking Metrics:")
    lines.append("-" * 40)
    lines.append(f"  MRR:                  {ranking_metrics['mrr']:.4f}")
    lines.append(f"  Mean Rank:            {ranking_metrics['mean_rank']:.2f}")
    lines.append(f"  Median Rank:          {ranking_metrics['median_rank']:.2f}")
    lines.append("")
    
    lines.append("Hits@K:")
    lines.append("-" * 40)
    for k in [5, 10, 20, 50]:
        if f'hits@{k}' in ranking_metrics:
            lines.append(f"  Hits@{k}:              {ranking_metrics[f'hits@{k}']:.4f}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_html_report(
    metrics: Dict,
    ranking_metrics: Dict,
    output_dir: Path,
    plots: List[str]
):
    """Generate HTML report with embedded plots."""
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HGT Model Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric-value {{ font-weight: bold; color: #2ecc71; }}
        .plot-container {{ margin: 20px 0; text-align: center; }}
        .plot-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>HGT Chemical-Disease Link Prediction - Evaluation Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overall Performance</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>AUROC</td><td class="metric-value">{metrics['auroc']:.4f}</td></tr>
            <tr><td>AUPRC</td><td class="metric-value">{metrics['auprc']:.4f}</td></tr>
            <tr><td>MRR</td><td class="metric-value">{ranking_metrics['mrr']:.4f}</td></tr>
            <tr><td>Hits@10</td><td class="metric-value">{ranking_metrics.get('hits@10', 0):.4f}</td></tr>
            <tr><td>Hits@50</td><td class="metric-value">{ranking_metrics.get('hits@50', 0):.4f}</td></tr>
        </table>
        
        <h2>ROC Curve</h2>
        <div class="plot-container">
            <img src="roc_curve.png" alt="ROC Curve">
        </div>
        
        <h2>Precision-Recall Curve</h2>
        <div class="plot-container">
            <img src="pr_curve.png" alt="PR Curve">
        </div>
        
        <h2>Score Distribution</h2>
        <div class="plot-container">
            <img src="score_distribution.png" alt="Score Distribution">
        </div>
        
        <h2>Calibration</h2>
        <div class="plot-container">
            <img src="calibration.png" alt="Calibration">
        </div>
        
        <h2>Ranking Metrics</h2>
        <div class="plot-container">
            <img src="ranking_metrics.png" alt="Ranking Metrics">
        </div>
        
        <h2>Per-Disease Analysis</h2>
        <div class="plot-container">
            <img src="per_disease_analysis.png" alt="Per-Disease Analysis">
        </div>
        
        <h2>Per-Chemical Analysis</h2>
        <div class="plot-container">
            <img src="per_chemical_analysis.png" alt="Per-Chemical Analysis">
        </div>
    </div>
</body>
</html>
    """
    
    with open(output_dir / 'report.html', 'w') as f:
        f.write(html)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive HGT model evaluation')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--processed-dir', type=str, default='./data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--num-neg-eval', type=int, default=50,
                        help='Number of negatives per positive for evaluation')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size for evaluation')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Which split to evaluate')
    parser.add_argument('--no-tsne', action='store_true',
                        help='Skip t-SNE visualization (faster)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation split ratio when not loading artifact')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test split ratio when not loading artifact')
    parser.add_argument('--split-strategy', type=str, default='stratified',
                        choices=['stratified', 'random'],
                        help='CD edge split strategy when not loading artifact')
    parser.add_argument('--stratify-bins', type=int, default=8,
                        help='Number of log-degree bins for stratified split')
    parser.add_argument('--no-enforce-train-node-coverage', action='store_true',
                        help='Disable best-effort train node coverage rebalancing')
    parser.add_argument('--eval-hard-negative-ratio', type=float, default=0.0,
                        help='Fraction [0,1] of degree-biased negatives during eval')
    parser.add_argument('--degree-alpha', type=float, default=0.75,
                        help='Exponent for degree-biased negative sampling')
    parser.add_argument('--split-artifact-path', type=str, default=None,
                        help='Optional path to saved split artifact; reuses exact split if provided')
    parser.add_argument(
        '--protocol-config',
        type=str,
        default='./configs/examples/eval_protocol.yaml',
        help='Evaluation protocol config (YAML)'
    )
    parser.add_argument(
        '--allow-noncomparable',
        action='store_true',
        help='Allow protocol violations (marks outputs non-comparable instead of failing)'
    )
    parser.add_argument(
        '--protocol-strict',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Fail on protocol violations unless --allow-noncomparable is set'
    )
    parser.add_argument(
        '--protocol-report-path',
        type=str,
        default=None,
        help='Optional JSON path to save protocol validation report'
    )
    
    args, _ = parse_args_with_config(parser)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    data, vocabs = build_graph_from_processed(
        args.processed_dir, 
        add_reverse_edges=True,
        include_extended=True
    )
    print_graph_summary(data)
    
    # Load entity names
    data_dict = load_processed_data(args.processed_dir)
    
    chem_names = dict(zip(
        data_dict['chemicals']['CHEM_ID'].to_list(),
        data_dict['chemicals']['CHEM_NAME'].to_list()
    ))
    dis_names = dict(zip(
        data_dict['diseases']['DS_ID'].to_list(),
        data_dict['diseases']['DS_NAME'].to_list()
    ))
    
    # Prepare splits
    print("\nPreparing data splits...")
    if args.split_artifact_path:
        print(f"Using split artifact: {args.split_artifact_path}")
    else:
        print("No split artifact provided; generating a fresh split (val=0.1, test=0.1).")

    arts = prepare_splits_and_loaders(
        data_full=data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_strategy=args.split_strategy,
        stratify_bins=args.stratify_bins,
        enforce_train_node_coverage=not args.no_enforce_train_node_coverage,
        batch_size=args.batch_size,
        split_artifact_load_path=args.split_artifact_path
    )
    if args.split_artifact_path and arts.split_metadata:
        print(
            'Loaded split metadata: '
            f"seed={arts.split_metadata.get('seed')}, "
            f"val_ratio={arts.split_metadata.get('val_ratio')}, "
            f"test_ratio={arts.split_metadata.get('test_ratio')}"
        )

    protocol = load_evaluation_protocol(args.protocol_config)
    protocol_result = validate_evaluation_protocol(
        protocol,
        split_artifact_path=args.split_artifact_path,
        split_metadata=arts.split_metadata,
        num_neg_eval=int(args.num_neg_eval),
        eval_hard_negative_ratio=float(args.eval_hard_negative_ratio),
        runtime_seed=int(args.seed),
        runtime_val_ratio=float(args.val_ratio),
        runtime_test_ratio=float(args.test_ratio),
        runtime_split_strategy=str(args.split_strategy),
        runtime_stratify_bins=int(args.stratify_bins),
        allow_noncomparable=bool(args.allow_noncomparable),
    )
    strict_mode = bool(args.protocol_strict) and not bool(args.allow_noncomparable)
    if protocol_result.violations:
        msg = format_protocol_violations(protocol_result.violations, prefix='Evaluation protocol violation')
        if strict_mode:
            raise ValueError(msg)
        print(msg)
        print('Proceeding in non-comparable mode due to --allow-noncomparable.')
    else:
        print('Evaluation protocol check passed.')
    protocol_payload = {
        'version': int(protocol.version),
        'config_path': str(Path(args.protocol_config).expanduser()),
        'strict_mode': bool(strict_mode),
        'allow_noncomparable': bool(args.allow_noncomparable),
        **protocol_result.to_dict(),
    }
    if args.protocol_report_path:
        report_path = Path(args.protocol_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(protocol_payload, indent=2))
        print(f'Saved protocol report: {report_path}')
    
    # Load model
    print("\n" + "="*60)
    print("Loading Model")
    print("="*60)
    
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
    model = HGTPredictor(
        num_nodes_dict=num_nodes_dict,
        metadata=arts.data_train.metadata(),
        node_input_dims=model_cfg['node_input_dims'] or node_input_dims,
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        dropout=0.0,  # eval mode; value does not affect results
        num_action_types=model_cfg['num_action_types'] or vocabs['action_type'].height,
        num_action_subjects=model_cfg['num_action_subjects'] or vocabs['action_subject'].height,
        num_pheno_action_types=model_cfg['num_pheno_action_types'],
    )
    
    # Load checkpoint weights
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint best metric: {ckpt.get('best_metrics', 'unknown')}")
    
    # Select loader
    split_seed = int((arts.split_metadata or {}).get('seed', args.seed))
    if args.split == 'test':
        loader = arts.test_loader
        known_pos = arts.known_pos_test
        sampling_seed = split_seed + 303
    else:
        loader = arts.val_loader
        known_pos = arts.known_pos_val
        sampling_seed = split_seed + 202
    print(f"\nEvaluating on {args.split} set...")
    
    # Collect predictions
    print("\n" + "="*60)
    print("Collecting Predictions")
    print("="*60)
    
    predictions = collect_all_predictions(
        model=model,
        loader=loader,
        known_pos=known_pos,
        device=device,
        num_neg_per_pos=args.num_neg_eval,
        amp=True,
        sampling_seed=sampling_seed,
        hard_negative_ratio=args.eval_hard_negative_ratio,
        degree_alpha=args.degree_alpha,
        global_chem_degree=arts.chem_train_degree,
        global_dis_degree=arts.dis_train_degree,
    )
    
    print(f"Positive samples: {len(predictions['pos_scores']):,}")
    print(f"Negative samples: {len(predictions['neg_scores']):,}")
    
    # Compute metrics
    print("\n" + "="*60)
    print("Computing Metrics")
    print("="*60)
    
    y_true = np.concatenate([
        np.ones(len(predictions['pos_scores'])),
        np.zeros(len(predictions['neg_scores']))
    ])
    y_score = np.concatenate([predictions['pos_scores'], predictions['neg_scores']])
    
    metrics = compute_metrics(y_true, y_score)
    ranking_metrics = compute_ranking_metrics(
        predictions['pos_scores'],
        predictions['neg_scores'],
        args.num_neg_eval
    )
    
    # Print metrics
    report = generate_metrics_table(metrics, ranking_metrics)
    print(report)
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(report)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({**metrics, **ranking_metrics, 'evaluation_protocol': protocol_payload}, f, indent=2)
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    
    print("Plotting ROC curve...")
    plot_roc_curve(y_true, y_score, output_dir / 'roc_curve.png')
    
    print("Plotting PR curve...")
    plot_pr_curve(y_true, y_score, output_dir / 'pr_curve.png')
    
    print("Plotting score distribution...")
    plot_score_distribution(predictions['pos_scores'], predictions['neg_scores'],
                           output_dir / 'score_distribution.png')
    
    print("Plotting calibration curve...")
    plot_calibration_curve(y_true, y_score, output_dir / 'calibration.png')
    
    print("Plotting ranking metrics...")
    plot_metrics_at_k(ranking_metrics, output_dir / 'ranking_metrics.png')
    
    print("Plotting per-disease analysis...")
    disease_metrics = plot_per_entity_performance(
        predictions, 'disease', dis_names,
        output_dir / 'per_disease_analysis.png'
    )
    
    print("Plotting per-chemical analysis...")
    chemical_metrics = plot_per_entity_performance(
        predictions, 'chemical', chem_names,
        output_dir / 'per_chemical_analysis.png'
    )
    
    # t-SNE visualization
    if not args.no_tsne:
        print("Generating t-SNE visualizations...")
        for node_type in ['chemical', 'disease']:
            names = chem_names if node_type == 'chemical' else dis_names
            plot_embedding_tsne(
                model, data, node_type, names,
                output_dir / f'{node_type}_embeddings_tsne.png',
                n_samples=min(2000, num_nodes_dict[node_type])
            )
    
    # Generate tables
    print("\n" + "="*60)
    print("Generating Tables")
    print("="*60)
    
    # Top predictions
    print("Generating top predictions table...")
    cd_full = data[('chemical', 'associated_with', 'disease')].edge_index.cpu().numpy()
    known_pos_set = set(zip(cd_full[0].tolist(), cd_full[1].tolist()))
    top_preds = generate_top_predictions_table(
        predictions, chem_names, dis_names, known_pos_set
    )
    top_preds.to_csv(output_dir / 'top_predictions.csv', index=False)
    print(f"Top 10 predictions:")
    print(top_preds.head(10).to_string(index=False))
    
    # Error analysis
    print("\nGenerating error analysis...")
    errors = generate_error_analysis(predictions, chem_names, dis_names)
    errors['false_negatives'].to_csv(output_dir / 'false_negatives.csv', index=False)
    errors['false_positives'].to_csv(output_dir / 'false_positives.csv', index=False)
    
    print(f"\nTop 5 False Negatives (missed associations):")
    print(errors['false_negatives'].head(5).to_string(index=False))
    
    print(f"\nTop 5 False Positives (incorrect predictions):")
    print(errors['false_positives'].head(5).to_string(index=False))
    
    # Save per-entity metrics
    if disease_metrics:
        disease_df = pd.DataFrame([
            {'disease_id': k, **v} for k, v in disease_metrics.items()
        ]).sort_values('auroc', ascending=False)
        disease_df.to_csv(output_dir / 'per_disease_metrics.csv', index=False)
    
    if chemical_metrics:
        chemical_df = pd.DataFrame([
            {'chemical_id': k, **v} for k, v in chemical_metrics.items()
        ]).sort_values('auroc', ascending=False)
        chemical_df.to_csv(output_dir / 'per_chemical_metrics.csv', index=False)
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    save_html_report(metrics, ranking_metrics, output_dir, [])
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nFiles generated:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")
    
    print(f"\nOpen {output_dir / 'report.html'} in a browser for the full report.")


if __name__ == '__main__':
    main()
