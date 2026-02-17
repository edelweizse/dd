"""
High-level explainability API for chemical-disease predictions.

Ties together metapath enumeration (Tier 1) and attention extraction (Tier 2)
into a single `explain_pair()` call that returns a fully annotated
ExplanationResult.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from torch_geometric.data import HeteroData
from typing import Any, Dict, List, Optional, Tuple

from .paths import (
    AdjacencyIndex,
    PathInstance,
    build_adjacency,
    enumerate_paths,
    METAPATH_TEMPLATES,
)
from .scoring import ScoredPath, score_paths


@dataclass
class ExplanationResult:
    """Complete explanation for a chemical-disease prediction."""
    chemical_id: str
    disease_id: str
    chemical_name: str
    disease_name: str
    probability: float
    label: int
    logit: float
    known: bool
    paths: List[ScoredPath]
    attention_available: bool
    metapath_summary: Dict[str, int] = field(default_factory=dict)

    @property
    def top_paths(self) -> List[ScoredPath]:
        """Return top-10 paths by combined score."""
        return self.paths[:10]

    def summary_text(self, max_paths: int = 5) -> str:
        """Return a compact text summary of the explanation."""
        lines = [
            f"Prediction: {self.chemical_name} <-> {self.disease_name}",
            f"  Probability: {self.probability:.4f}  "
            f"Label: {'Associated' if self.label else 'Not associated'}  "
            f"Known: {'Yes' if self.known else 'No'}",
            f"  Attention scoring: {'Yes' if self.attention_available else 'No'}",
            f"  Total paths found: {len(self.paths)}",
        ]
        if self.metapath_summary:
            lines.append("  Path types: " + ", ".join(
                f"{k}: {v}" for k, v in sorted(self.metapath_summary.items())
            ))
        lines.append("")
        for i, sp in enumerate(self.paths[:max_paths], 1):
            lines.append(
                f"  {i}. [{sp.evidence_type}] score={sp.combined_score:.4f}  "
                f"attn={sp.attention_score:.4f}  emb={sp.embedding_score:.4f}"
            )
            lines.append(f"     {sp.description}")
        return "\n".join(lines)


def explain_pair(
    *,
    data: HeteroData,
    chem_idx: int,
    disease_idx: int,
    chemical_id: str,
    disease_id: str,
    chemical_name: str = "Unknown",
    disease_name: str = "Unknown",
    probability: float = 0.0,
    label: int = 0,
    logit: float = 0.0,
    known: bool = False,
    embeddings: Optional[Dict[str, torch.Tensor]] = None,
    attention_weights: Optional[List[Dict[Tuple, torch.Tensor]]] = None,
    node_names: Optional[Dict[str, Dict[int, str]]] = None,
    adj: Optional[AdjacencyIndex] = None,
    max_paths_per_template: int = 100,
    max_total_paths: int = 500,
) -> ExplanationResult:
    """
    Generate a full explanation for a chemical-disease pair prediction.
    
    Args:
        data: HeteroData containing the full graph.
        chem_idx: Internal chemical node index.
        disease_idx: Internal disease node index.
        chemical_id: External chemical ID (MESH).
        disease_id: External disease ID (MESH/OMIM).
        chemical_name: Human-readable chemical name.
        disease_name: Human-readable disease name.
        probability: Prediction probability from the model.
        label: Predicted label (0/1).
        logit: Raw logit from the decoder.
        known: Whether this is a known CTD association.
        embeddings: Node embeddings dict from model.encode().
        attention_weights: Per-layer attention dicts from model.encode(return_attention=True).
        node_names: Optional name lookup dict: node_type -> {idx -> name}.
        adj: Pre-built adjacency index (reuse for speed).
        max_paths_per_template: Max paths per metapath template.
        max_total_paths: Overall cap on returned paths.
        
    Returns:
        ExplanationResult with scored, ranked paths.
    """
    # Step 1: Build adjacency if not provided
    if adj is None:
        adj = build_adjacency(data)

    # Step 2: Enumerate paths (Tier 1)
    raw_paths = enumerate_paths(
        data, chem_idx, disease_idx,
        adj=adj,
        max_paths_per_template=max_paths_per_template,
    )

    # Step 3: Score paths (Tier 1 + optional Tier 2)
    scored = score_paths(
        raw_paths,
        embeddings=embeddings,
        attention_weights=attention_weights,
        node_names=node_names,
    )

    # Step 4: Truncate
    if len(scored) > max_total_paths:
        scored = scored[:max_total_paths]

    # Step 5: Build metapath summary
    mp_summary: Dict[str, int] = {}
    for sp in scored:
        key = sp.evidence_type
        mp_summary[key] = mp_summary.get(key, 0) + 1

    return ExplanationResult(
        chemical_id=chemical_id,
        disease_id=disease_id,
        chemical_name=chemical_name,
        disease_name=disease_name,
        probability=probability,
        label=label,
        logit=logit,
        known=known,
        paths=scored,
        attention_available=attention_weights is not None,
        metapath_summary=mp_summary,
    )


def build_node_names(
    data_dict: Dict[str, Any],
) -> Dict[str, Dict[int, str]]:
    """
    Build a node_names lookup from the processed data dictionary.
    
    Args:
        data_dict: Dictionary from load_processed_data() containing
            'chemicals', 'diseases', 'genes', etc. DataFrames.
            
    Returns:
        Dict mapping node_type -> {node_idx -> human_name}.
    """
    names: Dict[str, Dict[int, str]] = {}

    if 'chemicals' in data_dict:
        df = data_dict['chemicals']
        names['chemical'] = {
            idx: (name if name is not None else f"chemical:{idx}")
            for idx, name in zip(
                df['CHEM_ID'].to_list(),
                df['CHEM_NAME'].to_list(),
            )
        }

    if 'diseases' in data_dict:
        df = data_dict['diseases']
        names['disease'] = {
            idx: (name if name is not None else f"disease:{idx}")
            for idx, name in zip(
                df['DS_ID'].to_list(),
                df['DS_NAME'].to_list(),
            )
        }

    if 'genes' in data_dict:
        df = data_dict['genes']
        if 'GENE_SYMBOL' in df.columns:
            names['gene'] = {
                gid: (sym if sym is not None else f"gene:{gid}")
                for gid, sym in zip(
                    df['GENE_ID'].to_list(),
                    df['GENE_SYMBOL'].to_list(),
                )
            }
        else:
            names['gene'] = {
                gid: f"NCBI:{ncbi}" if ncbi is not None else f"gene:{gid}"
                for gid, ncbi in zip(
                    df['GENE_ID'].to_list(),
                    df['GENE_NCBI_ID'].to_list(),
                )
            }

    if 'pathways' in data_dict:
        df = data_dict['pathways']
        name_col = 'PATHWAY_NAME' if 'PATHWAY_NAME' in df.columns else None
        if name_col:
            names['pathway'] = {
                pid: (name if name is not None else f"pathway:{pid}")
                for pid, name in zip(
                    df['PATHWAY_ID'].to_list(),
                    df[name_col].to_list(),
                )
            }
        else:
            names['pathway'] = {
                pid: f"pathway:{pid}"
                for pid in df['PATHWAY_ID'].to_list()
            }

    if 'go_terms' in data_dict:
        df = data_dict['go_terms']
        name_col = 'GO_NAME' if 'GO_NAME' in df.columns else None
        # GO_SOURCE_ID provides a human-readable fallback (e.g. "GO:0000016")
        source_col = 'GO_SOURCE_ID' if 'GO_SOURCE_ID' in df.columns else None
        if name_col:
            source_ids = df[source_col].to_list() if source_col else [None] * df.height
            names['go_term'] = {
                gid: (
                    name if name is not None
                    else (src if src is not None else f"go_term:{gid}")
                )
                for gid, name, src in zip(
                    df['GO_TERM_ID'].to_list(),
                    df[name_col].to_list(),
                    source_ids,
                )
            }
        else:
            names['go_term'] = {
                gid: f"go_term:{gid}"
                for gid in df['GO_TERM_ID'].to_list()
            }

    return names
