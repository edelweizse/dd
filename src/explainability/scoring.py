"""
Path scoring for chemical-disease explainability.

Given enumerated PathInstance objects and (optionally) per-edge attention
weights from the HGT model, compute a composite score for each path.

Scoring formula per path:
    path_score = attention_score * embedding_similarity

Where:
  - attention_score: geometric mean of edge attention weights along the path
    (falls back to 1.0 when attention is unavailable).
  - embedding_similarity: cosine similarity between the intermediate node
    embeddings and the target chemical/disease embeddings, averaged over
    intermediate nodes.  This captures "how relevant is this intermediary
    to both the chemical and disease?".
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .paths import PathInstance


# ---------------------------------------------------------------------------
# Human-readable evidence labels
# ---------------------------------------------------------------------------

_TEMPLATE_LABELS: Dict[str, str] = {
    "chem-gene-disease": "Shared gene target",
    "chem-pathway-disease": "Shared pathway",
    "chem-goterm-disease": "GO term bridge",
    "chem-phenoGO-disease": "Phenotype GO bridge",
    "chem-gene-gene-disease": "PPI-connected gene targets",
    "chem-gene-pathway-disease": "Gene-to-pathway bridge",
    "chem-pathway-gene-disease": "Pathway-to-gene bridge",
}

_EDGE_VERB: Dict[Tuple[str, str, str], str] = {
    ("chemical", "affects", "gene"): "affects",
    ("gene", "rev_targets", "disease"): "targeted by",
    ("chemical", "enriched_in", "pathway"): "enriched in",
    ("pathway", "rev_disrupts", "disease"): "disrupted by",
    ("chemical", "enriched_in", "go_term"): "enriched in",
    ("go_term", "associated_with", "disease"): "associated with",
    ("chemical", "affects_phenotype", "go_term"): "affects phenotype",
    ("gene", "interacts_with", "gene"): "interacts with",
    ("gene", "participates_in", "pathway"): "participates in",
    ("pathway", "rev_participates_in", "gene"): "involves gene",
}


@dataclass
class ScoredPath:
    """A path with computed relevance scores and human-readable description."""
    path: PathInstance
    attention_score: float          # geometric mean of edge attentions
    embedding_score: float          # intermediate-node embedding similarity
    combined_score: float           # final ranking score
    evidence_type: str              # human label (e.g. "Shared gene target")
    description: str                # full human-readable path string
    edge_attentions: List[float] = field(default_factory=list)  # per-edge


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _geomean(values: List[float]) -> float:
    """Geometric mean of positive floats, returns 0 if any <= 0."""
    if not values:
        return 0.0
    product = 1.0
    for v in values:
        if v <= 0:
            return 0.0
        product *= v
    return product ** (1.0 / len(values))


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    a = a.float()
    b = b.float()
    dot = (a * b).sum()
    norm = a.norm() * b.norm()
    if norm < 1e-12:
        return 0.0
    return (dot / norm).item()


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_paths(
    paths: List[PathInstance],
    *,
    embeddings: Optional[Dict[str, torch.Tensor]] = None,
    attention_weights: Optional[List[Dict[Tuple, torch.Tensor]]] = None,
    node_names: Optional[Dict[str, Dict[int, str]]] = None,
) -> List[ScoredPath]:
    """
    Score and annotate a list of PathInstance objects.
    
    Args:
        paths: Enumerated path instances from the metapath walker.
        embeddings: Node embeddings dict (node_type -> [N, d]).
            Used for intermediate-node embedding similarity scoring.
        attention_weights: Per-layer attention dicts from HGTPredictor.encode().
            attention_weights[layer][edge_type] -> [E] attention values.
            When provided, enables Tier-2 attention scoring.
        node_names: Optional name lookup for human-readable descriptions.
            Maps node_type -> {node_idx -> name_str}.
            
    Returns:
        List of ScoredPath sorted by combined_score descending.
    """
    scored: List[ScoredPath] = []
    node_names = node_names or {}

    for pi in paths:
        # --- Attention score (Tier 2) ---
        edge_attns: List[float] = []
        if attention_weights is not None:
            for hop_idx, (et, epos) in enumerate(zip(pi.edge_types, pi.edge_positions)):
                attn_val = 1.0
                # Use last layer's attention (most refined)
                last_layer = attention_weights[-1]
                if et in last_layer and epos >= 0:
                    weights_tensor = last_layer[et]
                    if epos < weights_tensor.size(0):
                        attn_val = weights_tensor[epos].item()
                edge_attns.append(attn_val)
        else:
            edge_attns = [1.0] * len(pi.edge_types)

        attn_score = _geomean(edge_attns)

        # --- Embedding similarity score ---
        emb_score = 1.0
        if embeddings is not None and len(pi.node_indices) > 2:
            chem_emb = embeddings.get("chemical")
            dis_emb = embeddings.get("disease")
            if chem_emb is not None and dis_emb is not None:
                chem_vec = chem_emb[pi.node_indices[0]]
                dis_vec = dis_emb[pi.node_indices[-1]]
                mid_sims: List[float] = []
                for mid_pos in range(1, len(pi.node_indices) - 1):
                    mid_type = pi.node_types[mid_pos]
                    mid_idx = pi.node_indices[mid_pos]
                    mid_emb = embeddings.get(mid_type)
                    if mid_emb is not None and mid_idx < mid_emb.size(0):
                        mid_vec = mid_emb[mid_idx]
                        sim_c = _cosine_sim(mid_vec, chem_vec)
                        sim_d = _cosine_sim(mid_vec, dis_vec)
                        # Average relevance to both endpoints
                        mid_sims.append((sim_c + sim_d) / 2.0)
                if mid_sims:
                    emb_score = max(0.0, sum(mid_sims) / len(mid_sims))

        combined = attn_score * (0.5 + 0.5 * emb_score)

        # --- Human-readable description ---
        evidence = _TEMPLATE_LABELS.get(pi.template_name, pi.template_name)
        desc = _build_description(pi, node_names)

        scored.append(ScoredPath(
            path=pi,
            attention_score=attn_score,
            embedding_score=emb_score,
            combined_score=combined,
            evidence_type=evidence,
            description=desc,
            edge_attentions=edge_attns,
        ))

    scored.sort(key=lambda sp: sp.combined_score, reverse=True)
    return scored


def _build_description(
    pi: PathInstance,
    node_names: Dict[str, Dict[int, str]],
) -> str:
    """Build a human-readable string for a path instance."""
    parts: List[str] = []
    for i, (ntype, nidx) in enumerate(zip(pi.node_types, pi.node_indices)):
        name_map = node_names.get(ntype, {})
        name = name_map.get(nidx, f"{ntype}:{nidx}")
        # Guard against None values in name lookups
        if name is None:
            name = f"{ntype}:{nidx}"
        parts.append(name)
        if i < len(pi.edge_types):
            et = pi.edge_types[i]
            verb = _EDGE_VERB.get(et, et[1])
            parts.append(f" --[{verb}]--> ")
    return "".join(parts)
