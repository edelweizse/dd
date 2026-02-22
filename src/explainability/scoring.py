"""
Path scoring for chemical-disease explainability.

Given enumerated PathInstance objects and (optionally) per-edge attention
weights from the model, compute a composite score for each path.

Combined score:
    combined_score = w_attn * attention_score + w_emb * embedding_score
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .paths import PathInstance


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
    attention_score: float
    embedding_score: float
    combined_score: float
    evidence_type: str
    description: str
    edge_attentions: List[float] = field(default_factory=list)


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


def _normalize_weights(weight_attention: float, weight_embedding: float) -> Tuple[float, float]:
    wa = max(0.0, float(weight_attention))
    we = max(0.0, float(weight_embedding))
    total = wa + we
    if total <= 0:
        return 0.5, 0.5
    return wa / total, we / total


def score_paths(
    paths: List[PathInstance],
    *,
    embeddings: Optional[Dict[str, torch.Tensor]] = None,
    attention_weights: Optional[List[Dict[Tuple[str, str, str], torch.Tensor]]] = None,
    node_names: Optional[Dict[str, Dict[int, str]]] = None,
    weight_attention: float = 0.5,
    weight_embedding: float = 0.5,
    top_k: Optional[int] = None,
) -> List[ScoredPath]:
    """
    Score and annotate a list of PathInstance objects.

    Args:
        paths: Enumerated path instances from the metapath walker.
        embeddings: Node embeddings dict (node_type -> [N, d]).
        attention_weights: Per-layer attention dicts from model.encode().
        node_names: Optional name lookup for human-readable descriptions.
        weight_attention: Non-negative attention weight for combined score.
        weight_embedding: Non-negative embedding-similarity weight.
        top_k: Optional top-k selection; when set, avoids sorting full list.

    Returns:
        List of ScoredPath sorted by combined_score descending.
    """
    node_names = node_names or {}
    use_attention = bool(attention_weights)
    last_layer = attention_weights[-1] if use_attention else None
    wa, we = _normalize_weights(weight_attention, weight_embedding)

    if top_k is not None and top_k <= 0:
        top_k = None

    heap: List[Tuple[float, int, ScoredPath]] = []
    scored_all: List[ScoredPath] = []

    for idx, pi in enumerate(paths):
        edge_attns: List[float] = []
        if last_layer is not None:
            for et, epos in zip(pi.edge_types, pi.edge_positions):
                attn_val = 1.0
                if et in last_layer and epos >= 0:
                    weights_tensor = last_layer[et]
                    if epos < weights_tensor.size(0):
                        attn_val = float(weights_tensor[epos].item())
                edge_attns.append(attn_val)
        else:
            edge_attns = [1.0] * len(pi.edge_types)

        attn_score = _geomean(edge_attns)

        emb_score = 0.0
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
                        mid_sims.append(max(0.0, (sim_c + sim_d) / 2.0))
                if mid_sims:
                    emb_score = sum(mid_sims) / len(mid_sims)

        combined = wa * attn_score + we * emb_score

        evidence = _TEMPLATE_LABELS.get(pi.template_name, pi.template_name)
        desc = _build_description(pi, node_names)

        item = ScoredPath(
            path=pi,
            attention_score=attn_score,
            embedding_score=emb_score,
            combined_score=combined,
            evidence_type=evidence,
            description=desc,
            edge_attentions=edge_attns,
        )

        if top_k is None:
            scored_all.append(item)
        else:
            entry = (item.combined_score, idx, item)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)

    if top_k is None:
        scored_all.sort(key=lambda sp: sp.combined_score, reverse=True)
        return scored_all

    top = [x[2] for x in sorted(heap, key=lambda x: x[0], reverse=True)]
    return top


def _build_description(
    pi: PathInstance,
    node_names: Dict[str, Dict[int, str]],
) -> str:
    """Build a human-readable string for a path instance."""
    parts: List[str] = []
    for i, (ntype, nidx) in enumerate(zip(pi.node_types, pi.node_indices)):
        name_map = node_names.get(ntype, {})
        name = name_map.get(nidx, f"{ntype}:{nidx}")
        if name is None:
            name = f"{ntype}:{nidx}"
        parts.append(name)
        if i < len(pi.edge_types):
            et = pi.edge_types[i]
            verb = _EDGE_VERB.get(et, et[1])
            parts.append(f" --[{verb}]--> ")
    return "".join(parts)
