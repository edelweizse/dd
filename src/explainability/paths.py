"""
Metapath enumeration for chemical-disease pair explainability.

Given a (chemical_idx, disease_idx) pair and the full HeteroData graph,
enumerate all concrete connecting paths that follow pre-defined metapath
templates.  Each template describes a typed sequence of edges that can
connect a chemical to a disease through intermediate nodes.

The implementation builds sparse adjacency structures once and then
walks forward from the chemical and backward from the disease,
intersecting at intermediate nodes to avoid full graph traversal.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from torch_geometric.data import HeteroData
from typing import Dict, List, Set, Tuple, Optional


# ---------------------------------------------------------------------------
# Metapath template definitions
# ---------------------------------------------------------------------------
# Each template is a list of (src_type, rel_type, dst_type) edge-type tuples
# describing a typed walk from 'chemical' to 'disease'.
# The walker will materialise concrete node indices along each template.

_EDGE = Tuple[str, str, str]

METAPATH_TEMPLATES: Dict[str, List[_EDGE]] = {
    # Length-2 paths (1 intermediate node)
    "chem-gene-disease": [
        ("chemical", "affects", "gene"),
        ("gene", "rev_targets", "disease"),
    ],
    "chem-pathway-disease": [
        ("chemical", "enriched_in", "pathway"),
        ("pathway", "rev_disrupts", "disease"),
    ],
    "chem-goterm-disease": [
        ("chemical", "enriched_in", "go_term"),
        ("go_term", "associated_with", "disease"),
    ],
    "chem-phenoGO-disease": [
        ("chemical", "affects_phenotype", "go_term"),
        ("go_term", "associated_with", "disease"),
    ],

    # Length-3 paths (2 intermediate nodes)
    "chem-gene-gene-disease": [
        ("chemical", "affects", "gene"),
        ("gene", "interacts_with", "gene"),
        ("gene", "rev_targets", "disease"),
    ],
    "chem-gene-pathway-disease": [
        ("chemical", "affects", "gene"),
        ("gene", "participates_in", "pathway"),
        ("pathway", "rev_disrupts", "disease"),
    ],
    "chem-pathway-gene-disease": [
        ("chemical", "enriched_in", "pathway"),
        ("pathway", "rev_participates_in", "gene"),
        ("gene", "rev_targets", "disease"),
    ],
    "chem-goterm-gene-disease": [
        ("chemical", "enriched_in", "go_term"),
        ("go_term", "rev_enriched_in", "chemical"),
        # This template is unusual — skip for now
    ],
}

# Only keep well-formed templates (start with chemical, end with disease)
METAPATH_TEMPLATES = {
    k: v for k, v in METAPATH_TEMPLATES.items()
    if v[0][0] == "chemical" and v[-1][2] == "disease"
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PathInstance:
    """A concrete path through the graph with typed node/edge sequences."""
    template_name: str
    node_indices: List[int]       # [chem_idx, ..., disease_idx]
    node_types: List[str]         # [chemical, ..., disease]
    edge_types: List[_EDGE]       # edge type tuples along the path
    # Indices into the edge_index for each hop (for attention lookup)
    edge_positions: List[int] = field(default_factory=list)


@dataclass
class AdjacencyIndex:
    """Pre-built adjacency lists for fast neighbor lookup."""
    # forward[edge_type][src_idx] -> set of dst_idx
    forward: Dict[_EDGE, Dict[int, Set[int]]]
    # For edge position lookup: edge_type -> {(src, dst) -> position in edge_index}
    edge_pos: Dict[_EDGE, Dict[Tuple[int, int], int]]


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------

def build_adjacency(
    data: HeteroData,
    edge_types: Optional[List[_EDGE]] = None,
) -> AdjacencyIndex:
    """
    Build forward adjacency dicts from a HeteroData graph.
    
    Args:
        data: HeteroData containing the full graph.
        edge_types: If given, only build for these edge types.
            Defaults to all edge types present in the templates.
            
    Returns:
        AdjacencyIndex with forward adjacency and edge-position maps.
    """
    if edge_types is None:
        needed: Set[_EDGE] = set()
        for template in METAPATH_TEMPLATES.values():
            for et in template:
                needed.add(et)
        edge_types = [et for et in needed if et in data.edge_types]

    forward: Dict[_EDGE, Dict[int, Set[int]]] = {}
    edge_pos: Dict[_EDGE, Dict[Tuple[int, int], int]] = {}

    for et in edge_types:
        ei = data[et].edge_index  # [2, E]
        src_arr = ei[0].cpu().tolist()
        dst_arr = ei[1].cpu().tolist()

        adj: Dict[int, Set[int]] = {}
        pos: Dict[Tuple[int, int], int] = {}
        for i, (s, d) in enumerate(zip(src_arr, dst_arr)):
            adj.setdefault(s, set()).add(d)
            pos[(s, d)] = i

        forward[et] = adj
        edge_pos[et] = pos

    return AdjacencyIndex(forward=forward, edge_pos=edge_pos)


# ---------------------------------------------------------------------------
# Path enumeration
# ---------------------------------------------------------------------------

def _enumerate_template(
    template_name: str,
    template: List[_EDGE],
    chem_idx: int,
    disease_idx: int,
    adj: AdjacencyIndex,
    max_paths: int = 100,
) -> List[PathInstance]:
    """
    Enumerate paths following a single metapath template between a
    chemical and a disease.
    
    Uses bidirectional search: expand forward from the chemical and
    backward from the disease, intersect at the middle node.  For
    length-2 templates, the middle is hop-1.  For length-3 templates,
    we expand forward 1 hop from chem, backward 1 hop from disease,
    then check if a middle edge connects them.
    """
    n_hops = len(template)

    if n_hops == 2:
        return _enumerate_2hop(template_name, template, chem_idx, disease_idx,
                               adj, max_paths)
    elif n_hops == 3:
        return _enumerate_3hop(template_name, template, chem_idx, disease_idx,
                               adj, max_paths)
    else:
        return []  # Only 2/3-hop templates supported


def _enumerate_2hop(
    name: str,
    template: List[_EDGE],
    chem_idx: int,
    disease_idx: int,
    adj: AdjacencyIndex,
    max_paths: int,
) -> List[PathInstance]:
    """Enumerate 2-hop paths: chemical -> X -> disease."""
    et0, et1 = template
    fwd0 = adj.forward.get(et0, {})
    fwd1 = adj.forward.get(et1, {})

    mid_from_chem = fwd0.get(chem_idx, set())
    mid_from_disease: Set[int] = set()
    # Walk backward from disease: find all X such that X -> disease
    for x, dsts in fwd1.items():
        if disease_idx in dsts:
            mid_from_disease.add(x)

    shared = mid_from_chem & mid_from_disease
    paths: List[PathInstance] = []
    for mid_idx in shared:
        if len(paths) >= max_paths:
            break
        pos0 = adj.edge_pos.get(et0, {}).get((chem_idx, mid_idx), -1)
        pos1 = adj.edge_pos.get(et1, {}).get((mid_idx, disease_idx), -1)
        paths.append(PathInstance(
            template_name=name,
            node_indices=[chem_idx, mid_idx, disease_idx],
            node_types=[et0[0], et0[2], et1[2]],
            edge_types=[et0, et1],
            edge_positions=[pos0, pos1],
        ))
    return paths


def _enumerate_3hop(
    name: str,
    template: List[_EDGE],
    chem_idx: int,
    disease_idx: int,
    adj: AdjacencyIndex,
    max_paths: int,
) -> List[PathInstance]:
    """Enumerate 3-hop paths: chemical -> A -> B -> disease."""
    et0, et1, et2 = template
    fwd0 = adj.forward.get(et0, {})
    fwd1 = adj.forward.get(et1, {})
    fwd2 = adj.forward.get(et2, {})

    # Forward 1 hop from chem: chem -> A set
    a_nodes = fwd0.get(chem_idx, set())
    # Backward 1 hop from disease: B set -> disease
    b_nodes: Set[int] = set()
    for b, dsts in fwd2.items():
        if disease_idx in dsts:
            b_nodes.add(b)

    if not a_nodes or not b_nodes:
        return []

    paths: List[PathInstance] = []
    for a in a_nodes:
        if len(paths) >= max_paths:
            break
        # Forward 1 hop from A via et1: A -> B candidates
        b_from_a = fwd1.get(a, set())
        shared_b = b_from_a & b_nodes
        for b in shared_b:
            if len(paths) >= max_paths:
                break
            pos0 = adj.edge_pos.get(et0, {}).get((chem_idx, a), -1)
            pos1 = adj.edge_pos.get(et1, {}).get((a, b), -1)
            pos2 = adj.edge_pos.get(et2, {}).get((b, disease_idx), -1)
            paths.append(PathInstance(
                template_name=name,
                node_indices=[chem_idx, a, b, disease_idx],
                node_types=[et0[0], et0[2], et1[2], et2[2]],
                edge_types=[et0, et1, et2],
                edge_positions=[pos0, pos1, pos2],
            ))
    return paths


def enumerate_paths(
    data: HeteroData,
    chem_idx: int,
    disease_idx: int,
    adj: Optional[AdjacencyIndex] = None,
    templates: Optional[Dict[str, List[_EDGE]]] = None,
    max_paths_per_template: int = 100,
) -> List[PathInstance]:
    """
    Enumerate all connecting paths between a chemical and disease.
    
    Args:
        data: HeteroData containing the full graph.
        chem_idx: Internal chemical node index.
        disease_idx: Internal disease node index.
        adj: Pre-built adjacency (reuse across calls for speed).
        templates: Metapath templates to use (defaults to all).
        max_paths_per_template: Cap per template to avoid explosion.
        
    Returns:
        List of PathInstance objects, one per concrete path found.
    """
    if adj is None:
        adj = build_adjacency(data)
    if templates is None:
        templates = METAPATH_TEMPLATES

    all_paths: List[PathInstance] = []
    for tname, template in templates.items():
        # Check all required edge types exist in adjacency
        if not all(et in adj.forward for et in template):
            continue
        paths = _enumerate_template(
            tname, template, chem_idx, disease_idx,
            adj, max_paths=max_paths_per_template,
        )
        all_paths.extend(paths)
    return all_paths
