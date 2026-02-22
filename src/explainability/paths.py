"""
Metapath enumeration for chemical-disease pair explainability.

Given a (chemical_idx, disease_idx) pair and the full HeteroData graph,
enumerate all concrete connecting paths that follow pre-defined metapath
templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from torch_geometric.data import HeteroData

from .index import AdjacencyIndex, build_adjacency_index
from .templates import DEFAULT_METAPATH_TEMPLATES, required_edge_types, validate_templates


# ---------------------------------------------------------------------------
# Metapath templates
# ---------------------------------------------------------------------------

_EDGE = Tuple[str, str, str]
METAPATH_TEMPLATES: Dict[str, List[_EDGE]] = DEFAULT_METAPATH_TEMPLATES


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PathInstance:
    """A concrete path through the graph with typed node/edge sequences."""

    template_name: str
    node_indices: List[int]  # [chem_idx, ..., disease_idx]
    node_types: List[str]  # [chemical, ..., disease]
    edge_types: List[_EDGE]  # edge type tuples along the path
    edge_positions: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------

def build_adjacency(
    data: HeteroData,
    edge_types: Optional[List[_EDGE]] = None,
) -> AdjacencyIndex:
    """
    Build adjacency maps for edge types.

    If ``edge_types`` is omitted, adjacency is built only for edges required by
    the default metapath templates.
    """
    active_edge_types = edge_types if edge_types is not None else required_edge_types(METAPATH_TEMPLATES)
    return build_adjacency_index(data, edge_types=active_edge_types)


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
    """Enumerate paths for one template."""
    n_hops = len(template)
    if n_hops == 2:
        return _enumerate_2hop(template_name, template, chem_idx, disease_idx, adj, max_paths)
    if n_hops == 3:
        return _enumerate_3hop(template_name, template, chem_idx, disease_idx, adj, max_paths)
    return []


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
    rev1 = adj.reverse.get(et1, {})

    mid_from_chem = fwd0.get(chem_idx, set())
    mid_from_disease = rev1.get(disease_idx, set())
    shared = mid_from_chem & mid_from_disease

    paths: List[PathInstance] = []
    for mid_idx in shared:
        if len(paths) >= max_paths:
            break
        pos0 = adj.edge_pos.get(et0, {}).get((chem_idx, mid_idx), -1)
        pos1 = adj.edge_pos.get(et1, {}).get((mid_idx, disease_idx), -1)
        paths.append(
            PathInstance(
                template_name=name,
                node_indices=[chem_idx, mid_idx, disease_idx],
                node_types=[et0[0], et0[2], et1[2]],
                edge_types=[et0, et1],
                edge_positions=[pos0, pos1],
            )
        )
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
    rev2 = adj.reverse.get(et2, {})

    a_nodes = fwd0.get(chem_idx, set())
    b_nodes = rev2.get(disease_idx, set())

    if not a_nodes or not b_nodes:
        return []

    paths: List[PathInstance] = []
    for a in a_nodes:
        if len(paths) >= max_paths:
            break
        b_from_a = fwd1.get(a, set())
        shared_b = b_from_a & b_nodes
        for b in shared_b:
            if len(paths) >= max_paths:
                break
            pos0 = adj.edge_pos.get(et0, {}).get((chem_idx, a), -1)
            pos1 = adj.edge_pos.get(et1, {}).get((a, b), -1)
            pos2 = adj.edge_pos.get(et2, {}).get((b, disease_idx), -1)
            paths.append(
                PathInstance(
                    template_name=name,
                    node_indices=[chem_idx, a, b, disease_idx],
                    node_types=[et0[0], et0[2], et1[2], et2[2]],
                    edge_types=[et0, et1, et2],
                    edge_positions=[pos0, pos1, pos2],
                )
            )
    return paths


def enumerate_paths(
    data: HeteroData,
    chem_idx: int,
    disease_idx: int,
    adj: Optional[AdjacencyIndex] = None,
    templates: Optional[Dict[str, List[_EDGE]]] = None,
    max_paths_per_template: int = 100,
) -> List[PathInstance]:
    """Enumerate all connecting paths between a chemical and disease."""
    active_templates = METAPATH_TEMPLATES if templates is None else templates
    validate_templates(active_templates)

    if adj is None:
        adj = build_adjacency(
            data,
            edge_types=required_edge_types(active_templates),
        )

    all_paths: List[PathInstance] = []
    for tname, template in active_templates.items():
        if not all(et in adj.forward for et in template):
            continue
        all_paths.extend(
            _enumerate_template(
                tname,
                template,
                chem_idx,
                disease_idx,
                adj,
                max_paths=max_paths_per_template,
            )
        )
    return all_paths


__all__ = [
    "_EDGE",
    "METAPATH_TEMPLATES",
    "PathInstance",
    "AdjacencyIndex",
    "build_adjacency",
    "enumerate_paths",
]
