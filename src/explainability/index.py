"""
Adjacency indexing for explainability path enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from torch_geometric.data import HeteroData

EdgeType = Tuple[str, str, str]


@dataclass
class AdjacencyIndex:
    """Pre-built adjacency maps for fast forward/backward lookups."""

    forward: Dict[EdgeType, Dict[int, Set[int]]]
    reverse: Dict[EdgeType, Dict[int, Set[int]]]
    edge_pos: Dict[EdgeType, Dict[Tuple[int, int], int]]
    duplicate_pairs: Dict[EdgeType, int] = field(default_factory=dict)


def build_adjacency_index(
    data: HeteroData,
    edge_types: Optional[List[EdgeType]] = None,
) -> AdjacencyIndex:
    """
    Build forward/reverse adjacency maps and edge position lookup.

    Edge position policy for duplicate (src, dst) edges is deterministic:
    keep the first occurrence index and count later duplicates.
    """
    if edge_types is None:
        edge_types = list(data.edge_types)

    forward: Dict[EdgeType, Dict[int, Set[int]]] = {}
    reverse: Dict[EdgeType, Dict[int, Set[int]]] = {}
    edge_pos: Dict[EdgeType, Dict[Tuple[int, int], int]] = {}
    duplicate_pairs: Dict[EdgeType, int] = {}

    for et in edge_types:
        if et not in data.edge_types:
            continue
        ei = data[et].edge_index  # [2, E]
        src_arr = ei[0].cpu().tolist()
        dst_arr = ei[1].cpu().tolist()

        fwd: Dict[int, Set[int]] = {}
        rev: Dict[int, Set[int]] = {}
        pos: Dict[Tuple[int, int], int] = {}
        dup_count = 0

        for i, (s, d) in enumerate(zip(src_arr, dst_arr)):
            fwd.setdefault(s, set()).add(d)
            rev.setdefault(d, set()).add(s)
            key = (s, d)
            if key in pos:
                dup_count += 1
            else:
                pos[key] = i

        forward[et] = fwd
        reverse[et] = rev
        edge_pos[et] = pos
        duplicate_pairs[et] = dup_count

    return AdjacencyIndex(
        forward=forward,
        reverse=reverse,
        edge_pos=edge_pos,
        duplicate_pairs=duplicate_pairs,
    )

