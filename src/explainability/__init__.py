"""
Explainability module for chemical-disease link predictions.

Provides two tiers of explanation:
  Tier 1: Metapath enumeration — find connecting paths between a chemical
          and disease through the heterogeneous graph.
  Tier 2: Attention weight extraction — retrieve per-edge attention weights
          from the HGT model to score each path's contribution.
"""

from .paths import (
    METAPATH_TEMPLATES,
    PathInstance,
    AdjacencyIndex,
    build_adjacency,
    enumerate_paths,
)
from .scoring import ScoredPath, score_paths
from .schema import ExplainContext, ExplainRequest, ExplanationResult
from .explain import explain, explain_pair, build_node_names
from .service import ExplainService

__all__ = [
    'METAPATH_TEMPLATES',
    'PathInstance',
    'AdjacencyIndex',
    'build_adjacency',
    'enumerate_paths',
    'ScoredPath',
    'score_paths',
    'ExplainRequest',
    'ExplainContext',
    'ExplanationResult',
    'ExplainService',
    'explain',
    'explain_pair',
    'build_node_names',
]
