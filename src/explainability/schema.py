"""
Typed request/response schema for explainability engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

import torch
from torch_geometric.data import HeteroData

if TYPE_CHECKING:
    from .paths import AdjacencyIndex
    from .scoring import ScoredPath

ExplainMode = Literal["path_attention"]
RuntimeProfile = Literal["fast", "balanced", "deep"]


@dataclass
class ExplainRequest:
    """Normalized explainer request."""

    chemical_id: str
    disease_id: str
    chem_idx: int
    disease_idx: int
    chemical_name: str = "Unknown"
    disease_name: str = "Unknown"
    probability: float = 0.0
    label: int = 0
    logit: float = 0.0
    known: bool = False
    mode: ExplainMode = "path_attention"
    runtime_profile: RuntimeProfile = "fast"
    template_set: str = "default"
    use_attention: bool = True
    max_paths_total: int = 500
    max_paths_per_template: int = 100
    node_names: Optional[Dict[str, Dict[int, str]]] = None
    custom_templates: Optional[Dict[str, List[tuple[str, str, str]]]] = None


@dataclass
class ExplainContext:
    """Runtime context used by the chosen explainer engine."""

    data: HeteroData
    embeddings: Optional[Dict[str, torch.Tensor]] = None
    attention_weights: Optional[List[Dict[tuple[str, str, str], torch.Tensor]]] = None
    adj: Optional["AdjacencyIndex"] = None
    model: Optional[torch.nn.Module] = None


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
    paths: List["ScoredPath"]
    attention_available: bool
    metapath_summary: Dict[str, int] = field(default_factory=dict)
    engine: str = "path_attention"
    runtime_profile: str = "fast"
    debug_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_paths(self) -> List["ScoredPath"]:
        """Return top-10 paths by combined score."""
        return self.paths[:10]

    def summary_text(self, max_paths: int = 5) -> str:
        """Return a compact text summary of the explanation."""
        lines = [
            f"Prediction: {self.chemical_name} <-> {self.disease_name}",
            f"  Probability: {self.probability:.4f}  "
            f"Label: {'Associated' if self.label else 'Not associated'}  "
            f"Known: {'Yes' if self.known else 'No'}",
            f"  Engine: {self.engine}  Profile: {self.runtime_profile}",
            f"  Attention scoring: {'Yes' if self.attention_available else 'No'}",
            f"  Total paths found: {len(self.paths)}",
        ]
        if self.metapath_summary:
            lines.append(
                "  Path types: "
                + ", ".join(f"{k}: {v}" for k, v in sorted(self.metapath_summary.items()))
            )
        if self.debug_metrics:
            shown = []
            for key in ("enumeration_ms", "scoring_ms", "total_ms", "num_raw_paths"):
                if key in self.debug_metrics:
                    shown.append(f"{key}={self.debug_metrics[key]}")
            if shown:
                lines.append("  Debug: " + ", ".join(shown))
        lines.append("")
        for i, sp in enumerate(self.paths[:max_paths], 1):
            lines.append(
                f"  {i}. [{sp.evidence_type}] score={sp.combined_score:.4f}  "
                f"attn={sp.attention_score:.4f}  emb={sp.embedding_score:.4f}"
            )
            lines.append(f"     {sp.description}")
        if not self.paths:
            lines.append("  No path-based evidence available for this mode/output.")
        return "\n".join(lines)
