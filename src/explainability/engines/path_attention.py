"""
Path + attention explainer engine.
"""

from __future__ import annotations

import time
from typing import Dict

from ..paths import enumerate_paths, build_adjacency
from ..schema import ExplainContext, ExplainRequest, ExplanationResult
from ..scoring import score_paths
from ..templates import required_edge_types, resolve_template_set


_PROFILE_WEIGHTS: Dict[str, tuple[float, float]] = {
    "fast": (0.2, 0.8),
    "balanced": (0.5, 0.5),
    "deep": (0.7, 0.3),
}


class PathAttentionExplainer:
    """Refined metapath explainer with profile-driven scoring/runtime knobs."""

    name = "path_attention"

    def explain(self, request: ExplainRequest, context: ExplainContext) -> ExplanationResult:
        t0 = time.perf_counter()
        templates = resolve_template_set(
            template_set=request.template_set,
            custom_templates=request.custom_templates,
        )

        adj = context.adj
        if adj is None:
            adj = build_adjacency(
                context.data,
                edge_types=required_edge_types(templates),
            )

        t_enum0 = time.perf_counter()
        raw_paths = enumerate_paths(
            context.data,
            chem_idx=request.chem_idx,
            disease_idx=request.disease_idx,
            adj=adj,
            templates=templates,
            max_paths_per_template=int(request.max_paths_per_template),
        )
        t_enum1 = time.perf_counter()

        wa, we = _PROFILE_WEIGHTS.get(request.runtime_profile, _PROFILE_WEIGHTS["fast"])
        attention_weights = context.attention_weights if request.use_attention else None

        t_score0 = time.perf_counter()
        scored = score_paths(
            raw_paths,
            embeddings=context.embeddings,
            attention_weights=attention_weights,
            node_names=request.node_names,
            weight_attention=wa,
            weight_embedding=we,
            top_k=int(request.max_paths_total),
        )
        t_score1 = time.perf_counter()

        mp_summary: Dict[str, int] = {}
        for sp in scored:
            mp_summary[sp.evidence_type] = mp_summary.get(sp.evidence_type, 0) + 1

        total_ms = (time.perf_counter() - t0) * 1000.0
        debug_metrics = {
            "num_raw_paths": int(len(raw_paths)),
            "num_scored_paths": int(len(scored)),
            "template_count": int(len(templates)),
            "enumeration_ms": round((t_enum1 - t_enum0) * 1000.0, 3),
            "scoring_ms": round((t_score1 - t_score0) * 1000.0, 3),
            "total_ms": round(total_ms, 3),
            "duplicate_edge_pairs": {
                "__".join(et): int(c)
                for et, c in adj.duplicate_pairs.items()
                if c > 0
            },
            "weight_attention": wa,
            "weight_embedding": we,
        }

        return ExplanationResult(
            chemical_id=request.chemical_id,
            disease_id=request.disease_id,
            chemical_name=request.chemical_name,
            disease_name=request.disease_name,
            probability=float(request.probability),
            label=int(request.label),
            logit=float(request.logit),
            known=bool(request.known),
            paths=scored,
            attention_available=bool(attention_weights),
            metapath_summary=mp_summary,
            engine=self.name,
            runtime_profile=request.runtime_profile,
            debug_metrics=debug_metrics,
        )
