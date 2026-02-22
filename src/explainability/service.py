"""
Service layer for selecting and executing explainability engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .engines import PathAttentionExplainer
from .schema import ExplainContext, ExplainRequest, ExplanationResult


@dataclass
class ExplainService:
    """Router over available explainer engines."""

    engines: Dict[str, object] = field(
        default_factory=lambda: {
            "path_attention": PathAttentionExplainer(),
        }
    )

    def explain(self, request: ExplainRequest, context: ExplainContext) -> ExplanationResult:
        if int(request.max_paths_total) < 1:
            raise ValueError("max_paths_total must be >= 1")
        if int(request.max_paths_per_template) < 1:
            raise ValueError("max_paths_per_template must be >= 1")

        mode = str(request.mode)
        if mode not in self.engines:
            raise ValueError(
                f'Unknown explain mode "{mode}". Available: {sorted(self.engines.keys())}'
            )
        engine = self.engines[mode]
        return engine.explain(request, context)


_DEFAULT_SERVICE = ExplainService()


def explain(request: ExplainRequest, context: ExplainContext) -> ExplanationResult:
    """Convenience function using default explain service instance."""
    return _DEFAULT_SERVICE.explain(request, context)
