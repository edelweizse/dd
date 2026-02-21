"""Backward-compatible shim for full-graph inference predictor imports."""

from .inference.full_graph import ChemDiseasePredictor, FullGraphPredictor

__all__ = [
    'ChemDiseasePredictor',
    'FullGraphPredictor',
]
