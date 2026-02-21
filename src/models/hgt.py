"""Backward-compatible shim for HGT architecture imports."""

from .architectures.hgt import EdgeAttrHeteroConv, HGTPredictor, HGTMainModel, create_model_from_data

__all__ = [
    'EdgeAttrHeteroConv',
    'HGTPredictor',
    'HGTMainModel',
    'create_model_from_data',
]
