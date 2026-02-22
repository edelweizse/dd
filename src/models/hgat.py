"""HGAT architecture shim for top-level model imports."""

from .architectures.hgat import (
    EdgeAttrHeteroConv,
    HGATMainModel,
    HGATPredictor,
    create_model_from_data,
    infer_hgat_hparams_from_state,
)

__all__ = [
    "EdgeAttrHeteroConv",
    "HGATPredictor",
    "HGATMainModel",
    "infer_hgat_hparams_from_state",
    "create_model_from_data",
]
