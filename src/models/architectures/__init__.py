"""Model architecture namespace."""

from .hgat import (
    EdgeAttrHeteroConv,
    HGATMainModel,
    HGATPredictor,
    create_model_from_data,
    infer_hgat_hparams_from_state,
)
from .generic_hgat import (
    BilinearLinkHead,
    EdgeAttrSpec,
    GenericEdgeAttrHeteroConv,
    GenericHGATEncoder,
    GenericLinkPredictor,
    GraphSchema,
    NodeInputSpec,
    infer_schema_from_data,
)

__all__ = [
    "EdgeAttrHeteroConv",
    "HGATPredictor",
    "HGATMainModel",
    "infer_hgat_hparams_from_state",
    "create_model_from_data",
    "NodeInputSpec",
    "EdgeAttrSpec",
    "GraphSchema",
    "infer_schema_from_data",
    "GenericEdgeAttrHeteroConv",
    "GenericHGATEncoder",
    "BilinearLinkHead",
    "GenericLinkPredictor",
]
