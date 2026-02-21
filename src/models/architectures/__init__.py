"""Model architecture namespace."""

from .hgt import EdgeAttrHeteroConv, HGTPredictor, HGTMainModel, create_model_from_data
from .generic_hgt import (
    NodeInputSpec,
    EdgeAttrSpec,
    GraphSchema,
    infer_schema_from_data,
    GenericEdgeAttrHeteroConv,
    GenericHGTEncoder,
    BilinearLinkHead,
    GenericLinkPredictor,
)

__all__ = [
    'EdgeAttrHeteroConv',
    'HGTPredictor',
    'HGTMainModel',
    'create_model_from_data',
    'NodeInputSpec',
    'EdgeAttrSpec',
    'GraphSchema',
    'infer_schema_from_data',
    'GenericEdgeAttrHeteroConv',
    'GenericHGTEncoder',
    'BilinearLinkHead',
    'GenericLinkPredictor',
]
