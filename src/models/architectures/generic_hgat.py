"""Canonical generic HGAT architecture exports."""

from .generic_hgt import (
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
    "NodeInputSpec",
    "EdgeAttrSpec",
    "GraphSchema",
    "infer_schema_from_data",
    "GenericEdgeAttrHeteroConv",
    "GenericHGATEncoder",
    "BilinearLinkHead",
    "GenericLinkPredictor",
]
