"""Baseline model namespace."""

from .core import (
    BASELINE_NAMES,
    CD_REL,
    ComparisonConfig,
    DegreePopularityBaseline,
    GenericHGTBaseline,
    HeteroSAGEBaseline,
    HGTNoEdgeAttrBaseline,
    LightGCNCDBaseline,
    MatrixFactorizationBaseline,
    PairMLPBaseline,
    RGCNCDBaseline,
    build_baseline,
    compare_main_and_baselines,
    evaluate_baseline,
    load_main_model_from_checkpoint,
    train_baseline,
)

__all__ = [
    "CD_REL",
    "BASELINE_NAMES",
    "ComparisonConfig",
    "DegreePopularityBaseline",
    "MatrixFactorizationBaseline",
    "PairMLPBaseline",
    "LightGCNCDBaseline",
    "RGCNCDBaseline",
    "HeteroSAGEBaseline",
    "HGTNoEdgeAttrBaseline",
    "GenericHGTBaseline",
    "build_baseline",
    "train_baseline",
    "evaluate_baseline",
    "load_main_model_from_checkpoint",
    "compare_main_and_baselines",
]
