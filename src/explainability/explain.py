"""
High-level explainability API for chemical-disease predictions.

Legacy ``explain_pair`` is kept as a compatibility wrapper over the new
request/context service API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from .schema import ExplainContext, ExplainRequest, ExplanationResult
from .service import explain as _run_explain


def explain(request: ExplainRequest, context: ExplainContext) -> ExplanationResult:
    """Execute explainability using the configured engine mode."""
    return _run_explain(request, context)


def explain_pair(
    *,
    data: HeteroData,
    chem_idx: int,
    disease_idx: int,
    chemical_id: str,
    disease_id: str,
    chemical_name: str = "Unknown",
    disease_name: str = "Unknown",
    probability: float = 0.0,
    label: int = 0,
    logit: float = 0.0,
    known: bool = False,
    embeddings: Optional[Dict[str, torch.Tensor]] = None,
    attention_weights: Optional[List[Dict[Tuple[str, str, str], torch.Tensor]]] = None,
    node_names: Optional[Dict[str, Dict[int, str]]] = None,
    adj=None,
    max_paths_per_template: int = 100,
    max_total_paths: int = 500,
) -> ExplanationResult:
    """
    Backward-compatible wrapper for path-attention explainability.
    """
    request = ExplainRequest(
        chemical_id=chemical_id,
        disease_id=disease_id,
        chem_idx=int(chem_idx),
        disease_idx=int(disease_idx),
        chemical_name=chemical_name,
        disease_name=disease_name,
        probability=float(probability),
        label=int(label),
        logit=float(logit),
        known=bool(known),
        mode="path_attention",
        runtime_profile="balanced",
        template_set="default",
        use_attention=attention_weights is not None,
        max_paths_total=int(max_total_paths),
        max_paths_per_template=int(max_paths_per_template),
        node_names=node_names,
    )
    context = ExplainContext(
        data=data,
        embeddings=embeddings,
        attention_weights=attention_weights,
        adj=adj,
    )
    return _run_explain(request, context)


def build_node_names(
    data_dict: Dict[str, Any],
) -> Dict[str, Dict[int, str]]:
    """
    Build node_names lookup from processed data dictionary.
    """
    names: Dict[str, Dict[int, str]] = {}

    if 'chemicals' in data_dict:
        df = data_dict['chemicals']
        names['chemical'] = {
            idx: (name if name is not None else f"chemical:{idx}")
            for idx, name in zip(
                df['CHEM_ID'].to_list(),
                df['CHEM_NAME'].to_list(),
            )
        }

    if 'diseases' in data_dict:
        df = data_dict['diseases']
        names['disease'] = {
            idx: (name if name is not None else f"disease:{idx}")
            for idx, name in zip(
                df['DS_ID'].to_list(),
                df['DS_NAME'].to_list(),
            )
        }

    if 'genes' in data_dict:
        df = data_dict['genes']
        if 'GENE_SYMBOL' in df.columns:
            names['gene'] = {
                gid: (sym if sym is not None else f"gene:{gid}")
                for gid, sym in zip(
                    df['GENE_ID'].to_list(),
                    df['GENE_SYMBOL'].to_list(),
                )
            }
        else:
            names['gene'] = {
                gid: f"NCBI:{ncbi}" if ncbi is not None else f"gene:{gid}"
                for gid, ncbi in zip(
                    df['GENE_ID'].to_list(),
                    df['GENE_NCBI_ID'].to_list(),
                )
            }

    if 'pathways' in data_dict:
        df = data_dict['pathways']
        name_col = 'PATHWAY_NAME' if 'PATHWAY_NAME' in df.columns else None
        if name_col:
            names['pathway'] = {
                pid: (name if name is not None else f"pathway:{pid}")
                for pid, name in zip(
                    df['PATHWAY_ID'].to_list(),
                    df[name_col].to_list(),
                )
            }
        else:
            names['pathway'] = {
                pid: f"pathway:{pid}"
                for pid in df['PATHWAY_ID'].to_list()
            }

    if 'go_terms' in data_dict:
        df = data_dict['go_terms']
        name_col = 'GO_NAME' if 'GO_NAME' in df.columns else None
        source_col = 'GO_SOURCE_ID' if 'GO_SOURCE_ID' in df.columns else None
        if name_col:
            source_ids = df[source_col].to_list() if source_col else [None] * df.height
            names['go_term'] = {
                gid: (
                    name if name is not None
                    else (src if src is not None else f"go_term:{gid}")
                )
                for gid, name, src in zip(
                    df['GO_TERM_ID'].to_list(),
                    df[name_col].to_list(),
                    source_ids,
                )
            }
        else:
            names['go_term'] = {
                gid: f"go_term:{gid}"
                for gid in df['GO_TERM_ID'].to_list()
            }

    return names


__all__ = [
    'ExplainRequest',
    'ExplainContext',
    'ExplanationResult',
    'explain',
    'explain_pair',
    'build_node_names',
]
