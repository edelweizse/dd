import pytest
import torch
from torch_geometric.data import HeteroData

from src.explainability.schema import ExplainContext, ExplainRequest
from src.explainability.service import ExplainService


def _tiny_graph() -> HeteroData:
    data = HeteroData()
    data['chemical'].x = torch.arange(2)
    data['disease'].x = torch.arange(2)
    data['gene'].x = torch.arange(2)
    data['chemical'].num_nodes = 2
    data['disease'].num_nodes = 2
    data['gene'].num_nodes = 2

    data['chemical', 'affects', 'gene'].edge_index = torch.tensor([[0], [1]])
    data['gene', 'rev_targets', 'disease'].edge_index = torch.tensor([[1], [0]])
    return data


def _tiny_embeddings():
    torch.manual_seed(1)
    return {
        'chemical': torch.randn(2, 8),
        'disease': torch.randn(2, 8),
        'gene': torch.randn(2, 8),
    }


def test_explain_service_path_attention_mode():
    svc = ExplainService()
    req = ExplainRequest(
        chemical_id='C0',
        disease_id='D0',
        chem_idx=0,
        disease_idx=0,
        mode='path_attention',
        runtime_profile='fast',
        max_paths_total=20,
        max_paths_per_template=20,
    )
    ctx = ExplainContext(
        data=_tiny_graph(),
        embeddings=_tiny_embeddings(),
        attention_weights=None,
    )
    out = svc.explain(req, ctx)
    assert out.engine == 'path_attention'
    assert out.runtime_profile == 'fast'
    assert len(out.paths) >= 1


def test_explain_service_unknown_mode():
    svc = ExplainService()
    req = ExplainRequest(
        chemical_id='C0',
        disease_id='D0',
        chem_idx=0,
        disease_idx=0,
        mode='path_attention',
        runtime_profile='deep',
        max_paths_total=10,
        max_paths_per_template=10,
    )
    req.mode = 'unknown_mode'  # type: ignore[assignment]
    ctx = ExplainContext(data=_tiny_graph())
    with pytest.raises(ValueError, match='Unknown explain mode'):
        svc.explain(req, ctx)
