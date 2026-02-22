import copy

import pytest
import torch
from torch_geometric.data import HeteroData

from src.models.architectures.generic_hgt import (
    EdgeAttrSpec,
    GenericLinkPredictor,
    GraphSchema,
    NodeInputSpec,
    infer_schema_from_data,
)
from src.models.architectures.hgt import (
    HGTPredictor,
    create_model_from_data,
    infer_hgt_hparams_from_state,
)


def _make_arch_data() -> HeteroData:
    data = HeteroData()
    torch.manual_seed(0)

    data['chemical'].x = torch.randn(3, 4)
    data['disease'].x = torch.randn(2, 3)
    data['gene'].x = torch.randn(2, 5)

    data['chemical'].num_nodes = 3
    data['disease'].num_nodes = 2
    data['gene'].num_nodes = 2

    cd_edge = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data['chemical', 'associated_with', 'disease'].edge_index = cd_edge
    data['disease', 'rev_associated_with', 'chemical'].edge_index = torch.flip(cd_edge, dims=[0])

    cg_edge = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    cg_attr = torch.tensor([[1, 2], [0, 1]], dtype=torch.long)
    data['chemical', 'affects', 'gene'].edge_index = cg_edge
    data['chemical', 'affects', 'gene'].edge_attr = cg_attr
    data['gene', 'rev_affects', 'chemical'].edge_index = torch.flip(cg_edge, dims=[0])
    data['gene', 'rev_affects', 'chemical'].edge_attr = cg_attr

    gd_edge = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    gd_attr = torch.tensor([[1.0, 0.5], [0.5, 1.5]], dtype=torch.float32)
    data['gene', 'rev_targets', 'disease'].edge_index = gd_edge
    data['gene', 'rev_targets', 'disease'].edge_attr = gd_attr
    data['disease', 'targets', 'gene'].edge_index = torch.flip(gd_edge, dims=[0])
    data['disease', 'targets', 'gene'].edge_attr = gd_attr

    return data


def test_hgt_predictor_encode_and_forward_shapes():
    data = _make_arch_data()
    model = HGTPredictor(
        num_nodes_dict={ntype: data[ntype].num_nodes for ntype in data.node_types},
        metadata=data.metadata(),
        node_input_dims={ntype: int(data[ntype].x.size(1)) for ntype in data.node_types},
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        num_action_types=4,
        num_action_subjects=4,
    )
    model.eval()

    edge_attr_dict = {
        et: data[et].edge_attr
        for et in data.edge_types
        if hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None
    }
    z, attn = model.encode(
        data.x_dict,
        data.edge_index_dict,
        edge_attr_dict,
        return_attention=True,
    )
    assert set(z.keys()) == {'chemical', 'disease', 'gene'}
    assert len(attn) == 2

    pos_edge = data['chemical', 'associated_with', 'disease'].edge_index
    neg_edge = torch.tensor([[2, 2], [0, 1]], dtype=torch.long)
    pos_logits, neg_logits = model(data, pos_edge, neg_edge)
    assert pos_logits.shape == (2,)
    assert neg_logits.shape == (2,)


def test_hgt_reverse_edge_ablation_runs_and_reduces_parameter_count():
    full_data = _make_arch_data()
    reduced_data = copy.deepcopy(full_data)
    del reduced_data['disease', 'rev_associated_with', 'chemical']
    del reduced_data['gene', 'rev_affects', 'chemical']
    del reduced_data['gene', 'rev_targets', 'disease']

    model_full = HGTPredictor(
        num_nodes_dict={ntype: full_data[ntype].num_nodes for ntype in full_data.node_types},
        metadata=full_data.metadata(),
        node_input_dims={ntype: int(full_data[ntype].x.size(1)) for ntype in full_data.node_types},
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        num_action_types=4,
        num_action_subjects=4,
    )
    model_reduced = HGTPredictor(
        num_nodes_dict={ntype: reduced_data[ntype].num_nodes for ntype in reduced_data.node_types},
        metadata=reduced_data.metadata(),
        node_input_dims={ntype: int(reduced_data[ntype].x.size(1)) for ntype in reduced_data.node_types},
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        num_action_types=4,
        num_action_subjects=4,
    )
    model_full.eval()
    model_reduced.eval()

    pos_edge = full_data['chemical', 'associated_with', 'disease'].edge_index
    neg_edge = torch.tensor([[2, 2], [0, 1]], dtype=torch.long)
    pos_full, neg_full = model_full(full_data, pos_edge, neg_edge)
    pos_reduced, neg_reduced = model_reduced(reduced_data, pos_edge, neg_edge)

    assert pos_full.shape == pos_reduced.shape == (2,)
    assert neg_full.shape == neg_reduced.shape == (2,)

    params_full = sum(p.numel() for p in model_full.parameters())
    params_reduced = sum(p.numel() for p in model_reduced.parameters())
    assert params_reduced < params_full


def test_create_model_from_data_infers_dense_node_inputs():
    data = _make_arch_data()
    model = create_model_from_data(
        data,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        num_action_types=4,
        num_action_subjects=4,
    )
    assert isinstance(model, HGTPredictor)
    assert set(model.node_proj.keys()) == set(data.node_types)
    assert len(model.node_emb) == 0


def test_infer_hgt_hparams_from_state_matches_constructed_model():
    data = _make_arch_data()
    model = HGTPredictor(
        num_nodes_dict={ntype: data[ntype].num_nodes for ntype in data.node_types},
        metadata=data.metadata(),
        node_input_dims={ntype: int(data[ntype].x.size(1)) for ntype in data.node_types},
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        num_action_types=4,
        num_action_subjects=4,
        num_pheno_action_types=3,
    )
    cfg = infer_hgt_hparams_from_state(model.state_dict())
    assert cfg['hidden_dim'] == 16
    assert cfg['num_layers'] == 2
    assert cfg['num_heads'] == 4
    assert cfg['num_action_types'] == 4
    assert cfg['num_action_subjects'] == 4
    assert cfg['num_pheno_action_types'] == 3
    assert cfg['node_input_dims'] == {'chemical': 4, 'disease': 3, 'gene': 5}


def test_generic_link_predictor_forward_and_attention():
    data = _make_arch_data()
    schema = GraphSchema(
        node_specs={
            'chemical': NodeInputSpec(mode='dense', in_dim=4),
            'disease': NodeInputSpec(mode='dense', in_dim=3),
            'gene': NodeInputSpec(mode='dense', in_dim=5),
        },
        edge_specs={
            ('chemical', 'affects', 'gene'): EdgeAttrSpec(
                kind='categorical',
                categorical_cardinalities=(4, 4),
            ),
            ('gene', 'rev_affects', 'chemical'): EdgeAttrSpec(
                kind='categorical',
                categorical_cardinalities=(4, 4),
            ),
            ('gene', 'rev_targets', 'disease'): EdgeAttrSpec(
                kind='continuous',
                continuous_dim=2,
            ),
            ('disease', 'targets', 'gene'): EdgeAttrSpec(
                kind='continuous',
                continuous_dim=2,
            ),
        },
    )
    model = GenericLinkPredictor(
        schema=schema,
        metadata=data.metadata(),
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        relation_types=[('chemical', 'associated_with', 'disease')],
    )
    model.eval()

    edge_attr_dict = {
        et: data[et].edge_attr
        for et in data.edge_types
        if hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None
    }
    z, attn = model.encode(
        data.x_dict,
        data.edge_index_dict,
        edge_attr_dict,
        return_attention=True,
    )
    assert set(z.keys()) == {'chemical', 'disease', 'gene'}
    assert len(attn) == 2

    pos_edge = data['chemical', 'associated_with', 'disease'].edge_index
    neg_edge = torch.tensor([[2, 2], [0, 1]], dtype=torch.long)
    pos_logits, neg_logits = model(
        batch_data=data,
        pos_edge_idx=pos_edge,
        neg_edge_idx=neg_edge,
        target_edge_type=('chemical', 'associated_with', 'disease'),
    )
    assert pos_logits.shape == (2,)
    assert neg_logits.shape == (2,)


def test_infer_schema_from_data_detects_dense_and_edge_attr_kinds():
    data = _make_arch_data()
    schema = infer_schema_from_data(data)
    assert schema.node_specs['chemical'].mode == 'dense'
    assert schema.edge_specs[('chemical', 'affects', 'gene')].kind == 'categorical'
    assert schema.edge_specs[('gene', 'rev_targets', 'disease')].kind == 'continuous'
    assert schema.edge_specs[('chemical', 'associated_with', 'disease')].kind == 'none'


def test_generic_predictor_raises_on_continuous_attr_dim_mismatch():
    data = _make_arch_data()
    bad_data = copy.deepcopy(data)
    bad_data['gene', 'rev_targets', 'disease'].edge_attr = torch.tensor(
        [[1.0], [2.0]], dtype=torch.float32
    )

    schema = GraphSchema(
        node_specs={
            'chemical': NodeInputSpec(mode='dense', in_dim=4),
            'disease': NodeInputSpec(mode='dense', in_dim=3),
            'gene': NodeInputSpec(mode='dense', in_dim=5),
        },
        edge_specs={
            ('gene', 'rev_targets', 'disease'): EdgeAttrSpec(kind='continuous', continuous_dim=2),
        },
    )
    model = GenericLinkPredictor(
        schema=schema,
        metadata=bad_data.metadata(),
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        relation_types=[('chemical', 'associated_with', 'disease')],
    )

    edge_attr_dict = {
        et: bad_data[et].edge_attr
        for et in bad_data.edge_types
        if hasattr(bad_data[et], 'edge_attr') and bad_data[et].edge_attr is not None
    }
    with pytest.raises(ValueError, match='continuous edge_attr dim mismatch'):
        model.encode(bad_data.x_dict, bad_data.edge_index_dict, edge_attr_dict)
