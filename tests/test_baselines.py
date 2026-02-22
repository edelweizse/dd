import polars as pl
import pytest
import torch
from torch_geometric.data import HeteroData

from src.data.splits import prepare_splits_and_loaders
from src.models.architectures.hgat import HGATPredictor
from src.models.baselines import (
    BASELINE_NAMES,
    ComparisonConfig,
    build_baseline,
    compare_main_and_baselines,
    evaluate_baseline,
    train_baseline,
)


def _make_tiny_hetero() -> HeteroData:
    data = HeteroData()

    data["chemical"].num_nodes = 4
    data["disease"].num_nodes = 3
    data["gene"].num_nodes = 3
    data["chemical"].x = torch.arange(4, dtype=torch.long)
    data["disease"].x = torch.arange(3, dtype=torch.long)
    data["gene"].x = torch.arange(3, dtype=torch.long)

    cd = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 2, 1, 2, 0, 1]],
        dtype=torch.long,
    )
    data["chemical", "associated_with", "disease"].edge_index = cd
    data["disease", "rev_associated_with", "chemical"].edge_index = torch.flip(cd, dims=[0])

    cg = torch.tensor([[0, 1, 1, 2, 3], [0, 1, 2, 1, 2]], dtype=torch.long)
    cg_attr = torch.tensor([[0, 1], [1, 1], [2, 0], [1, 2], [0, 2]], dtype=torch.long)
    data["chemical", "affects", "gene"].edge_index = cg
    data["chemical", "affects", "gene"].edge_attr = cg_attr
    data["gene", "rev_affects", "chemical"].edge_index = torch.flip(cg, dims=[0])
    data["gene", "rev_affects", "chemical"].edge_attr = cg_attr.clone()

    dg = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 2]], dtype=torch.long)
    dg_attr = torch.tensor([[1.0, 0.2], [0.2, 1.0], [1.5, 0.4], [0.6, 1.1]], dtype=torch.float32)
    data["disease", "targets", "gene"].edge_index = dg
    data["disease", "targets", "gene"].edge_attr = dg_attr
    data["gene", "rev_targets", "disease"].edge_index = torch.flip(dg, dims=[0])
    data["gene", "rev_targets", "disease"].edge_attr = dg_attr.clone()

    gg = torch.tensor([[0, 1, 2, 2], [1, 2, 0, 1]], dtype=torch.long)
    data["gene", "interacts_with", "gene"].edge_index = gg
    data["gene", "rev_interacts_with", "gene"].edge_index = torch.flip(gg, dims=[0])

    return data


@pytest.fixture()
def tiny_artifacts():
    data = _make_tiny_hetero()
    arts = prepare_splits_and_loaders(
        data_full=data,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=7,
        split_strategy="random",
        enforce_train_node_coverage=False,
        batch_size=4,
        num_neighbours=[2, 1],
    )
    return data, arts


@pytest.mark.parametrize("baseline_name", BASELINE_NAMES)
def test_baseline_build_train_and_eval_runs(tiny_artifacts, baseline_name):
    data, arts = tiny_artifacts
    model = build_baseline(
        baseline_name,
        data_train=arts.data_train,
        split_train_pos=arts.split.train_pos,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        device=torch.device("cpu"),
    )
    train_baseline(
        model,
        arts=arts,
        device=torch.device("cpu"),
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        num_neg_train=2,
    )
    val_metrics = evaluate_baseline(
        model,
        loader=arts.val_loader,
        known_pos=arts.known_pos_val,
        device=torch.device("cpu"),
        num_neg_eval=2,
    )
    assert "auprc" in val_metrics
    assert "auroc" in val_metrics
    assert int(val_metrics["n_pos"]) == int(arts.split.val_pos.size(1))


def test_compare_main_and_baselines_smoke(tiny_artifacts, tmp_path):
    data, arts = tiny_artifacts
    vocabs = {
        "action_type": pl.DataFrame({"ACTION_TYPE": ["a", "b", "c"]}),
        "action_subject": pl.DataFrame({"ACTION_SUBJECT": ["x", "y", "z"]}),
    }

    model = HGATPredictor(
        num_nodes_dict={nt: int(data[nt].num_nodes) for nt in data.node_types},
        metadata=arts.data_train.metadata(),
        node_input_dims={},
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        num_action_types=vocabs["action_type"].height,
        num_action_subjects=vocabs["action_subject"].height,
    )
    ckpt_path = tmp_path / "main.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    cfg = ComparisonConfig(
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        num_neg_train=2,
        num_neg_eval=2,
        ks=(5, 10),
    )
    results = compare_main_and_baselines(
        checkpoint_path=str(ckpt_path),
        data_full=data,
        vocabs=vocabs,
        arts=arts,
        baseline_names=["degree", "mf", "generic_hgat"],
        device=torch.device("cpu"),
        config=cfg,
    )
    assert "main_hgat" in results
    assert "degree" in results
    assert "mf" in results
    assert "generic_hgat" in results
    assert "val_auprc" in results["main_hgat"]
