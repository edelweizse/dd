import pytest
import torch
from torch_geometric.data import HeteroData

from src.data.splits import (
    LinkSplit,
    PackedPairFilter,
    _compute_split_counts,
    make_split_graph,
    negative_sample_cd_batch_local,
    prepare_splits_and_loaders,
    save_split_artifact,
    load_split_artifact,
    split_cd,
    validate_split_artifact_compatibility,
)


def _tiny_full_graph() -> HeteroData:
    data = HeteroData()
    data["chemical"].num_nodes = 6
    data["disease"].num_nodes = 5
    data["gene"].num_nodes = 4

    data["chemical"].node_id = torch.arange(6).view(-1, 1)
    data["disease"].node_id = torch.arange(5).view(-1, 1)
    data["gene"].node_id = torch.arange(4).view(-1, 1)
    data["chemical"].x = data["chemical"].node_id.clone()
    data["disease"].x = data["disease"].node_id.clone()
    data["gene"].x = data["gene"].node_id.clone()

    cd = torch.tensor(
        [
            [0, 0, 1, 2, 3, 4, 5, 5, 4, 3],
            [0, 1, 1, 2, 2, 3, 4, 0, 4, 1],
        ],
        dtype=torch.long,
    )
    data["chemical", "associated_with", "disease"].edge_index = cd
    data["disease", "rev_associated_with", "chemical"].edge_index = torch.flip(cd, dims=[0])

    # minimal additional relations for heterogeneous graph validity
    data["chemical", "affects", "gene"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data["disease", "targets", "gene"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data["gene", "interacts_with", "gene"].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return data


def test_compute_split_counts_rounds_and_keeps_non_empty():
    n_train, n_val, n_test = _compute_split_counts(11, val_ratio=0.2, test_ratio=0.2)
    assert n_train + n_val + n_test == 11
    assert n_train > 0 and n_val > 0 and n_test > 0


def test_split_cd_is_deterministic_with_seed():
    cd_idx = torch.tensor(
        [[0, 0, 1, 2, 2, 3, 4, 5], [0, 1, 1, 2, 3, 4, 0, 1]],
        dtype=torch.long,
    )
    s1 = split_cd(cd_idx, val_ratio=0.25, test_ratio=0.25, seed=123, stratify=True)
    s2 = split_cd(cd_idx, val_ratio=0.25, test_ratio=0.25, seed=123, stratify=True)
    assert torch.equal(s1.train_pos, s2.train_pos)
    assert torch.equal(s1.val_pos, s2.val_pos)
    assert torch.equal(s1.test_pos, s2.test_pos)


def test_split_cd_can_return_realized_strategy():
    cd_idx = torch.tensor(
        [[0, 0, 1, 2, 2, 3, 4, 5], [0, 1, 1, 2, 3, 4, 0, 1]],
        dtype=torch.long,
    )
    split, strategy = split_cd(
        cd_idx,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=123,
        stratify=True,
        return_strategy=True,
    )
    assert isinstance(split, LinkSplit)
    assert strategy in {"stratified", "random"}


def test_split_cd_enforces_train_node_coverage_when_feasible():
    cd_idx = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    split = split_cd(
        cd_idx,
        val_ratio=2.0 / 12.0,
        test_ratio=2.0 / 12.0,
        seed=3,
        stratify=False,
        enforce_train_node_coverage=True,
    )
    train_chem = int(split.train_pos[0].unique().numel())
    train_dis = int(split.train_pos[1].unique().numel())
    assert train_chem == int(cd_idx[0].unique().numel())
    assert train_dis == int(cd_idx[1].unique().numel())
    assert split.train_pos.size(1) == 8
    assert split.val_pos.size(1) == 2
    assert split.test_pos.size(1) == 2


def test_make_split_graph_replaces_forward_and_reverse_cd_edges():
    data = _tiny_full_graph()
    train_cd = data["chemical", "associated_with", "disease"].edge_index[:, :4]
    out = make_split_graph(data, train_cd)
    assert torch.equal(out["chemical", "associated_with", "disease"].edge_index, train_cd)
    assert torch.equal(
        out["disease", "rev_associated_with", "chemical"].edge_index,
        torch.flip(train_cd, dims=[0]),
    )


def test_split_artifact_roundtrip_and_compatibility(tmp_path):
    data = _tiny_full_graph()
    cd_all = data["chemical", "associated_with", "disease"].edge_index
    split = split_cd(cd_all, val_ratio=0.2, test_ratio=0.2, seed=7, stratify=False)
    path = tmp_path / "split.pt"

    save_split_artifact(
        artifact_path=path,
        split=split,
        data_full=data,
        seed=7,
        val_ratio=0.2,
        test_ratio=0.2,
        split_strategy="random",
        stratify_bins=8,
    )
    loaded_split, meta = load_split_artifact(path)
    validate_split_artifact_compatibility(loaded_split, meta, data)

    assert meta["seed"] == 7
    assert meta["split_strategy"] == "random"
    assert torch.equal(split.train_pos, loaded_split.train_pos)


def test_validate_split_artifact_detects_incompatible_graph():
    data = _tiny_full_graph()
    cd_all = data["chemical", "associated_with", "disease"].edge_index
    split = split_cd(cd_all, val_ratio=0.2, test_ratio=0.2, seed=9, stratify=False)
    metadata = {
        "num_chemical_nodes": int(data["chemical"].num_nodes),
        "num_disease_nodes": int(data["disease"].num_nodes),
        "num_cd_edges": int(cd_all.size(1)),
    }

    bad_data = data.clone()
    bad_data["chemical"].num_nodes = data["chemical"].num_nodes + 1
    with pytest.raises(ValueError):
        validate_split_artifact_compatibility(split, metadata, bad_data)


def test_validate_split_artifact_detects_relation_mismatch():
    data = _tiny_full_graph()
    cd_all = data["chemical", "associated_with", "disease"].edge_index
    split = split_cd(cd_all, val_ratio=0.2, test_ratio=0.2, seed=9, stratify=False)
    metadata = {
        "num_chemical_nodes": int(data["chemical"].num_nodes),
        "num_disease_nodes": int(data["disease"].num_nodes),
        "num_cd_edges": int(cd_all.size(1)),
        "cd_relation": ["chemical", "wrong_rel", "disease"],
    }
    with pytest.raises(ValueError):
        validate_split_artifact_compatibility(split, metadata, data)


def test_negative_sampling_is_deterministic_and_avoids_known_positives():
    batch = HeteroData()
    batch["chemical"].num_nodes = 4
    batch["disease"].num_nodes = 4
    batch["chemical"].node_id = torch.arange(4).view(-1, 1)
    batch["disease"].node_id = torch.arange(4).view(-1, 1)
    batch["chemical"].x = batch["chemical"].node_id.clone()
    batch["disease"].x = batch["disease"].node_id.clone()
    batch["chemical", "associated_with", "disease"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )

    pos_local = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    known_pos = PackedPairFilter(torch.tensor([[0, 1], [0, 1]], dtype=torch.long), num_dis=4)

    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    neg1 = negative_sample_cd_batch_local(
        batch_data=batch,
        pos_edge_index_local=pos_local,
        known_pos=known_pos,
        num_neg_per_pos=2,
        generator=g1,
    )
    neg2 = negative_sample_cd_batch_local(
        batch_data=batch,
        pos_edge_index_local=pos_local,
        known_pos=known_pos,
        num_neg_per_pos=2,
        generator=g2,
    )

    assert torch.equal(neg1, neg2)
    coll = known_pos.contains_mask_cpu(neg1[0], neg1[1])
    assert not bool(coll.any())


def test_negative_sampling_raises_when_no_valid_negative_exists():
    batch = HeteroData()
    batch["chemical"].num_nodes = 1
    batch["disease"].num_nodes = 1
    batch["chemical"].node_id = torch.tensor([[0]])
    batch["disease"].node_id = torch.tensor([[0]])
    batch["chemical"].x = batch["chemical"].node_id.clone()
    batch["disease"].x = batch["disease"].node_id.clone()
    batch["chemical", "associated_with", "disease"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    pos_local = torch.tensor([[0], [0]], dtype=torch.long)
    known_pos = PackedPairFilter(torch.tensor([[0], [0]], dtype=torch.long), num_dis=1)
    with pytest.raises(RuntimeError):
        negative_sample_cd_batch_local(
            batch_data=batch,
            pos_edge_index_local=pos_local,
            known_pos=known_pos,
            num_neg_per_pos=1,
            max_tries=3,
            generator=torch.Generator().manual_seed(1),
        )


def test_prepare_splits_and_loaders_builds_phase_specific_positive_sets():
    data = _tiny_full_graph()
    arts = prepare_splits_and_loaders(
        data_full=data,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        split_strategy="stratified",
        stratify_bins=4,
        batch_size=4,
        num_neighbours=[2, 2],
    )

    assert arts.split.train_pos.size(1) + arts.split.val_pos.size(1) + arts.split.test_pos.size(1) == \
        data["chemical", "associated_with", "disease"].edge_index.size(1)
    assert arts.known_pos_train._set.issubset(arts.known_pos_val._set)
    assert arts.known_pos_val._set.issubset(arts.known_pos_test._set)
