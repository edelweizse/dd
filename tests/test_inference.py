import polars as pl
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from src.models.inference.cached_embeddings import EmbeddingCachePredictor
from src.models.inference.full_graph import ChemDiseasePredictor


def _make_metadata_frames():
    disease_df = pl.DataFrame(
        {
            'DS_ID': [0, 1],
            'DS_OMIM_MESH_ID': ['D0', 'D1'],
            'DS_NAME': ['disease-0', 'disease-1'],
        }
    )
    chemical_df = pl.DataFrame(
        {
            'CHEM_ID': [0, 1, 2],
            'CHEM_MESH_ID': ['C0', 'C1', 'C2'],
            'CHEM_NAME': ['chem-0', 'chem-1', 'chem-2'],
        }
    )
    chem_disease_df = pl.DataFrame({'CHEM_ID': [0], 'DS_ID': [0]})
    return disease_df, chemical_df, chem_disease_df


def _make_inference_data() -> HeteroData:
    data = HeteroData()
    data['chemical'].x = torch.arange(3)
    data['disease'].x = torch.arange(2)
    data['chemical'].num_nodes = 3
    data['disease'].num_nodes = 2
    data['chemical', 'associated_with', 'disease'].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    return data


class _DummyModel(nn.Module):
    def __init__(self, embeddings, w_cd):
        super().__init__()
        self._embeddings = {k: v.clone() for k, v in embeddings.items()}
        self.W_cd = nn.Parameter(w_cd.clone())

    def encode(self, *_args, return_attention=False, **_kwargs):
        if return_attention:
            return self._embeddings, [{}]
        return self._embeddings


def test_full_graph_predictor_pair_and_topk():
    disease_df, chemical_df, _ = _make_metadata_frames()
    data = _make_inference_data()
    embeddings = {
        'chemical': torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        'disease': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    model = _DummyModel(embeddings=embeddings, w_cd=torch.eye(2))

    predictor = ChemDiseasePredictor(
        model=model,
        data=data,
        disease_df=disease_df,
        chemical_df=chemical_df,
        device=torch.device('cpu'),
    )

    pair = predictor.predict_pair('D0', 'C0')
    assert pair['known'] is True
    assert pair['probability'] > 0.7  # sigmoid(1.0)

    top = predictor.predict_chemicals_for_disease('D0', top_k=2, exclude_known=True)
    assert top.height == 2
    assert top['chemical_id'][0] == 'C1'


def test_full_graph_predictor_unknown_ids_raise():
    disease_df, chemical_df, _ = _make_metadata_frames()
    data = _make_inference_data()
    embeddings = {
        'chemical': torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        'disease': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    predictor = ChemDiseasePredictor(
        model=_DummyModel(embeddings=embeddings, w_cd=torch.eye(2)),
        data=data,
        disease_df=disease_df,
        chemical_df=chemical_df,
        device=torch.device('cpu'),
    )

    with pytest.raises(ValueError, match='Unknown disease ID'):
        predictor.predict_pair('DOES_NOT_EXIST', 'C0')
    with pytest.raises(ValueError, match='Unknown chemical ID'):
        predictor.predict_pair('D0', 'NOPE')


def test_cached_embedding_predictor_pair_and_topk():
    disease_df, chemical_df, _ = _make_metadata_frames()
    predictor = EmbeddingCachePredictor(
        W_cd=torch.eye(2),
        z_chem=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        z_dis=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        disease_df=disease_df,
        chemical_df=chemical_df,
        known_links={(0, 0)},
        device=torch.device('cpu'),
    )

    pair = predictor.predict_pair('D0', 'C0')
    assert pair['known'] is True
    assert pair['probability'] > 0.7

    top = predictor.predict_diseases_for_chemical('C0', top_k=1, exclude_known=True)
    assert top.height == 1
    assert top['disease_id'][0] == 'D1'


def test_cached_embedding_roundtrip_compute_and_load(tmp_path):
    disease_df, chemical_df, chem_disease_df = _make_metadata_frames()
    data = _make_inference_data()
    embeddings = {
        'chemical': torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        'disease': torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    model = _DummyModel(embeddings=embeddings, w_cd=torch.eye(2))

    EmbeddingCachePredictor.compute_and_save_embeddings(
        model=model,
        data=data,
        output_dir=str(tmp_path),
        device=torch.device('cpu'),
    )
    predictor = EmbeddingCachePredictor.from_cache(
        cache_dir=str(tmp_path),
        disease_df=disease_df,
        chemical_df=chemical_df,
        chem_disease_df=chem_disease_df,
        device=torch.device('cpu'),
    )

    assert predictor.is_known_link(0, 0)
    assert (tmp_path / 'chemical_embeddings.npy').exists()
    assert (tmp_path / 'disease_embeddings.npy').exists()
    assert (tmp_path / 'W_cd.pt').exists()
