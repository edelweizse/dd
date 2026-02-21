import numpy as np
import torch

from src.data.feature_encoders import (
    BooleanFieldEncoder,
    CategoryOneHotEncoder,
    FeatureEncoderPipeline,
    ListHashingEncoder,
    MultiCategoryEncoder,
    NumericFieldEncoder,
    TextHashingEncoder,
    UrlStatsEncoder,
    build_current_kg_node_encoder,
    build_default_metadata_encoder,
)


def test_numeric_encoder_standardize_and_log1p():
    records = [
        {"score": 0},
        {"score": 9},
        {"score": None},
        {"score": "24"},
    ]
    enc = NumericFieldEncoder(field="score", log1p=True, standardize=True, fill_value=0.0).fit(records)
    out = enc.transform(records)
    assert out.shape == (4, 1)
    assert np.isfinite(out).all()
    assert abs(float(out.mean())) < 1e-6


def test_category_one_hot_unknown_bucket():
    train = [{"kind": "A"}, {"kind": "B"}, {"kind": "A"}]
    test = [{"kind": "B"}, {"kind": "C"}, {"kind": None}]
    enc = CategoryOneHotEncoder(field="kind", add_unknown=True).fit(train)
    out = enc.transform(test)

    assert out.shape == (3, 3)  # A, B, <unk>
    # B
    assert out[0, 1] == 1.0
    # C -> unk
    assert out[1, 2] == 1.0
    # missing -> all zero
    assert out[2].sum() == 0.0


def test_multicategory_encoder_binary_and_unknown():
    train = [
        {"xrefs": ["MESH:C1", "CHEBI:1"]},
        {"xrefs": ["MESH:C2"]},
    ]
    test = [
        {"xrefs": ["MESH:C1", "UNKNOWN:X"]},
        {"xrefs": []},
    ]
    enc = MultiCategoryEncoder(field="xrefs", add_unknown=True, binary=True).fit(train)
    out = enc.transform(test)

    assert out.shape[0] == 2
    # known + unknown should both fire on first row
    assert out[0].sum() == 2.0
    # empty list should stay zero
    assert out[1].sum() == 0.0


def test_text_hashing_encoder_is_deterministic():
    records = [
        {"synonyms": ["alpha beta", "beta gamma"]},
        {"synonyms": ["delta"]},
    ]
    enc = TextHashingEncoder(field="synonyms", n_features=64, ngram_min=1, ngram_max=2)
    a = enc.transform(records)
    b = enc.transform(records)
    assert a.shape == (2, 64)
    assert np.allclose(a, b)
    # l2-normalized rows (or zero)
    norms = np.linalg.norm(a, axis=1)
    assert np.all((np.isclose(norms, 1.0) | np.isclose(norms, 0.0)))


def test_list_hashing_encoder_pipe_delimited_ids():
    records = [
        {"DS_PARENT_IDS": "MESH:D011041|MESH:D014883"},
        {"DS_PARENT_IDS": ""},
    ]
    enc = ListHashingEncoder(field="DS_PARENT_IDS", n_features=32, delimiter="|")
    out = enc.transform(records)
    assert out.shape == (2, 32)
    assert np.isfinite(out).all()
    assert out[0].sum() > 0.0
    assert out[1].sum() == 0.0


def test_url_stats_encoder_parses_basic_features():
    recs = [
        {"ctd_url": "https://ctdbase.org/detail.go?type=chem&acc=C114385"},
        {"ctd_url": None},
    ]
    enc = UrlStatsEncoder(field="ctd_url", domain_buckets=8)
    out = enc.transform(recs)
    assert out.shape == (2, 15)
    # present + https + has_query for first row
    assert out[0, 0] == 1.0
    assert out[0, 1] == 1.0
    assert out[0, 2] == 1.0
    # missing URL row should be all zeros
    assert out[1].sum() == 0.0


def test_default_metadata_pipeline_with_user_like_records():
    records = [
        {
            "synonyms": [
                "001 C8 NBD",
                "H-MeTyr-Arg-MeArg-D-Leu-NH(CH2)8NH-NBD",
                "MeTyr-Arg-MeArg-Leu-NH-NBD",
            ],
            "xrefs": ["MESH:C114385"],
            "parentIDs": ["MESH:D009842", "MESH:D010069"],
            "ctd_url": "http://ctdbase.org/detail.go?type=chem&acc=C114385",
            "parentTreeNumbers": ["D03.383.129.462.580", "D12.644.456"],
            "treeNumbers": ["D03.383.129.462.580/C114385", "D12.644.456/C114385"],
            "ctd": "yes",
        },
        {
            "synonyms": ["water", "oxidane"],
            "xrefs": ["MESH:C000001"],
            "parentIDs": [],
            "ctd_url": "",
            "parentTreeNumbers": [],
            "treeNumbers": [],
            "ctd": "no",
        },
    ]

    pipe = build_default_metadata_encoder(
        synonyms_hash_dim=64,
        id_hash_dim=32,
        tree_hash_dim=32,
        url_domain_buckets=8,
    ).fit(records)

    x_np = pipe.transform_numpy(records)
    x_t = pipe.transform_tensor(records)
    names = pipe.feature_names()

    assert x_np.shape == (2, pipe.output_dim)
    assert x_np.shape[1] == len(names)
    assert np.isfinite(x_np).all()
    assert isinstance(x_t, torch.Tensor)
    assert x_t.shape == x_np.shape
    assert x_t.dtype == torch.float32


def test_pipeline_composes_multiple_encoders():
    records = [
        {"score": 1.0, "ctd": "yes", "kind": "a"},
        {"score": 3.0, "ctd": "no", "kind": "b"},
    ]
    pipe = FeatureEncoderPipeline(
        encoders=[
            NumericFieldEncoder(field="score", standardize=True),
            BooleanFieldEncoder(field="ctd"),
            CategoryOneHotEncoder(field="kind", add_unknown=False),
        ]
    ).fit(records)

    out = pipe.transform_numpy(records)
    assert out.shape == (2, 4)  # 1 numeric + 1 bool + 2 one-hot
    assert np.isfinite(out).all()


def test_current_kg_disease_encoder_with_sample_like_row():
    records = [
        {
            "DS_ID": 0,
            "DS_OMIM_MESH_ID": "MESH:D014869",
            "DS_NAME": "Water Intoxication",
            "DS_DEFINITION": "A condition resulting from the excessive retention of water with sodium depletion.",
            "DS_PARENT_IDS": "MESH:D011041|MESH:D014883",
            "DS_TREE_NUMBERS": "C18.452.950.932|C25.723.932",
            "DS_PARENT_TREE_NUMBERS": "C18.452.950|C25.723",
            "DS_SYNONYMS": "Hyperhydration|Overhydration",
            "DS_SLIM_MAPPINGS": "Metabolic disease|Poisoning",
        }
    ]
    enc = build_current_kg_node_encoder(
        "disease",
        text_hash_dim=64,
        id_hash_dim=32,
        tree_hash_dim=32,
        misc_hash_dim=32,
    ).fit(records)
    out = enc.transform_numpy(records)
    assert out.shape[0] == 1
    assert out.shape[1] == enc.output_dim
    assert np.isfinite(out).all()
    assert out.sum() > 0.0


def test_current_kg_chemical_encoder_with_sample_like_row():
    records = [
        {
            "CHEM_ID": 0,
            "CHEM_MESH_ID": "C072610",
            "CHEM_NAME": "adenosine-N(6)-methyl-propylthioether-N-pyridoxamine",
            "CHEM_DEFINITION": None,
            "CHEM_PARENT_IDS": "",
            "CHEM_TREE_NUMBERS": "",
            "CHEM_PARENT_TREE_NUMBERS": "",
            "CHEM_SYNONYMS": "",
        }
    ]
    enc = build_current_kg_node_encoder(
        "chemical",
        text_hash_dim=64,
        id_hash_dim=32,
        tree_hash_dim=32,
        misc_hash_dim=32,
    ).fit(records)
    out = enc.transform_numpy(records)
    assert out.shape == (1, enc.output_dim)
    assert np.isfinite(out).all()
    assert out.sum() > 0.0


def test_current_kg_gene_encoder_with_sample_like_row():
    records = [
        {
            "GENE_ID": 6574,
            "GENE_NCBI_ID": 6574,
            "GENE_SYMBOL": "SLC20A1",
            "GENE_NAME": "solute carrier family 20 member 1",
            "GENE_BIOGRID_IDS": "",
            "GENE_ALT_IDS": "",
            "GENE_SYNONYMS": "",
            "GENE_PHARMGKB_IDS": "",
            "GENE_UNIPROT_IDS": "",
        }
    ]
    enc = build_current_kg_node_encoder(
        "gene",
        text_hash_dim=64,
        id_hash_dim=32,
        tree_hash_dim=32,
        misc_hash_dim=32,
    ).fit(records)
    out = enc.transform_tensor(records)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, enc.output_dim)
    assert torch.isfinite(out).all()
    assert float(out.sum().item()) > 0.0
