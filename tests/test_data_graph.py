import polars as pl
import pytest
import torch

from src.data.graph import (
    _assert_correct_ids,
    build_graph_from_processed,
    build_hetero_data,
    get_graph_summary,
)


def _base_graph_frames():
    cnodes = pl.DataFrame(
        {"CHEM_ID": [0, 1], "CHEM_MESH_ID": ["C1", "C2"], "CHEM_NAME": ["c1", "c2"]}
    ).with_columns(pl.col("CHEM_ID").cast(pl.UInt32))
    dnodes = pl.DataFrame(
        {"DS_ID": [0, 1], "DS_OMIM_MESH_ID": ["D1", "D2"], "DS_NAME": ["d1", "d2"]}
    ).with_columns(pl.col("DS_ID").cast(pl.UInt32))
    gnodes = pl.DataFrame({"GENE_ID": [0, 1], "GENE_NCBI_ID": [11, 12]}).with_columns(
        pl.col("GENE_ID").cast(pl.UInt32)
    )

    cd = pl.DataFrame({"CHEM_ID": [0], "DS_ID": [0]})
    cg = pl.DataFrame(
        {
            "CHEM_ID": [0, 1],
            "GENE_ID": [0, 1],
            "ACTION_TYPE": ["increases", "decreases"],
            "ACTION_SUBJECT": ["expression", "activity"],
            "LOG_PUBMED_COUNT": [1.0, 2.0],
        }
    )
    dg = pl.DataFrame(
        {
            "GENE_ID": [0],
            "DS_ID": [0],
            "DIRECT_EVIDENCE_TYPE": [1],
            "LOG_PUBMED_COUNT": [1.0],
        }
    )
    ppi = pl.DataFrame({"GENE_ID_1": [0], "GENE_ID_2": [1]})
    return cnodes, dnodes, gnodes, cd, cg, dg, ppi


def test_assert_correct_ids_accepts_unsigned_contiguous_ids():
    nodes = pl.DataFrame({"CHEM_ID": [0, 1, 2]}).with_columns(pl.col("CHEM_ID").cast(pl.UInt32))
    assert _assert_correct_ids(nodes, "CHEM_ID") == 3


def test_assert_correct_ids_rejects_non_unsigned_dtype():
    nodes = pl.DataFrame({"CHEM_ID": [0, 1, 2]}).with_columns(pl.col("CHEM_ID").cast(pl.Int64))
    with pytest.raises(TypeError):
        _assert_correct_ids(nodes, "CHEM_ID")


def test_build_hetero_data_creates_core_and_reverse_edges():
    cnodes, dnodes, gnodes, cd, cg, dg, ppi = _base_graph_frames()
    data, vocabs = build_hetero_data(
        cnodes=cnodes,
        dnodes=dnodes,
        gnodes=gnodes,
        cd=cd,
        cg=cg,
        dg=dg,
        ppi=ppi,
        add_reverse_edges=True,
    )

    assert data["chemical"].num_nodes == 2
    assert data["disease"].num_nodes == 2
    assert data["gene"].num_nodes == 2
    assert ("disease", "rev_associated_with", "chemical") in data.edge_types
    assert ("gene", "rev_affects", "chemical") in data.edge_types
    assert data["chemical", "affects", "gene"].edge_attr.shape[1] == 2
    assert vocabs["action_type"].height == 2
    assert vocabs["action_subject"].height == 2


def test_build_hetero_data_with_extended_nodes_and_attrs():
    cnodes, dnodes, gnodes, cd, cg, dg, ppi = _base_graph_frames()
    pathway_nodes = pl.DataFrame({"PATHWAY_ID": [0], "PATHWAY_SOURCE_ID": ["P1"], "PATHWAY_NAME": ["p"]}).with_columns(
        pl.col("PATHWAY_ID").cast(pl.UInt32)
    )
    go_term_nodes = pl.DataFrame(
        {"GO_TERM_ID": [0], "GO_SOURCE_ID": ["GO:1"], "GO_NAME": ["g1"], "GO_ONTOLOGY": ["Biological Process"]}
    ).with_columns(pl.col("GO_TERM_ID").cast(pl.UInt32))
    gene_pathway = pl.DataFrame({"GENE_ID": [0], "PATHWAY_ID": [0]})
    disease_pathway = pl.DataFrame({"DS_ID": [0], "PATHWAY_ID": [0], "INFERENCE_GENE_COUNT": [2]})
    chem_pathway = pl.DataFrame(
        {"CHEM_ID": [0], "PATHWAY_ID": [0], "NEG_LOG_PVALUE": [2.0], "TARGET_RATIO": [0.2], "FOLD_ENRICHMENT": [3.0]}
    )
    chem_go = pl.DataFrame(
        {
            "CHEM_ID": [0],
            "GO_TERM_ID": [0],
            "NEG_LOG_PVALUE": [2.0],
            "TARGET_RATIO": [0.2],
            "FOLD_ENRICHMENT": [3.0],
            "ONTOLOGY_TYPE": [0],
            "GO_LEVEL_NORM": [0.5],
        }
    )
    chem_pheno = pl.DataFrame({"CHEM_ID": [0], "GO_TERM_ID": [0], "PHENO_ACTION_TYPE": [1]})
    go_disease = pl.DataFrame(
        {"GO_TERM_ID": [0], "DS_ID": [0], "ONTOLOGY_TYPE": [0], "LOG_INFERENCE_CHEM": [1.0], "LOG_INFERENCE_GENE": [1.0]}
    )

    data, _ = build_hetero_data(
        cnodes=cnodes,
        dnodes=dnodes,
        gnodes=gnodes,
        cd=cd,
        cg=cg,
        dg=dg,
        ppi=ppi,
        pathway_nodes=pathway_nodes,
        go_term_nodes=go_term_nodes,
        gene_pathway=gene_pathway,
        disease_pathway=disease_pathway,
        chem_pathway=chem_pathway,
        chem_go=chem_go,
        chem_pheno=chem_pheno,
        go_disease=go_disease,
        add_reverse_edges=True,
    )

    assert data["pathway"].num_nodes == 1
    assert data["go_term"].num_nodes == 1
    assert ("pathway", "rev_disrupts", "disease") in data.edge_types
    assert ("go_term", "rev_enriched_in", "chemical") in data.edge_types
    assert hasattr(data["go_term"], "ontology_type")


def test_get_graph_summary_has_totals():
    cnodes, dnodes, gnodes, cd, cg, dg, ppi = _base_graph_frames()
    data, _ = build_hetero_data(
        cnodes=cnodes,
        dnodes=dnodes,
        gnodes=gnodes,
        cd=cd,
        cg=cg,
        dg=dg,
        ppi=ppi,
        add_reverse_edges=False,
    )
    summary = get_graph_summary(data)
    assert summary["total_nodes"] == 6
    assert summary["total_edges"] > 0
    assert "chemical__associated_with__disease" in summary["edge_types"]


def test_build_graph_from_processed_loads_and_writes_vocab_csv(tmp_path):
    cnodes, dnodes, gnodes, cd, cg, dg, ppi = _base_graph_frames()

    cnodes.write_parquet(tmp_path / "chemicals_nodes.parquet")
    dnodes.write_parquet(tmp_path / "diseases_nodes.parquet")
    gnodes.write_parquet(tmp_path / "genes_nodes.parquet")
    cd.with_row_index("CHEM_DS_IDX").write_parquet(tmp_path / "chem_disease_edges.parquet")
    cg.with_row_index("CHEM_GENE_IDX").write_parquet(tmp_path / "chem_gene_edges.parquet")
    dg.with_row_index("GENE_DS_IDX").write_parquet(tmp_path / "disease_gene_edges.parquet")
    ppi.with_row_index("PPI_IDX").write_parquet(tmp_path / "ppi_edges.parquet")
    ppi.with_row_index("PPI_DIR_IDX").write_parquet(tmp_path / "ppi_directed_edges.parquet")

    data, vocabs = build_graph_from_processed(
        processed_data_dir=str(tmp_path),
        add_reverse_edges=True,
        save_vocabs=True,
        include_extended=False,
    )

    assert data["chemical"].num_nodes == 2
    assert vocabs["action_type"].height == 2
    assert (tmp_path / "vocab_action_type.csv").exists()
    assert (tmp_path / "vocab_action_subject.csv").exists()
    assert torch.equal(
        data["disease", "rev_associated_with", "chemical"].edge_index,
        torch.flip(data["chemical", "associated_with", "disease"].edge_index, dims=[0]),
    )
