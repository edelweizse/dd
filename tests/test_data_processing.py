import math

import polars as pl

from src.data.processing import (
    build_edge_tables,
    build_go_term_entities,
    build_interaction_type_hierarchy,
    build_pathway_entities,
    load_processed_data,
)


def test_build_interaction_type_hierarchy_maps_parent_ids():
    ixn_types = pl.DataFrame(
        {
            "IXN_TYPE_NAME": ["root", "child", "orphan"],
            "IXN_TYPE_CODE": ["A", "B", "C"],
            "IXN_TYPE_DESC": ["root desc", "child desc", "orphan desc"],
            "IXN_TYPE_PARENT_CODE": [None, "A", "Z"],
        }
    ).lazy()

    out = build_interaction_type_hierarchy(ixn_types)
    rows = out.sort("IXN_TYPE_CODE").to_dicts()

    assert rows[0]["IXN_TYPE_CODE"] == "A"
    assert rows[0]["IXN_TYPE_PARENT_ID"] is None
    assert rows[1]["IXN_TYPE_CODE"] == "B"
    assert rows[1]["IXN_TYPE_PARENT_ID"] == rows[0]["IXN_TYPE_ID"]
    assert rows[2]["IXN_TYPE_CODE"] == "C"
    assert rows[2]["IXN_TYPE_PARENT_ID"] == -1


def test_build_pathway_entities_keeps_only_multi_source_pathways():
    genes_core = pl.DataFrame({"GENE_NCBI_ID": [11], "GENE_ID": [0]})
    ds_core = pl.DataFrame({"DS_OMIM_MESH_ID": ["D1"], "DS_ID": [0]})
    chems_core = pl.DataFrame({"CHEM_MESH_ID": ["C1"], "CHEM_ID": [0]})

    gene_pathways = pl.DataFrame(
        {
            "GENE_NCBI_ID": [11, 11],
            "PATHWAY_ID": ["P_SHARED", "P_GENE_ONLY"],
            "PATHWAY_NAME": ["shared", "gene only"],
        }
    ).lazy()
    disease_pathways = pl.DataFrame(
        {
            "DS_OMIM_MESH_ID": ["D1"],
            "PATHWAY_ID": ["P_SHARED"],
            "PATHWAY_NAME": ["shared"],
            "INFERENCE_GENE_SYMBOL": ["G"],
        }
    ).lazy()
    chem_pathways = pl.DataFrame(
        {
            "CHEM_MESH_ID": ["C1", "C1"],
            "PATHWAY_ID": ["P_SHARED", "P_CHEM_ONLY"],
            "PATHWAY_NAME": ["shared", "chem only"],
        }
    ).lazy()

    out = build_pathway_entities(
        gene_pathways=gene_pathways,
        disease_pathways=disease_pathways,
        chem_pathways=chem_pathways,
        genes_core=genes_core,
        ds_core=ds_core,
        chems_core=chems_core,
    )

    assert out["PATHWAY_SOURCE_ID"].to_list() == ["P_SHARED"]


def test_build_go_term_entities_requires_chemical_and_disease_links():
    chems_core = pl.DataFrame({"CHEM_MESH_ID": ["C1"], "CHEM_ID": [0]})
    ds_core = pl.DataFrame({"DS_OMIM_MESH_ID": ["D1"], "DS_ID": [0]})

    chem_go = pl.DataFrame(
        {
            "CHEM_MESH_ID": ["C1", "C1"],
            "GO_ID": ["GO_SHARED", "GO_CHEM_ONLY"],
            "GO_NAME": ["shared", "chem only"],
            "GO_ONTOLOGY": ["Biological Process", "Biological Process"],
        }
    ).lazy()
    pheno_ixns = pl.DataFrame(
        {
            "CHEM_MESH_ID": ["C1"],
            "GO_ID": ["GO_SHARED"],
            "GO_NAME": ["shared"],
        }
    ).lazy()
    pheno_disease_bp = pl.DataFrame(
        {
            "GO_ID": ["GO_SHARED", "GO_DISEASE_ONLY"],
            "GO_NAME": ["shared", "disease only"],
            "GO_ONTOLOGY": ["Biological Process", "Biological Process"],
            "DS_OMIM_MESH_ID": ["D1", "D1"],
        }
    ).lazy()
    pheno_disease_mf = pl.DataFrame(
        {"GO_ID": [], "GO_NAME": [], "GO_ONTOLOGY": [], "DS_OMIM_MESH_ID": []},
        schema={
            "GO_ID": pl.String,
            "GO_NAME": pl.String,
            "GO_ONTOLOGY": pl.String,
            "DS_OMIM_MESH_ID": pl.String,
        },
    ).lazy()
    pheno_disease_cc = pl.DataFrame(
        {"GO_ID": [], "GO_NAME": [], "GO_ONTOLOGY": [], "DS_OMIM_MESH_ID": []},
        schema={
            "GO_ID": pl.String,
            "GO_NAME": pl.String,
            "GO_ONTOLOGY": pl.String,
            "DS_OMIM_MESH_ID": pl.String,
        },
    ).lazy()

    out = build_go_term_entities(
        chem_go=chem_go,
        pheno_ixns=pheno_ixns,
        pheno_disease_bp=pheno_disease_bp,
        pheno_disease_mf=pheno_disease_mf,
        pheno_disease_cc=pheno_disease_cc,
        chems_core=chems_core,
        ds_core=ds_core,
    )

    assert out["GO_SOURCE_ID"].to_list() == ["GO_SHARED"]


def test_build_edge_tables_parses_interactions_and_deduplicates_ppi():
    genes_core = pl.DataFrame({"GENE_NCBI_ID": [11, 12], "GENE_ID": [0, 1]})
    ds_core = pl.DataFrame({"DS_OMIM_MESH_ID": ["D1"], "DS_ID": [0]})
    chems_core = pl.DataFrame({"CHEM_MESH_ID": ["C1"], "CHEM_ID": [0]})

    ctd_chg = pl.DataFrame(
        {
            "CHEM_MESH_ID": ["C1", "C1"],
            "CHEM_NAME": ["chem", "chem"],
            "GENE_NCBI_ID": [11, 11],
            "INTERACTIONS": ["increases^expression|decreases^activity", "affects^binding"],
            "PUBMED_IDS": ["1|2|3", "1"],
        }
    ).lazy()
    ctd_chd = pl.DataFrame(
        {"CHEM_MESH_ID": ["C1"], "DS_OMIM_MESH_ID": ["D1"], "DS_NAME": ["disease"]}
    ).lazy()
    ctd_dg = pl.DataFrame(
        {
            "GENE_NCBI_ID": [11],
            "DS_OMIM_MESH_ID": ["D1"],
            "DIRECT_EVIDENCE": ["therapeutic"],
            "PUBMED_IDS": ["9|10"],
        }
    ).lazy()
    ppi = pl.DataFrame(
        {
            "GENE_NCBI_ID_1": [11, 12, 12],
            "GENE_NCBI_ID_2": [12, 11, 12],  # includes reverse duplicate + self loop
        }
    ).lazy()

    cg, dg, cd, ppi_u, ppi_d = build_edge_tables(
        ctd_chg=ctd_chg,
        ctd_chd=ctd_chd,
        ctd_dg=ctd_dg,
        ppi=ppi,
        genes_core=genes_core,
        ds_core=ds_core,
        chems_core=chems_core,
    )

    # interaction list explodes into two rows and keeps parsed action fields
    assert set(cg["ACTION_TYPE"].to_list()) == {"increases", "decreases"}
    assert set(cg["ACTION_SUBJECT"].to_list()) == {"expression", "activity"}
    assert all(v > 0 for v in cg["LOG_PUBMED_COUNT"].to_list())

    # disease-gene direct evidence encoded
    assert dg.height == 1
    assert dg["DIRECT_EVIDENCE_TYPE"].to_list()[0] == 1
    assert math.isclose(dg["LOG_PUBMED_COUNT"].to_list()[0], math.log(3), rel_tol=1e-6)

    assert cd.height == 1

    # undirected deduped to one edge; directed has both directions
    assert ppi_u.height == 1
    assert ppi_d.height == 2


def test_load_processed_data_reads_required_and_optional_tables(tmp_path):
    required = {
        "genes_nodes.parquet": pl.DataFrame({"GENE_ID": [0], "GENE_NCBI_ID": [11]}),
        "diseases_nodes.parquet": pl.DataFrame({"DS_ID": [0], "DS_OMIM_MESH_ID": ["D1"]}),
        "chemicals_nodes.parquet": pl.DataFrame({"CHEM_ID": [0], "CHEM_MESH_ID": ["C1"]}),
        "chem_gene_edges.parquet": pl.DataFrame({"CHEM_GENE_IDX": [0], "CHEM_ID": [0], "GENE_ID": [0], "ACTION_TYPE": ["a"], "ACTION_SUBJECT": ["b"], "LOG_PUBMED_COUNT": [0.0]}),
        "disease_gene_edges.parquet": pl.DataFrame({"GENE_DS_IDX": [0], "GENE_ID": [0], "DS_ID": [0], "DIRECT_EVIDENCE_TYPE": [1], "LOG_PUBMED_COUNT": [0.0]}),
        "chem_disease_edges.parquet": pl.DataFrame({"CHEM_DS_IDX": [0], "CHEM_ID": [0], "DS_ID": [0]}),
        "ppi_edges.parquet": pl.DataFrame({"PPI_IDX": [0], "GENE_ID_1": [0], "GENE_ID_2": [0]}),
        "ppi_directed_edges.parquet": pl.DataFrame({"PPI_DIR_IDX": [0], "GENE_ID_1": [0], "GENE_ID_2": [0]}),
    }
    optional = {
        "pathways_nodes.parquet": pl.DataFrame({"PATHWAY_ID": [0], "PATHWAY_SOURCE_ID": ["P1"], "PATHWAY_NAME": ["p"]}),
    }
    for filename, df in {**required, **optional}.items():
        df.write_parquet(tmp_path / filename)

    out = load_processed_data(str(tmp_path))

    for key in ("genes", "diseases", "chemicals", "chem_gene", "disease_gene", "chem_disease", "ppi", "ppi_directed"):
        assert key in out
    assert "pathways" in out
