"""
Graph construction module for building PyG HeteroData.
"""

import polars as pl
import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, List
from pathlib import Path


def _assert_correct_ids(nodes: pl.DataFrame, id_col: str) -> int:
    """
    Validate node ID column:
    - IDs are unique
    - min == 0
    - max == n-1
    - All unsigned integers
    
    Args:
        nodes: DataFrame containing node data.
        id_col: Name of the ID column to validate.
        
    Returns:
        Number of nodes (n).
        
    Raises:
        TypeError: If IDs are not unsigned integers.
        ValueError: If IDs are invalid.
    """
    s = nodes.select(pl.col(id_col)).to_series()
    
    if not s.dtype.is_unsigned_integer():
        raise TypeError(f'ID column {id_col} must be of integer type, got {s.dtype}.')
    
    n = nodes.height
    if n == 0:
        raise ValueError(f'Node DataFrame for {id_col} is empty.')
    
    min_id = s.min()
    max_id = s.max()
    n_unique = s.n_unique()
    
    if n_unique != n:
        raise ValueError(
            f'ID column {id_col} must have unique values, '
            f'got {n_unique} unique values for {n} rows.'
        )
    if min_id != 0:
        raise ValueError(f'ID column {id_col} must have minimum value 0, got {min_id}.')
    if max_id != n - 1:
        raise ValueError(f'ID column {id_col} must have maximum value {n - 1}, got {max_id}.')
    
    present = np.zeros(n, dtype=np.bool_)
    present[np.asarray(s.to_numpy(), dtype=np.int64)] = True
    if not present.all():
        missing = np.where(~present)[0][:10]
        raise ValueError(f'ID column {id_col} is missing values: {missing}...')
    
    return n


def _cat_ids_from_col(df: pl.DataFrame, col: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Create integer IDs for a categorical string column.
    
    Args:
        df: DataFrame containing the categorical column.
        col: Name of the column to convert to IDs.
        
    Returns:
        Tuple of (df_with_id, vocab_df) where:
        - df_with_id: Original df with {col}_ID added and {col} removed.
        - vocab_df: Mapping DataFrame with {col}_ID and {col}.
    """
    vocab = (
        df.select(col)
        .unique()
        .sort(col)
        .with_row_index(f'{col}_ID')
    )
    
    df = df.join(vocab, on=col, how='left').drop(col)
    
    return df, vocab.select([f'{col}_ID', col])


def build_hetero_data(
    *,
    cnodes: pl.DataFrame,
    dnodes: pl.DataFrame,
    gnodes: pl.DataFrame,
    cd: pl.DataFrame,
    cg: pl.DataFrame,
    dg: pl.DataFrame,
    ppi: pl.DataFrame,
    ppi_directed: Optional[pl.DataFrame] = None,
    pathway_nodes: Optional[pl.DataFrame] = None,
    go_term_nodes: Optional[pl.DataFrame] = None,
    gene_pathway: Optional[pl.DataFrame] = None,
    disease_pathway: Optional[pl.DataFrame] = None,
    chem_pathway: Optional[pl.DataFrame] = None,
    chem_go: Optional[pl.DataFrame] = None,
    chem_pheno: Optional[pl.DataFrame] = None,
    go_disease: Optional[pl.DataFrame] = None,
    add_reverse_edges: bool = True,
) -> Tuple[HeteroData, Dict[str, pl.DataFrame]]:
    """
    Build a PyG HeteroData from processed DataFrames.
    
    Args:
        cnodes: Chemical nodes DataFrame (CHEM_ID, CHEM_MESH_ID, CHEM_NAME).
        dnodes: Disease nodes DataFrame (DS_ID, DS_OMIM_MESH_ID, DS_NAME).
        gnodes: Gene nodes DataFrame (GENE_ID, GENE_NCBI_ID).
        cd: Chemical-Disease edges DataFrame.
        cg: Chemical-Gene edges DataFrame with ACTION_TYPE, ACTION_SUBJECT.
        dg: Disease-Gene edges DataFrame.
        ppi: PPI edges DataFrame (undirected).
        ppi_directed: Optional PPI edges containing both directions.
        pathway_nodes: Pathway nodes DataFrame (optional).
        go_term_nodes: GO term nodes DataFrame (optional).
        gene_pathway: Gene-Pathway edges DataFrame (optional).
        disease_pathway: Disease-Pathway edges DataFrame (optional).
        chem_pathway: Chemical-Pathway edges DataFrame with enrichment attrs (optional).
        chem_go: Chemical-GO enriched edges DataFrame (optional).
        chem_pheno: Chemical-Phenotype edges DataFrame (optional).
        go_disease: GO-Disease edges DataFrame (optional).
        add_reverse_edges: Whether to add reverse edges for message passing.
        
    Returns:
        Tuple of (HeteroData, vocabs) where vocabs contains action type/subject mappings.
    """
    num_chem = _assert_correct_ids(cnodes, 'CHEM_ID')
    num_ds = _assert_correct_ids(dnodes, 'DS_ID')
    num_gene = _assert_correct_ids(gnodes, 'GENE_ID')
    
    cg, action_type_vocab = _cat_ids_from_col(cg, 'ACTION_TYPE')
    cg, action_subject_vocab = _cat_ids_from_col(cg, 'ACTION_SUBJECT')
    vocabs = {
        'action_type': action_type_vocab,
        'action_subject': action_subject_vocab
    }
    
    data = HeteroData()
    # CORE NODES (chemical, disease, gene)
    data['chemical'].num_nodes = num_chem
    data['disease'].num_nodes = num_ds
    data['gene'].num_nodes = num_gene

    # Keep stable global IDs for sampling/collision checks.
    data['chemical'].node_id = torch.arange(num_chem).view(-1, 1)
    data['disease'].node_id = torch.arange(num_ds).view(-1, 1)
    data['gene'].node_id = torch.arange(num_gene).view(-1, 1)

    data['chemical'].x = data['chemical'].node_id.clone()
    data['disease'].x = data['disease'].node_id.clone()
    data['gene'].x = data['gene'].node_id.clone()

    # NEW NODES (pathway, go_term)
    if pathway_nodes is not None and pathway_nodes.height > 0:
        num_pathway = _assert_correct_ids(pathway_nodes, 'PATHWAY_ID')
        data['pathway'].num_nodes = num_pathway
        data['pathway'].node_id = torch.arange(num_pathway).view(-1, 1)
        data['pathway'].x = data['pathway'].node_id.clone()
    
    if go_term_nodes is not None and go_term_nodes.height > 0:
        num_go_term = _assert_correct_ids(go_term_nodes, 'GO_TERM_ID')
        data['go_term'].num_nodes = num_go_term
        data['go_term'].node_id = torch.arange(num_go_term).view(-1, 1)
        data['go_term'].x = data['go_term'].node_id.clone()
        
        # Add ontology type as node feature if available
        if 'GO_ONTOLOGY' in go_term_nodes.columns:
            ontology_map = {'Biological Process': 0, 'Molecular Function': 1, 'Cellular Component': 2}
            ontology_ids = go_term_nodes['GO_ONTOLOGY'].replace_strict(ontology_map, default=0).to_numpy().copy()
            data['go_term'].ontology_type = torch.from_numpy(ontology_ids).long()
    
    # CORE EDGES
    # Chemical-Disease edges
    data['chemical', 'associated_with', 'disease'].edge_index = torch.from_numpy(
        cd.select(['CHEM_ID', 'DS_ID']).to_numpy().T.astype(np.int64)
    ).long()
    
    # Chemical-Gene edges with attributes
    data['chemical', 'affects', 'gene'].edge_index = torch.from_numpy(
        cg.select(['CHEM_ID', 'GENE_ID']).to_numpy().T.astype(np.int64)
    ).long()
    
    cg_edge_attr = torch.from_numpy(
        cg.select(['ACTION_TYPE_ID', 'ACTION_SUBJECT_ID']).to_numpy().astype(np.int64)
    ).long()
    data['chemical', 'affects', 'gene'].edge_attr = cg_edge_attr
    
    # Disease-Gene edges
    data['disease', 'targets', 'gene'].edge_index = torch.from_numpy(
        dg.select(['DS_ID', 'GENE_ID']).to_numpy().T.astype(np.int64)
    ).long()
    dg_attr_cols = ['DIRECT_EVIDENCE_TYPE', 'LOG_PUBMED_COUNT']
    if all(c in dg.columns for c in dg_attr_cols):
        dg_attr = dg.select(dg_attr_cols).to_numpy().astype(np.float32)
        data['disease', 'targets', 'gene'].edge_attr = torch.from_numpy(dg_attr)
    
    # Gene-Gene (PPI) edges.
    # Prefer directed PPI edges when available to avoid storing a redundant
    # reverse edge type for this symmetric relation.
    ppi_edges = ppi_directed if ppi_directed is not None and ppi_directed.height > 0 else ppi
    data['gene', 'interacts_with', 'gene'].edge_index = torch.from_numpy(
        ppi_edges.select(['GENE_ID_1', 'GENE_ID_2']).to_numpy().T.astype(np.int64)
    ).long()

    # PATHWAY EDGES
    if gene_pathway is not None and gene_pathway.height > 0:
        data['gene', 'participates_in', 'pathway'].edge_index = torch.from_numpy(
            gene_pathway.select(['GENE_ID', 'PATHWAY_ID']).to_numpy().T.astype(np.int64)
        ).long()
    
    if disease_pathway is not None and disease_pathway.height > 0:
        data['disease', 'disrupts', 'pathway'].edge_index = torch.from_numpy(
            disease_pathway.select(['DS_ID', 'PATHWAY_ID']).to_numpy().T.astype(np.int64)
        ).long()
        
        # Edge attribute: inference gene count (log-scaled)
        if 'INFERENCE_GENE_COUNT' in disease_pathway.columns:
            inference_count = disease_pathway['INFERENCE_GENE_COUNT'].fill_null(1).to_numpy()
            data['disease', 'disrupts', 'pathway'].edge_attr = torch.from_numpy(
                np.log1p(inference_count).astype(np.float32)
            ).view(-1, 1)
    
    if chem_pathway is not None and chem_pathway.height > 0:
        data['chemical', 'enriched_in', 'pathway'].edge_index = torch.from_numpy(
            chem_pathway.select(['CHEM_ID', 'PATHWAY_ID']).to_numpy().T.astype(np.int64)
        ).long()
        
        # Edge attributes: enrichment statistics
        attr_cols = ['NEG_LOG_PVALUE', 'TARGET_RATIO', 'FOLD_ENRICHMENT']
        if all(c in chem_pathway.columns for c in attr_cols):
            chem_pathway_attr = chem_pathway.select(attr_cols).to_numpy().astype(np.float32)
            data['chemical', 'enriched_in', 'pathway'].edge_attr = torch.from_numpy(chem_pathway_attr)

    # GO TERM EDGES
    if chem_go is not None and chem_go.height > 0:
        data['chemical', 'enriched_in', 'go_term'].edge_index = torch.from_numpy(
            chem_go.select(['CHEM_ID', 'GO_TERM_ID']).to_numpy().T.astype(np.int64)
        ).long()
        
        # Edge attributes: enrichment statistics + ontology + GO level
        attr_cols = ['NEG_LOG_PVALUE', 'TARGET_RATIO', 'FOLD_ENRICHMENT', 'ONTOLOGY_TYPE', 'GO_LEVEL_NORM']
        if all(c in chem_go.columns for c in attr_cols):
            chem_go_attr = chem_go.select(attr_cols).to_numpy().astype(np.float32)
            data['chemical', 'enriched_in', 'go_term'].edge_attr = torch.from_numpy(chem_go_attr)
    
    if chem_pheno is not None and chem_pheno.height > 0:
        data['chemical', 'affects_phenotype', 'go_term'].edge_index = torch.from_numpy(
            chem_pheno.select(['CHEM_ID', 'GO_TERM_ID']).to_numpy().T.astype(np.int64)
        ).long()
        
        # Edge attribute: action type (increases=0, decreases=1, affects=2)
        if 'PHENO_ACTION_TYPE' in chem_pheno.columns:
            action_type = chem_pheno['PHENO_ACTION_TYPE'].to_numpy().astype(np.int64)
            data['chemical', 'affects_phenotype', 'go_term'].edge_attr = torch.from_numpy(action_type).view(-1, 1)
    
    if go_disease is not None and go_disease.height > 0:
        data['go_term', 'associated_with', 'disease'].edge_index = torch.from_numpy(
            go_disease.select(['GO_TERM_ID', 'DS_ID']).to_numpy().T.astype(np.int64)
        ).long()
        
        # Edge attributes: ontology type + inference counts
        attr_cols = ['ONTOLOGY_TYPE', 'LOG_INFERENCE_CHEM', 'LOG_INFERENCE_GENE']
        if all(c in go_disease.columns for c in attr_cols):
            go_disease_attr = go_disease.select(attr_cols).to_numpy().astype(np.float32)
            data['go_term', 'associated_with', 'disease'].edge_attr = torch.from_numpy(go_disease_attr)
    
    # REVERSE EDGES
    if add_reverse_edges:
        # Original reverse edges
        data['disease', 'rev_associated_with', 'chemical'].edge_index = torch.flip(
            data['chemical', 'associated_with', 'disease'].edge_index, dims=[0]
        )
        
        data['gene', 'rev_affects', 'chemical'].edge_index = torch.flip(
            data['chemical', 'affects', 'gene'].edge_index, dims=[0]
        )
        data['gene', 'rev_affects', 'chemical'].edge_attr = cg_edge_attr.clone()
        
        data['gene', 'rev_targets', 'disease'].edge_index = torch.flip(
            data['disease', 'targets', 'gene'].edge_index, dims=[0]
        )
        if hasattr(data['disease', 'targets', 'gene'], 'edge_attr'):
            data['gene', 'rev_targets', 'disease'].edge_attr = \
                data['disease', 'targets', 'gene'].edge_attr.clone()
        
        # Keep gene-gene reverse edge only when inputs are undirected.
        if ppi_directed is None or ppi_directed.height == 0:
            data['gene', 'rev_interacts_with', 'gene'].edge_index = torch.flip(
                data['gene', 'interacts_with', 'gene'].edge_index, dims=[0]
            )
        
        # Pathway reverse edges
        if ('gene', 'participates_in', 'pathway') in data.edge_types:
            data['pathway', 'rev_participates_in', 'gene'].edge_index = torch.flip(
                data['gene', 'participates_in', 'pathway'].edge_index, dims=[0]
            )
        
        if ('disease', 'disrupts', 'pathway') in data.edge_types:
            data['pathway', 'rev_disrupts', 'disease'].edge_index = torch.flip(
                data['disease', 'disrupts', 'pathway'].edge_index, dims=[0]
            )
            if hasattr(data['disease', 'disrupts', 'pathway'], 'edge_attr'):
                data['pathway', 'rev_disrupts', 'disease'].edge_attr = \
                    data['disease', 'disrupts', 'pathway'].edge_attr.clone()
        
        if ('chemical', 'enriched_in', 'pathway') in data.edge_types:
            data['pathway', 'rev_enriched_in', 'chemical'].edge_index = torch.flip(
                data['chemical', 'enriched_in', 'pathway'].edge_index, dims=[0]
            )
            if hasattr(data['chemical', 'enriched_in', 'pathway'], 'edge_attr'):
                data['pathway', 'rev_enriched_in', 'chemical'].edge_attr = \
                    data['chemical', 'enriched_in', 'pathway'].edge_attr.clone()
        
        # GO term reverse edges
        if ('chemical', 'enriched_in', 'go_term') in data.edge_types:
            data['go_term', 'rev_enriched_in', 'chemical'].edge_index = torch.flip(
                data['chemical', 'enriched_in', 'go_term'].edge_index, dims=[0]
            )
            if hasattr(data['chemical', 'enriched_in', 'go_term'], 'edge_attr'):
                data['go_term', 'rev_enriched_in', 'chemical'].edge_attr = \
                    data['chemical', 'enriched_in', 'go_term'].edge_attr.clone()
        
        if ('chemical', 'affects_phenotype', 'go_term') in data.edge_types:
            data['go_term', 'rev_affects_phenotype', 'chemical'].edge_index = torch.flip(
                data['chemical', 'affects_phenotype', 'go_term'].edge_index, dims=[0]
            )
            if hasattr(data['chemical', 'affects_phenotype', 'go_term'], 'edge_attr'):
                data['go_term', 'rev_affects_phenotype', 'chemical'].edge_attr = \
                    data['chemical', 'affects_phenotype', 'go_term'].edge_attr.clone()
        
        if ('go_term', 'associated_with', 'disease') in data.edge_types:
            data['disease', 'rev_associated_with', 'go_term'].edge_index = torch.flip(
                data['go_term', 'associated_with', 'disease'].edge_index, dims=[0]
            )
            if hasattr(data['go_term', 'associated_with', 'disease'], 'edge_attr'):
                data['disease', 'rev_associated_with', 'go_term'].edge_attr = \
                    data['go_term', 'associated_with', 'disease'].edge_attr.clone()
    
    data.validate(raise_on_error=True)
    
    return data, vocabs


def build_graph_from_processed(
    processed_data_dir: str = './data/processed',
    add_reverse_edges: bool = True,
    save_vocabs: bool = True,
    include_extended: bool = True,
) -> Tuple[HeteroData, Dict[str, pl.DataFrame]]:
    """
    Build HeteroData from processed parquet files.
    
    Args:
        processed_data_dir: Path to processed data directory.
        add_reverse_edges: Whether to add reverse edges.
        save_vocabs: Whether to save vocabulary files.
        include_extended: Whether to include pathway and GO term nodes/edges.
        
    Returns:
        Tuple of (HeteroData, vocabs).
    """
    from .processing import load_processed_data
    
    data_dict = load_processed_data(processed_data_dir)

    # Prepare optional arguments for extended graph
    extended_args = {}
    if include_extended:
        # Add pathway data if available
        if 'pathways' in data_dict:
            extended_args['pathway_nodes'] = data_dict['pathways']
        if 'gene_pathway' in data_dict:
            extended_args['gene_pathway'] = data_dict['gene_pathway']
        if 'disease_pathway' in data_dict:
            extended_args['disease_pathway'] = data_dict['disease_pathway']
        if 'chem_pathway' in data_dict:
            extended_args['chem_pathway'] = data_dict['chem_pathway']
        
        # Add GO term data if available
        if 'go_terms' in data_dict:
            extended_args['go_term_nodes'] = data_dict['go_terms']
        if 'chem_go' in data_dict:
            extended_args['chem_go'] = data_dict['chem_go']
        if 'chem_pheno' in data_dict:
            extended_args['chem_pheno'] = data_dict['chem_pheno']
        if 'go_disease' in data_dict:
            extended_args['go_disease'] = data_dict['go_disease']
    
    data, vocabs = build_hetero_data(
        cnodes=data_dict['chemicals'],
        dnodes=data_dict['diseases'],
        gnodes=data_dict['genes'],
        cd=data_dict['chem_disease'],
        cg=data_dict['chem_gene'],
        dg=data_dict['disease_gene'],
        ppi=data_dict['ppi'],
        ppi_directed=data_dict.get('ppi_directed'),
        add_reverse_edges=add_reverse_edges,
        **extended_args
    )
    
    if save_vocabs:
        processed_path = Path(processed_data_dir)
        for name, vocab_df in vocabs.items():
            vocab_df.write_csv(processed_path / f'vocab_{name}.csv')
    
    return data, vocabs


def load_vocabs(processed_data_dir: str = './data/processed') -> Dict[str, pl.DataFrame]:
    """
    Load vocabulary DataFrames from CSV files.
    
    Args:
        processed_data_dir: Path to processed data directory.
        
    Returns:
        Dictionary with 'action_type' and 'action_subject' vocabularies.
    """
    processed_path = Path(processed_data_dir)
    return {
        'action_type': pl.read_csv(processed_path / 'vocab_action_type.csv'),
        'action_subject': pl.read_csv(processed_path / 'vocab_action_subject.csv')
    }


def get_graph_summary(data: HeteroData) -> Dict:
    """
    Get a summary of the heterogeneous graph structure.
    
    Args:
        data: HeteroData object.
        
    Returns:
        Dictionary with node counts, edge counts, and metadata.
    """
    summary = {
        'node_types': {},
        'edge_types': {},
        'total_nodes': 0,
        'total_edges': 0
    }
    
    for node_type in data.node_types:
        count = data[node_type].num_nodes
        summary['node_types'][node_type] = count
        summary['total_nodes'] += count
    
    for edge_type in data.edge_types:
        count = data[edge_type].edge_index.size(1)
        edge_key = '__'.join(edge_type)
        summary['edge_types'][edge_key] = {
            'count': count,
            'has_attr': hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None
        }
        summary['total_edges'] += count
    
    return summary


def print_graph_summary(data: HeteroData):
    """Print a formatted summary of the graph structure."""
    summary = get_graph_summary(data)
    
    print("=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)
    
    print(f"\nNode Types ({len(summary['node_types'])}):")
    for ntype, count in sorted(summary['node_types'].items()):
        print(f"  {ntype}: {count:,}")
    
    print(f"\nEdge Types ({len(summary['edge_types'])}):")
    for etype, info in sorted(summary['edge_types'].items()):
        attr_str = " [+attr]" if info['has_attr'] else ""
        print(f"  {etype}: {info['count']:,}{attr_str}")
    
    print(f"\nTotal: {summary['total_nodes']:,} nodes, {summary['total_edges']:,} edges")
    print("=" * 60)
