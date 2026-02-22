"""
Data processing module for CTD (Comparative Toxicogenomics Database) data.

This module handles loading and preprocessing of:
- Chemical-Gene interactions (CTD_chem_gene_ixns.tsv)
- Chemical-Disease associations (CTD_curated_chemicals_diseases.tsv)
- Disease-Gene associations (CTD_curated_genes_diseases.tsv)
- Protein-Protein interactions (PP-Decagon_ppi.csv)
- Gene-Pathway associations (CTD_genes_pathways.tsv)
- Disease-Pathway associations (CTD_diseases_pathways.tsv)
- Chemical-Pathway enriched associations (CTD_chem_pathways_enriched.tsv)
- Chemical-GO enriched associations (CTD_chem_go_enriched.tsv)
- Chemical-Phenotype interactions (CTD_pheno_term_ixns.tsv)
- Phenotype-Disease associations (CTD_Phenotype-Disease_*.tsv)
- Chemical-Gene interaction types (CTD_chem_gene_ixn_types.tsv)
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


def load_raw_data(data_dir: str = './data/raw') -> Dict[str, pl.LazyFrame]:
    """
    Load all raw CTD and PPI data files as lazy frames.
    
    Args:
        data_dir: Path to directory containing raw data files.
        
    Returns:
        Dictionary with all raw data lazy frames.
    """
    data_path = Path(data_dir)
    
    # CORE INTERACTION DATA
    ctd_chg = pl.scan_csv(
        data_path / 'CTD_chem_gene_ixns.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'ChemicalName': pl.String,
            'ChemicalID': pl.String,
            'GeneID': pl.Int64,
            'InteractionActions': pl.String,
            'PubMedIDs': pl.String
        })
    ).select(['ChemicalName', 'ChemicalID', 'GeneID', 'InteractionActions', 'PubMedIDs']).rename({
        'ChemicalName': 'CHEM_NAME',
        'ChemicalID': 'CHEM_MESH_ID',
        'GeneID': 'GENE_NCBI_ID',
        'InteractionActions': 'INTERACTIONS',
        'PubMedIDs': 'PUBMED_IDS'
    })
    
    ctd_chd = pl.scan_csv(
        data_path / 'CTD_curated_chemicals_diseases.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'ChemicalName': pl.String,
            'ChemicalID': pl.String,
            'DiseaseName': pl.String,
            'DiseaseID': pl.String,
            'DirectEvidence': pl.String,
            'PubMedIDs': pl.String
        })
    ).select(['ChemicalName', 'ChemicalID', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'PubMedIDs']).rename({
        'ChemicalName': 'CHEM_NAME',
        'ChemicalID': 'CHEM_MESH_ID',
        'DiseaseName': 'DS_NAME',
        'DiseaseID': 'DS_OMIM_MESH_ID',
        'DirectEvidence': 'DIRECT_EVIDENCE',
        'PubMedIDs': 'PUBMED_IDS'
    })
    
    ctd_dg = pl.scan_csv(
        data_path / 'CTD_curated_genes_diseases.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'GeneID': pl.Int64,
            'DiseaseName': pl.String,
            'DiseaseID': pl.String,
            'OmimIDs': pl.String,
            'DirectEvidence': pl.String,
            'PubMedIDs': pl.String
        })
    ).select(['GeneID', 'DiseaseName', 'DiseaseID', 'OmimIDs', 'DirectEvidence', 'PubMedIDs']).rename({
        'GeneID': 'GENE_NCBI_ID',
        'DiseaseName': 'DS_NAME',
        'DiseaseID': 'DS_OMIM_MESH_ID',
        'OmimIDs': 'DS_OMIM_IDS',
        'DirectEvidence': 'DIRECT_EVIDENCE',
        'PubMedIDs': 'PUBMED_IDS'
    })
    
    ppi = pl.scan_csv(
        data_path / 'PP-Decagon_ppi.csv',
        has_header=False,
        new_columns=['GENE_NCBI_ID_1', 'GENE_NCBI_ID_2']
    )
    
    # ANNOTATION/VOCABULARY DATA
    chem_annots = pl.scan_csv(
        data_path / 'CTD_chemicals.tsv',
        separator='\t',
    ).select([
        'ChemicalID',
        'Definition',
        'ParentIDs',
        'TreeNumbers',
        'ParentTreeNumbers',
        'CTDCuratedSynonyms'
    ]).rename({
        'ChemicalID': 'CHEM_MESH_ID',
        'Definition': 'CHEM_DEFINITION',
        'ParentIDs': 'CHEM_PARENT_IDS',
        'TreeNumbers': 'CHEM_TREE_NUMBERS',
        'ParentTreeNumbers': 'CHEM_PARENT_TREE_NUMBERS',
        'CTDCuratedSynonyms': 'CHEM_SYNONYMS'
    })

    ds_annots = pl.scan_csv(
        data_path / 'CTD_diseases.tsv',
        separator='\t',
    ).select([
        'DiseaseID',
        'Definition',
        'ParentIDs',
        'TreeNumbers',
        'ParentTreeNumbers',
        'Synonyms',
        'SlimMappings'
    ]).rename({
        'DiseaseID': 'DS_OMIM_MESH_ID',
        'Definition': 'DS_DEFINITION',
        'ParentIDs': 'DS_PARENT_IDS',
        'TreeNumbers': 'DS_TREE_NUMBERS',
        'ParentTreeNumbers': 'DS_PARENT_TREE_NUMBERS',
        'Synonyms': 'DS_SYNONYMS',
        'SlimMappings': 'DS_SLIM_MAPPINGS'
    })

    gene_annots = pl.scan_csv(
        data_path / 'CTD_genes.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'BioGRIDIDs': pl.String
        })
    ).rename({
        'GeneID': 'GENE_NCBI_ID',
        'GeneSymbol': 'GENE_SYMBOL',
        'GeneName': 'GENE_NAME',
        'BioGRIDIDs': 'GENE_BIOGRID_IDS',
        'AltGeneIDs': 'GENE_ALT_IDS',
        'Synonyms': 'GENE_SYNONYMS',
        'PharmGKBIDs': 'GENE_PHARMGKB_IDS',
        'UniProtIDs': 'GENE_UNIPROT_IDS'
    })
    
    # PATHWAY DATA    
    gene_pathways = pl.scan_csv(
        data_path / 'CTD_genes_pathways.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'GeneID': pl.Int64,
        })
    ).select(['GeneSymbol', 'GeneID', 'PathwayName', 'PathwayID']).rename({
        'GeneID': 'GENE_NCBI_ID',
        'GeneSymbol': 'GENE_SYMBOL',
        'PathwayName': 'PATHWAY_NAME',
        'PathwayID': 'PATHWAY_ID'
    })
    
    disease_pathways = pl.scan_csv(
        data_path / 'CTD_diseases_pathways.tsv',
        separator='\t',
    ).select(['DiseaseName', 'DiseaseID', 'PathwayName', 'PathwayID', 'InferenceGeneSymbol']).rename({
        'DiseaseName': 'DS_NAME',
        'DiseaseID': 'DS_OMIM_MESH_ID',
        'PathwayName': 'PATHWAY_NAME',
        'PathwayID': 'PATHWAY_ID',
        'InferenceGeneSymbol': 'INFERENCE_GENE_SYMBOL'
    })
    
    chem_pathways = pl.scan_csv(
        data_path / 'CTD_chem_pathways_enriched.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'PValue': pl.Float64,
            'CorrectedPValue': pl.Float64,
            'TargetMatchQty': pl.Int64,
            'TargetTotalQty': pl.Int64,
            'BackgroundMatchQty': pl.Int64,
            'BackgroundTotalQty': pl.Int64
        })
    ).select([
        'ChemicalName', 'ChemicalID', 'PathwayName', 'PathwayID',
        'PValue', 'CorrectedPValue', 'TargetMatchQty', 'TargetTotalQty',
        'BackgroundMatchQty', 'BackgroundTotalQty'
    ]).rename({
        'ChemicalName': 'CHEM_NAME',
        'ChemicalID': 'CHEM_MESH_ID',
        'PathwayName': 'PATHWAY_NAME',
        'PathwayID': 'PATHWAY_ID'
    })
    
    # GO TERM / PHENOTYPE DATA
    chem_go = pl.scan_csv(
        data_path / 'CTD_chem_go_enriched.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'PValue': pl.Float64,
            'CorrectedPValue': pl.Float64,
            'TargetMatchQty': pl.Int64,
            'TargetTotalQty': pl.Int64,
            'BackgroundMatchQty': pl.Int64,
            'BackgroundTotalQty': pl.Int64,
            'HighestGOLevel': pl.Int64
        })
    ).select([
        'ChemicalName', 'ChemicalID', 'Ontology', 'GOTermName', 'GOTermID',
        'HighestGOLevel', 'PValue', 'CorrectedPValue', 'TargetMatchQty',
        'TargetTotalQty', 'BackgroundMatchQty', 'BackgroundTotalQty'
    ]).rename({
        'ChemicalName': 'CHEM_NAME',
        'ChemicalID': 'CHEM_MESH_ID',
        'Ontology': 'GO_ONTOLOGY',
        'GOTermName': 'GO_NAME',
        'GOTermID': 'GO_ID',
        'HighestGOLevel': 'GO_LEVEL'
    })
    
    pheno_ixns = pl.scan_csv(
        data_path / 'CTD_pheno_term_ixns.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'organismid': pl.Int64,
        })
    ).select([
        'chemicalname', 'chemicalid', 'phenotypename', 'phenotypeid',
        'organism', 'organismid', 'interaction', 'interactionactions',
        'anatomyterms', 'inferencegenesymbols'
    ]).rename({
        'chemicalname': 'CHEM_NAME',
        'chemicalid': 'CHEM_MESH_ID',
        'phenotypename': 'GO_NAME',
        'phenotypeid': 'GO_ID',
        'organism': 'ORGANISM',
        'organismid': 'ORGANISM_ID',
        'interaction': 'INTERACTION',
        'interactionactions': 'INTERACTION_ACTIONS',
        'anatomyterms': 'ANATOMY_TERMS',
        'inferencegenesymbols': 'INFERENCE_GENE_SYMBOLS'
    })
    
    # Phenotype-Disease associations (3 ontology files)
    pheno_disease_bp = pl.scan_csv(
        data_path / 'CTD_Phenotype-Disease_biological_process_associations.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'InferenceChemicalQty': pl.Int64,
            'InferenceGeneQty': pl.Int64
        })
    ).select([
        'GOName', 'GOID', 'DiseaseName', 'DiseaseID',
        'InferenceChemicalQty', 'InferenceChemicalNames',
        'InferenceGeneQty', 'InferenceGeneSymbols'
    ]).rename({
        'GOName': 'GO_NAME',
        'GOID': 'GO_ID',
        'DiseaseName': 'DS_NAME',
        'DiseaseID': 'DS_OMIM_MESH_ID',
        'InferenceChemicalQty': 'INFERENCE_CHEM_QTY',
        'InferenceChemicalNames': 'INFERENCE_CHEM_NAMES',
        'InferenceGeneQty': 'INFERENCE_GENE_QTY',
        'InferenceGeneSymbols': 'INFERENCE_GENE_SYMBOLS'
    }).with_columns(pl.lit('Biological Process').alias('GO_ONTOLOGY'))
    
    pheno_disease_mf = pl.scan_csv(
        data_path / 'CTD_Phenotype-Disease_molecular_function_associations.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'InferenceChemicalQty': pl.Int64,
            'InferenceGeneQty': pl.Int64
        })
    ).select([
        'GOName', 'GOID', 'DiseaseName', 'DiseaseID',
        'InferenceChemicalQty', 'InferenceChemicalNames',
        'InferenceGeneQty', 'InferenceGeneSymbols'
    ]).rename({
        'GOName': 'GO_NAME',
        'GOID': 'GO_ID',
        'DiseaseName': 'DS_NAME',
        'DiseaseID': 'DS_OMIM_MESH_ID',
        'InferenceChemicalQty': 'INFERENCE_CHEM_QTY',
        'InferenceChemicalNames': 'INFERENCE_CHEM_NAMES',
        'InferenceGeneQty': 'INFERENCE_GENE_QTY',
        'InferenceGeneSymbols': 'INFERENCE_GENE_SYMBOLS'
    }).with_columns(pl.lit('Molecular Function').alias('GO_ONTOLOGY'))
    
    pheno_disease_cc = pl.scan_csv(
        data_path / 'CTD_Phenotype-Disease_cellular_component_associations.tsv',
        separator='\t',
        schema_overrides=pl.Schema({
            'InferenceChemicalQty': pl.Int64,
            'InferenceGeneQty': pl.Int64
        })
    ).select([
        'GOName', 'GOID', 'DiseaseName', 'DiseaseID',
        'InferenceChemicalQty', 'InferenceChemicalNames',
        'InferenceGeneQty', 'InferenceGeneSymbols'
    ]).rename({
        'GOName': 'GO_NAME',
        'GOID': 'GO_ID',
        'DiseaseName': 'DS_NAME',
        'DiseaseID': 'DS_OMIM_MESH_ID',
        'InferenceChemicalQty': 'INFERENCE_CHEM_QTY',
        'InferenceChemicalNames': 'INFERENCE_CHEM_NAMES',
        'InferenceGeneQty': 'INFERENCE_GENE_QTY',
        'InferenceGeneSymbols': 'INFERENCE_GENE_SYMBOLS'
    }).with_columns(pl.lit('Cellular Component').alias('GO_ONTOLOGY'))
    
    # INTERACTION TYPE HIERARCHY
    ixn_types = pl.scan_csv(
        data_path / 'CTD_chem_gene_ixn_types.tsv',
        separator='\t',
    ).select(['TypeName', 'Code', 'Description', 'ParentCode']).rename({
        'TypeName': 'IXN_TYPE_NAME',
        'Code': 'IXN_TYPE_CODE',
        'Description': 'IXN_TYPE_DESC',
        'ParentCode': 'IXN_TYPE_PARENT_CODE'
    })

    return {
        # Core interaction data
        'ctd_chg': ctd_chg,
        'ctd_chd': ctd_chd,
        'ctd_dg': ctd_dg,
        'ppi': ppi,
        # Annotations
        'chem_annots': chem_annots,
        'ds_annots': ds_annots,
        'gene_annots': gene_annots,
        # Pathway data
        'gene_pathways': gene_pathways,
        'disease_pathways': disease_pathways,
        'chem_pathways': chem_pathways,
        # GO/Phenotype data
        'chem_go': chem_go,
        'pheno_ixns': pheno_ixns,
        'pheno_disease_bp': pheno_disease_bp,
        'pheno_disease_mf': pheno_disease_mf,
        'pheno_disease_cc': pheno_disease_cc,
        # Interaction type hierarchy
        'ixn_types': ixn_types
    }


def build_core_entities(
    ctd_chg: pl.LazyFrame,
    ctd_chd: pl.LazyFrame,
    ctd_dg: pl.LazyFrame,
    ppi: pl.LazyFrame,
    chem_annots: pl.LazyFrame,
    ds_annots: pl.LazyFrame,
    gene_annots: pl.LazyFrame
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Build core entity tables (genes, diseases, chemicals) by finding
    entities that appear in all relevant datasets.
    
    Args:
        ctd_chg: Chemical-Gene interactions lazy frame.
        ctd_chd: Chemical-Disease associations lazy frame.
        ctd_dg: Disease-Gene associations lazy frame.
        ppi: Protein-Protein interactions lazy frame.
        chem_annots: Chemical annotations lazy frame.
        ds_annots: Disease annotations lazy frame.
        gene_annots: Gene annotations lazy frame.
        
    Returns:
        Tuple of (genes_core, diseases_core, chemicals_core) DataFrames.
    """
    # Genes: must appear in ctd_chg, ctd_dg, and ppi
    genes_core = (
        ctd_chg.select(pl.col('GENE_NCBI_ID').unique())
        .join(
            ctd_dg.select(pl.col('GENE_NCBI_ID').unique()),
            on='GENE_NCBI_ID',
            how='inner'
        )
        .join(
            pl.concat([
                ppi.select('GENE_NCBI_ID_1').rename({'GENE_NCBI_ID_1': 'GENE_NCBI_ID'}),
                ppi.select('GENE_NCBI_ID_2').rename({'GENE_NCBI_ID_2': 'GENE_NCBI_ID'})
            ], how='vertical').unique(),
            on='GENE_NCBI_ID',
            how='inner'
        )
        .with_columns(pl.col('GENE_NCBI_ID').cast(pl.UInt32))
    ).with_row_index('GENE_ID').collect()
    
    # Diseases: must appear in both ctd_dg and ctd_chd
    ds_core = (
        ctd_dg.select(['DS_OMIM_MESH_ID', 'DS_NAME']).unique('DS_OMIM_MESH_ID')
        .join(
            ctd_chd.select(['DS_OMIM_MESH_ID', 'DS_NAME']).unique('DS_OMIM_MESH_ID'),
            on='DS_OMIM_MESH_ID',
            how='inner'
        )
    ).select(pl.col('DS_OMIM_MESH_ID', 'DS_NAME')).with_row_index('DS_ID').collect()
    
    # Chemicals: must appear in both ctd_chg and ctd_chd
    chems_core = (
        ctd_chg.select(['CHEM_MESH_ID', 'CHEM_NAME']).unique('CHEM_MESH_ID')
        .join(
            ctd_chd.select(['CHEM_MESH_ID', 'CHEM_NAME']).unique('CHEM_MESH_ID'),
            on='CHEM_MESH_ID',
            how='inner'
        )
    ).select(pl.col('CHEM_MESH_ID', 'CHEM_NAME')).with_row_index('CHEM_ID').collect()
    
    genes_core = genes_core.join(gene_annots.collect(), on='GENE_NCBI_ID', how='left')
    ds_core = ds_core.join(ds_annots.collect(), on='DS_OMIM_MESH_ID', how='left')
    chems_core = chems_core.join(chem_annots.collect(), on='CHEM_MESH_ID', how='left')

    return genes_core, ds_core, chems_core


def build_pathway_entities(
    gene_pathways: pl.LazyFrame,
    disease_pathways: pl.LazyFrame,
    chem_pathways: pl.LazyFrame,
    genes_core: pl.DataFrame,
    ds_core: pl.DataFrame,
    chems_core: pl.DataFrame
) -> pl.DataFrame:
    """
    Build pathway node table from pathways that are connected to core entities.
    
    Pathways must appear in at least TWO of the three sources (gene-pathway,
    disease-pathway, chemical-pathway) to ensure they have sufficient connectivity.
    
    Args:
        gene_pathways: Gene-Pathway associations lazy frame.
        disease_pathways: Disease-Pathway associations lazy frame.
        chem_pathways: Chemical-Pathway enriched associations lazy frame.
        genes_core: Core genes DataFrame.
        ds_core: Core diseases DataFrame.
        chems_core: Core chemicals DataFrame.
        
    Returns:
        Pathway nodes DataFrame with PATHWAY_ID, PATHWAY_SOURCE_ID, PATHWAY_NAME.
    """
    # Get pathways connected to core genes
    pathways_from_genes = (
        gene_pathways
        .join(genes_core.lazy().select(['GENE_NCBI_ID']), on='GENE_NCBI_ID', how='inner')
        .select(['PATHWAY_ID', 'PATHWAY_NAME'])
        .unique('PATHWAY_ID')
        .rename({'PATHWAY_NAME': 'PATHWAY_NAME_GENE'})
        .with_columns(pl.lit(1).alias('IN_GENE_PATHWAYS'))
    )
    
    # Get pathways connected to core diseases
    pathways_from_diseases = (
        disease_pathways
        .join(ds_core.lazy().select(['DS_OMIM_MESH_ID']), on='DS_OMIM_MESH_ID', how='inner')
        .select(['PATHWAY_ID', 'PATHWAY_NAME'])
        .unique('PATHWAY_ID')
        .rename({'PATHWAY_NAME': 'PATHWAY_NAME_DISEASE'})
        .with_columns(pl.lit(1).alias('IN_DISEASE_PATHWAYS'))
    )
    
    # Get pathways connected to core chemicals
    pathways_from_chems = (
        chem_pathways
        .join(chems_core.lazy().select(['CHEM_MESH_ID']), on='CHEM_MESH_ID', how='inner')
        .select(['PATHWAY_ID', 'PATHWAY_NAME'])
        .unique('PATHWAY_ID')
        .rename({'PATHWAY_NAME': 'PATHWAY_NAME_CHEM'})
        .with_columns(pl.lit(1).alias('IN_CHEM_PATHWAYS'))
    )
    
    # Union all pathways and count sources
    all_pathways = (
        pathways_from_genes
        .join(pathways_from_diseases, on='PATHWAY_ID', how='full', coalesce=True)
        .join(pathways_from_chems, on='PATHWAY_ID', how='full', coalesce=True)
        .with_columns([
            pl.col('IN_GENE_PATHWAYS').fill_null(0),
            pl.col('IN_DISEASE_PATHWAYS').fill_null(0),
            pl.col('IN_CHEM_PATHWAYS').fill_null(0),
        ])
        .with_columns(
            pl.coalesce(
                [
                    pl.col('PATHWAY_NAME_GENE'),
                    pl.col('PATHWAY_NAME_DISEASE'),
                    pl.col('PATHWAY_NAME_CHEM'),
                ]
            ).alias('PATHWAY_NAME')
        )
        .with_columns(
            (pl.col('IN_GENE_PATHWAYS') + pl.col('IN_DISEASE_PATHWAYS') + pl.col('IN_CHEM_PATHWAYS'))
            .alias('SOURCE_COUNT')
        )
        # Must appear in at least 2 sources for sufficient connectivity
        .filter(pl.col('SOURCE_COUNT') >= 2)
        .select(['PATHWAY_ID', 'PATHWAY_NAME'])
        .sort('PATHWAY_ID')
    ).collect()
    
    # Rename PATHWAY_ID to PATHWAY_SOURCE_ID and add internal ID
    pathways_core = (
        all_pathways
        .rename({'PATHWAY_ID': 'PATHWAY_SOURCE_ID'})
        .with_row_index('PATHWAY_ID')
    )
    
    return pathways_core


def build_go_term_entities(
    chem_go: pl.LazyFrame,
    pheno_ixns: pl.LazyFrame,
    pheno_disease_bp: pl.LazyFrame,
    pheno_disease_mf: pl.LazyFrame,
    pheno_disease_cc: pl.LazyFrame,
    chems_core: pl.DataFrame,
    ds_core: pl.DataFrame
) -> pl.DataFrame:
    """
    Build unified GO term node table merging GO terms and phenotypes.
    
    GO terms must appear in at least TWO sources or have connections to both
    chemicals and diseases to ensure sufficient connectivity.
    
    Args:
        chem_go: Chemical-GO enriched associations lazy frame.
        pheno_ixns: Chemical-Phenotype interactions lazy frame.
        pheno_disease_bp: Phenotype-Disease BP associations lazy frame.
        pheno_disease_mf: Phenotype-Disease MF associations lazy frame.
        pheno_disease_cc: Phenotype-Disease CC associations lazy frame.
        chems_core: Core chemicals DataFrame.
        ds_core: Core diseases DataFrame.
        
    Returns:
        GO term nodes DataFrame with GO_TERM_ID, GO_SOURCE_ID, GO_NAME, GO_ONTOLOGY.
    """
    # Get GO terms from chemical-GO enriched (connected to core chemicals)
    go_from_chem_enriched = (
        chem_go
        .join(chems_core.lazy().select(['CHEM_MESH_ID']), on='CHEM_MESH_ID', how='inner')
        .select(['GO_ID', 'GO_NAME', 'GO_ONTOLOGY'])
        .unique('GO_ID')
        .with_columns(pl.lit(1).alias('IN_CHEM_GO'))
    )
    
    # Get GO terms from phenotype interactions (connected to core chemicals)
    go_from_pheno_ixns = (
        pheno_ixns
        .join(chems_core.lazy().select(['CHEM_MESH_ID']), on='CHEM_MESH_ID', how='inner')
        .select(['GO_ID', 'GO_NAME'])
        .unique('GO_ID')
        # Phenotype interactions are biological processes
        .with_columns([
            pl.lit('Biological Process').alias('GO_ONTOLOGY'),
            pl.lit(1).alias('IN_PHENO_IXNS')
        ])
    )
    
    # Get GO terms from phenotype-disease associations (connected to core diseases)
    pheno_disease_all = pl.concat([pheno_disease_bp, pheno_disease_mf, pheno_disease_cc], how='vertical')
    go_from_pheno_disease = (
        pheno_disease_all
        .join(ds_core.lazy().select(['DS_OMIM_MESH_ID']), on='DS_OMIM_MESH_ID', how='inner')
        .select(['GO_ID', 'GO_NAME', 'GO_ONTOLOGY'])
        .unique('GO_ID')
        .with_columns(pl.lit(1).alias('IN_PHENO_DISEASE'))
    )
    
    # Union all GO terms
    all_go_terms = (
        go_from_chem_enriched
        .join(go_from_pheno_ixns.select(['GO_ID', 'IN_PHENO_IXNS']), on='GO_ID', how='full', coalesce=True)
        .join(go_from_pheno_disease.select(['GO_ID', 'IN_PHENO_DISEASE']), on='GO_ID', how='full', coalesce=True)
        .with_columns([
            pl.col('IN_CHEM_GO').fill_null(0),
            pl.col('IN_PHENO_IXNS').fill_null(0),
            pl.col('IN_PHENO_DISEASE').fill_null(0),
        ])
        .with_columns(
            (pl.col('IN_CHEM_GO') + pl.col('IN_PHENO_IXNS') + pl.col('IN_PHENO_DISEASE'))
            .alias('SOURCE_COUNT')
        )
        # Must have chemical link AND disease link for meaningful connectivity
        .with_columns(
            ((pl.col('IN_CHEM_GO') + pl.col('IN_PHENO_IXNS')) > 0).alias('HAS_CHEM_LINK')
        )
        .with_columns(
            (pl.col('IN_PHENO_DISEASE') > 0).alias('HAS_DISEASE_LINK')
        )
        # Keep GO terms with both chemical and disease links
        .filter(pl.col('HAS_CHEM_LINK') & pl.col('HAS_DISEASE_LINK'))
        .select(['GO_ID', 'GO_NAME', 'GO_ONTOLOGY'])
        .sort('GO_ID')
    ).collect()
    
    # Rename GO_ID to GO_SOURCE_ID and add internal ID
    go_terms_core = (
        all_go_terms
        .rename({'GO_ID': 'GO_SOURCE_ID'})
        .with_row_index('GO_TERM_ID')
    )
    
    return go_terms_core


def build_edge_tables(
    ctd_chg: pl.LazyFrame,
    ctd_chd: pl.LazyFrame,
    ctd_dg: pl.LazyFrame,
    ppi: pl.LazyFrame,
    genes_core: pl.DataFrame,
    ds_core: pl.DataFrame,
    chems_core: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Build edge tables filtered to core entities.
    
    Args:
        ctd_chg: Chemical-Gene interactions lazy frame.
        ctd_chd: Chemical-Disease associations lazy frame.
        ctd_dg: Disease-Gene associations lazy frame.
        ppi: Protein-Protein interactions lazy frame.
        genes_core: Core genes DataFrame.
        ds_core: Core diseases DataFrame.
        chems_core: Core chemicals DataFrame.
        
    Returns:
        Tuple of (chem_gene_edges, disease_gene_edges, chem_disease_edges, 
                  ppi_edges, ppi_directed_edges) DataFrames.
    """
    # Chemical-Gene edges with interaction parsing
    ctd_chg_final = (
        ctd_chg.collect()
        .join(genes_core.select(['GENE_NCBI_ID', 'GENE_ID']), on='GENE_NCBI_ID', how='inner')
        .join(chems_core.select(['CHEM_MESH_ID', 'CHEM_ID']), on='CHEM_MESH_ID', how='inner')
        .with_columns([
            pl.col('INTERACTIONS').cast(pl.Utf8).fill_null('').alias('INTERACTIONS'),
            pl.col('INTERACTIONS').str.contains(r'\|').alias('has_list'),
            pl.col('INTERACTIONS').str.count_matches(r'\|').alias('n_pipes'),
            pl.col('INTERACTIONS').str.len_chars().alias('n_chars'),
        ])
        .group_by(['CHEM_MESH_ID', 'GENE_NCBI_ID'])
        .agg([
            pl.col('INTERACTIONS')
                .sort_by(['has_list', 'n_pipes', 'n_chars'], descending=True)
                .first()
                .alias('INTERACTIONS'),
            pl.col('PUBMED_IDS')
                .sort_by(['has_list', 'n_pipes', 'n_chars'], descending=True)
                .first()
                .alias('PUBMED_IDS'),
            pl.first('CHEM_ID').alias('CHEM_ID'),
            pl.first('GENE_ID').alias('GENE_ID'),
        ])
        .with_columns(
            pl.when(pl.col('INTERACTIONS') == '')
                .then(pl.lit(None, dtype=pl.List(pl.Utf8)))
                .otherwise(pl.col('INTERACTIONS').str.split('|'))
                .alias('INTERACTION_ITEM')
        )
        .explode('INTERACTION_ITEM')
        .drop_nulls('INTERACTION_ITEM')
        .with_columns(
            pl.col('INTERACTION_ITEM')
                .str.split_exact('^', 1)
                .struct.rename_fields(['ACTION_TYPE', 'ACTION_SUBJECT'])
                .alias('parts')
        )
        .unnest('parts')
        .with_columns(
            (
                pl.col('PUBMED_IDS').cast(pl.Utf8).fill_null('').str.count_matches(r'\|') +
                pl.when(pl.col('PUBMED_IDS').cast(pl.Utf8).fill_null('') != '').then(1).otherwise(0)
            ).alias('PUBMED_COUNT')
        )
        .with_columns((pl.col('PUBMED_COUNT') + 1).log().alias('LOG_PUBMED_COUNT'))
        .select(['CHEM_ID', 'GENE_ID', 'ACTION_TYPE', 'ACTION_SUBJECT', 'LOG_PUBMED_COUNT'])
        .unique()
        .with_row_index('CHEM_GENE_IDX')
    )
    
    # Disease-Gene edges
    ctd_dg_final = (
        ctd_dg.collect()
        .join(genes_core.select(['GENE_NCBI_ID', 'GENE_ID']), on='GENE_NCBI_ID')
        .join(ds_core.select(['DS_OMIM_MESH_ID', 'DS_ID']), on='DS_OMIM_MESH_ID')
        .with_columns([
            pl.col('DIRECT_EVIDENCE').cast(pl.Utf8).fill_null('unspecified').alias('DIRECT_EVIDENCE'),
            (
                pl.col('PUBMED_IDS').cast(pl.Utf8).fill_null('').str.count_matches(r'\|') +
                pl.when(pl.col('PUBMED_IDS').cast(pl.Utf8).fill_null('') != '').then(1).otherwise(0)
            ).alias('PUBMED_COUNT')
        ])
        .group_by(['GENE_ID', 'DS_ID'])
        .agg([
            pl.first('DIRECT_EVIDENCE').alias('DIRECT_EVIDENCE'),
            pl.max('PUBMED_COUNT').alias('PUBMED_COUNT')
        ])
        .with_columns([
            pl.col('DIRECT_EVIDENCE').replace_strict({
                'marker/mechanism': 0,
                'therapeutic': 1,
            }, default=2).alias('DIRECT_EVIDENCE_TYPE'),
            (pl.col('PUBMED_COUNT') + 1).log().alias('LOG_PUBMED_COUNT')
        ])
        .select(['GENE_ID', 'DS_ID', 'DIRECT_EVIDENCE_TYPE', 'LOG_PUBMED_COUNT'])
        .with_row_index('GENE_DS_IDX')
    )
    
    # Chemical-Disease edges
    ctd_chd_final = (
        ctd_chd.collect()
        .join(chems_core.select(['CHEM_MESH_ID', 'CHEM_ID']), on='CHEM_MESH_ID')
        .join(ds_core.select(['DS_OMIM_MESH_ID', 'DS_ID']), on='DS_OMIM_MESH_ID')
        .unique(['CHEM_MESH_ID', 'DS_OMIM_MESH_ID'])
        .select(['CHEM_ID', 'DS_ID'])
        .with_row_index('CHEM_DS_IDX')
    )
    
    # PPI edges (undirected)
    ppi_final = (
        ppi.collect()
        .join(
            genes_core.select([
                pl.col('GENE_NCBI_ID').alias('GENE_NCBI_ID_1'),
                pl.col('GENE_ID').alias('GENE_ID_1'),
            ]),
            on='GENE_NCBI_ID_1',
            how='inner',
        )
        .join(
            genes_core.select([
                pl.col('GENE_NCBI_ID').alias('GENE_NCBI_ID_2'),
                pl.col('GENE_ID').alias('GENE_ID_2'),
            ]),
            on='GENE_NCBI_ID_2',
            how='inner',
        )
        .filter(pl.col('GENE_ID_1') != pl.col('GENE_ID_2'))
        .with_columns([
            pl.min_horizontal('GENE_ID_1', 'GENE_ID_2').alias('GENE_ID_1'),
            pl.max_horizontal('GENE_ID_1', 'GENE_ID_2').alias('GENE_ID_2'),
        ])
        .select(['GENE_ID_1', 'GENE_ID_2'])
        .unique()
        .with_row_index('PPI_IDX')
    )
    
    # PPI directed edges (both directions)
    ppi_directed = pl.concat([
        ppi_final.select([
            pl.col('GENE_ID_1').alias('GENE_ID_1'),
            pl.col('GENE_ID_2').alias('GENE_ID_2'),
        ]),
        ppi_final.select([
            pl.col('GENE_ID_2').alias('GENE_ID_1'),
            pl.col('GENE_ID_1').alias('GENE_ID_2'),
        ]),
    ], how='vertical').with_row_index('PPI_DIR_IDX')
    
    return ctd_chg_final, ctd_dg_final, ctd_chd_final, ppi_final, ppi_directed


def build_pathway_edge_tables(
    gene_pathways: pl.LazyFrame,
    disease_pathways: pl.LazyFrame,
    chem_pathways: pl.LazyFrame,
    genes_core: pl.DataFrame,
    ds_core: pl.DataFrame,
    chems_core: pl.DataFrame,
    pathways_core: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Build pathway-related edge tables filtered to core entities.
    
    Args:
        gene_pathways: Gene-Pathway associations lazy frame.
        disease_pathways: Disease-Pathway associations lazy frame.
        chem_pathways: Chemical-Pathway enriched associations lazy frame.
        genes_core: Core genes DataFrame.
        ds_core: Core diseases DataFrame.
        chems_core: Core chemicals DataFrame.
        pathways_core: Core pathways DataFrame.
        
    Returns:
        Tuple of (gene_pathway_edges, disease_pathway_edges, chem_pathway_edges) DataFrames.
    """
    # Gene-Pathway edges
    gene_pathway_edges = (
        gene_pathways.collect()
        .join(genes_core.select(['GENE_NCBI_ID', 'GENE_ID']), on='GENE_NCBI_ID', how='inner')
        .join(
            pathways_core.select([
                pl.col('PATHWAY_SOURCE_ID').alias('PATHWAY_ID'),
                pl.col('PATHWAY_ID').alias('PATHWAY_NODE_ID')
            ]),
            on='PATHWAY_ID',
            how='inner'
        )
        .select(['GENE_ID', 'PATHWAY_NODE_ID'])
        .rename({'PATHWAY_NODE_ID': 'PATHWAY_ID'})
        .unique()
        .with_row_index('GENE_PATHWAY_IDX')
    )
    
    # Disease-Pathway edges (with inference gene count as attribute)
    disease_pathway_edges = (
        disease_pathways.collect()
        .join(ds_core.select(['DS_OMIM_MESH_ID', 'DS_ID']), on='DS_OMIM_MESH_ID', how='inner')
        .join(
            pathways_core.select([
                pl.col('PATHWAY_SOURCE_ID').alias('PATHWAY_ID'),
                pl.col('PATHWAY_ID').alias('PATHWAY_NODE_ID')
            ]),
            on='PATHWAY_ID',
            how='inner'
        )
        # Count inference genes per disease-pathway pair
        .group_by(['DS_ID', 'PATHWAY_NODE_ID'])
        .agg([
            pl.col('INFERENCE_GENE_SYMBOL').n_unique().alias('INFERENCE_GENE_COUNT')
        ])
        .rename({'PATHWAY_NODE_ID': 'PATHWAY_ID'})
        .with_row_index('DS_PATHWAY_IDX')
    )
    
    # Chemical-Pathway edges with enrichment statistics as edge attributes
    chem_pathway_edges = (
        chem_pathways.collect()
        .join(chems_core.select(['CHEM_MESH_ID', 'CHEM_ID']), on='CHEM_MESH_ID', how='inner')
        .join(
            pathways_core.select([
                pl.col('PATHWAY_SOURCE_ID').alias('PATHWAY_ID'),
                pl.col('PATHWAY_ID').alias('PATHWAY_NODE_ID')
            ]),
            on='PATHWAY_ID',
            how='inner'
        )
        .select([
            'CHEM_ID', 'PATHWAY_NODE_ID',
            'PValue', 'CorrectedPValue', 'TargetMatchQty', 'TargetTotalQty',
            'BackgroundMatchQty', 'BackgroundTotalQty'
        ])
        .rename({'PATHWAY_NODE_ID': 'PATHWAY_ID'})
        # Compute continuous edge features
        .with_columns([
            # -log10(corrected p-value), capped at 10 for numerical stability
            pl.when(pl.col('CorrectedPValue') > 0)
                .then((-pl.col('CorrectedPValue').log10()).clip(0, 10))
                .otherwise(pl.lit(10.0))
                .alias('NEG_LOG_PVALUE'),
            # Enrichment ratio
            (pl.col('TargetMatchQty') / pl.col('TargetTotalQty').clip(lower_bound=1))
            .alias('TARGET_RATIO'),
            # Fold enrichment
            ((pl.col('TargetMatchQty') / pl.col('TargetTotalQty').clip(lower_bound=1)) / 
             (pl.col('BackgroundMatchQty') / pl.col('BackgroundTotalQty').clip(lower_bound=1)).clip(lower_bound=1e-6))
            .clip(0, 100)
            .alias('FOLD_ENRICHMENT')
        ])
        .select(['CHEM_ID', 'PATHWAY_ID', 'NEG_LOG_PVALUE', 'TARGET_RATIO', 'FOLD_ENRICHMENT'])
        .unique(['CHEM_ID', 'PATHWAY_ID'])
        .with_row_index('CHEM_PATHWAY_IDX')
    )
    
    return gene_pathway_edges, disease_pathway_edges, chem_pathway_edges


def build_go_term_edge_tables(
    chem_go: pl.LazyFrame,
    pheno_ixns: pl.LazyFrame,
    pheno_disease_bp: pl.LazyFrame,
    pheno_disease_mf: pl.LazyFrame,
    pheno_disease_cc: pl.LazyFrame,
    chems_core: pl.DataFrame,
    ds_core: pl.DataFrame,
    go_terms_core: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Build GO term-related edge tables filtered to core entities.
    
    Args:
        chem_go: Chemical-GO enriched associations lazy frame.
        pheno_ixns: Chemical-Phenotype interactions lazy frame.
        pheno_disease_bp: Phenotype-Disease BP associations lazy frame.
        pheno_disease_mf: Phenotype-Disease MF associations lazy frame.
        pheno_disease_cc: Phenotype-Disease CC associations lazy frame.
        chems_core: Core chemicals DataFrame.
        ds_core: Core diseases DataFrame.
        go_terms_core: Core GO terms DataFrame.
        
    Returns:
        Tuple of (chem_go_edges, chem_pheno_edges, go_disease_edges) DataFrames.
    """
    # Create ontology mapping for categorical encoding
    ontology_map = {'Biological Process': 0, 'Molecular Function': 1, 'Cellular Component': 2}
    
    # Chemical-GO enriched edges with enrichment statistics
    chem_go_edges = (
        chem_go.collect()
        .join(chems_core.select(['CHEM_MESH_ID', 'CHEM_ID']), on='CHEM_MESH_ID', how='inner')
        .join(
            go_terms_core.select([
                pl.col('GO_SOURCE_ID').alias('GO_ID'),
                pl.col('GO_TERM_ID').alias('GO_NODE_ID')
            ]),
            on='GO_ID',
            how='inner'
        )
        .select([
            'CHEM_ID', 'GO_NODE_ID', 'GO_ONTOLOGY',
            'PValue', 'CorrectedPValue', 'TargetMatchQty', 'TargetTotalQty',
            'BackgroundMatchQty', 'BackgroundTotalQty', 'GO_LEVEL'
        ])
        .rename({'GO_NODE_ID': 'GO_TERM_ID'})
        # Compute continuous edge features
        .with_columns([
            # -log10(corrected p-value), capped at 10
            pl.when(pl.col('CorrectedPValue') > 0)
                .then((-pl.col('CorrectedPValue').log10()).clip(0, 10))
                .otherwise(pl.lit(10.0))
                .alias('NEG_LOG_PVALUE'),
            # Target ratio
            (pl.col('TargetMatchQty') / pl.col('TargetTotalQty').clip(lower_bound=1))
            .alias('TARGET_RATIO'),
            # Fold enrichment
            ((pl.col('TargetMatchQty') / pl.col('TargetTotalQty').clip(lower_bound=1)) / 
             (pl.col('BackgroundMatchQty') / pl.col('BackgroundTotalQty').clip(lower_bound=1)).clip(lower_bound=1e-6))
            .clip(0, 100)
            .alias('FOLD_ENRICHMENT'),
            # Ontology type as categorical
            pl.col('GO_ONTOLOGY').replace_strict(ontology_map, default=0).alias('ONTOLOGY_TYPE'),
            # GO level normalized (0-1)
            (pl.col('GO_LEVEL') / 10.0).clip(0, 1).alias('GO_LEVEL_NORM')
        ])
        .select(['CHEM_ID', 'GO_TERM_ID', 'NEG_LOG_PVALUE', 'TARGET_RATIO', 
                 'FOLD_ENRICHMENT', 'ONTOLOGY_TYPE', 'GO_LEVEL_NORM'])
        .unique(['CHEM_ID', 'GO_TERM_ID'])
        .with_row_index('CHEM_GO_IDX')
    )
    
    # Chemical-Phenotype interaction edges with action type
    action_type_map = {'increases': 0, 'decreases': 1, 'affects': 2}
    
    chem_pheno_edges = (
        pheno_ixns.collect()
        .join(chems_core.select(['CHEM_MESH_ID', 'CHEM_ID']), on='CHEM_MESH_ID', how='inner')
        .join(
            go_terms_core.select([
                pl.col('GO_SOURCE_ID').alias('GO_ID'),
                pl.col('GO_TERM_ID').alias('GO_NODE_ID')
            ]),
            on='GO_ID',
            how='inner'
        )
        .select(['CHEM_ID', 'GO_NODE_ID', 'INTERACTION_ACTIONS'])
        .rename({'GO_NODE_ID': 'GO_TERM_ID'})
        # Parse action type from INTERACTION_ACTIONS (e.g., "decreases^phenotype")
        .with_columns(
            pl.col('INTERACTION_ACTIONS')
                .str.extract(r'^(\w+)\^', 1)
                .fill_null('affects')
                .alias('ACTION_TYPE_STR')
        )
        .with_columns(
            pl.col('ACTION_TYPE_STR').replace_strict(action_type_map, default=2).alias('PHENO_ACTION_TYPE')
        )
        .group_by(['CHEM_ID', 'GO_TERM_ID'])
        .agg([
            # Take most specific action (increases/decreases over affects)
            pl.col('PHENO_ACTION_TYPE').min().alias('PHENO_ACTION_TYPE')
        ])
        .with_row_index('CHEM_PHENO_IDX')
    )
    
    # GO term-Disease edges (phenotype-disease associations)
    pheno_disease_all = pl.concat([
        pheno_disease_bp, pheno_disease_mf, pheno_disease_cc
    ], how='vertical')
    
    go_disease_edges = (
        pheno_disease_all.collect()
        .join(ds_core.select(['DS_OMIM_MESH_ID', 'DS_ID']), on='DS_OMIM_MESH_ID', how='inner')
        .join(
            go_terms_core.select([
                pl.col('GO_SOURCE_ID').alias('GO_ID'),
                pl.col('GO_TERM_ID').alias('GO_NODE_ID'),
                pl.col('GO_ONTOLOGY')
            ]),
            on='GO_ID',
            how='inner'
        )
        .select([
            'GO_NODE_ID', 'DS_ID', 'GO_ONTOLOGY',
            'INFERENCE_CHEM_QTY', 'INFERENCE_GENE_QTY'
        ])
        .rename({'GO_NODE_ID': 'GO_TERM_ID'})
        # Aggregate duplicate (GO, disease) pairs
        .group_by(['GO_TERM_ID', 'DS_ID'])
        .agg([
            pl.first('GO_ONTOLOGY').alias('GO_ONTOLOGY'),
            pl.col('INFERENCE_CHEM_QTY').max().alias('INFERENCE_CHEM_QTY'),
            pl.col('INFERENCE_GENE_QTY').max().alias('INFERENCE_GENE_QTY')
        ])
        .with_columns([
            # Ontology type as categorical
            pl.col('GO_ONTOLOGY').replace_strict(ontology_map, default=0).alias('ONTOLOGY_TYPE'),
            # Log-scaled inference counts as edge features
            (pl.col('INFERENCE_CHEM_QTY') + 1).log().alias('LOG_INFERENCE_CHEM'),
            (pl.col('INFERENCE_GENE_QTY') + 1).log().alias('LOG_INFERENCE_GENE')
        ])
        .select(['GO_TERM_ID', 'DS_ID', 'ONTOLOGY_TYPE', 'LOG_INFERENCE_CHEM', 'LOG_INFERENCE_GENE'])
        .with_row_index('GO_DS_IDX')
    )
    
    return chem_go_edges, chem_pheno_edges, go_disease_edges


def build_interaction_type_hierarchy(
    ixn_types: pl.LazyFrame
) -> pl.DataFrame:
    """
    Build interaction type hierarchy table with parent codes for hierarchical embeddings.
    
    Args:
        ixn_types: Interaction types lazy frame.
        
    Returns:
        DataFrame with IXN_TYPE_ID, IXN_TYPE_CODE, IXN_TYPE_NAME, IXN_TYPE_PARENT_ID.
    """
    ixn_types_df = ixn_types.collect()
    
    # Create ID mapping
    ixn_types_with_id = ixn_types_df.with_row_index('IXN_TYPE_ID')
    
    # Create code to ID mapping
    code_to_id = dict(zip(
        ixn_types_with_id['IXN_TYPE_CODE'].to_list(),
        ixn_types_with_id['IXN_TYPE_ID'].to_list()
    ))
    
    # Map parent codes to parent IDs
    ixn_types_final = (
        ixn_types_with_id
        .with_columns(
            pl.col('IXN_TYPE_PARENT_CODE')
                .map_elements(lambda x: code_to_id.get(x, -1) if x else -1, return_dtype=pl.Int64)
                .alias('IXN_TYPE_PARENT_ID')
        )
        .select(['IXN_TYPE_ID', 'IXN_TYPE_CODE', 'IXN_TYPE_NAME', 'IXN_TYPE_DESC', 'IXN_TYPE_PARENT_ID'])
    )
    
    return ixn_types_final


def process_and_save(
    raw_data_dir: str = './data/raw',
    processed_data_dir: str = './data/processed'
) -> Dict[str, pl.DataFrame]:
    """
    Process raw data and save to parquet files.
    
    Args:
        raw_data_dir: Path to raw data directory.
        processed_data_dir: Path to processed data directory.
        
    Returns:
        Dictionary containing all processed DataFrames.
    """
    import os
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Load raw data
    print("Loading raw data...")
    raw_data = load_raw_data(raw_data_dir)
    
    # BUILD CORE ENTITIES
    print("\nBuilding core entities (chemicals, diseases, genes)...")
    genes_core, ds_core, chems_core = build_core_entities(
        raw_data['ctd_chg'],
        raw_data['ctd_chd'],
        raw_data['ctd_dg'],
        raw_data['ppi'],
        raw_data['chem_annots'],
        raw_data['ds_annots'],
        raw_data['gene_annots']
    )
    
    print(f'  Genes: {genes_core.shape[0]}')
    print(f'  Diseases: {ds_core.shape[0]}')
    print(f'  Chemicals: {chems_core.shape[0]}')
    
    # BUILD PATHWAY ENTITIES
    print("\nBuilding pathway entities...")
    pathways_core = build_pathway_entities(
        raw_data['gene_pathways'],
        raw_data['disease_pathways'],
        raw_data['chem_pathways'],
        genes_core,
        ds_core,
        chems_core
    )
    print(f'  Pathways: {pathways_core.shape[0]}')
    
    # BUILD GO TERM ENTITIES
    print("\nBuilding GO term entities...")
    go_terms_core = build_go_term_entities(
        raw_data['chem_go'],
        raw_data['pheno_ixns'],
        raw_data['pheno_disease_bp'],
        raw_data['pheno_disease_mf'],
        raw_data['pheno_disease_cc'],
        chems_core,
        ds_core
    )
    print(f'  GO terms: {go_terms_core.shape[0]}')
    
    # BUILD INTERACTION TYPE HIERARCHY
    print("\nBuilding interaction type hierarchy...")
    ixn_types = build_interaction_type_hierarchy(raw_data['ixn_types'])
    print(f'  Interaction types: {ixn_types.shape[0]}')
    
    # BUILD CORE EDGE TABLES
    print("\nBuilding original edge tables...")
    cg, dg, cd, ppi, ppi_directed = build_edge_tables(
        raw_data['ctd_chg'],
        raw_data['ctd_chd'],
        raw_data['ctd_dg'],
        raw_data['ppi'],
        genes_core,
        ds_core,
        chems_core
    )
    
    print(f'  Chemical-Gene edges: {cg.shape[0]}')
    print(f'  Disease-Gene edges: {dg.shape[0]}')
    print(f'  Chemical-Disease edges: {cd.shape[0]}')
    print(f'  PPI edges: {ppi.shape[0]}')
    
    # BUILD PATHWAY EDGE TABLES
    print("\nBuilding pathway edge tables...")
    gene_pathway_edges, disease_pathway_edges, chem_pathway_edges = build_pathway_edge_tables(
        raw_data['gene_pathways'],
        raw_data['disease_pathways'],
        raw_data['chem_pathways'],
        genes_core,
        ds_core,
        chems_core,
        pathways_core
    )
    
    print(f'  Gene-Pathway edges: {gene_pathway_edges.shape[0]}')
    print(f'  Disease-Pathway edges: {disease_pathway_edges.shape[0]}')
    print(f'  Chemical-Pathway edges: {chem_pathway_edges.shape[0]}')
    
    # BUILD GO TERM EDGE TABLES
    print("\nBuilding GO term edge tables...")
    chem_go_edges, chem_pheno_edges, go_disease_edges = build_go_term_edge_tables(
        raw_data['chem_go'],
        raw_data['pheno_ixns'],
        raw_data['pheno_disease_bp'],
        raw_data['pheno_disease_mf'],
        raw_data['pheno_disease_cc'],
        chems_core,
        ds_core,
        go_terms_core
    )
    
    print(f'  Chemical-GO (enriched) edges: {chem_go_edges.shape[0]}')
    print(f'  Chemical-Phenotype edges: {chem_pheno_edges.shape[0]}')
    print(f'  GO-Disease edges: {go_disease_edges.shape[0]}')
    
    total_nodes = genes_core.shape[0] + ds_core.shape[0] + chems_core.shape[0] + \
                  pathways_core.shape[0] + go_terms_core.shape[0]
    total_edges = cg.shape[0] + dg.shape[0] + cd.shape[0] + ppi.shape[0] + \
                  gene_pathway_edges.shape[0] + disease_pathway_edges.shape[0] + \
                  chem_pathway_edges.shape[0] + chem_go_edges.shape[0] + \
                  chem_pheno_edges.shape[0] + go_disease_edges.shape[0]
    
    print(f'\n{"="*60}')
    print(f'TOTAL NODES: {total_nodes:,}')
    print(f'TOTAL EDGES: {total_edges:,}')
    print(f'{"="*60}')
    
    print(f'\nSaving to {processed_data_dir}...')
    processed_path = Path(processed_data_dir)
    
    # Node tables
    genes_core.write_parquet(processed_path / 'genes_nodes.parquet')
    ds_core.write_parquet(processed_path / 'diseases_nodes.parquet')
    chems_core.write_parquet(processed_path / 'chemicals_nodes.parquet')
    pathways_core.write_parquet(processed_path / 'pathways_nodes.parquet')
    go_terms_core.write_parquet(processed_path / 'go_terms_nodes.parquet')
    
    # Core edge tables
    cg.write_parquet(processed_path / 'chem_gene_edges.parquet')
    dg.write_parquet(processed_path / 'disease_gene_edges.parquet')
    cd.write_parquet(processed_path / 'chem_disease_edges.parquet')
    ppi.write_parquet(processed_path / 'ppi_edges.parquet')
    ppi_directed.write_parquet(processed_path / 'ppi_directed_edges.parquet')
    
    # Edge tables (pathway)
    gene_pathway_edges.write_parquet(processed_path / 'gene_pathway_edges.parquet')
    disease_pathway_edges.write_parquet(processed_path / 'disease_pathway_edges.parquet')
    chem_pathway_edges.write_parquet(processed_path / 'chem_pathway_edges.parquet')
    
    # Edge tables (GO terms)
    chem_go_edges.write_parquet(processed_path / 'chem_go_edges.parquet')
    chem_pheno_edges.write_parquet(processed_path / 'chem_pheno_edges.parquet')
    go_disease_edges.write_parquet(processed_path / 'go_disease_edges.parquet')
    
    # Vocabulary tables
    ixn_types.write_parquet(processed_path / 'interaction_types.parquet')
    
    print('Writing done!')
    
    return {
        # Nodes
        'genes': genes_core,
        'diseases': ds_core,
        'chemicals': chems_core,
        'pathways': pathways_core,
        'go_terms': go_terms_core,
        # Core edges
        'chem_gene': cg,
        'disease_gene': dg,
        'chem_disease': cd,
        'ppi': ppi,
        'ppi_directed': ppi_directed,
        # Pathway edges
        'gene_pathway': gene_pathway_edges,
        'disease_pathway': disease_pathway_edges,
        'chem_pathway': chem_pathway_edges,
        # GO term edges
        'chem_go': chem_go_edges,
        'chem_pheno': chem_pheno_edges,
        'go_disease': go_disease_edges,
        # Vocabulary
        'interaction_types': ixn_types
    }


def load_processed_data(processed_data_dir: str = './data/processed') -> Dict[str, pl.DataFrame]:
    """
    Load processed data from parquet files.
    
    Args:
        processed_data_dir: Path to processed data directory.
        
    Returns:
        Dictionary containing all processed DataFrames.
    """
    processed_path = Path(processed_data_dir)
    
    result = {
        # Nodes
        'genes': pl.read_parquet(processed_path / 'genes_nodes.parquet'),
        'diseases': pl.read_parquet(processed_path / 'diseases_nodes.parquet'),
        'chemicals': pl.read_parquet(processed_path / 'chemicals_nodes.parquet'),
        # Core edges
        'chem_gene': pl.read_parquet(processed_path / 'chem_gene_edges.parquet'),
        'disease_gene': pl.read_parquet(processed_path / 'disease_gene_edges.parquet'),
        'chem_disease': pl.read_parquet(processed_path / 'chem_disease_edges.parquet'),
        'ppi': pl.read_parquet(processed_path / 'ppi_edges.parquet'),
        'ppi_directed': pl.read_parquet(processed_path / 'ppi_directed_edges.parquet'),
    }
    
    # Load new node/edge tables if they exist
    new_files = {
        'pathways': 'pathways_nodes.parquet',
        'go_terms': 'go_terms_nodes.parquet',
        'gene_pathway': 'gene_pathway_edges.parquet',
        'disease_pathway': 'disease_pathway_edges.parquet',
        'chem_pathway': 'chem_pathway_edges.parquet',
        'chem_go': 'chem_go_edges.parquet',
        'chem_pheno': 'chem_pheno_edges.parquet',
        'go_disease': 'go_disease_edges.parquet',
        'interaction_types': 'interaction_types.parquet'
    }
    
    for key, filename in new_files.items():
        filepath = processed_path / filename
        if filepath.exists():
            result[key] = pl.read_parquet(filepath)
    
    return result


if __name__ == '__main__':
    process_and_save()
