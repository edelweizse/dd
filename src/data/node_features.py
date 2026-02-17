"""
Node feature engineering for inductive heterogeneous graph learning.

This module builds biologically meaningful node features for all node types:
- chemical: structure + text + hierarchy
- disease: text + hierarchy
- gene: sequence composition + text
- pathway: text + source ontology
- go_term: text + ontology

Feature tables are saved as parquet files and can be plugged into graph building.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .processing import load_processed_data


AA_ORDER = list('ACDEFGHIKLMNPQRSTVWY')
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}


@dataclass
class NodeFeatureConfig:
    text_dim: int = 128
    chem_fp_bits: int = 1024
    include_pubchem: bool = True
    include_uniprot: bool = True
    include_umls: bool = False
    umls_api_key: Optional[str] = None
    disgenet_file: Optional[str] = None
    request_timeout_s: int = 20
    sleep_s: float = 0.02
    max_pubchem_fetch: Optional[int] = None
    max_uniprot_fetch: Optional[int] = None
    max_umls_fetch: Optional[int] = None
    random_state: int = 42


def _clean_text(x: object) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and np.isnan(x):
        return ''
    text = str(x).strip()
    return '' if text.lower() in {'none', 'nan', 'null'} else text


def _split_multi_value(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r'[|;,]', text)
    return [p.strip() for p in parts if p.strip()]


def _count_multi_value(text: str) -> float:
    return float(len(_split_multi_value(_clean_text(text))))


def _build_text_row(parts: Iterable[object]) -> str:
    cleaned = [_clean_text(p) for p in parts]
    cleaned = [c for c in cleaned if c]
    return ' [SEP] '.join(cleaned)


def _safe_float_matrix(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat


def _zscore_columns(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if mat.size == 0:
        return mat
    mean = mat.mean(axis=0, keepdims=True)
    std = mat.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (mat - mean) / std


def _fit_text_embedding(
    texts: List[str],
    text_dim: int,
    random_state: int,
) -> np.ndarray:
    texts = [t if t else '[EMPTY]' for t in texts]
    if len(texts) == 0:
        return np.zeros((0, text_dim), dtype=np.float32)

    min_df = 1 if len(texts) < 100 else 2
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
        max_features=50000,
    )
    x = vectorizer.fit_transform(texts)

    max_comp = min(text_dim, max(1, x.shape[0] - 1), max(1, x.shape[1] - 1))
    if max_comp >= 2:
        svd = TruncatedSVD(n_components=max_comp, random_state=random_state)
        emb = svd.fit_transform(x).astype(np.float32)
    else:
        dense = x.toarray().astype(np.float32)
        emb = dense[:, :max_comp]

    if emb.shape[1] < text_dim:
        pad = np.zeros((emb.shape[0], text_dim - emb.shape[1]), dtype=np.float32)
        emb = np.concatenate([emb, pad], axis=1)
    elif emb.shape[1] > text_dim:
        emb = emb[:, :text_dim]

    return _safe_float_matrix(emb)


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _http_get_json(url: str, timeout: int) -> Optional[Dict[str, object]]:
    try:
        import requests
    except Exception:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _http_get_text(url: str, timeout: int) -> Optional[str]:
    try:
        import requests
    except Exception:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception:
        return None


def _first_token(val: object) -> str:
    text = _clean_text(val)
    if not text:
        return ''
    return re.split(r'[|;,\s]+', text)[0].strip()


def _fetch_smiles_pubchem(
    cid: str,
    timeout: int,
) -> str:
    url = (
        f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/'
        'property/CanonicalSMILES/JSON'
    )
    payload = _http_get_json(url, timeout=timeout)
    if not payload:
        return ''
    try:
        return str(payload['PropertyTable']['Properties'][0]['CanonicalSMILES'])
    except Exception:
        return ''


def _fetch_uniprot_fasta(accession: str, timeout: int) -> str:
    url = f'https://rest.uniprot.org/uniprotkb/{accession}.fasta'
    text = _http_get_text(url, timeout=timeout)
    if not text:
        return ''
    seq = ''.join(line.strip() for line in text.splitlines() if line and not line.startswith('>'))
    return re.sub(r'[^A-Z]', '', seq)


def _fetch_umls_definition(term: str, api_key: str, timeout: int) -> str:
    term = _clean_text(term)
    if not term:
        return ''
    search_url = (
        'https://uts-ws.nlm.nih.gov/rest/search/current'
        f'?string={quote(term)}&searchType=exact&apiKey={quote(api_key)}'
    )
    search_payload = _http_get_json(search_url, timeout=timeout)
    if not search_payload:
        return ''

    try:
        results = search_payload['result']['results']
    except Exception:
        return ''
    if not results:
        return ''

    cui = ''
    for item in results:
        ui = _clean_text(item.get('ui'))
        if ui and ui != 'NONE':
            cui = ui
            break
    if not cui:
        return ''

    def_url = (
        f'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{quote(cui)}/definitions'
        f'?apiKey={quote(api_key)}'
    )
    def_payload = _http_get_json(def_url, timeout=timeout)
    if not def_payload:
        return ''
    try:
        defs = def_payload['result']
    except Exception:
        return ''
    if not defs:
        return ''

    for item in defs:
        value = _clean_text(item.get('value'))
        if value:
            return value
    return ''


def _sequence_features(seq: str) -> np.ndarray:
    seq = _clean_text(seq).upper()
    if not seq:
        return np.zeros(len(AA_ORDER) + 7, dtype=np.float32)

    counts = np.zeros(len(AA_ORDER), dtype=np.float32)
    valid = 0
    for aa in seq:
        idx = AA_INDEX.get(aa)
        if idx is not None:
            counts[idx] += 1
            valid += 1

    length = max(valid, 1)
    comp = counts / float(length)
    aromatic = sum(comp[AA_INDEX[a]] for a in ('F', 'W', 'Y'))
    acidic = sum(comp[AA_INDEX[a]] for a in ('D', 'E'))
    basic = sum(comp[AA_INDEX[a]] for a in ('K', 'R', 'H'))
    polar = sum(comp[AA_INDEX[a]] for a in ('S', 'T', 'N', 'Q'))
    gly_pro = sum(comp[AA_INDEX[a]] for a in ('G', 'P'))
    unknown_frac = max(0.0, (len(seq) - valid) / float(max(len(seq), 1)))
    tail = np.array(
        [
            np.log1p(float(length)),
            aromatic,
            acidic,
            basic,
            polar,
            gly_pro,
            unknown_frac,
        ],
        dtype=np.float32,
    )
    return np.concatenate([comp, tail], axis=0)


def _smiles_features(smiles: str, fp_bits: int) -> np.ndarray:
    smiles = _clean_text(smiles)
    if not smiles:
        return np.zeros(fp_bits + 9, dtype=np.float32)

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    except Exception:
        return np.zeros(fp_bits + 9, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fp_bits + 9, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_bits)
    fp_arr = np.zeros((fp_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    desc = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            float(rdMolDescriptors.CalcNumHBA(mol)),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            float(rdMolDescriptors.CalcTPSA(mol)),
            float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            1.0,
        ],
        dtype=np.float32,
    )
    desc[:-1] = np.log1p(np.clip(desc[:-1], a_min=0.0, a_max=None))
    return np.concatenate([fp_arr, desc], axis=0)


def _to_feature_df(id_col: str, ids: np.ndarray, feat: np.ndarray) -> pl.DataFrame:
    feat = _safe_float_matrix(feat)
    cols = {id_col: ids.astype(np.uint32)}
    for i in range(feat.shape[1]):
        cols[f'f_{i:04d}'] = feat[:, i]
    return pl.DataFrame(cols)


def _load_chem_annots(raw_data_dir: Path) -> pl.DataFrame:
    path = raw_data_dir / 'CTD_chemicals.tsv'
    return pl.read_csv(path, separator='\t', comment_prefix='#').select([
        pl.col('ChemicalID').alias('CHEM_MESH_ID'),
        pl.col('PubChemCID').alias('CHEM_PUBCHEM_CID'),
        pl.col('InChIKey').alias('CHEM_INCHIKEY'),
        pl.col('Definition').alias('CHEM_DEFINITION_RAW'),
        pl.col('MESHSynonyms').alias('CHEM_MESH_SYNONYMS'),
        pl.col('CTDCuratedSynonyms').alias('CHEM_CURATED_SYNONYMS'),
        pl.col('TreeNumbers').alias('CHEM_TREE_NUMBERS_RAW'),
        pl.col('ParentIDs').alias('CHEM_PARENT_IDS_RAW'),
    ])


def _load_gene_annots(raw_data_dir: Path) -> pl.DataFrame:
    path = raw_data_dir / 'CTD_genes.tsv'
    return pl.read_csv(
        path,
        separator='\t',
        comment_prefix='#',
        schema_overrides={
            'BioGRIDIDs': pl.String,
            'UniProtIDs': pl.String,
            'Synonyms': pl.String,
            'GeneName': pl.String,
            'GeneID': pl.Int64,
        },
    ).select([
        pl.col('GeneID').cast(pl.UInt32).alias('GENE_NCBI_ID'),
        pl.col('UniProtIDs').alias('GENE_UNIPROT_IDS_RAW'),
        pl.col('GeneName').alias('GENE_NAME_RAW'),
        pl.col('Synonyms').alias('GENE_SYNONYMS_RAW'),
    ])


def build_node_feature_tables(
    processed_data_dir: str = './data/processed',
    raw_data_dir: str = './data/raw',
    output_dir: Optional[str] = None,
    config: Optional[NodeFeatureConfig] = None,
) -> Dict[str, pl.DataFrame]:
    """
    Build and save inductive node feature tables for all node types.

    Returns dict keyed by node type.
    """
    cfg = config or NodeFeatureConfig()
    processed_path = Path(processed_data_dir)
    raw_path = Path(raw_data_dir)
    out_path = Path(output_dir) if output_dir else (processed_path / 'features')
    out_path.mkdir(parents=True, exist_ok=True)
    cache_path = out_path / 'cache'
    cache_path.mkdir(parents=True, exist_ok=True)

    data = load_processed_data(processed_data_dir)
    chem_nodes = data['chemicals']
    ds_nodes = data['diseases']
    gene_nodes = data['genes']
    pathway_nodes = data.get('pathways', pl.DataFrame())
    go_nodes = data.get('go_terms', pl.DataFrame())

    chem_annots = _load_chem_annots(raw_path)
    gene_annots = _load_gene_annots(raw_path)

    disgenet_disease_stats: Dict[str, Tuple[float, float, float]] = {}
    disgenet_gene_stats: Dict[str, Tuple[float, float, float]] = {}
    if cfg.disgenet_file:
        dg_path = Path(cfg.disgenet_file)
        if dg_path.exists():
            sep = '\t' if dg_path.suffix.lower() == '.tsv' else ','
            dg_df = pl.read_csv(dg_path, separator=sep, ignore_errors=True)
            dg_cols = set(dg_df.columns)

            disease_col = next((c for c in ['diseaseName', 'disease_name', 'disease'] if c in dg_cols), None)
            gene_col = next((c for c in ['geneSymbol', 'gene_symbol', 'gene'] if c in dg_cols), None)
            score_col = next((c for c in ['score', 'Score'] if c in dg_cols), None)

            if disease_col is not None:
                tmp = dg_df.with_columns([
                    pl.col(disease_col).cast(pl.Utf8).str.to_lowercase().alias('_d_name'),
                    (pl.col(score_col).cast(pl.Float64) if score_col else pl.lit(1.0)).alias('_score')
                ]).filter(pl.col('_d_name').is_not_null())
                agg = tmp.group_by('_d_name').agg([
                    pl.len().alias('DG_COUNT'),
                    pl.col('_score').mean().alias('DG_MEAN_SCORE'),
                    pl.col('_score').max().alias('DG_MAX_SCORE')
                ])
                for row in agg.iter_rows(named=True):
                    disgenet_disease_stats[row['_d_name']] = (
                        float(row['DG_COUNT']),
                        float(row['DG_MEAN_SCORE']),
                        float(row['DG_MAX_SCORE']),
                    )

            if gene_col is not None:
                tmp = dg_df.with_columns([
                    pl.col(gene_col).cast(pl.Utf8).str.to_uppercase().alias('_g_symbol'),
                    (pl.col(score_col).cast(pl.Float64) if score_col else pl.lit(1.0)).alias('_score')
                ]).filter(pl.col('_g_symbol').is_not_null())
                agg = tmp.group_by('_g_symbol').agg([
                    pl.len().alias('DG_COUNT'),
                    pl.col('_score').mean().alias('DG_MEAN_SCORE'),
                    pl.col('_score').max().alias('DG_MAX_SCORE')
                ])
                for row in agg.iter_rows(named=True):
                    disgenet_gene_stats[row['_g_symbol']] = (
                        float(row['DG_COUNT']),
                        float(row['DG_MEAN_SCORE']),
                        float(row['DG_MAX_SCORE']),
                    )

    umls_key = cfg.umls_api_key or os.environ.get('UMLS_API_KEY')
    umls_cache_file = cache_path / 'umls_definitions.json'
    umls_cache = _load_json(umls_cache_file)
    if cfg.include_umls and umls_key:
        terms = []
        terms.extend([_clean_text(x) for x in chem_nodes['CHEM_NAME'].to_list()])
        terms.extend([_clean_text(x) for x in ds_nodes['DS_NAME'].to_list()])
        uniq_terms = [t for t in dict.fromkeys(terms) if t]
        if cfg.max_umls_fetch is not None:
            uniq_terms = uniq_terms[:cfg.max_umls_fetch]
        for term in uniq_terms:
            if term in umls_cache:
                continue
            umls_cache[term] = _fetch_umls_definition(term, api_key=umls_key, timeout=cfg.request_timeout_s)
            if cfg.sleep_s > 0:
                time.sleep(cfg.sleep_s)
        _save_json(umls_cache_file, umls_cache)

    # ------------------------------------------------------------------
    # Chemicals: text + structure + hierarchy
    # ------------------------------------------------------------------
    chem = chem_nodes.join(chem_annots, on='CHEM_MESH_ID', how='left')

    chem_texts = [
        _build_text_row([
            row.get('CHEM_NAME'),
            row.get('CHEM_DEFINITION'),
            row.get('CHEM_DEFINITION_RAW'),
            row.get('CHEM_SYNONYMS'),
            row.get('CHEM_MESH_SYNONYMS'),
            row.get('CHEM_CURATED_SYNONYMS'),
            umls_cache.get(_clean_text(row.get('CHEM_NAME')), ''),
        ])
        for row in chem.iter_rows(named=True)
    ]
    chem_text_feat = _fit_text_embedding(chem_texts, cfg.text_dim, cfg.random_state)

    smiles_cache_file = cache_path / 'pubchem_smiles.json'
    smiles_cache = _load_json(smiles_cache_file)
    if cfg.include_pubchem:
        cid_series = chem['CHEM_PUBCHEM_CID'] if 'CHEM_PUBCHEM_CID' in chem.columns else pl.Series([])
        cids = [_clean_text(v) for v in cid_series.to_list()] if cid_series.len() > 0 else [''] * chem.height
        if cfg.max_pubchem_fetch is not None:
            remaining = cfg.max_pubchem_fetch
        else:
            remaining = None
        for cid in cids:
            if not cid or cid in smiles_cache:
                continue
            if remaining is not None and remaining <= 0:
                break
            smiles_cache[cid] = _fetch_smiles_pubchem(cid, timeout=cfg.request_timeout_s)
            if remaining is not None:
                remaining -= 1
            if cfg.sleep_s > 0:
                time.sleep(cfg.sleep_s)
        _save_json(smiles_cache_file, smiles_cache)

    chem_struct = []
    chem_meta = []
    for row in chem.iter_rows(named=True):
        cid = _clean_text(row.get('CHEM_PUBCHEM_CID'))
        smiles = _clean_text(smiles_cache.get(cid, '')) if cid else ''
        chem_struct.append(_smiles_features(smiles, fp_bits=cfg.chem_fp_bits))

        tree_numbers = _clean_text(row.get('CHEM_TREE_NUMBERS') or row.get('CHEM_TREE_NUMBERS_RAW'))
        parents = _clean_text(row.get('CHEM_PARENT_IDS') or row.get('CHEM_PARENT_IDS_RAW'))
        tree_parts = _split_multi_value(tree_numbers)
        avg_depth = 0.0
        if tree_parts:
            avg_depth = float(np.mean([t.count('.') + 1 for t in tree_parts]))
        chem_meta.append([
            _count_multi_value(tree_numbers),
            _count_multi_value(parents),
            avg_depth,
            1.0 if smiles else 0.0,
        ])

    chem_struct_feat = _safe_float_matrix(np.vstack(chem_struct))
    chem_meta_feat = _zscore_columns(_safe_float_matrix(np.asarray(chem_meta, dtype=np.float32)))
    chem_feat = np.concatenate([chem_struct_feat, chem_meta_feat, chem_text_feat], axis=1)
    chemical_features_df = _to_feature_df('CHEM_ID', chem['CHEM_ID'].to_numpy(), chem_feat)

    # ------------------------------------------------------------------
    # Diseases: text + hierarchy
    # ------------------------------------------------------------------
    ds_texts = [
        _build_text_row([
            row.get('DS_NAME'),
            row.get('DS_DEFINITION'),
            row.get('DS_SYNONYMS'),
            row.get('DS_SLIM_MAPPINGS'),
            row.get('DS_OMIM_MESH_ID'),
            umls_cache.get(_clean_text(row.get('DS_NAME')), ''),
        ])
        for row in ds_nodes.iter_rows(named=True)
    ]
    ds_text_feat = _fit_text_embedding(ds_texts, cfg.text_dim, cfg.random_state)

    ds_meta = []
    for row in ds_nodes.iter_rows(named=True):
        tree_numbers = _clean_text(row.get('DS_TREE_NUMBERS'))
        parents = _clean_text(row.get('DS_PARENT_IDS'))
        slim = _clean_text(row.get('DS_SLIM_MAPPINGS'))
        tree_parts = _split_multi_value(tree_numbers)
        avg_depth = 0.0
        if tree_parts:
            avg_depth = float(np.mean([t.count('.') + 1 for t in tree_parts]))
        dg_key = _clean_text(row.get('DS_NAME')).lower()
        dg_count, dg_mean, dg_max = disgenet_disease_stats.get(dg_key, (0.0, 0.0, 0.0))
        ds_meta.append([
            _count_multi_value(tree_numbers),
            _count_multi_value(parents),
            _count_multi_value(slim),
            avg_depth,
            np.log1p(dg_count),
            dg_mean,
            dg_max,
        ])
    ds_meta_feat = _zscore_columns(_safe_float_matrix(np.asarray(ds_meta, dtype=np.float32)))
    ds_feat = np.concatenate([ds_meta_feat, ds_text_feat], axis=1)
    disease_features_df = _to_feature_df('DS_ID', ds_nodes['DS_ID'].to_numpy(), ds_feat)

    # ------------------------------------------------------------------
    # Genes: sequence + text
    # ------------------------------------------------------------------
    genes = gene_nodes.join(gene_annots, on='GENE_NCBI_ID', how='left')
    seq_cache_file = cache_path / 'uniprot_sequences.json'
    seq_cache = _load_json(seq_cache_file)

    if cfg.include_uniprot:
        accs = []
        for val in genes['GENE_UNIPROT_IDS'].to_list() if 'GENE_UNIPROT_IDS' in genes.columns else []:
            acc = _first_token(val)
            if acc:
                accs.append(acc)
        accs_unique = list(dict.fromkeys(accs))
        if cfg.max_uniprot_fetch is not None:
            accs_unique = accs_unique[:cfg.max_uniprot_fetch]
        for acc in accs_unique:
            if acc in seq_cache:
                continue
            seq_cache[acc] = _fetch_uniprot_fasta(acc, timeout=cfg.request_timeout_s)
            if cfg.sleep_s > 0:
                time.sleep(cfg.sleep_s)
        _save_json(seq_cache_file, seq_cache)

    gene_seq_feat = []
    gene_texts = []
    gene_meta = []
    for row in genes.iter_rows(named=True):
        acc = _first_token(row.get('GENE_UNIPROT_IDS'))
        seq = _clean_text(seq_cache.get(acc, '')) if acc else ''
        gene_seq_feat.append(_sequence_features(seq))

        gene_texts.append(_build_text_row([
            row.get('GENE_SYMBOL'),
            row.get('GENE_NAME'),
            row.get('GENE_NAME_RAW'),
            row.get('GENE_SYNONYMS'),
            row.get('GENE_SYNONYMS_RAW'),
        ]))

        g_key = _clean_text(row.get('GENE_SYMBOL')).upper()
        dg_count, dg_mean, dg_max = disgenet_gene_stats.get(g_key, (0.0, 0.0, 0.0))
        gene_meta.append([np.log1p(dg_count), dg_mean, dg_max])

    gene_seq_feat = _zscore_columns(_safe_float_matrix(np.vstack(gene_seq_feat)))
    gene_text_feat = _fit_text_embedding(gene_texts, cfg.text_dim, cfg.random_state)
    gene_meta_feat = _zscore_columns(_safe_float_matrix(np.asarray(gene_meta, dtype=np.float32)))
    gene_feat = np.concatenate([gene_seq_feat, gene_meta_feat, gene_text_feat], axis=1)
    gene_features_df = _to_feature_df('GENE_ID', genes['GENE_ID'].to_numpy(), gene_feat)

    # ------------------------------------------------------------------
    # Pathways: text + source prefix
    # ------------------------------------------------------------------
    pathway_features_df = pl.DataFrame({'PATHWAY_ID': []}, schema={'PATHWAY_ID': pl.UInt32})
    if pathway_nodes.height > 0:
        path_texts = [
            _build_text_row([
                row.get('PATHWAY_NAME'),
                row.get('PATHWAY_SOURCE_ID'),
            ])
            for row in pathway_nodes.iter_rows(named=True)
        ]
        path_text_feat = _fit_text_embedding(path_texts, cfg.text_dim, cfg.random_state)
        source_prefixes = [
            _clean_text(v).split(':')[0] if _clean_text(v) else 'UNK'
            for v in pathway_nodes['PATHWAY_SOURCE_ID'].to_list()
        ]
        uniq = sorted(set(source_prefixes))
        prefix_to_idx = {p: i for i, p in enumerate(uniq)}
        prefix_feat = np.zeros((len(source_prefixes), len(prefix_to_idx)), dtype=np.float32)
        for i, p in enumerate(source_prefixes):
            prefix_feat[i, prefix_to_idx[p]] = 1.0
        path_feat = np.concatenate([prefix_feat, path_text_feat], axis=1)
        pathway_features_df = _to_feature_df('PATHWAY_ID', pathway_nodes['PATHWAY_ID'].to_numpy(), path_feat)

    # ------------------------------------------------------------------
    # GO terms: ontology + text
    # ------------------------------------------------------------------
    go_features_df = pl.DataFrame({'GO_TERM_ID': []}, schema={'GO_TERM_ID': pl.UInt32})
    if go_nodes.height > 0:
        go_texts = [
            _build_text_row([
                row.get('GO_NAME'),
                row.get('GO_SOURCE_ID'),
                row.get('GO_ONTOLOGY'),
            ])
            for row in go_nodes.iter_rows(named=True)
        ]
        go_text_feat = _fit_text_embedding(go_texts, cfg.text_dim, cfg.random_state)

        ontology_map = {
            'Biological Process': 0,
            'Molecular Function': 1,
            'Cellular Component': 2,
        }
        go_one_hot = np.zeros((go_nodes.height, 3), dtype=np.float32)
        for i, ont in enumerate(go_nodes['GO_ONTOLOGY'].to_list()):
            idx = ontology_map.get(_clean_text(ont), 0)
            go_one_hot[i, idx] = 1.0

        go_feat = np.concatenate([go_one_hot, go_text_feat], axis=1)
        go_features_df = _to_feature_df('GO_TERM_ID', go_nodes['GO_TERM_ID'].to_numpy(), go_feat)

    feature_tables = {
        'chemical': chemical_features_df,
        'disease': disease_features_df,
        'gene': gene_features_df,
        'pathway': pathway_features_df,
        'go_term': go_features_df,
    }

    filename_map = {
        'chemical': 'chemical_node_features.parquet',
        'disease': 'disease_node_features.parquet',
        'gene': 'gene_node_features.parquet',
        'pathway': 'pathway_node_features.parquet',
        'go_term': 'go_term_node_features.parquet',
    }
    for node_type, df in feature_tables.items():
        df.write_parquet(out_path / filename_map[node_type])

    meta = {
        'config': {
            'text_dim': cfg.text_dim,
            'chem_fp_bits': cfg.chem_fp_bits,
            'include_pubchem': cfg.include_pubchem,
            'include_uniprot': cfg.include_uniprot,
            'include_umls': cfg.include_umls,
            'disgenet_file': cfg.disgenet_file,
        },
        'dimensions': {
            node_type: int(df.width - 1) if df.width > 0 else 0
            for node_type, df in feature_tables.items()
        },
    }
    _save_json(out_path / 'node_feature_metadata.json', meta)

    return feature_tables
