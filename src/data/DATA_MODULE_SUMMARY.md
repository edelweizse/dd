# src/data Module Summary

This document summarizes the full data pipeline in `src/data`, including processing, graph construction, splitting, artifact reuse, and negative sampling.

## Scope

Package files:
- `src/data/__init__.py`
- `src/data/processing.py`
- `src/data/graph.py`
- `src/data/splits.py`

## End-to-End Data Flow

1. Raw CTD/PPI files are loaded lazily with Polars (`load_raw_data`, `src/data/processing.py:24`).
2. Core entities are intersected and reindexed to contiguous IDs (`build_core_entities`, `src/data/processing.py:358`).
3. Optional pathway and GO-term entities are built with connectivity constraints (`build_pathway_entities`, `src/data/processing.py:429`; `build_go_term_entities`, `src/data/processing.py:511`).
4. Edge tables are constructed with feature engineering (`build_edge_tables`, `src/data/processing.py:607`; pathway/GO variants at `src/data/processing.py:765`, `src/data/processing.py:870`).
5. Processed parquet files are persisted (`process_and_save`, `src/data/processing.py:1057`) and reloaded (`load_processed_data`, `src/data/processing.py:1244`).
6. A PyG `HeteroData` is built from processed tables (`build_hetero_data`, `src/data/graph.py:89`) and optional reverse edges are added (`src/data/graph.py:270`).
7. CD edges are split into train/val/test with optional stratification and train-node coverage rebalancing (`split_cd`, `src/data/splits.py:393`).
8. A train-only CD graph is materialized for message passing (`make_split_graph`, `src/data/splits.py:706`).
9. Phase-specific known-positive filters + loaders are prepared (`prepare_splits_and_loaders`, `src/data/splits.py:954`).
10. Negatives are generated per batch with collision rejection (`negative_sample_cd_batch_local`, `src/data/splits.py:733`).

## `__init__.py` Exports

`src/data/__init__.py:5` re-exports high-level APIs from the three modules:
- Processing: load/process/save functions.
- Graph: graph builders and vocab loading.
- Splits: split structures, split generation, loaders, and negative sampling.

This provides a flat import surface (`from src.data import ...`) without submodule imports.

## `processing.py` Detailed Logic

### 1) Raw ingestion (`load_raw_data`)
Source: `src/data/processing.py:24`

What it does:
- Defines schema overrides for selected columns to avoid dtype drift.
- Renames source columns into a canonical schema (`CHEM_*`, `DS_*`, `GENE_*`, `GO_*`, etc.).
- Returns a dictionary of `pl.LazyFrame` objects for all source datasets.

Important behavior:
- Uses lazy scans (`pl.scan_csv`) for large files.
- Fails early if expected files/columns are missing.
- Includes three phenotype-disease ontology files and tags ontology explicitly.

### 2) Core entity construction (`build_core_entities`)
Source: `src/data/processing.py:358`

What it does:
- Genes are kept only if present in all required views: chem-gene, disease-gene, and PPI (`:383-400`).
- Diseases are kept only if present in both disease-gene and chem-disease (`:402-410`).
- Chemicals are kept only if present in both chem-gene and chem-disease (`:412-420`).
- Adds contiguous internal IDs using `with_row_index`.
- Left-joins annotation tables for metadata enrichment (`:422-424`).

Implication:
- Strong intersection policy improves consistency but drops sparse entities.

### 3) Pathway node construction (`build_pathway_entities`)
Source: `src/data/processing.py:429`

What it does:
- Collects pathway IDs linked to core genes, diseases, and chemicals separately.
- Outer-coalesces the three sets and computes source presence flags (`:482-494`).
- Requires pathway appearance in at least 2 sources (`SOURCE_COUNT >= 2`, `:495-497`).
- Renames external pathway ID to `PATHWAY_SOURCE_ID` and assigns contiguous `PATHWAY_ID` (`:501-506`).

Caveat:
- Uses `how='outer_coalesce'`, which is deprecated in newer Polars versions.

### 4) GO-term node construction (`build_go_term_entities`)
Source: `src/data/processing.py:511`

What it does:
- Builds GO candidates from:
  - chemical-GO enrichment,
  - chemical-phenotype interactions,
  - phenotype-disease associations (`:538-568`).
- Computes source flags and keeps GO terms that have both:
  - at least one chemical-side link (`HAS_CHEM_LINK`),
  - at least one disease-side link (`HAS_DISEASE_LINK`) (`:585-593`).
- Renames `GO_ID` to `GO_SOURCE_ID` and assigns contiguous `GO_TERM_ID` (`:597-601`).

Implication:
- Ensures GO terms are bridgeable between chemical and disease spaces.

### 5) Core edge tables (`build_edge_tables`)
Source: `src/data/processing.py:607`

#### Chemical-Gene edges
- Joins to core IDs and parses `INTERACTIONS` strings (`:633-680`).
- Deduplicates per `(CHEM_MESH_ID, GENE_NCBI_ID)` by preferring richer interaction strings (`has_list`, `n_pipes`, `n_chars`, `:638-652`).
- Splits interaction items by `|`, then action tuple by `^` into `ACTION_TYPE` and `ACTION_SUBJECT` (`:656-670`).
- Computes `LOG_PUBMED_COUNT = log(PUBMED_COUNT + 1)` (`:672-678`).

#### Disease-Gene edges
- Joins to core IDs (`:684-688`).
- Normalizes direct evidence; maps to categorical `DIRECT_EVIDENCE_TYPE` (`:689-705`).
- Aggregates duplicates by max pubmed count (`:695-699`).

#### Chemical-Disease edges
- Joins and deduplicates unique chemical-disease pairs (`:712-719`).

#### PPI
- Maps both endpoints to core gene IDs (`:721-739`).
- Removes self loops and canonicalizes endpoint order for undirected uniqueness (`:740-747`).
- Also emits directed double-edge variant (`:750-761`).

### 6) Pathway edge tables (`build_pathway_edge_tables`)
Source: `src/data/processing.py:765`

What it does:
- Gene-pathway: simple join/filter/deduplicate (`:790-805`).
- Disease-pathway: aggregates unique inference gene symbol count as edge attribute (`INFERENCE_GENE_COUNT`, `:807-826`).
- Chemical-pathway: computes engineered attributes (`:846-863`):
  - `NEG_LOG_PVALUE` capped to `[0, 10]`,
  - `TARGET_RATIO`,
  - `FOLD_ENRICHMENT` clipped to `[0, 100]`.

### 7) GO-term edge tables (`build_go_term_edge_tables`)
Source: `src/data/processing.py:870`

What it does:
- Defines ontology mapping BP/MF/CC -> 0/1/2 (`:896-898`).
- Chemical-GO edges with enrichment features + ontology + normalized GO level (`:900-941`).
- Chemical-phenotype edges:
  - Extracts action token from `INTERACTION_ACTIONS` via regex (`:959-965`),
  - maps to increases/decreases/affects (0/1/2),
  - keeps minimum code as “most specific” (`:969-973`).
- GO-disease edges:
  - merges all ontology disease files,
  - aggregates duplicates with max inference counts,
  - computes log-scaled inference features (`:982-1015`).

### 8) Interaction hierarchy (`build_interaction_type_hierarchy`)
Source: `src/data/processing.py:1020`

What it does:
- Assigns `IXN_TYPE_ID` by row order.
- Maps `IXN_TYPE_PARENT_CODE` to `IXN_TYPE_PARENT_ID`; missing parents -> `-1` (`:1044-1049`).

### 9) Orchestration + persistence (`process_and_save`)
Source: `src/data/processing.py:1057`

What it does:
- Runs all entity and edge builders in sequence (`:1078-1168`).
- Prints counts and totals (`:1174-1184`).
- Writes node, edge, and vocab parquet artifacts (`:1189-1215`).
- Returns all processed tables in-memory (`:1218-1241`).

### 10) Processed loading (`load_processed_data`)
Source: `src/data/processing.py:1244`

What it does:
- Loads required core parquet files.
- Loads optional extended files only if they exist (`:1269-1286`).

## `graph.py` Detailed Logic

### ID and vocab helpers
- `_assert_correct_ids` (`src/data/graph.py:13`): enforces unsigned, contiguous, unique IDs from `0..n-1`; raises on violations.
- `_cat_ids_from_col` (`src/data/graph.py:64`): creates stable integer vocab IDs for categorical columns.

### Graph construction (`build_hetero_data`)
Source: `src/data/graph.py:89`

What it does:
- Validates node table IDs for chemical/disease/gene (`:132-134`).
- Converts `ACTION_TYPE` and `ACTION_SUBJECT` to integer vocab IDs (`:136-141`).
- Initializes `HeteroData` with:
  - `num_nodes` per type,
  - `node_id` tensors,
  - `x` initialized to ID column clones (`:143-156`).
- Adds optional pathway/GO nodes and optional GO node ontology feature (`:158-176`).
- Populates edge indices + optional edge attributes for all available relations (`:177-267`).

Reverse-edge behavior:
- If `add_reverse_edges=True`, reverse edge types are created by flipping forward edge indices, with edge attributes cloned when present (`:270-337`).
- Includes reverse edges for core, pathway, and GO relations.

Validation:
- Calls `data.validate(raise_on_error=True)` before return (`:339`).

### Processed-graph loader (`build_graph_from_processed`)
Source: `src/data/graph.py:344`

What it does:
- Reads processed tables via `load_processed_data`.
- Optionally includes extended pathway/GO artifacts (`include_extended`) (`:366-387`).
- Builds graph with `build_hetero_data` (`:389-399`).
- Optionally writes vocab CSVs (`:401-404`).

### Utility functions
- `load_vocabs` (`src/data/graph.py:409`): loads action vocab CSVs.
- `get_graph_summary` (`src/data/graph.py:426`): counts nodes/edges and attr presence.
- `print_graph_summary` (`src/data/graph.py:460`): prints formatted summary.

## `splits.py` Detailed Logic

### Data structures
- `LinkSplit` (`src/data/splits.py:21`): holds positive edge tensors for train/val/test.
- `SplitArtifacts` (`src/data/splits.py:29`): bundles train graph, split, phase filters, loaders, degree vectors, and metadata.

### Known-positive membership (`PackedPairFilter`)
Source: `src/data/splits.py:50`

What it does:
- Packs pair `(chem, dis)` into int64 key `chem * num_dis + dis`.
- Stores keys in a set and sorted tensor for fast vectorized membership via `torch.isin` (`:69-93`).

### Split sizing (`_compute_split_counts`)
Source: `src/data/splits.py:110`

What it does:
- Validates ratios and edge count.
- Computes counts via floor + fractional remainder assignment (`:123-132`).
- Ensures each split non-empty by borrowing from largest split (`:134-145`).

### Stratification labels (`_degree_strata_labels`)
Source: `src/data/splits.py:152`

What it does:
- Computes degree of each endpoint type.
- Uses log2-binned degree for chem/dis, then joint bin label.
- Rare labels (`count < min_class_size`) are merged into one fallback label (`:177-181`).

### Train-node coverage rebalance (`_rebalance_split_for_train_node_coverage`)
Source: `src/data/splits.py:185`

What it does:
- Best-effort swap-based rebalance to improve train coverage of chemical/disease nodes while preserving split sizes.
- Builds per-source candidate maps for val/test edges (`:227-241`).
- For missing train nodes, pulls a covering edge from val or test and swaps out a “safe” train edge whose endpoints both have train count > 1 (`:311-376`).
- Stops when no safe swap exists.

Guarantees:
- Split cardinalities remain unchanged.
- Coverage is best-effort, not absolute.

### Main split function (`split_cd`)
Source: `src/data/splits.py:393`

What it does:
- Validates CD edge tensor shape/type.
- Computes counts.
- If stratified:
  - uses `StratifiedShuffleSplit` for test split then val split over remaining edges (`:441-457`),
  - falls back to random on `ValueError` (`:459-461`).
- Optional train-node coverage rebalance (`:469-476`).
- Returns `LinkSplit`, optionally with realized strategy (`stratified`/`random`).

### Split artifacts
- `_normalize_split_edge_tensor` (`src/data/splits.py:488`): validates split tensor shape and normalizes to contiguous CPU `long`.
- `_build_split_metadata` (`src/data/splits.py:499`): records seed/ratios/strategy/bins/coverage flag and graph dimensions.
- `save_split_artifact` (`src/data/splits.py:530`): writes split + metadata payload.
- `load_split_artifact` (`src/data/splits.py:581`): loads and validates payload structure.
- `validate_split_artifact_compatibility` (`src/data/splits.py:614`): verifies node counts, edge counts, relation tuple, split total, and ID ranges.

### Train graph creation (`make_split_graph`)
Source: `src/data/splits.py:706`

What it does:
- Clones full graph.
- Replaces forward CD edge index with `train_pos`.
- If reverse CD relation exists, rewrites it as flipped train edges.
- Validates graph.

### Negative sampling (`negative_sample_cd_batch_local`)
Source: `src/data/splits.py:733`

What it does:
- For each positive edge, creates `num_neg_per_pos` negatives by corrupting exactly one endpoint.
- Optional hard-negative mode:
  - degree-weighted sampling with exponent `degree_alpha`,
  - can use global train degrees if provided, otherwise local batch CD degrees.
- Collision handling:
  - checks `(chem, dis)` against `known_pos`,
  - resamples only collided entries,
  - keeps original corruption direction.
- Raises runtime error if collisions remain after `max_tries`.

### Loader creation (`make_link_loaders`)
Source: `src/data/splits.py:890`

What it does:
- Creates 3 `LinkNeighborLoader`s on the same train graph.
- Uses `edge_label_index` = train/val/test positives respectively.
- Sets `neg_sampling_ratio=0`; negatives are externally generated by sampler.
- Uses deterministic generators with seed offsets.

### Split orchestration (`prepare_splits_and_loaders`)
Source: `src/data/splits.py:954`

What it does:
- Validates split strategy.
- Either loads split artifact (and compatibility-checks) or creates a new split.
- Optionally saves split artifact.
- Builds train-only CD graph (`make_split_graph`).
- Builds phase-specific known-positive filters:
  - train filter: train positives only,
  - val filter: train + val positives,
  - test filter: train + val + test positives.
- Builds loaders and train-degree vectors.
- Returns `SplitArtifacts`.

Important semantic point:
- Train/val/test do not each get separate message-passing graphs.
- All loaders run on one train-CD graph; only supervision labels differ by split.

## Assumptions and Constraints

### Data/ID invariants
- Processed node IDs must be contiguous and unsigned (`src/data/graph.py:13`).
- Split tensors are expected as shape `[2, E]` with dtype `torch.long` (`src/data/splits.py:421`, `src/data/splits.py:488`).

### Reproducibility
- Split generation, loader sampling order, and negative sampling all use seeded generators with deterministic offsets (`src/data/splits.py:243`, `src/data/splits.py:914-916`, `src/data/splits.py:1077`).

### Leakage controls
- Message passing uses train-only CD adjacency (`src/data/splits.py:706`).
- Val/test known-positive filters include earlier splits to prevent false negatives during evaluation (`src/data/splits.py:1068-1074`).

## Notable Caveats / Operational Notes

1. `outer_coalesce` in Polars is deprecated and may require migration in future Polars versions (`src/data/processing.py:484-485`, `src/data/processing.py:573-574`).
2. Core-entity intersection can be strict and may drop many sparse entities by design (`src/data/processing.py:383-420`).
3. GO/pathway filtering requires cross-source support, favoring connectivity over recall (`src/data/processing.py:495-497`, `src/data/processing.py:592`).
4. Split rebalancing is best-effort; full train-node coverage is not always feasible (`src/data/splits.py:333-385`).
5. Stratified split falls back silently to random if strata are degenerate (`src/data/splits.py:459-461`).

## Suggested Reading Order

1. `src/data/processing.py` (`process_and_save` and its helpers)
2. `src/data/graph.py` (`build_graph_from_processed` -> `build_hetero_data`)
3. `src/data/splits.py` (`prepare_splits_and_loaders` -> splitting/sampling internals)
4. `src/data/__init__.py` (public import surface)
