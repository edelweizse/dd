"""
End-to-end tests for the explainability module.

Tests cover:
1. Metapath enumeration on a synthetic graph
2. Path scoring with mock embeddings and attention weights
3. explain_pair() high-level API
4. Attention weight extraction from HGTPredictor.encode()
5. build_node_names() helper
"""

import pytest
import torch
from torch_geometric.data import HeteroData

from src.explainability.paths import (
    AdjacencyIndex,
    PathInstance,
    build_adjacency,
    enumerate_paths,
    METAPATH_TEMPLATES,
)
from src.explainability.scoring import ScoredPath, score_paths, _geomean, _cosine_sim
from src.explainability.explain import ExplanationResult, explain_pair, build_node_names


# ---------------------------------------------------------------------------
# Fixtures: synthetic graph
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_graph() -> HeteroData:
    """
    Build a small heterogeneous graph with known paths.

    Nodes:
        chemical: 3
        disease: 2
        gene: 4
        pathway: 2
        go_term: 2

    Edges that create known metapaths between chem=0, disease=0:
        chem 0 --affects--> gene 1
        gene 1 --rev_targets--> disease 0          (2-hop: chem-gene-disease)
        chem 0 --enriched_in--> pathway 0
        pathway 0 --rev_disrupts--> disease 0       (2-hop: chem-pathway-disease)
        gene 1 --interacts_with--> gene 2
        gene 2 --rev_targets--> disease 0           (3-hop: chem-gene-gene-disease)
    """
    data = HeteroData()

    # Node features (simple embedding-lookup indices)
    data['chemical'].x = torch.arange(3)
    data['disease'].x = torch.arange(2)
    data['gene'].x = torch.arange(4)
    data['pathway'].x = torch.arange(2)
    data['go_term'].x = torch.arange(2)

    data['chemical'].num_nodes = 3
    data['disease'].num_nodes = 2
    data['gene'].num_nodes = 4
    data['pathway'].num_nodes = 2
    data['go_term'].num_nodes = 2

    # --- Edges ---
    # chem 0 -> gene 1, chem 1 -> gene 3
    data['chemical', 'affects', 'gene'].edge_index = torch.tensor(
        [[0, 1], [1, 3]]
    )
    # gene 1 -> disease 0, gene 2 -> disease 0
    data['gene', 'rev_targets', 'disease'].edge_index = torch.tensor(
        [[1, 2], [0, 0]]
    )
    # chem 0 -> pathway 0
    data['chemical', 'enriched_in', 'pathway'].edge_index = torch.tensor(
        [[0], [0]]
    )
    # pathway 0 -> disease 0
    data['pathway', 'rev_disrupts', 'disease'].edge_index = torch.tensor(
        [[0], [0]]
    )
    # gene 1 -> gene 2  (interacts_with for 3-hop)
    data['gene', 'interacts_with', 'gene'].edge_index = torch.tensor(
        [[1], [2]]
    )
    # go_term edges (none connect to our target pair, but needed so templates parse)
    data['chemical', 'enriched_in', 'go_term'].edge_index = torch.tensor(
        [[], []]
    ).long().reshape(2, 0)
    data['go_term', 'associated_with', 'disease'].edge_index = torch.tensor(
        [[], []]
    ).long().reshape(2, 0)
    data['chemical', 'affects_phenotype', 'go_term'].edge_index = torch.tensor(
        [[], []]
    ).long().reshape(2, 0)
    # gene -> pathway (for chem-gene-pathway-disease template)
    data['gene', 'participates_in', 'pathway'].edge_index = torch.tensor(
        [[], []]
    ).long().reshape(2, 0)
    # pathway -> gene (for chem-pathway-gene-disease template)
    data['pathway', 'rev_participates_in', 'gene'].edge_index = torch.tensor(
        [[], []]
    ).long().reshape(2, 0)

    return data


@pytest.fixture
def mock_embeddings():
    """Node embeddings (random but deterministic)."""
    torch.manual_seed(42)
    return {
        'chemical': torch.randn(3, 16),
        'disease': torch.randn(2, 16),
        'gene': torch.randn(4, 16),
        'pathway': torch.randn(2, 16),
        'go_term': torch.randn(2, 16),
    }


# ---------------------------------------------------------------------------
# Tests: Adjacency construction
# ---------------------------------------------------------------------------

class TestBuildAdjacency:
    def test_adjacency_has_expected_edge_types(self, synthetic_graph):
        adj = build_adjacency(synthetic_graph)
        assert ('chemical', 'affects', 'gene') in adj.forward
        assert ('gene', 'rev_targets', 'disease') in adj.forward
        assert ('chemical', 'enriched_in', 'pathway') in adj.forward

    def test_forward_neighbours(self, synthetic_graph):
        adj = build_adjacency(synthetic_graph)
        et = ('chemical', 'affects', 'gene')
        assert 1 in adj.forward[et][0]  # chem 0 -> gene 1

    def test_edge_pos_recorded(self, synthetic_graph):
        adj = build_adjacency(synthetic_graph)
        et = ('chemical', 'affects', 'gene')
        assert (0, 1) in adj.edge_pos[et]
        pos = adj.edge_pos[et][(0, 1)]
        assert isinstance(pos, int) and pos >= 0


# ---------------------------------------------------------------------------
# Tests: Path enumeration
# ---------------------------------------------------------------------------

class TestEnumeratePaths:
    def test_finds_2hop_chem_gene_disease(self, synthetic_graph):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        templates_found = {p.template_name for p in paths}
        assert 'chem-gene-disease' in templates_found

    def test_finds_2hop_chem_pathway_disease(self, synthetic_graph):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        templates_found = {p.template_name for p in paths}
        assert 'chem-pathway-disease' in templates_found

    def test_finds_3hop_chem_gene_gene_disease(self, synthetic_graph):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        templates_found = {p.template_name for p in paths}
        assert 'chem-gene-gene-disease' in templates_found

    def test_correct_node_sequence_2hop(self, synthetic_graph):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        gene_paths = [p for p in paths if p.template_name == 'chem-gene-disease']
        assert len(gene_paths) == 1
        p = gene_paths[0]
        assert p.node_indices == [0, 1, 0]  # chem 0 -> gene 1 -> disease 0
        assert p.node_types == ['chemical', 'gene', 'disease']

    def test_correct_node_sequence_3hop(self, synthetic_graph):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        gene_gene_paths = [p for p in paths if p.template_name == 'chem-gene-gene-disease']
        assert len(gene_gene_paths) == 1
        p = gene_gene_paths[0]
        assert p.node_indices == [0, 1, 2, 0]  # chem 0 -> gene 1 -> gene 2 -> disease 0

    def test_no_paths_for_disconnected_pair(self, synthetic_graph):
        adj = build_adjacency(synthetic_graph)
        # chem 2, disease 1 have no connecting edges
        paths = enumerate_paths(synthetic_graph, chem_idx=2, disease_idx=1, adj=adj)
        assert len(paths) == 0

    def test_max_paths_per_template_respected(self, synthetic_graph):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(
            synthetic_graph, chem_idx=0, disease_idx=0,
            adj=adj, max_paths_per_template=1,
        )
        # Each template should produce at most 1 path
        from collections import Counter
        counts = Counter(p.template_name for p in paths)
        assert all(c <= 1 for c in counts.values())


# ---------------------------------------------------------------------------
# Tests: Scoring helpers
# ---------------------------------------------------------------------------

class TestScoringHelpers:
    def test_geomean_basic(self):
        assert abs(_geomean([4.0, 9.0]) - 6.0) < 1e-6

    def test_geomean_single(self):
        assert abs(_geomean([5.0]) - 5.0) < 1e-6

    def test_geomean_zero_element(self):
        assert _geomean([0.0, 5.0]) == 0.0

    def test_geomean_empty(self):
        assert _geomean([]) == 0.0

    def test_cosine_sim_identical(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        assert abs(_cosine_sim(v, v) - 1.0) < 1e-5

    def test_cosine_sim_orthogonal(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert abs(_cosine_sim(a, b)) < 1e-5


# ---------------------------------------------------------------------------
# Tests: Path scoring
# ---------------------------------------------------------------------------

class TestScorePaths:
    def test_returns_sorted_scored_paths(self, synthetic_graph, mock_embeddings):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        scored = score_paths(paths, embeddings=mock_embeddings)
        assert len(scored) == len(paths)
        # Check sorted descending
        for i in range(len(scored) - 1):
            assert scored[i].combined_score >= scored[i + 1].combined_score

    def test_scored_path_fields(self, synthetic_graph, mock_embeddings):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        scored = score_paths(paths, embeddings=mock_embeddings)
        sp = scored[0]
        assert isinstance(sp, ScoredPath)
        assert sp.combined_score >= 0
        assert sp.attention_score > 0  # default 1.0 when no attention
        assert sp.evidence_type  # non-empty
        assert sp.description  # non-empty

    def test_no_attention_defaults_to_one(self, synthetic_graph, mock_embeddings):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        scored = score_paths(paths, embeddings=mock_embeddings, attention_weights=None)
        for sp in scored:
            assert sp.attention_score == 1.0

    def test_with_mock_attention(self, synthetic_graph, mock_embeddings):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        # Build mock attention weights: last layer, each edge type has uniform 0.5
        mock_attn_layer = {}
        for et in synthetic_graph.edge_types:
            num_edges = synthetic_graph[et].edge_index.size(1)
            if num_edges > 0:
                mock_attn_layer[et] = torch.full((num_edges,), 0.5)
        mock_attention = [mock_attn_layer]  # single layer
        scored = score_paths(paths, embeddings=mock_embeddings, attention_weights=mock_attention)
        for sp in scored:
            assert abs(sp.attention_score - 0.5) < 1e-5

    def test_node_names_in_description(self, synthetic_graph, mock_embeddings):
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        node_names = {
            'chemical': {0: 'Aspirin'},
            'disease': {0: 'Diabetes'},
            'gene': {1: 'PTGS2', 2: 'TP53'},
            'pathway': {0: 'MAPK'},
        }
        scored = score_paths(paths, embeddings=mock_embeddings, node_names=node_names)
        gene_path = [sp for sp in scored if sp.evidence_type == 'Shared gene target']
        assert len(gene_path) == 1
        assert 'Aspirin' in gene_path[0].description
        assert 'PTGS2' in gene_path[0].description
        assert 'Diabetes' in gene_path[0].description

    def test_none_node_names_produce_fallback(self, synthetic_graph, mock_embeddings):
        """Node names dict with None values must not crash join()."""
        adj = build_adjacency(synthetic_graph)
        paths = enumerate_paths(synthetic_graph, chem_idx=0, disease_idx=0, adj=adj)
        # Deliberately inject None values to simulate nulls from data
        node_names = {
            'chemical': {0: 'Aspirin'},
            'disease': {0: None},       # <-- None value
            'gene': {1: None, 2: 'TP53'},  # <-- None for gene 1
            'pathway': {0: 'MAPK'},
        }
        scored = score_paths(paths, embeddings=mock_embeddings, node_names=node_names)
        for sp in scored:
            assert isinstance(sp.description, str)
            # No None should appear as literal text; fallback uses "type:idx"
            assert 'None' not in sp.description


# ---------------------------------------------------------------------------
# Tests: explain_pair API
# ---------------------------------------------------------------------------

class TestExplainPair:
    def test_returns_explanation_result(self, synthetic_graph, mock_embeddings):
        result = explain_pair(
            data=synthetic_graph,
            chem_idx=0,
            disease_idx=0,
            chemical_id='D000001',
            disease_id='MESH:D000002',
            chemical_name='TestChem',
            disease_name='TestDisease',
            probability=0.8,
            label=1,
            logit=1.5,
            known=False,
            embeddings=mock_embeddings,
        )
        assert isinstance(result, ExplanationResult)
        assert result.chemical_id == 'D000001'
        assert result.disease_id == 'MESH:D000002'
        assert result.probability == 0.8
        assert not result.attention_available  # no attention passed
        assert len(result.paths) > 0

    def test_metapath_summary_populated(self, synthetic_graph, mock_embeddings):
        result = explain_pair(
            data=synthetic_graph,
            chem_idx=0,
            disease_idx=0,
            chemical_id='D000001',
            disease_id='MESH:D000002',
            embeddings=mock_embeddings,
        )
        assert len(result.metapath_summary) > 0
        assert 'Shared gene target' in result.metapath_summary

    def test_summary_text_nonempty(self, synthetic_graph, mock_embeddings):
        result = explain_pair(
            data=synthetic_graph,
            chem_idx=0,
            disease_idx=0,
            chemical_id='D000001',
            disease_id='MESH:D000002',
            embeddings=mock_embeddings,
        )
        text = result.summary_text()
        assert 'Prediction' in text
        assert 'TestChem' in text or 'Unknown' in text

    def test_top_paths_limited_to_10(self, synthetic_graph, mock_embeddings):
        result = explain_pair(
            data=synthetic_graph,
            chem_idx=0,
            disease_idx=0,
            chemical_id='D000001',
            disease_id='MESH:D000002',
            embeddings=mock_embeddings,
        )
        assert len(result.top_paths) <= 10

    def test_attention_available_flag(self, synthetic_graph, mock_embeddings):
        mock_attn_layer = {}
        for et in synthetic_graph.edge_types:
            num_edges = synthetic_graph[et].edge_index.size(1)
            if num_edges > 0:
                mock_attn_layer[et] = torch.full((num_edges,), 0.5)
        result = explain_pair(
            data=synthetic_graph,
            chem_idx=0,
            disease_idx=0,
            chemical_id='D000001',
            disease_id='MESH:D000002',
            embeddings=mock_embeddings,
            attention_weights=[mock_attn_layer],
        )
        assert result.attention_available

    def test_reuses_adjacency(self, synthetic_graph, mock_embeddings):
        adj = build_adjacency(synthetic_graph)
        result = explain_pair(
            data=synthetic_graph,
            chem_idx=0,
            disease_idx=0,
            chemical_id='D000001',
            disease_id='MESH:D000002',
            embeddings=mock_embeddings,
            adj=adj,
        )
        assert len(result.paths) > 0


# ---------------------------------------------------------------------------
# Tests: build_node_names
# ---------------------------------------------------------------------------

class TestBuildNodeNames:
    def test_builds_from_polars_dicts(self):
        import polars as pl
        data_dict = {
            'chemicals': pl.DataFrame({
                'CHEM_ID': [0, 1],
                'CHEM_MESH_ID': ['D000001', 'D000002'],
                'CHEM_NAME': ['Aspirin', 'Ibuprofen'],
            }),
            'diseases': pl.DataFrame({
                'DS_ID': [0, 1],
                'DS_OMIM_MESH_ID': ['MESH:D100', 'MESH:D200'],
                'DS_NAME': ['Diabetes', 'Cancer'],
            }),
            'genes': pl.DataFrame({
                'GENE_ID': [0, 1, 2],
                'GENE_NCBI_ID': [7166, 7157, 1234],
                'GENE_SYMBOL': ['PTGS2', 'TP53', 'ABC'],
            }),
        }
        names = build_node_names(data_dict)
        assert 'chemical' in names
        assert names['chemical'][0] == 'Aspirin'
        assert 'disease' in names
        assert names['disease'][1] == 'Cancer'
        assert 'gene' in names
        assert names['gene'][0] == 'PTGS2'

    def test_handles_none_go_names(self):
        """GO_NAME can be None for ~19% of GO terms — must not propagate None."""
        import polars as pl
        data_dict = {
            'chemicals': pl.DataFrame({
                'CHEM_ID': [0],
                'CHEM_MESH_ID': ['D000001'],
                'CHEM_NAME': ['Aspirin'],
            }),
            'diseases': pl.DataFrame({
                'DS_ID': [0],
                'DS_OMIM_MESH_ID': ['MESH:D100'],
                'DS_NAME': ['Diabetes'],
            }),
            'go_terms': pl.DataFrame({
                'GO_TERM_ID': [0, 1, 2],
                'GO_SOURCE_ID': ['GO:0000016', 'GO:0000080', None],
                'GO_NAME': ['lactase activity', None, None],
            }),
        }
        names = build_node_names(data_dict)
        assert 'go_term' in names
        # Named GO term keeps its name
        assert names['go_term'][0] == 'lactase activity'
        # None name falls back to GO_SOURCE_ID
        assert names['go_term'][1] == 'GO:0000080'
        # Both None falls back to generic label
        assert names['go_term'][2] == 'go_term:2'
        # No None values anywhere
        for ntype, nmap in names.items():
            for idx, name in nmap.items():
                assert name is not None, f"None name for {ntype}:{idx}"

    def test_handles_none_gene_symbols(self):
        """GENE_SYMBOL can have null values — must not propagate None."""
        import polars as pl
        data_dict = {
            'genes': pl.DataFrame({
                'GENE_ID': [0, 1],
                'GENE_NCBI_ID': [7166, 7157],
                'GENE_SYMBOL': ['PTGS2', None],
            }),
        }
        names = build_node_names(data_dict)
        assert names['gene'][0] == 'PTGS2'
        assert names['gene'][1] == 'gene:1'
        assert names['gene'][1] is not None


# ---------------------------------------------------------------------------
# Tests: HGTPredictor attention extraction
# ---------------------------------------------------------------------------

class TestAttentionExtraction:
    @pytest.fixture
    def small_model_and_data(self):
        """Create a minimal HGTPredictor and HeteroData for attention test."""
        from src.models.hgt import HGTPredictor
        import polars as pl

        data = HeteroData()
        data['chemical'].x = torch.arange(5)
        data['disease'].x = torch.arange(3)
        data['gene'].x = torch.arange(4)
        data['chemical'].num_nodes = 5
        data['disease'].num_nodes = 3
        data['gene'].num_nodes = 4

        # Edges
        data['chemical', 'affects', 'gene'].edge_index = torch.tensor(
            [[0, 1, 2], [0, 1, 2]]
        )
        data['gene', 'rev_targets', 'disease'].edge_index = torch.tensor(
            [[0, 1], [0, 1]]
        )

        # Need vocabs for action_type/action_subject
        action_type_vocab = pl.DataFrame({'action_type': ['A', 'B']})
        action_subject_vocab = pl.DataFrame({'action_subject': ['X', 'Y', 'Z']})

        num_nodes_dict = {
            'chemical': 5,
            'disease': 3,
            'gene': 4,
        }
        metadata = data.metadata()

        model = HGTPredictor(
            num_nodes_dict=num_nodes_dict,
            metadata=metadata,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            num_action_types=action_type_vocab.height,
            num_action_subjects=action_subject_vocab.height,
        )
        model.eval()
        return model, data

    def test_encode_returns_embeddings_only(self, small_model_and_data):
        model, data = small_model_and_data
        edge_attr_dict = {}
        for et in data.edge_types:
            edge_store = data[et]
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                edge_attr_dict[et] = edge_store.edge_attr
        with torch.no_grad():
            result = model.encode(data.x_dict, data.edge_index_dict, edge_attr_dict or None)
        assert isinstance(result, dict)  # Should be just embeddings
        assert 'chemical' in result
        assert 'disease' in result

    def test_encode_returns_attention_when_requested(self, small_model_and_data):
        model, data = small_model_and_data
        with torch.no_grad():
            result = model.encode(
                data.x_dict, data.edge_index_dict, None,
                return_attention=True,
            )
        assert isinstance(result, tuple)
        embeddings, attn_list = result
        assert isinstance(embeddings, dict)
        assert isinstance(attn_list, list)
        assert len(attn_list) == 2  # 2 layers

    def test_attention_shapes(self, small_model_and_data):
        model, data = small_model_and_data
        with torch.no_grad():
            embeddings, attn_list = model.encode(
                data.x_dict, data.edge_index_dict, None,
                return_attention=True,
            )
        for layer_attn in attn_list:
            assert isinstance(layer_attn, dict)
            for et, weights in layer_attn.items():
                num_edges = data[et].edge_index.size(1)
                assert weights.shape == (num_edges,), (
                    f"Expected shape ({num_edges},) for {et}, got {weights.shape}"
                )

    def test_attention_values_in_range(self, small_model_and_data):
        model, data = small_model_and_data
        with torch.no_grad():
            _, attn_list = model.encode(
                data.x_dict, data.edge_index_dict, None,
                return_attention=True,
            )
        for layer_attn in attn_list:
            for et, weights in layer_attn.items():
                assert torch.all(weights >= 0), f"Negative attention in {et}"
                assert torch.all(weights <= 1.0 + 1e-5), f"Attention > 1 in {et}"
