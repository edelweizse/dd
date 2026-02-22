"""
Inference module for chemical-disease link prediction.

This module provides a high-level interface for:
- Predicting single chemical-disease pairs
- Finding top chemicals for a disease
- Finding top diseases for a chemical
- Batch predictions
"""

import torch
import polars as pl
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Any, Optional

from ..architectures.hgt import HGTPredictor
from ._shared import (
    bilinear_score,
    build_id_mappings,
    predict_pair_common,
    rank_chemicals_for_disease_common,
    rank_diseases_for_chemical_common,
)
from ...explainability.paths import AdjacencyIndex, build_adjacency
from ...explainability.explain import build_node_names
from ...explainability.schema import ExplainContext, ExplainRequest, ExplanationResult
from ...explainability.service import explain as run_explain


class ChemDiseasePredictor:
    """
    Inference wrapper for chemical-disease link prediction.
    
    Provides three prediction modes:
    1. predict_pair: Given (disease_id, chemical_id) -> (label, probability)
    2. predict_chemicals_for_disease: Given disease_id -> top-k chemicals
    3. predict_diseases_for_chemical: Given chemical_id -> top-k diseases
    """
    
    def __init__(
        self,
        model: HGTPredictor,
        data: HeteroData,
        disease_df: pl.DataFrame,   # DS_ID, DS_OMIM_MESH_ID, DS_NAME
        chemical_df: pl.DataFrame,  # CHEM_ID, CHEM_MESH_ID, CHEM_NAME
        device: Optional[torch.device] = None,
        threshold: float = 0.5
    ):
        """
        Args:
            model: Trained HGTPredictor model.
            data: Full HeteroData graph (not train split).
            disease_df: Disease metadata DataFrame.
            chemical_df: Chemical metadata DataFrame.
            device: Torch device (default: auto-detect).
            threshold: Classification threshold for predictions.
        """
        self.model = model
        self.data = data
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        mappings = build_id_mappings(disease_df, chemical_df)
        self.disease_to_id = mappings['disease_to_id']
        self.id_to_disease = mappings['id_to_disease']
        self.disease_names = mappings['disease_names']
        self.chemical_to_id = mappings['chemical_to_id']
        self.id_to_chemical = mappings['id_to_chemical']
        self.chemical_names = mappings['chemical_names']
        self.num_diseases = mappings['num_diseases']
        self.num_chemicals = mappings['num_chemicals']
        
        # Pre-compute node embeddings for fast inference
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pre-compute all node embeddings from the full graph."""
        print("Pre-computing node embeddings...")
        
        # Build edge_attr_dict
        edge_attr_dict = {}
        for edge_type in self.data.edge_types:
            edge_store = self.data[edge_type]
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                edge_attr_dict[edge_type] = edge_store.edge_attr.to(self.device)
        
        # Move data to device - only move node types that exist in the model
        x_dict = {k: v.to(self.device) for k, v in self.data.x_dict.items()}
        
        edge_index_dict = {k: v.to(self.device) for k, v in self.data.edge_index_dict.items()}
        
        with torch.no_grad():
            self.embeddings = self.model.encode(x_dict, edge_index_dict, edge_attr_dict)
        
        # Keep on device for fast inference
        self.z_chem = self.embeddings['chemical']  # [num_chemicals, hidden_dim]
        self.z_dis = self.embeddings['disease']    # [num_diseases, hidden_dim]
        
        # Store embeddings for other node types if they exist
        self.z_gene = self.embeddings.get('gene')
        self.z_pathway = self.embeddings.get('pathway')
        self.z_go_term = self.embeddings.get('go_term')
        
        print(f"Embeddings computed: chemicals={self.z_chem.shape}, diseases={self.z_dis.shape}")
        if self.z_pathway is not None:
            print(f"  pathways={self.z_pathway.shape}")
        if self.z_go_term is not None:
            print(f"  go_terms={self.z_go_term.shape}")
    
    def _compute_score(
        self,
        chem_ids: torch.Tensor,
        dis_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction scores for (chemical, disease) pairs."""
        return bilinear_score(
            z_chem=self.z_chem,
            z_dis=self.z_dis,
            decoder_weight=self.model.W_cd,
            chem_ids=chem_ids,
            dis_ids=dis_ids,
        )
    
    def predict_pair(
        self,
        disease_id: str,
        chemical_id: str
    ) -> Dict[str, Any]:
        """
        Predict association between a disease and a chemical.
        
        Args:
            disease_id: DS_OMIM_MESH_ID (e.g., "MESH:D003920" or "OMIM:222100")
            chemical_id: CHEM_MESH_ID (e.g., "D000082")
            
        Returns:
            Dict with keys: 'disease_id', 'chemical_id', 'disease_name', 'chemical_name',
                          'probability', 'label', 'logit', 'known'
        """
        return predict_pair_common(
            disease_id=disease_id,
            chemical_id=chemical_id,
            disease_to_id=self.disease_to_id,
            chemical_to_id=self.chemical_to_id,
            disease_names=self.disease_names,
            chemical_names=self.chemical_names,
            device=self.device,
            threshold=self.threshold,
            compute_score=self._compute_score,
            is_known_link=self.is_known_link,
        )
    
    def _get_known_links(self) -> set:
        """Get set of known (chem_idx, dis_idx) pairs from the graph."""
        if not hasattr(self, '_known_links_cache'):
            cd_edge_index = self.data[('chemical', 'associated_with', 'disease')].edge_index
            self._known_links_cache = set(
                zip(cd_edge_index[0].cpu().tolist(), cd_edge_index[1].cpu().tolist())
            )
        return self._known_links_cache
    
    def is_known_link(self, chem_idx: int, dis_idx: int) -> bool:
        """Check if a chemical-disease pair is a known association."""
        return (chem_idx, dis_idx) in self._get_known_links()
    
    def predict_chemicals_for_disease(
        self,
        disease_id: str,
        top_k: int = 10,
        exclude_known: bool = True
    ) -> pl.DataFrame:
        """
        Get top-k chemicals most likely associated with a disease.
        
        Args:
            disease_id: DS_OMIM_MESH_ID
            top_k: Number of top chemicals to return
            exclude_known: If True, exclude chemicals already known to be associated
            
        Returns:
            DataFrame with columns: rank, chemical_id, chemical_name, probability, logit, known
        """
        return rank_chemicals_for_disease_common(
            disease_id=disease_id,
            disease_to_id=self.disease_to_id,
            id_to_chemical=self.id_to_chemical,
            chemical_names=self.chemical_names,
            num_chemicals=self.num_chemicals,
            device=self.device,
            top_k=top_k,
            exclude_known=exclude_known,
            compute_score=self._compute_score,
            is_known_link=self.is_known_link,
        )
    
    def predict_diseases_for_chemical(
        self,
        chemical_id: str,
        top_k: int = 10,
        exclude_known: bool = True
    ) -> pl.DataFrame:
        """
        Get top-k diseases most likely associated with a chemical.
        
        Args:
            chemical_id: CHEM_MESH_ID
            top_k: Number of top diseases to return
            exclude_known: If True, exclude diseases already known to be associated
            
        Returns:
            DataFrame with columns: rank, disease_id, disease_name, probability, logit, known
        """
        return rank_diseases_for_chemical_common(
            chemical_id=chemical_id,
            chemical_to_id=self.chemical_to_id,
            id_to_disease=self.id_to_disease,
            disease_names=self.disease_names,
            num_diseases=self.num_diseases,
            device=self.device,
            top_k=top_k,
            exclude_known=exclude_known,
            compute_score=self._compute_score,
            is_known_link=self.is_known_link,
        )
    
    def batch_predict_pairs(
        self,
        pairs: List[Tuple[str, str]]
    ) -> pl.DataFrame:
        """
        Predict associations for multiple (disease_id, chemical_id) pairs.
        
        Args:
            pairs: List of (disease_id, chemical_id) tuples
            
        Returns:
            DataFrame with predictions for all pairs
        """
        results = []
        for disease_id, chemical_id in pairs:
            try:
                pred = self.predict_pair(disease_id, chemical_id)
                results.append(pred)
            except ValueError as e:
                results.append({
                    'disease_id': disease_id,
                    'chemical_id': chemical_id,
                    'disease_name': 'Unknown',
                    'chemical_name': 'Unknown',
                    'probability': None,
                    'label': None,
                    'logit': None,
                    'error': str(e)
                })
        
        return pl.DataFrame(results)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the computed embeddings."""
        stats = {
            'chemical': {
                'shape': list(self.z_chem.shape),
                'mean': self.z_chem.mean().item(),
                'std': self.z_chem.std().item()
            },
            'disease': {
                'shape': list(self.z_dis.shape),
                'mean': self.z_dis.mean().item(),
                'std': self.z_dis.std().item()
            }
        }
        
        if self.z_gene is not None:
            stats['gene'] = {
                'shape': list(self.z_gene.shape),
                'mean': self.z_gene.mean().item(),
                'std': self.z_gene.std().item()
            }
        
        if self.z_pathway is not None:
            stats['pathway'] = {
                'shape': list(self.z_pathway.shape),
                'mean': self.z_pathway.mean().item(),
                'std': self.z_pathway.std().item()
            }
        
        if self.z_go_term is not None:
            stats['go_term'] = {
                'shape': list(self.z_go_term.shape),
                'mean': self.z_go_term.mean().item(),
                'std': self.z_go_term.std().item()
            }
        
        return stats
    
    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def _ensure_adjacency(self) -> AdjacencyIndex:
        """Lazily build and cache the adjacency index for path enumeration."""
        if not hasattr(self, '_adj_cache'):
            self._adj_cache = build_adjacency(self.data)
        return self._adj_cache

    def _encode_with_attention(self):
        """
        Re-run model.encode() with return_attention=True.
        
        This is more expensive than using pre-computed embeddings because it
        re-runs the full forward pass, but it gives us per-edge attention
        weights for Tier-2 scoring.
        
        Returns:
            Tuple of (embeddings_dict, attention_list) from model.encode().
        """
        edge_attr_dict = {}
        for edge_type in self.data.edge_types:
            edge_store = self.data[edge_type]
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                edge_attr_dict[edge_type] = edge_store.edge_attr.to(self.device)

        x_dict = {k: v.to(self.device) for k, v in self.data.x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in self.data.edge_index_dict.items()}

        with torch.no_grad():
            embeddings, attention = self.model.encode(
                x_dict, edge_index_dict, edge_attr_dict,
                return_attention=True,
            )
        return embeddings, attention

    def explain_prediction(
        self,
        disease_id: str,
        chemical_id: str,
        *,
        mode: str = 'path_attention',
        runtime_profile: str = 'fast',
        template_set: str = 'default',
        use_attention: bool = True,
        node_names: Optional[Dict[str, Dict[int, str]]] = None,
        data_dict: Optional[Dict[str, Any]] = None,
        max_paths_total: int = 500,
        max_paths_per_template: int = 100,
    ) -> ExplanationResult:
        """
        Explain a chemical-disease prediction with ranked graph paths.
        
        Combines Tier 1 (metapath enumeration) and optionally Tier 2
        (attention weight extraction) to produce scored explanatory paths.
        
        Args:
            disease_id: External disease ID (MESH/OMIM format).
            chemical_id: External chemical ID (MESH format).
            mode: Explainer mode (only "path_attention" is supported).
            runtime_profile: Runtime profile: {"fast", "balanced", "deep"}.
            template_set: Named metapath template set.
            use_attention: If True, re-run encode() with attention extraction.
                This is more expensive but gives better path scoring.
            node_names: Optional name lookup dict for path descriptions.
                If not provided but data_dict is given, names are built
                automatically from the metadata DataFrames.
            data_dict: Processed data dictionary (from load_processed_data).
                Used to build node_names if not provided directly.
            max_paths_total: Maximum number of scored paths retained.
            max_paths_per_template: Max paths per metapath template.
                
        Returns:
            ExplanationResult with scored, ranked explanatory paths.
        """
        if mode != 'path_attention':
            raise ValueError(
                'Unsupported explain mode. Only mode="path_attention" is available.'
            )

        # Validate IDs
        if disease_id not in self.disease_to_id:
            raise ValueError(f"Unknown disease ID: {disease_id}")
        if chemical_id not in self.chemical_to_id:
            raise ValueError(f"Unknown chemical ID: {chemical_id}")

        dis_idx = self.disease_to_id[disease_id]
        chem_idx = self.chemical_to_id[chemical_id]

        # Get prediction result
        pred = self.predict_pair(disease_id, chemical_id)

        # Build node names for readable descriptions
        if node_names is None and data_dict is not None:
            node_names = build_node_names(data_dict)

        # Attention extraction (Tier 2)
        attention_weights = None
        embeddings = self.embeddings  # pre-computed (CPU-detached)
        if use_attention and mode == 'path_attention':
            embeddings, attention_weights = self._encode_with_attention()

        request = ExplainRequest(
            chemical_id=chemical_id,
            disease_id=disease_id,
            chem_idx=chem_idx,
            disease_idx=dis_idx,
            chemical_name=pred['chemical_name'],
            disease_name=pred['disease_name'],
            probability=pred['probability'],
            label=pred['label'],
            logit=pred['logit'],
            known=pred.get('known', False),
            node_names=node_names,
            mode='path_attention',
            runtime_profile=runtime_profile,  # type: ignore[arg-type]
            template_set=template_set,
            use_attention=use_attention,
            max_paths_total=max_paths_total,
            max_paths_per_template=max_paths_per_template,
        )
        context = ExplainContext(
            data=self.data,
            embeddings=embeddings,
            attention_weights=attention_weights,
            adj=self._ensure_adjacency(),
            model=self.model,
        )
        return run_explain(request, context)


FullGraphPredictor = ChemDiseasePredictor

__all__ = [
    'ChemDiseasePredictor',
    'FullGraphPredictor',
]
