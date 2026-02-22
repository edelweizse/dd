"""
Memory-efficient inference module for chemical-disease link prediction.

This module provides the cache-based predictor for memory-efficient inference:
1. EmbeddingCachePredictor: Load pre-computed embeddings from disk

For large graphs that don't fit in memory, use this predictor.
"""

import torch
import polars as pl
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, Any, Optional
from pathlib import Path

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


class EmbeddingCachePredictor:
    """
    Predictor that uses pre-computed embeddings saved to disk.
    
    This is the most memory-efficient option for inference:
    1. Run compute_and_save_embeddings() once (can be done on a high-memory machine)
    2. Load only the chemical/disease embeddings for inference
    
    Usage:
        # Step 1: Compute embeddings (one-time, needs full graph in memory)
        EmbeddingCachePredictor.compute_and_save_embeddings(
            model, data, './embeddings/'
        )
        
        # Step 2: Fast inference (low memory)
        predictor = EmbeddingCachePredictor.from_cache(
            model, './embeddings/', disease_df, chemical_df
        )
        result = predictor.predict_pair('MESH:D003920', 'D008687')
    """
    
    def __init__(
        self,
        W_cd: torch.Tensor,
        z_chem: torch.Tensor,
        z_dis: torch.Tensor,
        disease_df: pl.DataFrame,
        chemical_df: pl.DataFrame,
        known_links: Optional[set] = None,
        device: Optional[torch.device] = None,
        threshold: float = 0.5
    ):
        """
        Args:
            W_cd: Bilinear decoder weight matrix [hidden_dim, hidden_dim]
            z_chem: Chemical embeddings [num_chemicals, hidden_dim]
            z_dis: Disease embeddings [num_diseases, hidden_dim]
            disease_df: Disease metadata DataFrame
            chemical_df: Chemical metadata DataFrame
            known_links: Set of known (chem_id, disease_id) tuples (internal IDs)
            device: Torch device
            threshold: Classification threshold
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Move tensors to device
        self.W_cd = W_cd.to(self.device)
        self.z_chem = z_chem.to(self.device)
        self.z_dis = z_dis.to(self.device)
        
        # Known links (set of (chem_idx, dis_idx) tuples)
        self.known_links = known_links or set()
        
        mappings = build_id_mappings(disease_df, chemical_df)
        self.disease_to_id = mappings['disease_to_id']
        self.id_to_disease = mappings['id_to_disease']
        self.disease_names = mappings['disease_names']
        self.chemical_to_id = mappings['chemical_to_id']
        self.id_to_chemical = mappings['id_to_chemical']
        self.chemical_names = mappings['chemical_names']
        self.num_diseases = mappings['num_diseases']
        self.num_chemicals = mappings['num_chemicals']
    
    @staticmethod
    def compute_and_save_embeddings(
        model: HGTPredictor,
        data: HeteroData,
        output_dir: str,
        device: Optional[torch.device] = None,
        batch_size: int = 10000
    ):
        """
        Compute embeddings and save prediction cache files.
        
        This needs to be run once on a machine with enough memory to hold the graph.
        After this, inference can be done with minimal memory.
        Saved files are: `chemical_embeddings.npy`, `disease_embeddings.npy`, `W_cd.pt`.
        
        Args:
            model: Trained HGTPredictor model
            data: Full HeteroData graph
            output_dir: Directory to save embeddings
            device: Torch device
            batch_size: Batch size for saving (not for computation)
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model = model.to(device)
        model.eval()
        
        # Build edge_attr_dict
        edge_attr_dict = {}
        for edge_type in data.edge_types:
            edge_store = data[edge_type]
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                edge_attr_dict[edge_type] = edge_store.edge_attr.to(device)
        
        # Move data to device
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
        
        print("Computing embeddings...")
        with torch.no_grad():
            embeddings = model.encode(x_dict, edge_index_dict, edge_attr_dict)
        
        # Save only tensors required by cached prediction
        print(f"Saving embeddings to {output_dir}...")
        required_node_types = ('chemical', 'disease')
        missing_node_types = [ntype for ntype in required_node_types if ntype not in embeddings]
        if missing_node_types:
            raise RuntimeError(
                f"Missing required node embeddings for cache prediction: {missing_node_types}"
            )

        for node_type in required_node_types:
            emb = embeddings[node_type]
            emb_np = emb.cpu().numpy()
            np.save(output_path / f'{node_type}_embeddings.npy', emb_np)
            print(f"  {node_type}: {emb_np.shape}")
        
        # Save decoder weights
        torch.save(model.W_cd.cpu(), output_path / 'W_cd.pt')
        print("  W_cd saved")
        
        print("Done!")
    
    @classmethod
    def from_cache(
        cls,
        cache_dir: str,
        disease_df: pl.DataFrame,
        chemical_df: pl.DataFrame,
        chem_disease_df: Optional[pl.DataFrame] = None,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
        load_all: bool = False
    ) -> 'EmbeddingCachePredictor':
        """
        Load predictor from cached embeddings.
        
        Args:
            cache_dir: Directory containing saved embeddings
            disease_df: Disease metadata DataFrame
            chemical_df: Chemical metadata DataFrame
            chem_disease_df: Chemical-disease edges DataFrame (for known links)
            device: Torch device
            threshold: Classification threshold
            load_all: If True, also load gene/pathway/go_term embeddings
            
        Returns:
            EmbeddingCachePredictor instance
        """
        cache_path = Path(cache_dir)
        
        print(f"Loading embeddings from {cache_dir}...")
        
        # Load required embeddings
        z_chem = torch.from_numpy(np.load(cache_path / 'chemical_embeddings.npy'))
        z_dis = torch.from_numpy(np.load(cache_path / 'disease_embeddings.npy'))
        W_cd = torch.load(cache_path / 'W_cd.pt')
        
        print(f"  chemical: {z_chem.shape}")
        print(f"  disease: {z_dis.shape}")
        
        # Build known links set
        known_links = set()
        if chem_disease_df is not None:
            print(f"  Loading {chem_disease_df.height:,} known links...")
            for row in chem_disease_df.iter_rows(named=True):
                known_links.add((row['CHEM_ID'], row['DS_ID']))
        
        predictor = cls(
            W_cd=W_cd,
            z_chem=z_chem,
            z_dis=z_dis,
            disease_df=disease_df,
            chemical_df=chemical_df,
            known_links=known_links,
            device=device,
            threshold=threshold
        )
        
        # Optionally load other embeddings
        if load_all:
            for node_type in ['gene', 'pathway', 'go_term']:
                path = cache_path / f'{node_type}_embeddings.npy'
                if path.exists():
                    emb = torch.from_numpy(np.load(path))
                    setattr(predictor, f'z_{node_type}', emb.to(predictor.device))
                    print(f"  {node_type}: {emb.shape}")
        
        return predictor
    
    def _compute_score(
        self,
        chem_ids: torch.Tensor,
        dis_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction scores for (chemical, disease) pairs."""
        return bilinear_score(
            z_chem=self.z_chem,
            z_dis=self.z_dis,
            decoder_weight=self.W_cd,
            chem_ids=chem_ids,
            dis_ids=dis_ids,
        )
    
    def is_known_link(self, chem_idx: int, dis_idx: int) -> bool:
        """Check if a chemical-disease pair is a known association."""
        return (chem_idx, dis_idx) in self.known_links
    
    def predict_pair(
        self,
        disease_id: str,
        chemical_id: str
    ) -> Dict[str, Any]:
        """Predict association between a disease and a chemical."""
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
    
    def predict_chemicals_for_disease(
        self,
        disease_id: str,
        top_k: int = 10,
        exclude_known: bool = False
    ) -> pl.DataFrame:
        """Get top-k chemicals most likely associated with a disease.
        
        Args:
            disease_id: Disease ID (MESH/OMIM format)
            top_k: Number of results to return
            exclude_known: If True, exclude known associations from results
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
        exclude_known: bool = False
    ) -> pl.DataFrame:
        """Get top-k diseases most likely associated with a chemical.
        
        Args:
            chemical_id: Chemical ID (MESH format)
            top_k: Number of results to return
            exclude_known: If True, exclude known associations from results
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

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        disease_id: str,
        chemical_id: str,
        *,
        mode: str = 'path_attention',
        runtime_profile: str = 'fast',
        template_set: str = 'default',
        data: Optional[HeteroData] = None,
        node_names: Optional[Dict[str, Dict[int, str]]] = None,
        data_dict: Optional[Dict[str, Any]] = None,
        adj: Optional[AdjacencyIndex] = None,
        max_paths_total: int = 500,
        max_paths_per_template: int = 100,
    ) -> ExplanationResult:
        """
        Explain a chemical-disease prediction with ranked graph paths.
        
        Since EmbeddingCachePredictor does not hold the graph or model,
        the ``data`` (HeteroData) argument is **required** for
        explainability.  Attention-based scoring (Tier 2) is not available
        in cache mode; only Tier 1 (metapath enumeration + embedding
        similarity) is used.
        
        Args:
            disease_id: External disease ID (MESH/OMIM format).
            chemical_id: External chemical ID (MESH format).
            mode: Explainer mode. Cached mode supports only "path_attention".
            runtime_profile: Runtime profile: {"fast", "balanced", "deep"}.
            template_set: Named metapath template set.
            data: HeteroData graph (required for path enumeration).
            node_names: Optional name lookup dict for path descriptions.
            data_dict: Processed data dictionary for auto-building node_names.
            adj: Pre-built adjacency index (reuse for speed).
            max_paths_total: Maximum number of scored paths retained.
            max_paths_per_template: Max paths per metapath template.
                
        Returns:
            ExplanationResult with scored, ranked explanatory paths.
            
        Raises:
            ValueError: If ``data`` is not provided.
        """
        if data is None:
            raise ValueError(
                "EmbeddingCachePredictor.explain_prediction() requires the "
                "'data' argument (HeteroData graph) for path enumeration. "
                "Load the graph via build_graph_from_processed() and pass it in."
            )
        if mode != 'path_attention':
            raise ValueError(
                'CachedEmbeddingPredictor supports only mode="path_attention". '
                'No alternative explain modes are available in cached mode.'
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

        # Build adjacency if not provided
        if adj is None:
            adj = build_adjacency(data)

        # Build embeddings dict from cached tensors for scoring
        embeddings: Dict[str, 'torch.Tensor'] = {
            'chemical': self.z_chem,
            'disease': self.z_dis,
        }
        for ntype in ('gene', 'pathway', 'go_term'):
            z = getattr(self, f'z_{ntype}', None)
            if z is not None:
                embeddings[ntype] = z

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
            use_attention=False,
            max_paths_total=max_paths_total,
            max_paths_per_template=max_paths_per_template,
        )
        context = ExplainContext(
            data=data,
            embeddings=embeddings,
            attention_weights=None,
            adj=adj,
        )
        return run_explain(request, context)


CachedEmbeddingPredictor = EmbeddingCachePredictor

__all__ = [
    'EmbeddingCachePredictor',
    'CachedEmbeddingPredictor',
]
