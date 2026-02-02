"""
Memory-efficient inference module for chemical-disease link prediction.

This module provides memory-efficient alternatives to ChemDiseasePredictor:
1. EmbeddingCachePredictor: Load pre-computed embeddings from disk
2. MiniBatchPredictor: Use neighbor sampling for on-the-fly inference

For large graphs that don't fit in memory, use one of these approaches.
"""

import os
import torch
import polars as pl
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .hgt import HGTPredictor


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
        
        # Build ID mappings
        self.disease_to_id = dict(zip(
            disease_df['DS_OMIM_MESH_ID'].to_list(),
            disease_df['DS_ID'].to_list()
        ))
        self.id_to_disease = dict(zip(
            disease_df['DS_ID'].to_list(),
            disease_df['DS_OMIM_MESH_ID'].to_list()
        ))
        self.disease_names = dict(zip(
            disease_df['DS_OMIM_MESH_ID'].to_list(),
            disease_df['DS_NAME'].to_list()
        ))
        
        self.chemical_to_id = dict(zip(
            chemical_df['CHEM_MESH_ID'].to_list(),
            chemical_df['CHEM_ID'].to_list()
        ))
        self.id_to_chemical = dict(zip(
            chemical_df['CHEM_ID'].to_list(),
            chemical_df['CHEM_MESH_ID'].to_list()
        ))
        self.chemical_names = dict(zip(
            chemical_df['CHEM_MESH_ID'].to_list(),
            chemical_df['CHEM_NAME'].to_list()
        ))
        
        self.num_diseases = len(self.disease_to_id)
        self.num_chemicals = len(self.chemical_to_id)
    
    @staticmethod
    def compute_and_save_embeddings(
        model: HGTPredictor,
        data: HeteroData,
        output_dir: str,
        device: Optional[torch.device] = None,
        batch_size: int = 10000
    ):
        """
        Compute embeddings for all nodes and save to disk.
        
        This needs to be run once on a machine with enough memory to hold the graph.
        After this, inference can be done with minimal memory.
        
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
        x_dict = {}
        for k, v in data.x_dict.items():
            if k in model.node_emb:
                x_dict[k] = v.to(device)
        
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
        
        print("Computing embeddings...")
        with torch.no_grad():
            embeddings = model.encode(x_dict, edge_index_dict, edge_attr_dict)
        
        # Save embeddings
        print(f"Saving embeddings to {output_dir}...")
        for node_type, emb in embeddings.items():
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
        c = self.z_chem[chem_ids]
        d = self.z_dis[dis_ids]
        logits = (c @ self.W_cd * d).sum(dim=-1)
        return logits
    
    def is_known_link(self, chem_idx: int, dis_idx: int) -> bool:
        """Check if a chemical-disease pair is a known association."""
        return (chem_idx, dis_idx) in self.known_links
    
    def predict_pair(
        self,
        disease_id: str,
        chemical_id: str
    ) -> Dict[str, Any]:
        """Predict association between a disease and a chemical."""
        if disease_id not in self.disease_to_id:
            raise ValueError(f"Unknown disease ID: {disease_id}")
        if chemical_id not in self.chemical_to_id:
            raise ValueError(f"Unknown chemical ID: {chemical_id}")
        
        dis_idx = self.disease_to_id[disease_id]
        chem_idx = self.chemical_to_id[chemical_id]
        
        chem_tensor = torch.tensor([chem_idx], device=self.device)
        dis_tensor = torch.tensor([dis_idx], device=self.device)
        
        with torch.no_grad():
            logit = self._compute_score(chem_tensor, dis_tensor).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
        
        return {
            'disease_id': disease_id,
            'chemical_id': chemical_id,
            'disease_name': self.disease_names.get(disease_id, 'Unknown'),
            'chemical_name': self.chemical_names.get(chemical_id, 'Unknown'),
            'probability': prob,
            'label': int(prob >= self.threshold),
            'logit': logit,
            'known': self.is_known_link(chem_idx, dis_idx)
        }
    
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
        if disease_id not in self.disease_to_id:
            raise ValueError(f"Unknown disease ID: {disease_id}")
        
        dis_idx = self.disease_to_id[disease_id]
        
        all_chem_ids = torch.arange(self.num_chemicals, device=self.device)
        dis_ids = torch.full((self.num_chemicals,), dis_idx, device=self.device)
        
        with torch.no_grad():
            logits = self._compute_score(all_chem_ids, dis_ids)
            probs = torch.sigmoid(logits)
        
        # Sort by probability
        sorted_indices = torch.argsort(probs, descending=True).cpu().tolist()
        
        results = []
        for idx in sorted_indices:
            is_known = self.is_known_link(idx, dis_idx)
            
            # Skip known if requested
            if exclude_known and is_known:
                continue
            
            chem_mesh_id = self.id_to_chemical[idx]
            results.append({
                'rank': len(results) + 1,
                'chemical_id': chem_mesh_id,
                'chemical_name': self.chemical_names.get(chem_mesh_id, 'Unknown'),
                'probability': probs[idx].item(),
                'logit': logits[idx].item(),
                'known': is_known
            })
            
            if len(results) >= top_k:
                break
        
        return pl.DataFrame(results)
    
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
        if chemical_id not in self.chemical_to_id:
            raise ValueError(f"Unknown chemical ID: {chemical_id}")
        
        chem_idx = self.chemical_to_id[chemical_id]
        
        all_dis_ids = torch.arange(self.num_diseases, device=self.device)
        chem_ids = torch.full((self.num_diseases,), chem_idx, device=self.device)
        
        with torch.no_grad():
            logits = self._compute_score(chem_ids, all_dis_ids)
            probs = torch.sigmoid(logits)
        
        # Sort by probability
        sorted_indices = torch.argsort(probs, descending=True).cpu().tolist()
        
        results = []
        for idx in sorted_indices:
            is_known = self.is_known_link(chem_idx, idx)
            
            # Skip known if requested
            if exclude_known and is_known:
                continue
            
            dis_mesh_id = self.id_to_disease[idx]
            results.append({
                'rank': len(results) + 1,
                'disease_id': dis_mesh_id,
                'disease_name': self.disease_names.get(dis_mesh_id, 'Unknown'),
                'probability': probs[idx].item(),
                'logit': logits[idx].item(),
                'known': is_known
            })
            
            if len(results) >= top_k:
                break
        
        return pl.DataFrame(results)


class MiniBatchPredictor:
    """
    Memory-efficient predictor using neighbor sampling.
    
    Instead of loading the full graph, this samples local subgraphs around
    query nodes for inference. This trades off accuracy for memory efficiency.
    
    Note: This approach may give slightly different results compared to
    full-graph inference due to the limited receptive field.
    
    Usage:
        predictor = MiniBatchPredictor(
            model, data, disease_df, chemical_df,
            num_neighbors=[10, 5]  # neighbors per layer
        )
        result = predictor.predict_pair('MESH:D003920', 'D008687')
    """
    
    def __init__(
        self,
        model: HGTPredictor,
        data: HeteroData,
        disease_df: pl.DataFrame,
        chemical_df: pl.DataFrame,
        num_neighbors: List[int] = [10, 5],
        device: Optional[torch.device] = None,
        threshold: float = 0.5
    ):
        """
        Args:
            model: Trained HGTPredictor model
            data: Full HeteroData graph (kept on CPU)
            disease_df: Disease metadata DataFrame
            chemical_df: Chemical metadata DataFrame
            num_neighbors: Number of neighbors to sample per layer
            device: Torch device for model
            threshold: Classification threshold
        """
        self.model = model
        self.data = data  # Keep on CPU
        self.num_neighbors = num_neighbors
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Build ID mappings
        self.disease_to_id = dict(zip(
            disease_df['DS_OMIM_MESH_ID'].to_list(),
            disease_df['DS_ID'].to_list()
        ))
        self.id_to_disease = dict(zip(
            disease_df['DS_ID'].to_list(),
            disease_df['DS_OMIM_MESH_ID'].to_list()
        ))
        self.disease_names = dict(zip(
            disease_df['DS_OMIM_MESH_ID'].to_list(),
            disease_df['DS_NAME'].to_list()
        ))
        
        self.chemical_to_id = dict(zip(
            chemical_df['CHEM_MESH_ID'].to_list(),
            chemical_df['CHEM_ID'].to_list()
        ))
        self.id_to_chemical = dict(zip(
            chemical_df['CHEM_ID'].to_list(),
            chemical_df['CHEM_MESH_ID'].to_list()
        ))
        self.chemical_names = dict(zip(
            chemical_df['CHEM_MESH_ID'].to_list(),
            chemical_df['CHEM_NAME'].to_list()
        ))
        
        self.num_diseases = len(self.disease_to_id)
        self.num_chemicals = len(self.chemical_to_id)
    
    def _sample_subgraph(
        self,
        chem_idx: int,
        dis_idx: int
    ) -> HeteroData:
        """Sample a subgraph around the query chemical and disease nodes."""
        # Create input nodes dict for both chemical and disease
        input_nodes = {
            'chemical': torch.tensor([chem_idx]),
            'disease': torch.tensor([dis_idx])
        }
        
        # Use NeighborLoader to sample subgraph
        loader = NeighborLoader(
            self.data,
            num_neighbors={
                edge_type: self.num_neighbors 
                for edge_type in self.data.edge_types
            },
            input_nodes=input_nodes,
            batch_size=2,  # We're querying 2 nodes (1 chem + 1 disease)
            shuffle=False
        )
        
        # Get the sampled subgraph
        batch = next(iter(loader))
        return batch
    
    def predict_pair(
        self,
        disease_id: str,
        chemical_id: str
    ) -> Dict[str, Any]:
        """Predict association using neighbor-sampled subgraph."""
        if disease_id not in self.disease_to_id:
            raise ValueError(f"Unknown disease ID: {disease_id}")
        if chemical_id not in self.chemical_to_id:
            raise ValueError(f"Unknown chemical ID: {chemical_id}")
        
        dis_idx = self.disease_to_id[disease_id]
        chem_idx = self.chemical_to_id[chemical_id]
        
        # Sample subgraph
        batch = self._sample_subgraph(chem_idx, dis_idx)
        batch = batch.to(self.device)
        
        # Build edge_attr_dict
        edge_attr_dict = {}
        for edge_type in batch.edge_types:
            edge_store = batch[edge_type]
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                edge_attr_dict[edge_type] = edge_store.edge_attr
        
        # Encode
        with torch.no_grad():
            z = self.model.encode(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
        
        # Find local indices of query nodes
        # The first nodes in each type should be our query nodes
        chem_local_idx = 0  # First chemical in batch
        dis_local_idx = 0   # First disease in batch
        
        # Decode
        c = z['chemical'][chem_local_idx:chem_local_idx+1]
        d = z['disease'][dis_local_idx:dis_local_idx+1]
        logit = (c @ self.model.W_cd * d).sum(dim=-1).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        
        return {
            'disease_id': disease_id,
            'chemical_id': chemical_id,
            'disease_name': self.disease_names.get(disease_id, 'Unknown'),
            'chemical_name': self.chemical_names.get(chemical_id, 'Unknown'),
            'probability': prob,
            'label': int(prob >= self.threshold),
            'logit': logit
        }
    
    def predict_chemicals_for_disease_batched(
        self,
        disease_id: str,
        top_k: int = 10,
        batch_size: int = 100
    ) -> pl.DataFrame:
        """
        Get top-k chemicals using batched inference.
        
        This is slower than full-graph inference but uses less memory.
        """
        if disease_id not in self.disease_to_id:
            raise ValueError(f"Unknown disease ID: {disease_id}")
        
        dis_idx = self.disease_to_id[disease_id]
        
        all_probs = []
        all_logits = []
        
        # Process chemicals in batches
        for start_idx in range(0, self.num_chemicals, batch_size):
            end_idx = min(start_idx + batch_size, self.num_chemicals)
            batch_chem_ids = list(range(start_idx, end_idx))
            
            batch_probs = []
            batch_logits = []
            
            for chem_idx in batch_chem_ids:
                try:
                    chem_mesh_id = self.id_to_chemical[chem_idx]
                    result = self.predict_pair(
                        self.id_to_disease[dis_idx],
                        chem_mesh_id
                    )
                    batch_probs.append(result['probability'])
                    batch_logits.append(result['logit'])
                except Exception:
                    batch_probs.append(0.0)
                    batch_logits.append(-float('inf'))
            
            all_probs.extend(batch_probs)
            all_logits.extend(batch_logits)
        
        # Get top-k
        probs_tensor = torch.tensor(all_probs)
        logits_tensor = torch.tensor(all_logits)
        top_k_vals, top_k_indices = torch.topk(probs_tensor, min(top_k, self.num_chemicals))
        
        results = []
        for rank, idx in enumerate(top_k_indices.tolist(), 1):
            chem_mesh_id = self.id_to_chemical[idx]
            results.append({
                'rank': rank,
                'chemical_id': chem_mesh_id,
                'chemical_name': self.chemical_names.get(chem_mesh_id, 'Unknown'),
                'probability': all_probs[idx],
                'logit': all_logits[idx]
            })
        
        return pl.DataFrame(results)
