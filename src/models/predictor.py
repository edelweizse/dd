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

from .hgt import HGTPredictor


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
        
        # Build ID mappings
        # Disease: DS_OMIM_MESH_ID -> DS_ID (internal)
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
        
        # Chemical: CHEM_MESH_ID -> CHEM_ID (internal)
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
        x_dict = {}
        for k, v in self.data.x_dict.items():
            if k in self.model.node_emb:
                x_dict[k] = v.to(self.device)
        
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
        c = self.z_chem[chem_ids]  # [N, hidden_dim]
        d = self.z_dis[dis_ids]    # [N, hidden_dim]
        logits = (c @ self.model.W_cd * d).sum(dim=-1)  # [N]
        return logits
    
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
        # Validate IDs
        if disease_id not in self.disease_to_id:
            raise ValueError(f"Unknown disease ID: {disease_id}")
        if chemical_id not in self.chemical_to_id:
            raise ValueError(f"Unknown chemical ID: {chemical_id}")
        
        dis_idx = self.disease_to_id[disease_id]
        chem_idx = self.chemical_to_id[chemical_id]
        
        # Compute score
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
        if disease_id not in self.disease_to_id:
            raise ValueError(f"Unknown disease ID: {disease_id}")
        
        dis_idx = self.disease_to_id[disease_id]
        
        # Score all chemicals against this disease
        all_chem_ids = torch.arange(self.num_chemicals, device=self.device)
        dis_ids = torch.full((self.num_chemicals,), dis_idx, device=self.device)
        
        with torch.no_grad():
            logits = self._compute_score(all_chem_ids, dis_ids)
            probs = torch.sigmoid(logits)
        
        # Get known associations
        cd_edge_index = self.data[('chemical', 'associated_with', 'disease')].edge_index
        mask = cd_edge_index[1] == dis_idx
        known_chems = set(cd_edge_index[0][mask].cpu().tolist())
        
        # Sort by probability
        sorted_indices = torch.argsort(probs, descending=True).cpu().tolist()
        
        results = []
        rank = 1
        for idx in sorted_indices:
            is_known = idx in known_chems
            
            if exclude_known and is_known:
                continue
            
            chem_mesh_id = self.id_to_chemical[idx]
            results.append({
                'rank': rank,
                'chemical_id': chem_mesh_id,
                'chemical_name': self.chemical_names.get(chem_mesh_id, 'Unknown'),
                'probability': probs[idx].item(),
                'logit': logits[idx].item(),
                'known': is_known
            })
            rank += 1
            
            if rank > top_k:
                break
        
        return pl.DataFrame(results)
    
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
        if chemical_id not in self.chemical_to_id:
            raise ValueError(f"Unknown chemical ID: {chemical_id}")
        
        chem_idx = self.chemical_to_id[chemical_id]
        
        # Score all diseases against this chemical
        all_dis_ids = torch.arange(self.num_diseases, device=self.device)
        chem_ids = torch.full((self.num_diseases,), chem_idx, device=self.device)
        
        with torch.no_grad():
            logits = self._compute_score(chem_ids, all_dis_ids)
            probs = torch.sigmoid(logits)
        
        # Get known associations
        cd_edge_index = self.data[('chemical', 'associated_with', 'disease')].edge_index
        mask = cd_edge_index[0] == chem_idx
        known_diseases = set(cd_edge_index[1][mask].cpu().tolist())
        
        # Sort by probability
        sorted_indices = torch.argsort(probs, descending=True).cpu().tolist()
        
        results = []
        rank = 1
        for idx in sorted_indices:
            is_known = idx in known_diseases
            
            if exclude_known and is_known:
                continue
            
            dis_mesh_id = self.id_to_disease[idx]
            results.append({
                'rank': rank,
                'disease_id': dis_mesh_id,
                'disease_name': self.disease_names.get(dis_mesh_id, 'Unknown'),
                'probability': probs[idx].item(),
                'logit': logits[idx].item(),
                'known': is_known
            })
            rank += 1
            
            if rank > top_k:
                break
        
        return pl.DataFrame(results)
    
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
