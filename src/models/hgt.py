"""
Heterogeneous Graph Neural Network models for chemical-disease link prediction.

This module implements:
- EdgeAttrHeteroConv: Heterogeneous graph convolution with edge attribute support
- HGTPredictor: Full model with encoder and bilinear decoder

Extended to support:
- 5 node types: chemical, disease, gene, pathway, go_term
- Multiple edge attribute types (categorical + continuous)
- Pathway and GO term connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
from typing import Dict, List, Tuple, Optional, Union


class EdgeAttrHeteroConv(nn.Module):
    """
    Heterogeneous Graph Convolution with edge attribute support.
    
    For edge types with attributes, uses edge embeddings/projections
    to modulate message passing. For other edge types, uses standard linear transform.
    
    Supports:
    - Categorical edge attributes (via embeddings)
    - Continuous edge attributes (via linear projections)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        metadata: Tuple,
        num_action_types: int = 0,
        num_action_subjects: int = 0,
        num_pheno_action_types: int = 3,  # increases, decreases, affects
        edge_attr_dim: int = 32,
        continuous_attr_dims: Optional[Dict[str, int]] = None,
        heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            metadata: Tuple of (node_types, edge_types) from HeteroData.
            num_action_types: Number of action type categories (chem-gene).
            num_action_subjects: Number of action subject categories (chem-gene).
            num_pheno_action_types: Number of phenotype action types.
            edge_attr_dim: Embedding dimension for categorical edge attributes.
            continuous_attr_dims: Dict mapping edge type key to continuous attr dimension.
            heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        if out_channels % heads != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by heads ({heads}).')
        self.dropout = dropout
        self.edge_attr_dim = edge_attr_dim
        
        node_types, edge_types = metadata
        
        # Edge types with categorical attributes (embeddings)
        self.chem_gene_attr_types = {
            ('chemical', 'affects', 'gene'),
            ('gene', 'rev_affects', 'chemical')
        }
        
        self.pheno_action_attr_types = {
            ('chemical', 'affects_phenotype', 'go_term'),
            ('go_term', 'rev_affects_phenotype', 'chemical')
        }
        
        # Edge types with continuous attributes
        self.continuous_attr_types = {
            # Pathway edges: NEG_LOG_PVALUE, TARGET_RATIO, FOLD_ENRICHMENT (3 dims)
            ('chemical', 'enriched_in', 'pathway'): 3,
            ('pathway', 'rev_enriched_in', 'chemical'): 3,
            # Disease-pathway: LOG_INFERENCE_GENE_COUNT (1 dim)
            ('disease', 'disrupts', 'pathway'): 1,
            ('pathway', 'rev_disrupts', 'disease'): 1,
            # Chem-GO enriched: NEG_LOG_PVALUE, TARGET_RATIO, FOLD_ENRICHMENT, ONTOLOGY_TYPE, GO_LEVEL_NORM (5 dims)
            ('chemical', 'enriched_in', 'go_term'): 5,
            ('go_term', 'rev_enriched_in', 'chemical'): 5,
            # GO-Disease: ONTOLOGY_TYPE, LOG_INFERENCE_CHEM, LOG_INFERENCE_GENE (3 dims)
            ('go_term', 'associated_with', 'disease'): 3,
            ('disease', 'rev_associated_with', 'go_term'): 3,
            # Disease-Gene: DIRECT_EVIDENCE_TYPE, LOG_PUBMED_COUNT (2 dims)
            ('disease', 'targets', 'gene'): 2,
            ('gene', 'rev_targets', 'disease'): 2,
        }
        
        if continuous_attr_dims is not None:
            self.continuous_attr_types.update(continuous_attr_dims)
        
        # EMBEDDINGS for categorical attributes
        self.use_chem_gene_attr = num_action_types > 0 and num_action_subjects > 0
        if self.use_chem_gene_attr:
            self.action_type_emb = nn.Embedding(num_action_types, edge_attr_dim)
            self.action_subject_emb = nn.Embedding(num_action_subjects, edge_attr_dim)
        
        self.use_pheno_action_attr = num_pheno_action_types > 0
        if self.use_pheno_action_attr:
            self.pheno_action_emb = nn.Embedding(num_pheno_action_types, edge_attr_dim)
        
        # Per-edge-type transformations
        self.lin_src = nn.ModuleDict()
        self.lin_dst = nn.ModuleDict()
        self.lin_edge_cat = nn.ModuleDict()  # For categorical edge attributes
        self.lin_edge_cont = nn.ModuleDict()  # For continuous edge attributes
        
        self._edge_keys = []
        
        for edge_type in edge_types:
            edge_key = '__'.join(edge_type)
            self._edge_keys.append(edge_key)
            
            self.lin_src[edge_key] = nn.Linear(in_channels, out_channels)
            self.lin_dst[edge_key] = nn.Linear(in_channels, out_channels)
            
            # Categorical edge attribute gate (chem-gene)
            if edge_type in self.chem_gene_attr_types and self.use_chem_gene_attr:
                self.lin_edge_cat[edge_key] = nn.Sequential(
                    nn.Linear(edge_attr_dim * 2, out_channels),
                    nn.Sigmoid()
                )
            
            # Categorical edge attribute gate (pheno action)
            if edge_type in self.pheno_action_attr_types and self.use_pheno_action_attr:
                self.lin_edge_cat[edge_key] = nn.Sequential(
                    nn.Linear(edge_attr_dim, out_channels),
                    nn.Sigmoid()
                )
            
            # Continuous edge attribute projection
            if edge_type in self.continuous_attr_types:
                attr_dim = self.continuous_attr_types[edge_type]
                self.lin_edge_cont[edge_key] = nn.Sequential(
                    nn.Linear(attr_dim, out_channels),
                    nn.GELU(),
                    nn.Linear(out_channels, out_channels),
                    nn.Sigmoid()
                )
            
            # Register attention weights as parameters
            attn_param = nn.Parameter(torch.zeros(1, heads, out_channels // heads))
            nn.init.xavier_uniform_(attn_param)
            self.register_parameter(f'attn_{edge_key}', attn_param)
        
        # Output projection per node type
        self.lin_out = nn.ModuleDict({
            ntype: nn.Linear(out_channels, out_channels) for ntype in node_types
        })
    
    def _get_attn(self, edge_key: str) -> torch.Tensor:
        """Get attention parameter for edge type."""
        return getattr(self, f'attn_{edge_key}')
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Union[Dict[str, torch.Tensor],
               Tuple[Dict[str, torch.Tensor], Dict[Tuple, torch.Tensor]]]:
        """
        Forward pass through heterogeneous convolution.
        
        Args:
            x_dict: Node features per type.
            edge_index_dict: Edge indices per type.
            edge_attr_dict: Edge attributes per type (optional).
            return_attention: If True, also return per-edge attention weights
                averaged across heads. Shape per edge type: [E].
            
        Returns:
            If return_attention is False: Updated node features per type.
            If return_attention is True: Tuple of (node_features, attention_dict)
                where attention_dict maps edge_type -> [E] mean attention weights.
        """
        out_dict = {ntype: [] for ntype in x_dict.keys()}
        attn_dict: Dict[Tuple, torch.Tensor] = {}
        
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type
            edge_key = '__'.join(edge_type)
            
            if edge_key not in self.lin_src:
                continue  # Skip unknown edge types
            
            # Skip if no edges
            if edge_index.numel() == 0:
                continue
            
            src_x = x_dict[src_type]
            dst_x = x_dict[dst_type]
            
            src_idx, dst_idx = edge_index[0], edge_index[1]
            
            # Transform source and destination features
            msg_src = self.lin_src[edge_key](src_x[src_idx])  # [E, out_channels]
            msg_dst = self.lin_dst[edge_key](dst_x[dst_idx])  # [E, out_channels]
            
            # Compute message
            msg = msg_src + msg_dst
            
            # Apply edge attribute gating
            if edge_attr_dict is not None and edge_type in edge_attr_dict:
                edge_attr = edge_attr_dict[edge_type]
                
                # Categorical: chem-gene action type/subject
                if (edge_type in self.chem_gene_attr_types and 
                    self.use_chem_gene_attr and
                    edge_key in self.lin_edge_cat):
                    # edge_attr: [E, 2] - (action_type_id, action_subject_id)
                    type_emb = self.action_type_emb(edge_attr[:, 0].long())
                    subj_emb = self.action_subject_emb(edge_attr[:, 1].long())
                    edge_emb = torch.cat([type_emb, subj_emb], dim=-1)
                    gate = self.lin_edge_cat[edge_key](edge_emb)
                    msg = msg * gate
                
                # Categorical: phenotype action type
                elif (edge_type in self.pheno_action_attr_types and
                      self.use_pheno_action_attr and
                      edge_key in self.lin_edge_cat):
                    # edge_attr: [E, 1] - (pheno_action_type)
                    edge_emb = self.pheno_action_emb(edge_attr.view(-1).long())
                    gate = self.lin_edge_cat[edge_key](edge_emb)
                    msg = msg * gate
                
                # Continuous: enrichment statistics, etc.
                elif edge_type in self.continuous_attr_types and edge_key in self.lin_edge_cont:
                    # edge_attr: [E, attr_dim]
                    gate = self.lin_edge_cont[edge_key](edge_attr.float())
                    msg = msg * gate
            
            num_dst = dst_x.size(0)
            head_dim = self.out_channels // self.heads
            msg_heads = msg.view(-1, self.heads, head_dim)
            src_heads = msg_src.view(-1, self.heads, head_dim)
            dst_heads = msg_dst.view(-1, self.heads, head_dim)

            # Dynamic attention logits (query-key style) + learned per-edge-type bias term.
            attn_param = self._get_attn(edge_key)
            attn_logits = (src_heads * dst_heads).sum(dim=-1) / (head_dim ** 0.5)
            attn_logits = attn_logits + (msg_heads * attn_param).sum(dim=-1)

            # Normalize across incoming neighbors for each destination node and head.
            attn_weights = softmax(attn_logits, dst_idx, num_nodes=num_dst)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            msg = (msg_heads * attn_weights.unsqueeze(-1)).reshape(-1, self.out_channels)

            # Store attention weights (head-averaged) if requested
            if return_attention:
                # attn_weights: [E, num_heads] -> mean over heads -> [E]
                attn_dict[edge_type] = attn_weights.mean(dim=-1).detach()

            # Aggregate messages to destination nodes
            aggr = torch.zeros(num_dst, self.out_channels, device=msg.device, dtype=msg.dtype)
            aggr.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(msg), msg)
            
            out_dict[dst_type].append(aggr)
        
        # Combine messages from different edge types
        result = {}
        for ntype, msgs in out_dict.items():
            if len(msgs) == 0:
                result[ntype] = x_dict[ntype]  # No incoming edges, keep original
            else:
                combined = torch.stack(msgs, dim=0).mean(dim=0)  # Average over edge types
                result[ntype] = self.lin_out[ntype](combined)
        
        if return_attention:
            return result, attn_dict
        return result


class HGTPredictor(nn.Module):
    """
    Heterogeneous Graph Transformer encoder + CD link decoder.
    
    Features:
    - Edge attribute support for multiple edge types
    - Residual connections
    - Multi-head attention via EdgeAttrHeteroConv
    - Support for extended node types (pathway, go_term)
    """
    
    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        metadata: Tuple,
        node_input_dims: Optional[Dict[str, int]] = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_action_types: int = 0,
        num_action_subjects: int = 0,
        num_pheno_action_types: int = 3
    ):
        """
        Args:
            num_nodes_dict: Number of nodes per type {'chemical': N, 'disease': M, ...}.
            metadata: Tuple of (node_types, edge_types) from HeteroData.
            node_input_dims: Optional input feature dimensions per node type.
                If present for a node type, the model uses a learned projection
                from features -> hidden_dim (inductive mode). Otherwise it falls
                back to embedding lookup by node ID (transductive mode).
            hidden_dim: Hidden dimension for embeddings.
            num_layers: Number of message passing layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            num_action_types: Number of action type categories.
            num_action_subjects: Number of action subject categories.
            num_pheno_action_types: Number of phenotype action types.
        """
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_action_types = num_action_types
        self.num_action_subjects = num_action_subjects
        self.node_input_dims = node_input_dims or {}
        
        # Node inputs: inductive projections where features exist, otherwise
        # transductive embedding tables.
        self.node_proj = nn.ModuleDict()
        self.node_emb = nn.ModuleDict()
        for ntype, n_nodes in num_nodes_dict.items():
            in_dim = int(self.node_input_dims.get(ntype, 0))
            if in_dim > 0:
                self.node_proj[ntype] = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            else:
                self.node_emb[ntype] = nn.Embedding(int(n_nodes), hidden_dim)
        
        # Graph convolution layers with edge attribute support
        self.convs = nn.ModuleList([
            EdgeAttrHeteroConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                num_action_types=num_action_types,
                num_action_subjects=num_action_subjects,
                num_pheno_action_types=num_pheno_action_types,
                edge_attr_dim=32,
                heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in num_nodes_dict.keys()})
            for _ in range(num_layers)
        ])
        
        # Decoder: bilinear for chemical-disease prediction
        self.W_cd = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W_cd)
    
    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Union[Dict[str, torch.Tensor],
               Tuple[Dict[str, torch.Tensor], List[Dict[Tuple, torch.Tensor]]]]:
        """
        Encode node features through heterogeneous GNN layers.
        
        Args:
            x_dict: Node features (global IDs) per type.
            edge_index_dict: Edge indices per type.
            edge_attr_dict: Edge attributes per type (optional).
            return_attention: If True, also return per-layer attention weights.
            
        Returns:
            If return_attention is False: Encoded node embeddings per type.
            If return_attention is True: Tuple of (embeddings, attention_list)
                where attention_list[i] is a dict mapping edge_type -> [E] for layer i.
        """
        h = self.initial_node_states(x_dict)
        all_attn: List[Dict[Tuple, torch.Tensor]] = []
        
        # Message passing layers
        for conv, norm_dict in zip(self.convs, self.norms):
            if return_attention:
                h_new, layer_attn = conv(h, edge_index_dict, edge_attr_dict,
                                         return_attention=True)
                all_attn.append(layer_attn)
            else:
                h_new = conv(h, edge_index_dict, edge_attr_dict)
            # Residual + LayerNorm + Dropout + Activation
            h = {
                k: F.gelu(norm_dict[k](F.dropout(h_new[k], p=self.dropout, training=self.training) + h[k]))
                for k in h.keys()
                if k in norm_dict  # Only process node types we have norms for
            }
        
        if return_attention:
            return h, all_attn
        return h

    def initial_node_states(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Initialize per-node-type hidden states from inputs."""
        h: Dict[str, torch.Tensor] = {}
        for ntype, x in x_dict.items():
            if ntype in self.node_proj:
                x_float = x.float()
                if x_float.dim() == 1:
                    x_float = x_float.unsqueeze(-1)
                h[ntype] = self.node_proj[ntype](x_float)
            elif ntype in self.node_emb:
                node_ids = x.view(-1).long()
                h[ntype] = self.node_emb[ntype](node_ids)
        return h
    
    def decode(
        self,
        z_chem: torch.Tensor,
        z_dis: torch.Tensor,
        edge_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode chemical-disease scores using bilinear transformation.
        
        Args:
            z_chem: [N_chem, hidden_dim] chemical embeddings.
            z_dis: [N_dis, hidden_dim] disease embeddings.
            edge_idx: [2, E] edge indices (chemical, disease).
            
        Returns:
            logits: [E] prediction scores.
        """
        c = z_chem[edge_idx[0]]  # [E, d]
        d = z_dis[edge_idx[1]]   # [E, d]
        # Bilinear: c^T W d
        return (c @ self.W_cd * d).sum(dim=-1)
    
    def forward(
        self,
        batch_data: HeteroData,
        pos_edge_idx: torch.Tensor,
        neg_edge_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            batch_data: HeteroData from LinkNeighborLoader.
            pos_edge_idx: [2, B] positive edge indices (local).
            neg_edge_idx: [2, N] negative edge indices (local).
            
        Returns:
            Tuple of (pos_logits, neg_logits) prediction scores.
        """
        # Build edge_attr_dict from batch_data
        edge_attr_dict = {}
        for edge_type in batch_data.edge_types:
            edge_store = batch_data[edge_type]
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                edge_attr_dict[edge_type] = edge_store.edge_attr
        
        z = self.encode(batch_data.x_dict, batch_data.edge_index_dict, edge_attr_dict)
        pos_logits = self.decode(z['chemical'], z['disease'], pos_edge_idx)
        neg_logits = self.decode(z['chemical'], z['disease'], neg_edge_idx)
        return pos_logits, neg_logits


def create_model_from_data(
    data: HeteroData,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.2,
    num_action_types: int = 0,
    num_action_subjects: int = 0
) -> HGTPredictor:
    """
    Create an HGTPredictor model from HeteroData.
    
    Args:
        data: HeteroData object with graph structure.
        hidden_dim: Hidden dimension for embeddings.
        num_layers: Number of message passing layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        num_action_types: Number of action type categories.
        num_action_subjects: Number of action subject categories.
        
    Returns:
        Initialized HGTPredictor model.
    """
    # Extract num_nodes_dict from data
    num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
    node_input_dims: Dict[str, int] = {}
    for ntype in data.node_types:
        x = data[ntype].x
        if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dim() == 2:
            node_input_dims[ntype] = int(x.size(1))
    
    # Get metadata
    metadata = (list(data.node_types), list(data.edge_types))
    
    return HGTPredictor(
        num_nodes_dict=num_nodes_dict,
        metadata=metadata,
        node_input_dims=node_input_dims,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_action_types=num_action_types,
        num_action_subjects=num_action_subjects
    )
