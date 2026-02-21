"""
Graph-agnostic heterogeneous GNN components (schema-driven).

This module intentionally avoids hardcoded node/edge type logic and can be
configured for arbitrary heterogeneous graphs via GraphSchema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax


EdgeType = Tuple[str, str, str]


@dataclass(frozen=True)
class NodeInputSpec:
    """How a node type should be initialized."""

    mode: Literal['embedding', 'dense']
    num_nodes: Optional[int] = None
    in_dim: Optional[int] = None

    def validate(self, node_type: str) -> None:
        if self.mode not in {'embedding', 'dense'}:
            raise ValueError(f'Invalid mode "{self.mode}" for node type "{node_type}".')
        if self.mode == 'embedding':
            if self.num_nodes is None or int(self.num_nodes) <= 0:
                raise ValueError(
                    f'node_specs["{node_type}"] requires num_nodes>0 in embedding mode.'
                )
        if self.mode == 'dense':
            if self.in_dim is None or int(self.in_dim) <= 0:
                raise ValueError(
                    f'node_specs["{node_type}"] requires in_dim>0 in dense mode.'
                )


@dataclass(frozen=True)
class EdgeAttrSpec:
    """
    Edge attribute schema.

    Layout convention for `kind="mixed"`:
    - First `len(categorical_cardinalities)` columns are categorical IDs
    - Remaining `continuous_dim` columns are continuous
    """

    kind: Literal['none', 'continuous', 'categorical', 'mixed'] = 'none'
    continuous_dim: int = 0
    categorical_cardinalities: Tuple[int, ...] = ()

    def validate(self, edge_type: EdgeType) -> None:
        if self.kind not in {'none', 'continuous', 'categorical', 'mixed'}:
            raise ValueError(f'Invalid edge kind "{self.kind}" for edge type {edge_type}.')
        if self.kind in {'continuous', 'mixed'} and int(self.continuous_dim) <= 0:
            raise ValueError(
                f'Edge type {edge_type} requires continuous_dim>0 for kind="{self.kind}".'
            )
        if self.kind in {'categorical', 'mixed'}:
            if not self.categorical_cardinalities:
                raise ValueError(
                    f'Edge type {edge_type} requires categorical_cardinalities for kind="{self.kind}".'
                )
            if any(int(v) <= 0 for v in self.categorical_cardinalities):
                raise ValueError(
                    f'Edge type {edge_type} has invalid categorical cardinality values: '
                    f'{self.categorical_cardinalities}.'
                )
        if self.kind == 'none' and (self.continuous_dim > 0 or self.categorical_cardinalities):
            raise ValueError(
                f'Edge type {edge_type} kind="none" must not declare dims/cardinalities.'
            )

    @property
    def num_cat_cols(self) -> int:
        return len(self.categorical_cardinalities)


@dataclass(frozen=True)
class GraphSchema:
    """Schema for node inputs and edge attributes."""

    node_specs: Dict[str, NodeInputSpec]
    edge_specs: Dict[EdgeType, EdgeAttrSpec]

    def validate(self, metadata: Tuple[List[str], List[EdgeType]]) -> None:
        node_types, edge_types = metadata
        missing_nodes = [nt for nt in node_types if nt not in self.node_specs]
        if missing_nodes:
            raise ValueError(f'Missing node specs for node types: {missing_nodes}')
        for ntype in node_types:
            self.node_specs[ntype].validate(ntype)
        # Missing edge specs default to "none"; validate present ones.
        for etype, espec in self.edge_specs.items():
            espec.validate(etype)
        unknown_specs = [etype for etype in self.edge_specs if etype not in edge_types]
        if unknown_specs:
            raise ValueError(
                f'edge_specs contain edge types not present in metadata: {unknown_specs}'
            )


def infer_schema_from_data(data: HeteroData) -> GraphSchema:
    """
    Best-effort schema inference from a HeteroData object.

    Production use should prefer explicit GraphSchema.
    """
    node_specs: Dict[str, NodeInputSpec] = {}
    for ntype in data.node_types:
        x = data[ntype].x
        if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dim() == 2:
            node_specs[ntype] = NodeInputSpec(
                mode='dense',
                in_dim=int(x.size(1)),
            )
        else:
            node_specs[ntype] = NodeInputSpec(
                mode='embedding',
                num_nodes=int(data[ntype].num_nodes),
            )

    edge_specs: Dict[EdgeType, EdgeAttrSpec] = {}
    for etype in data.edge_types:
        store = data[etype]
        if not hasattr(store, 'edge_attr') or store.edge_attr is None:
            edge_specs[etype] = EdgeAttrSpec(kind='none')
            continue
        edge_attr = store.edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        if edge_attr.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            card = tuple(int(edge_attr[:, i].max().item()) + 1 for i in range(edge_attr.size(1)))
            edge_specs[etype] = EdgeAttrSpec(
                kind='categorical',
                categorical_cardinalities=card,
            )
        elif edge_attr.is_floating_point():
            edge_specs[etype] = EdgeAttrSpec(
                kind='continuous',
                continuous_dim=int(edge_attr.size(1)),
            )
        else:
            edge_specs[etype] = EdgeAttrSpec(kind='none')

    schema = GraphSchema(node_specs=node_specs, edge_specs=edge_specs)
    schema.validate((list(data.node_types), list(data.edge_types)))
    return schema


class GenericEdgeAttrHeteroConv(nn.Module):
    """Schema-driven heterogeneous convolution with optional edge-attribute gates."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        metadata: Tuple[List[str], List[EdgeType]],
        edge_specs: Dict[EdgeType, EdgeAttrSpec],
        *,
        edge_attr_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if out_channels % heads != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by heads ({heads}).')
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_attr_dim = edge_attr_dim

        node_types, edge_types = metadata
        self.edge_specs: Dict[str, EdgeAttrSpec] = {}

        self.lin_src = nn.ModuleDict()
        self.lin_dst = nn.ModuleDict()
        self.lin_out = nn.ModuleDict({ntype: nn.Linear(out_channels, out_channels) for ntype in node_types})

        self.edge_cat_embs = nn.ModuleDict()
        self.edge_gate_proj = nn.ModuleDict()

        for edge_type in edge_types:
            edge_key = '__'.join(edge_type)
            spec = edge_specs.get(edge_type, EdgeAttrSpec(kind='none'))
            self.edge_specs[edge_key] = spec

            self.lin_src[edge_key] = nn.Linear(in_channels, out_channels)
            self.lin_dst[edge_key] = nn.Linear(in_channels, out_channels)

            if spec.kind in {'categorical', 'mixed'}:
                embs = nn.ModuleList(
                    [nn.Embedding(int(card), edge_attr_dim) for card in spec.categorical_cardinalities]
                )
                self.edge_cat_embs[edge_key] = embs

            gate_in_dim = 0
            if spec.kind in {'categorical', 'mixed'}:
                gate_in_dim += spec.num_cat_cols * edge_attr_dim
            if spec.kind in {'continuous', 'mixed'}:
                gate_in_dim += int(spec.continuous_dim)

            if gate_in_dim > 0:
                self.edge_gate_proj[edge_key] = nn.Sequential(
                    nn.Linear(gate_in_dim, out_channels),
                    nn.GELU(),
                    nn.Linear(out_channels, out_channels),
                    nn.Sigmoid(),
                )

            attn_param = nn.Parameter(torch.zeros(1, heads, out_channels // heads))
            nn.init.xavier_uniform_(attn_param)
            self.register_parameter(f'attn_{edge_key}', attn_param)

    def _get_attn(self, edge_key: str) -> torch.Tensor:
        return getattr(self, f'attn_{edge_key}')

    def _edge_gate(
        self,
        edge_key: str,
        edge_attr: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        spec = self.edge_specs[edge_key]
        if spec.kind == 'none':
            return None
        if edge_attr is None:
            return None
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        feats: List[torch.Tensor] = []
        if spec.kind in {'categorical', 'mixed'}:
            n_cat = spec.num_cat_cols
            if edge_attr.size(1) < n_cat:
                raise ValueError(
                    f'edge_attr for "{edge_key}" has {edge_attr.size(1)} cols, expected >= {n_cat}.'
                )
            cat = edge_attr[:, :n_cat].long()
            emb_list = []
            for i, emb in enumerate(self.edge_cat_embs[edge_key]):
                col = cat[:, i].clamp(min=0, max=emb.num_embeddings - 1)
                emb_list.append(emb(col))
            feats.append(torch.cat(emb_list, dim=-1))

        if spec.kind in {'continuous', 'mixed'}:
            if spec.kind == 'continuous':
                cont = edge_attr
            else:
                cont = edge_attr[:, spec.num_cat_cols:]
            if cont.size(1) != int(spec.continuous_dim):
                raise ValueError(
                    f'continuous edge_attr dim mismatch for "{edge_key}": '
                    f'got {cont.size(1)}, expected {spec.continuous_dim}.'
                )
            feats.append(cont.float())

        gate_feat = torch.cat(feats, dim=-1)
        return self.edge_gate_proj[edge_key](gate_feat)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
        return_attention: bool = False,
    ):
        out_dict = {ntype: [] for ntype in x_dict.keys()}
        attn_dict: Dict[EdgeType, torch.Tensor] = {}

        for edge_type, edge_index in edge_index_dict.items():
            edge_key = '__'.join(edge_type)
            if edge_key not in self.lin_src or edge_index.numel() == 0:
                continue

            src_type, _, dst_type = edge_type
            src_x = x_dict[src_type]
            dst_x = x_dict[dst_type]
            src_idx, dst_idx = edge_index[0], edge_index[1]

            msg_src = self.lin_src[edge_key](src_x[src_idx])
            msg_dst = self.lin_dst[edge_key](dst_x[dst_idx])
            msg = msg_src + msg_dst

            edge_attr = edge_attr_dict.get(edge_type) if edge_attr_dict is not None else None
            gate = self._edge_gate(edge_key, edge_attr)
            if gate is not None:
                msg = msg * gate

            num_dst = dst_x.size(0)
            head_dim = self.out_channels // self.heads
            msg_heads = msg.view(-1, self.heads, head_dim)
            src_heads = msg_src.view(-1, self.heads, head_dim)
            dst_heads = msg_dst.view(-1, self.heads, head_dim)

            attn_param = self._get_attn(edge_key)
            attn_logits = (src_heads * dst_heads).sum(dim=-1) / (head_dim ** 0.5)
            attn_logits = attn_logits + (msg_heads * attn_param).sum(dim=-1)
            attn_weights = softmax(attn_logits, dst_idx, num_nodes=num_dst)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            msg = (msg_heads * attn_weights.unsqueeze(-1)).reshape(-1, self.out_channels)

            if return_attention:
                attn_dict[edge_type] = attn_weights.mean(dim=-1).detach()

            aggr = torch.zeros(num_dst, self.out_channels, device=msg.device, dtype=msg.dtype)
            aggr.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(msg), msg)
            out_dict[dst_type].append(aggr)

        result = {}
        for ntype, msgs in out_dict.items():
            if not msgs:
                result[ntype] = x_dict[ntype]
            else:
                combined = torch.stack(msgs, dim=0).mean(dim=0)
                result[ntype] = self.lin_out[ntype](combined)

        if return_attention:
            return result, attn_dict
        return result


class GenericHGTEncoder(nn.Module):
    """Graph-agnostic heterogeneous encoder driven by GraphSchema."""

    def __init__(
        self,
        schema: GraphSchema,
        metadata: Tuple[List[str], List[EdgeType]],
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        edge_attr_dim: int = 32,
    ):
        super().__init__()
        schema.validate(metadata)
        self.schema = schema
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        node_types, _ = metadata
        self.node_proj = nn.ModuleDict()
        self.node_emb = nn.ModuleDict()
        for ntype in node_types:
            spec = schema.node_specs[ntype]
            if spec.mode == 'dense':
                self.node_proj[ntype] = nn.Sequential(
                    nn.Linear(int(spec.in_dim), hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            else:
                self.node_emb[ntype] = nn.Embedding(int(spec.num_nodes), hidden_dim)

        self.convs = nn.ModuleList(
            [
                GenericEdgeAttrHeteroConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    edge_specs=schema.edge_specs,
                    edge_attr_dim=edge_attr_dim,
                    heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in node_types}) for _ in range(num_layers)]
        )

    def initial_node_states(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
        return_attention: bool = False,
    ):
        h = self.initial_node_states(x_dict)
        all_attn: List[Dict[EdgeType, torch.Tensor]] = []

        for conv, norm_dict in zip(self.convs, self.norms):
            if return_attention:
                h_new, layer_attn = conv(
                    h,
                    edge_index_dict,
                    edge_attr_dict,
                    return_attention=True,
                )
                all_attn.append(layer_attn)
            else:
                h_new = conv(h, edge_index_dict, edge_attr_dict)

            h = {
                ntype: F.gelu(
                    norm_dict[ntype](
                        F.dropout(h_new[ntype], p=self.dropout, training=self.training) + h[ntype]
                    )
                )
                for ntype in h.keys()
                if ntype in norm_dict
            }

        if return_attention:
            return h, all_attn
        return h


class BilinearLinkHead(nn.Module):
    """Relation-aware bilinear link decoder for arbitrary edge types."""

    def __init__(
        self,
        hidden_dim: int,
        relation_types: List[EdgeType],
    ):
        super().__init__()
        self.rel_weights = nn.ParameterDict()
        for etype in relation_types:
            key = '__'.join(etype)
            param = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            nn.init.xavier_uniform_(param)
            self.rel_weights[key] = param

    def score(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_type: EdgeType,
        edge_idx: torch.Tensor,
    ) -> torch.Tensor:
        key = '__'.join(edge_type)
        if key not in self.rel_weights:
            raise KeyError(f'No bilinear relation weight configured for edge type: {edge_type}')
        src_type, _, dst_type = edge_type
        src = z_dict[src_type][edge_idx[0]]
        dst = z_dict[dst_type][edge_idx[1]]
        W = self.rel_weights[key]
        return (src @ W * dst).sum(dim=-1)


class GenericLinkPredictor(nn.Module):
    """
    End-to-end generic link prediction model:
    schema-driven encoder + relation-aware bilinear head.
    """

    def __init__(
        self,
        schema: GraphSchema,
        metadata: Tuple[List[str], List[EdgeType]],
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        edge_attr_dim: int = 32,
        relation_types: Optional[List[EdgeType]] = None,
    ):
        super().__init__()
        self.encoder = GenericHGTEncoder(
            schema=schema,
            metadata=metadata,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            edge_attr_dim=edge_attr_dim,
        )
        rels = relation_types if relation_types is not None else list(metadata[1])
        self.head = BilinearLinkHead(hidden_dim=hidden_dim, relation_types=rels)

    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
        return_attention: bool = False,
    ):
        return self.encoder.encode(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            return_attention=return_attention,
        )

    def forward(
        self,
        batch_data: HeteroData,
        pos_edge_idx: torch.Tensor,
        neg_edge_idx: torch.Tensor,
        *,
        target_edge_type: EdgeType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_attr_dict = {}
        for edge_type in batch_data.edge_types:
            edge_store = batch_data[edge_type]
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                edge_attr_dict[edge_type] = edge_store.edge_attr
        z = self.encode(
            x_dict=batch_data.x_dict,
            edge_index_dict=batch_data.edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            return_attention=False,
        )
        pos_logits = self.head.score(z, target_edge_type, pos_edge_idx)
        neg_logits = self.head.score(z, target_edge_type, neg_edge_idx)
        return pos_logits, neg_logits


__all__ = [
    'NodeInputSpec',
    'EdgeAttrSpec',
    'GraphSchema',
    'infer_schema_from_data',
    'GenericEdgeAttrHeteroConv',
    'GenericHGTEncoder',
    'BilinearLinkHead',
    'GenericLinkPredictor',
]

