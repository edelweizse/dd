"""Shared helpers for heterogeneous model architectures."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax


EdgeType = Tuple[str, str, str]
EdgeGateFn = Callable[[EdgeType, str, Optional[torch.Tensor]], Optional[torch.Tensor]]
AttnGetterFn = Callable[[str], torch.Tensor]


def init_node_states(
    x_dict: Dict[str, torch.Tensor],
    node_proj,
    node_emb,
) -> Dict[str, torch.Tensor]:
    """Initialize hidden states with dense projections or embedding lookups."""
    h: Dict[str, torch.Tensor] = {}
    for ntype, x in x_dict.items():
        if ntype in node_proj:
            x_float = x.float()
            if x_float.dim() == 1:
                x_float = x_float.unsqueeze(-1)
            h[ntype] = node_proj[ntype](x_float)
        elif ntype in node_emb:
            node_ids = x.view(-1).long()
            h[ntype] = node_emb[ntype](node_ids)
    return h


def encode_residual_stack(
    h: Dict[str, torch.Tensor],
    convs,
    norms,
    edge_index_dict: Dict[EdgeType, torch.Tensor],
    edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]],
    *,
    dropout: float,
    training: bool,
    return_attention: bool,
):
    """Run message-passing stack with residual+norm+dropout+GELU."""
    all_attn: List[Dict[EdgeType, torch.Tensor]] = []

    for conv, norm_dict in zip(convs, norms):
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
                    F.dropout(h_new[ntype], p=dropout, training=training) + h[ntype]
                )
            )
            for ntype in h.keys()
            if ntype in norm_dict
        }

    if return_attention:
        return h, all_attn
    return h


def hetero_conv_forward(
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[EdgeType, torch.Tensor],
    edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]],
    *,
    lin_src,
    lin_dst,
    lin_out,
    out_channels: int,
    heads: int,
    dropout: float,
    training: bool,
    get_attn: AttnGetterFn,
    edge_gate: EdgeGateFn,
    return_attention: bool,
):
    """Shared heterogeneous attention message passing core."""
    out_dict = {ntype: [] for ntype in x_dict.keys()}
    attn_dict: Dict[EdgeType, torch.Tensor] = {}

    for edge_type, edge_index in edge_index_dict.items():
        edge_key = "__".join(edge_type)
        if edge_key not in lin_src or edge_index.numel() == 0:
            continue

        src_type, _, dst_type = edge_type
        src_x = x_dict[src_type]
        dst_x = x_dict[dst_type]
        src_idx, dst_idx = edge_index[0], edge_index[1]

        msg_src = lin_src[edge_key](src_x[src_idx])
        msg_dst = lin_dst[edge_key](dst_x[dst_idx])
        msg = msg_src + msg_dst

        edge_attr = edge_attr_dict.get(edge_type) if edge_attr_dict is not None else None
        gate = edge_gate(edge_type, edge_key, edge_attr)
        if gate is not None:
            msg = msg * gate

        num_dst = dst_x.size(0)
        head_dim = out_channels // heads
        msg_heads = msg.view(-1, heads, head_dim)
        src_heads = msg_src.view(-1, heads, head_dim)
        dst_heads = msg_dst.view(-1, heads, head_dim)

        attn_param = get_attn(edge_key)
        attn_logits = (src_heads * dst_heads).sum(dim=-1) / (head_dim ** 0.5)
        attn_logits = attn_logits + (msg_heads * attn_param).sum(dim=-1)
        attn_weights = softmax(attn_logits, dst_idx, num_nodes=num_dst)
        attn_weights = F.dropout(attn_weights, p=dropout, training=training)
        msg = (msg_heads * attn_weights.unsqueeze(-1)).reshape(-1, out_channels)

        if return_attention:
            attn_dict[edge_type] = attn_weights.mean(dim=-1).detach()

        aggr = dst_x.new_zeros((num_dst, out_channels))
        aggr.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(msg), msg)
        out_dict[dst_type].append(aggr)

    result = {}
    for ntype, msgs in out_dict.items():
        if not msgs:
            result[ntype] = x_dict[ntype]
        else:
            combined = torch.stack(msgs, dim=0).mean(dim=0)
            result[ntype] = lin_out[ntype](combined)

    if return_attention:
        return result, attn_dict
    return result
