"""Baseline models and comparison utilities for CD link prediction."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, RGCNConv, SAGEConv

from src.data.splits import SplitArtifacts, negative_sample_cd_batch_local
from src.models.architectures._shared import init_node_states
from src.models.architectures.generic_hgt import GenericLinkPredictor, infer_schema_from_data
from src.models.architectures.hgt import HGTPredictor, infer_hgt_hparams_from_state
from src.training.utils import eval_epoch, sampled_ranking_metrics


CD_REL: Tuple[str, str, str] = ("chemical", "associated_with", "disease")


def _safe_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    yt = y_true.detach().cpu().numpy()
    ys = y_score.detach().cpu().numpy()
    if len(set(yt.tolist())) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(yt, ys))


def _safe_ap(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    yt = y_true.detach().cpu().numpy()
    ys = y_score.detach().cpu().numpy()
    if len(set(yt.tolist())) < 2:
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(yt, ys))


def _batch_global_ids(batch: HeteroData, node_type: str) -> torch.Tensor:
    store = batch[node_type]
    if hasattr(store, "node_id") and store.node_id is not None:
        return store.node_id.view(-1).long()
    if hasattr(store, "n_id") and store.n_id is not None:
        return store.n_id.view(-1).long()
    return store.x.view(-1).long()


def _build_cd_bipartite_buffers(
    train_pos: torch.Tensor,
    num_chem: int,
    num_dis: int,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build bidirectional CD edge index and normalization for CD-only models."""
    chem = train_pos[0].long().to(device)
    dis = train_pos[1].long().to(device) + int(num_chem)
    src = torch.cat([chem, dis], dim=0)
    dst = torch.cat([dis, chem], dim=0)
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.cat(
        [
            torch.zeros(chem.size(0), dtype=torch.long, device=device),
            torch.ones(chem.size(0), dtype=torch.long, device=device),
        ],
        dim=0,
    )

    deg = torch.bincount(src, minlength=int(num_chem + num_dis)).float().clamp_min(1.0)
    norm = (deg[src] * deg[dst]).rsqrt()
    return edge_index, edge_type, norm


class DegreePopularityBaseline(nn.Module):
    """Non-trainable degree-based heuristic scorer."""

    kind = "pair"
    trainable = False

    def __init__(self, num_chem: int, num_dis: int):
        super().__init__()
        self.register_buffer("chem_deg", torch.zeros(int(num_chem), dtype=torch.float32))
        self.register_buffer("dis_deg", torch.zeros(int(num_dis), dtype=torch.float32))

    def fit_from_split(self, split_train_pos: torch.Tensor) -> None:
        self.chem_deg.zero_()
        self.dis_deg.zero_()
        c = torch.bincount(split_train_pos[0].cpu(), minlength=self.chem_deg.numel()).float()
        d = torch.bincount(split_train_pos[1].cpu(), minlength=self.dis_deg.numel()).float()
        self.chem_deg.copy_(torch.log1p(c))
        self.dis_deg.copy_(torch.log1p(d))

    def score_pairs(self, chem_ids: torch.Tensor, dis_ids: torch.Tensor) -> torch.Tensor:
        c = self.chem_deg[chem_ids.long()]
        d = self.dis_deg[dis_ids.long()]
        return c + d


class MatrixFactorizationBaseline(nn.Module):
    """CD-only bilinear matrix factorization."""

    kind = "pair"
    trainable = True

    def __init__(self, num_chem: int, num_dis: int, hidden_dim: int):
        super().__init__()
        self.chem_emb = nn.Embedding(int(num_chem), int(hidden_dim))
        self.dis_emb = nn.Embedding(int(num_dis), int(hidden_dim))
        self.W = nn.Parameter(torch.empty(int(hidden_dim), int(hidden_dim)))
        nn.init.xavier_uniform_(self.chem_emb.weight)
        nn.init.xavier_uniform_(self.dis_emb.weight)
        nn.init.xavier_uniform_(self.W)

    def score_pairs(self, chem_ids: torch.Tensor, dis_ids: torch.Tensor) -> torch.Tensor:
        c = self.chem_emb(chem_ids.long())
        d = self.dis_emb(dis_ids.long())
        return (c @ self.W * d).sum(dim=-1)


class PairMLPBaseline(nn.Module):
    """Embedding + MLP pair scorer on CD pairs."""

    kind = "pair"
    trainable = True

    def __init__(self, num_chem: int, num_dis: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.chem_emb = nn.Embedding(int(num_chem), int(hidden_dim))
        self.dis_emb = nn.Embedding(int(num_dis), int(hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(2 * int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.xavier_uniform_(self.chem_emb.weight)
        nn.init.xavier_uniform_(self.dis_emb.weight)

    def score_pairs(self, chem_ids: torch.Tensor, dis_ids: torch.Tensor) -> torch.Tensor:
        c = self.chem_emb(chem_ids.long())
        d = self.dis_emb(dis_ids.long())
        return self.mlp(torch.cat([c, d], dim=-1)).view(-1)


class LightGCNCDBaseline(nn.Module):
    """LightGCN baseline on CD bipartite graph (train CD edges only)."""

    kind = "pair"
    trainable = True

    def __init__(
        self,
        num_chem: int,
        num_dis: int,
        hidden_dim: int,
        num_layers: int,
        split_train_pos: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.num_chem = int(num_chem)
        self.num_dis = int(num_dis)
        self.num_layers = int(max(1, num_layers))
        self.node_emb = nn.Embedding(self.num_chem + self.num_dis, int(hidden_dim))
        nn.init.xavier_uniform_(self.node_emb.weight)
        edge_index, _, norm = _build_cd_bipartite_buffers(
            split_train_pos,
            self.num_chem,
            self.num_dis,
            device=device,
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("norm", norm)
        self.W = nn.Parameter(torch.empty(int(hidden_dim), int(hidden_dim)))
        nn.init.xavier_uniform_(self.W)

    def _propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.node_emb.weight
        outs = [x]
        src = self.edge_index[0]
        dst = self.edge_index[1]
        for _ in range(self.num_layers):
            msg = x[src] * self.norm.unsqueeze(-1)
            x_new = torch.zeros_like(x)
            x_new.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
            x = x_new
            outs.append(x)
        z = torch.stack(outs, dim=0).mean(dim=0)
        return z[: self.num_chem], z[self.num_chem :]

    def score_pairs(self, chem_ids: torch.Tensor, dis_ids: torch.Tensor) -> torch.Tensor:
        zc, zd = self._propagate()
        c = zc[chem_ids.long()]
        d = zd[dis_ids.long()]
        return (c @ self.W * d).sum(dim=-1)


class RGCNCDBaseline(nn.Module):
    """R-GCN baseline on CD-only bipartite graph with forward/reverse relations."""

    kind = "pair"
    trainable = True

    def __init__(
        self,
        num_chem: int,
        num_dis: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        split_train_pos: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.num_chem = int(num_chem)
        self.num_dis = int(num_dis)
        self.dropout = float(dropout)
        self.node_emb = nn.Embedding(self.num_chem + self.num_dis, int(hidden_dim))
        nn.init.xavier_uniform_(self.node_emb.weight)
        edge_index, edge_type, _ = _build_cd_bipartite_buffers(
            split_train_pos,
            self.num_chem,
            self.num_dis,
            device=device,
        )
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)
        self.convs = nn.ModuleList(
            [RGCNConv(int(hidden_dim), int(hidden_dim), num_relations=2) for _ in range(int(max(1, num_layers)))]
        )
        self.W = nn.Parameter(torch.empty(int(hidden_dim), int(hidden_dim)))
        nn.init.xavier_uniform_(self.W)

    def _encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.node_emb.weight
        for i, conv in enumerate(self.convs):
            x = conv(x, self.edge_index, self.edge_type)
            if i + 1 < len(self.convs):
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x[: self.num_chem], x[self.num_chem :]

    def score_pairs(self, chem_ids: torch.Tensor, dis_ids: torch.Tensor) -> torch.Tensor:
        zc, zd = self._encode()
        c = zc[chem_ids.long()]
        d = zd[dis_ids.long()]
        return (c @ self.W * d).sum(dim=-1)


class HeteroSAGEBaseline(nn.Module):
    """Heterogeneous GraphSAGE encoder + CD bilinear decoder."""

    kind = "loader"
    trainable = True

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        node_input_dims: Dict[str, int],
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.node_proj = nn.ModuleDict()
        self.node_emb = nn.ModuleDict()
        for ntype, n_nodes in num_nodes_dict.items():
            in_dim = int(node_input_dims.get(ntype, 0))
            if in_dim > 0:
                self.node_proj[ntype] = nn.Sequential(
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.GELU(),
                )
            else:
                self.node_emb[ntype] = nn.Embedding(int(n_nodes), int(hidden_dim))

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        node_types, edge_types = metadata
        for _ in range(int(max(1, num_layers))):
            conv = HeteroConv(
                {
                    etype: SAGEConv((int(hidden_dim), int(hidden_dim)), int(hidden_dim))
                    for etype in edge_types
                },
                aggr="mean",
            )
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(int(hidden_dim)) for nt in node_types}))

        self.W_cd = nn.Parameter(torch.empty(int(hidden_dim), int(hidden_dim)))
        nn.init.xavier_uniform_(self.W_cd)

    def encode(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = init_node_states(x_dict, self.node_proj, self.node_emb)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index_dict)
            h = {
                ntype: F.gelu(
                    norm[ntype](
                        F.dropout(h_new.get(ntype, h[ntype]), p=self.dropout, training=self.training) + h[ntype]
                    )
                )
                for ntype in h.keys()
            }
        return h

    def decode(self, z_chem: torch.Tensor, z_dis: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
        c = z_chem[edge_idx[0]]
        d = z_dis[edge_idx[1]]
        return (c @ self.W_cd * d).sum(dim=-1)

    def forward_batch(self, batch_data: HeteroData, pos_edge_idx: torch.Tensor, neg_edge_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(batch_data.x_dict, batch_data.edge_index_dict)
        return self.decode(z["chemical"], z["disease"], pos_edge_idx), self.decode(
            z["chemical"], z["disease"], neg_edge_idx
        )


class HGTNoEdgeAttrBaseline(nn.Module):
    """HGT baseline with edge-attribute gates disabled in forward pass."""

    kind = "loader"
    trainable = True

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        node_input_dims: Dict[str, int],
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.model = HGTPredictor(
            num_nodes_dict=num_nodes_dict,
            metadata=metadata,
            node_input_dims=node_input_dims,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
            num_action_types=0,
            num_action_subjects=0,
            num_pheno_action_types=0,
        )

    def forward_batch(self, batch_data: HeteroData, pos_edge_idx: torch.Tensor, neg_edge_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.model.encode(batch_data.x_dict, batch_data.edge_index_dict, edge_attr_dict=None)
        return self.model.decode(z["chemical"], z["disease"], pos_edge_idx), self.model.decode(
            z["chemical"], z["disease"], neg_edge_idx
        )


class GenericHGTBaseline(nn.Module):
    """Schema-driven GenericHGT baseline focused on CD relation."""

    kind = "loader"
    trainable = True

    def __init__(
        self,
        data_train: HeteroData,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        schema = infer_schema_from_data(data_train)
        self.model = GenericLinkPredictor(
            schema=schema,
            metadata=data_train.metadata(),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
            relation_types=[CD_REL],
        )

    def forward_batch(self, batch_data: HeteroData, pos_edge_idx: torch.Tensor, neg_edge_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            batch_data=batch_data,
            pos_edge_idx=pos_edge_idx,
            neg_edge_idx=neg_edge_idx,
            target_edge_type=CD_REL,
        )


BASELINE_NAMES: Tuple[str, ...] = (
    "degree",
    "mf",
    "mlp",
    "lightgcn_cd",
    "rgcn_cd",
    "heterosage",
    "hgt_no_edge_attr",
    "generic_hgt",
)


def build_baseline(
    name: str,
    *,
    data_train: HeteroData,
    split_train_pos: torch.Tensor,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    name = name.strip().lower()
    num_nodes_dict = {nt: int(data_train[nt].num_nodes) for nt in data_train.node_types}
    node_input_dims = {
        nt: int(data_train[nt].x.size(1))
        for nt in data_train.node_types
        if isinstance(data_train[nt].x, torch.Tensor)
        and data_train[nt].x.dim() == 2
        and data_train[nt].x.is_floating_point()
    }
    if name == "degree":
        m = DegreePopularityBaseline(num_nodes_dict["chemical"], num_nodes_dict["disease"])
        m.fit_from_split(split_train_pos)
        return m
    if name == "mf":
        return MatrixFactorizationBaseline(num_nodes_dict["chemical"], num_nodes_dict["disease"], hidden_dim)
    if name == "mlp":
        return PairMLPBaseline(num_nodes_dict["chemical"], num_nodes_dict["disease"], hidden_dim, dropout)
    if name == "lightgcn_cd":
        return LightGCNCDBaseline(
            num_nodes_dict["chemical"],
            num_nodes_dict["disease"],
            hidden_dim,
            num_layers,
            split_train_pos,
            device=device,
        )
    if name == "rgcn_cd":
        return RGCNCDBaseline(
            num_nodes_dict["chemical"],
            num_nodes_dict["disease"],
            hidden_dim,
            num_layers,
            dropout,
            split_train_pos,
            device=device,
        )
    if name == "heterosage":
        return HeteroSAGEBaseline(
            num_nodes_dict=num_nodes_dict,
            metadata=data_train.metadata(),
            node_input_dims=node_input_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if name == "hgt_no_edge_attr":
        return HGTNoEdgeAttrBaseline(
            num_nodes_dict=num_nodes_dict,
            metadata=data_train.metadata(),
            node_input_dims=node_input_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    if name == "generic_hgt":
        return GenericHGTBaseline(
            data_train=data_train,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    raise ValueError(f"Unknown baseline: {name}. Available: {BASELINE_NAMES}")


def _train_step_pair_model(
    model: nn.Module,
    batch: HeteroData,
    pos_edge: torch.Tensor,
    neg_edge: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chem_gid = _batch_global_ids(batch, "chemical").to(pos_edge.device)
    dis_gid = _batch_global_ids(batch, "disease").to(pos_edge.device)
    pos_chem = chem_gid[pos_edge[0]]
    pos_dis = dis_gid[pos_edge[1]]
    neg_chem = chem_gid[neg_edge[0]]
    neg_dis = dis_gid[neg_edge[1]]
    return model.score_pairs(pos_chem, pos_dis), model.score_pairs(neg_chem, neg_dis)


def train_baseline(
    model: nn.Module,
    *,
    arts: SplitArtifacts,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    num_neg_train: int,
    degree_alpha: float = 0.75,
    progress_every: int = 0,
    max_batches: int = 0,
    progress_prefix: str | None = None,
) -> Dict[str, float]:
    """Train baseline in-place when trainable; returns training stats."""
    model = model.to(device)
    if not bool(getattr(model, "trainable", True)):
        return {"train_loss": float("nan"), "epochs": 0}

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_known_pos = getattr(arts, "known_pos", arts.known_pos_test)
    train_loss = float("nan")
    total_batches = len(arts.train_loader) if hasattr(arts.train_loader, "__len__") else None
    for epoch_idx in range(1, int(max(1, epochs)) + 1):
        model.train()
        loss_sum = 0.0
        n_pos = 0
        for batch_idx, batch in enumerate(arts.train_loader, start=1):
            if int(max_batches) > 0 and batch_idx > int(max_batches):
                break
            batch = batch.to(device)
            cd_edge_store = batch[CD_REL]
            pos_edge = cd_edge_store.edge_label_index
            neg_edge = negative_sample_cd_batch_local(
                batch_data=batch,
                pos_edge_index_local=pos_edge,
                known_pos=train_known_pos,
                num_neg_per_pos=int(num_neg_train),
                hard_negative_ratio=0.0,
                degree_alpha=float(degree_alpha),
                global_chem_degree=arts.chem_train_degree,
                global_dis_degree=arts.dis_train_degree,
                generator=None,
            )
            optimizer.zero_grad()
            if getattr(model, "kind", "pair") == "loader":
                pos_logits, neg_logits = model.forward_batch(batch, pos_edge, neg_edge)
            else:
                pos_logits, neg_logits = _train_step_pair_model(model, batch, pos_edge, neg_edge)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_logits, neg_logits], dim=0),
                torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0),
            )
            loss.backward()
            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            bsz = int(pos_edge.size(1))
            loss_sum += float(loss.item()) * bsz
            n_pos += bsz
            if int(progress_every) > 0 and (batch_idx % int(progress_every) == 0):
                total = (
                    min(int(total_batches), int(max_batches))
                    if total_batches is not None and int(max_batches) > 0
                    else total_batches
                )
                total_str = str(total) if total is not None else "?"
                prefix = f"{progress_prefix} " if progress_prefix else ""
                running_loss = loss_sum / max(n_pos, 1)
                print(
                    f"{prefix}train epoch {epoch_idx}/{int(max(1, epochs))} "
                    f"batch {batch_idx}/{total_str} loss={running_loss:.4f}",
                    flush=True,
                )
        train_loss = loss_sum / max(n_pos, 1)
    return {"train_loss": train_loss, "epochs": int(max(1, epochs))}


@torch.no_grad()
def evaluate_baseline(
    model: nn.Module,
    *,
    loader,
    known_pos,
    device: torch.device,
    num_neg_eval: int,
    ks: Sequence[int] = (5, 10, 20, 50),
    degree_alpha: float = 0.75,
    global_chem_degree: Optional[torch.Tensor] = None,
    global_dis_degree: Optional[torch.Tensor] = None,
    progress_every: int = 0,
    max_batches: int = 0,
    progress_prefix: str | None = None,
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()

    all_scores: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    mrr_sum = 0.0
    hits_sums = {int(k): 0.0 for k in ks}
    n_pos_total = 0

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    t0 = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        if int(max_batches) > 0 and batch_idx > int(max_batches):
            break
        batch = batch.to(device)
        pos_edge = batch[CD_REL].edge_label_index
        neg_edge = negative_sample_cd_batch_local(
            batch_data=batch,
            pos_edge_index_local=pos_edge,
            known_pos=known_pos,
            num_neg_per_pos=int(num_neg_eval),
            hard_negative_ratio=0.0,
            degree_alpha=float(degree_alpha),
            global_chem_degree=global_chem_degree,
            global_dis_degree=global_dis_degree,
            generator=None,
        )
        if getattr(model, "kind", "pair") == "loader":
            pos_logits, neg_logits = model.forward_batch(batch, pos_edge, neg_edge)
        else:
            pos_logits, neg_logits = _train_step_pair_model(model, batch, pos_edge, neg_edge)

        score = torch.sigmoid(torch.cat([pos_logits, neg_logits], dim=0)).detach().cpu()
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0).detach().cpu()
        all_scores.append(score)
        all_labels.append(labels)

        r = sampled_ranking_metrics(pos_logits, neg_logits, int(num_neg_eval), ks=tuple(int(k) for k in ks))
        bsz = int(pos_logits.numel())
        n_pos_total += bsz
        mrr_sum += float(r["mrr"]) * bsz
        for k in ks:
            hits_sums[int(k)] += float(r[f"hits_{int(k)}"]) * bsz
        if int(progress_every) > 0 and (batch_idx % int(progress_every) == 0):
            total = (
                min(int(total_batches), int(max_batches))
                if total_batches is not None and int(max_batches) > 0
                else total_batches
            )
            total_str = str(total) if total is not None else "?"
            prefix = f"{progress_prefix} " if progress_prefix else ""
            print(
                f"{prefix}eval batch {batch_idx}/{total_str} "
                f"(elapsed={time.time() - t0:.1f}s)",
                flush=True,
            )

    y_score = torch.cat(all_scores, dim=0)
    y_true = torch.cat(all_labels, dim=0)
    out: Dict[str, float] = {
        "auroc": _safe_auc(y_true, y_score),
        "auprc": _safe_ap(y_true, y_score),
        "mrr": mrr_sum / max(n_pos_total, 1),
        "n_pos": float(n_pos_total),
    }
    for k in ks:
        out[f"hits_{int(k)}"] = hits_sums[int(k)] / max(n_pos_total, 1)
    return out


def load_main_model_from_checkpoint(
    *,
    checkpoint_path: str,
    data_train: HeteroData,
    data_full: HeteroData,
    vocabs: Dict[str, pl.DataFrame],
    device: torch.device,
) -> HGTPredictor:
    """Load HGTPredictor from checkpoint with inferred architecture."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = infer_hgt_hparams_from_state(ckpt["model_state"])
    num_nodes_dict = {nt: int(data_full[nt].num_nodes) for nt in data_full.node_types}
    node_input_dims = {
        nt: int(data_full[nt].x.size(1))
        for nt in data_full.node_types
        if isinstance(data_full[nt].x, torch.Tensor)
        and data_full[nt].x.dim() == 2
        and data_full[nt].x.is_floating_point()
    }
    model = HGTPredictor(
        num_nodes_dict=num_nodes_dict,
        metadata=data_train.metadata(),
        node_input_dims=model_cfg["node_input_dims"] or node_input_dims,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=0.0,
        num_action_types=int(model_cfg["num_action_types"] or vocabs["action_type"].height),
        num_action_subjects=int(model_cfg["num_action_subjects"] or vocabs["action_subject"].height),
        num_pheno_action_types=int(model_cfg["num_pheno_action_types"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model


@dataclass
class ComparisonConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_neg_train: int = 2
    num_neg_eval: int = 20
    ks: Tuple[int, ...] = (5, 10, 20, 50)
    progress_every: int = 25
    max_train_batches: int = 0
    max_eval_batches: int = 0


def compare_main_and_baselines(
    *,
    checkpoint_path: str,
    data_full: HeteroData,
    vocabs: Dict[str, pl.DataFrame],
    arts: SplitArtifacts,
    baseline_names: Iterable[str],
    device: torch.device,
    config: ComparisonConfig,
) -> Dict[str, Dict[str, float]]:
    """Evaluate main HGT and baselines on val/test using a shared split artifact."""
    results: Dict[str, Dict[str, float]] = {}

    t0 = time.time()
    print("[main_hgt] Loading checkpoint...", flush=True)
    main_model = load_main_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        data_train=arts.data_train,
        data_full=data_full,
        vocabs=vocabs,
        device=device,
    )
    print("[main_hgt] Evaluating validation split...", flush=True)
    val_main = eval_epoch(
        main_model,
        arts.val_loader,
        arts.known_pos_val,
        device,
        num_neg_per_pos=int(config.num_neg_eval),
        ks=config.ks,
        amp=False,
        hard_negative_ratio=0.0,
        degree_alpha=0.75,
        sampling_seed=None,
        global_chem_degree=arts.chem_train_degree,
        global_dis_degree=arts.dis_train_degree,
        progress_every=int(config.progress_every),
        max_batches=int(config.max_eval_batches),
        progress_prefix="[main_hgt][val]",
    )
    print("[main_hgt] Evaluating test split...", flush=True)
    test_main = eval_epoch(
        main_model,
        arts.test_loader,
        arts.known_pos_test,
        device,
        num_neg_per_pos=int(config.num_neg_eval),
        ks=config.ks,
        amp=False,
        hard_negative_ratio=0.0,
        degree_alpha=0.75,
        sampling_seed=None,
        global_chem_degree=arts.chem_train_degree,
        global_dis_degree=arts.dis_train_degree,
        progress_every=int(config.progress_every),
        max_batches=int(config.max_eval_batches),
        progress_prefix="[main_hgt][test]",
    )
    results["main_hgt"] = {
        **{f"val_{k}": float(v) for k, v in val_main.items()},
        **{f"test_{k}": float(v) for k, v in test_main.items()},
        "train_seconds": 0.0,
        "eval_seconds": float(time.time() - t0),
    }
    print(f"[main_hgt] Done in {results['main_hgt']['eval_seconds']:.1f}s", flush=True)

    for name in baseline_names:
        name = name.strip().lower()
        bt0 = time.time()
        print(f"[{name}] Building baseline...", flush=True)
        baseline = build_baseline(
            name,
            data_train=arts.data_train,
            split_train_pos=arts.split.train_pos,
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            num_heads=int(config.num_heads),
            dropout=float(config.dropout),
            device=device,
        )
        print(f"[{name}] Training...", flush=True)
        train_stats = train_baseline(
            baseline,
            arts=arts,
            device=device,
            epochs=int(config.epochs),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
            grad_clip=float(config.grad_clip),
            num_neg_train=int(config.num_neg_train),
            degree_alpha=0.75,
            progress_every=int(config.progress_every),
            max_batches=int(config.max_train_batches),
            progress_prefix=f"[{name}]",
        )
        train_time = time.time() - bt0

        et0 = time.time()
        print(f"[{name}] Evaluating validation split...", flush=True)
        val_metrics = evaluate_baseline(
            baseline,
            loader=arts.val_loader,
            known_pos=arts.known_pos_val,
            device=device,
            num_neg_eval=int(config.num_neg_eval),
            ks=config.ks,
            degree_alpha=0.75,
            global_chem_degree=arts.chem_train_degree,
            global_dis_degree=arts.dis_train_degree,
            progress_every=int(config.progress_every),
            max_batches=int(config.max_eval_batches),
            progress_prefix=f"[{name}][val]",
        )
        print(f"[{name}] Evaluating test split...", flush=True)
        test_metrics = evaluate_baseline(
            baseline,
            loader=arts.test_loader,
            known_pos=arts.known_pos_test,
            device=device,
            num_neg_eval=int(config.num_neg_eval),
            ks=config.ks,
            degree_alpha=0.75,
            global_chem_degree=arts.chem_train_degree,
            global_dis_degree=arts.dis_train_degree,
            progress_every=int(config.progress_every),
            max_batches=int(config.max_eval_batches),
            progress_prefix=f"[{name}][test]",
        )
        eval_time = time.time() - et0

        results[name] = {
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            **{f"test_{k}": float(v) for k, v in test_metrics.items()},
            "train_seconds": float(train_time),
            "eval_seconds": float(eval_time),
            "train_loss": float(train_stats.get("train_loss", float("nan"))),
        }
        print(
            f"[{name}] Done (train={train_time:.1f}s, eval={eval_time:.1f}s, "
            f"val_auprc={results[name].get('val_auprc', float('nan')):.4f})",
            flush=True,
        )

    return results
