"""
Create a larger, presentation-friendly subgraph that shows overall graph structure.

This visualization selects 2-5 diseases and connects each to representative
chemicals and genes, plus chemical-gene links to expose heterogeneous structure.

Usage:
    python -m scripts.visualize_graph_structure_multidisease
    python -m scripts.visualize_graph_structure_multidisease --disease-count 5
    python -m scripts.visualize_graph_structure_multidisease --disease-ids MESH:D003920,MESH:D001943
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl


NODE_COLORS = {
    "disease": "#ef4444",
    "chemical": "#0ea5e9",
    "gene": "#22c55e",
}

EDGE_COLORS = {
    "chem_disease": "#64748b",
    "disease_gene": "#3b82f6",
    "chem_gene": "#10b981",
}


def _shorten(text: str, max_len: int) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def _load_tables(processed_dir: Path) -> Dict[str, pl.DataFrame]:
    return {
        "chemicals": pl.read_parquet(processed_dir / "chemicals_nodes.parquet"),
        "diseases": pl.read_parquet(processed_dir / "diseases_nodes.parquet"),
        "genes": pl.read_parquet(processed_dir / "genes_nodes.parquet"),
        "chem_disease": pl.read_parquet(processed_dir / "chem_disease_edges.parquet"),
        "disease_gene": pl.read_parquet(processed_dir / "disease_gene_edges.parquet"),
        "chem_gene": pl.read_parquet(processed_dir / "chem_gene_edges.parquet"),
    }


def _pick_diseases(tables: Dict[str, pl.DataFrame], disease_count: int, disease_ids_arg: str) -> pl.DataFrame:
    diseases = tables["diseases"].select(["DS_ID", "DS_OMIM_MESH_ID", "DS_NAME"])

    if disease_ids_arg:
        req_ids = [x.strip() for x in disease_ids_arg.split(",") if x.strip()]
        selected = diseases.filter(pl.col("DS_OMIM_MESH_ID").is_in(req_ids))
        if selected.height != len(req_ids):
            found = set(selected["DS_OMIM_MESH_ID"].to_list())
            missing = [x for x in req_ids if x not in found]
            raise ValueError(f"Unknown disease IDs: {missing}")
        return selected

    preferred = [
        "MESH:D003920",  # Diabetes Mellitus
        "MESH:D001943",  # Breast Neoplasms
        "MESH:D006973",  # Hypertension
        "MESH:D010300",  # Parkinson Disease
        "MESH:D012640",  # Seizures
    ]
    picked = diseases.filter(pl.col("DS_OMIM_MESH_ID").is_in(preferred))
    if picked.height >= disease_count:
        return (
            picked
            .with_columns(pl.col("DS_OMIM_MESH_ID").replace_strict({x: i for i, x in enumerate(preferred)}, default=9999).alias("ord"))
            .sort("ord")
            .drop("ord")
            .head(disease_count)
        )

    chem_disease = tables["chem_disease"]
    disease_gene = tables["disease_gene"]
    ranking = (
        chem_disease.group_by("DS_ID").len().rename({"len": "chem_count"})
        .join(disease_gene.group_by("DS_ID").len().rename({"len": "gene_count"}), on="DS_ID", how="left")
        .fill_null(0)
        .with_columns((pl.col("chem_count") + pl.col("gene_count")).alias("total"))
        .join(diseases, on="DS_ID", how="left")
        .sort("total", descending=True)
        .head(disease_count)
        .select(["DS_ID", "DS_OMIM_MESH_ID", "DS_NAME"])
    )
    return ranking


def _build_graph(
    tables: Dict[str, pl.DataFrame],
    selected_diseases: pl.DataFrame,
    num_chems_per_disease: int,
    num_genes_per_disease: int,
    max_chem_gene_edges_per_disease: int,
) -> Tuple[nx.Graph, Dict[str, str], pl.DataFrame, pl.DataFrame]:
    diseases = tables["diseases"].select(["DS_ID", "DS_OMIM_MESH_ID", "DS_NAME"])
    chemicals = tables["chemicals"].select(["CHEM_ID", "CHEM_MESH_ID", "CHEM_NAME"])
    genes = tables["genes"].select(["GENE_ID", "GENE_NCBI_ID", "GENE_SYMBOL"])
    chem_disease = tables["chem_disease"].select(["CHEM_ID", "DS_ID"])
    disease_gene = tables["disease_gene"].select(["GENE_ID", "DS_ID"])
    chem_gene = tables["chem_gene"].select(["CHEM_ID", "GENE_ID", "ACTION_TYPE"])

    g = nx.Graph()
    labels: Dict[str, str] = {}
    node_rows: List[Dict] = []
    edge_rows: List[Dict] = []

    for drow in selected_diseases.iter_rows(named=True):
        ds_id = int(drow["DS_ID"])
        ds_mesh = drow["DS_OMIM_MESH_ID"]
        ds_name = drow["DS_NAME"]
        dkey = f"disease:{ds_id}"

        g.add_node(dkey, node_type="disease", internal_id=ds_id, external_id=ds_mesh, raw_label=ds_name)
        labels[dkey] = f"{_shorten(ds_name, 28)}\n({ds_mesh})"
        node_rows.append({
            "node_key": dkey,
            "node_type": "disease",
            "internal_id": ds_id,
            "external_id": ds_mesh,
            "label": ds_name,
        })

        disease_chems = chem_disease.filter(pl.col("DS_ID") == ds_id).select(["CHEM_ID"]).unique()
        disease_genes = disease_gene.filter(pl.col("DS_ID") == ds_id).select(["GENE_ID"]).unique()
        if disease_chems.height == 0 or disease_genes.height == 0:
            continue

        bridge = (
            chem_gene.join(disease_chems, on="CHEM_ID", how="inner")
            .join(disease_genes, on="GENE_ID", how="inner")
        )
        if bridge.height == 0:
            continue

        top_chems = (
            bridge.group_by("CHEM_ID")
            .agg(pl.col("GENE_ID").n_unique().alias("gene_count"), pl.len().alias("interaction_count"))
            .sort(["gene_count", "interaction_count"], descending=[True, True])
            .head(num_chems_per_disease)
            .select(["CHEM_ID"])
        )

        top_genes = (
            bridge.join(top_chems, on="CHEM_ID", how="inner")
            .group_by("GENE_ID")
            .agg(pl.col("CHEM_ID").n_unique().alias("chem_count"), pl.len().alias("interaction_count"))
            .sort(["chem_count", "interaction_count"], descending=[True, True])
            .head(num_genes_per_disease)
            .select(["GENE_ID"])
        )

        selected_bridge = (
            bridge.join(top_chems, on="CHEM_ID", how="inner")
            .join(top_genes, on="GENE_ID", how="inner")
        )

        cg_pairs = (
            selected_bridge.group_by(["CHEM_ID", "GENE_ID"])
            .agg(pl.len().alias("weight"), pl.col("ACTION_TYPE").drop_nulls().unique().sort().alias("actions"))
            .sort("weight", descending=True)
            .head(max_chem_gene_edges_per_disease)
        )

        chem_info = top_chems.join(chemicals, on="CHEM_ID", how="left")
        for crow in chem_info.iter_rows(named=True):
            ckey = f"chemical:{crow['CHEM_ID']}"
            if ckey not in g:
                g.add_node(
                    ckey,
                    node_type="chemical",
                    internal_id=int(crow["CHEM_ID"]),
                    external_id=crow["CHEM_MESH_ID"],
                    raw_label=crow["CHEM_NAME"],
                )
                labels[ckey] = _shorten(crow["CHEM_NAME"], 18)
                node_rows.append({
                    "node_key": ckey,
                    "node_type": "chemical",
                    "internal_id": int(crow["CHEM_ID"]),
                    "external_id": crow["CHEM_MESH_ID"],
                    "label": crow["CHEM_NAME"],
                })
            g.add_edge(ckey, dkey, relation="chem_disease", weight=1)
            edge_rows.append({"source": ckey, "target": dkey, "relation": "chem_disease", "weight": 1, "label": ""})

        gene_info = top_genes.join(genes, on="GENE_ID", how="left")
        for grow in gene_info.iter_rows(named=True):
            gkey = f"gene:{grow['GENE_ID']}"
            if gkey not in g:
                g.add_node(
                    gkey,
                    node_type="gene",
                    internal_id=int(grow["GENE_ID"]),
                    external_id=str(grow["GENE_NCBI_ID"]),
                    raw_label=grow["GENE_SYMBOL"],
                )
                labels[gkey] = grow["GENE_SYMBOL"]
                node_rows.append({
                    "node_key": gkey,
                    "node_type": "gene",
                    "internal_id": int(grow["GENE_ID"]),
                    "external_id": str(grow["GENE_NCBI_ID"]),
                    "label": grow["GENE_SYMBOL"],
                })
            g.add_edge(dkey, gkey, relation="disease_gene", weight=1)
            edge_rows.append({"source": dkey, "target": gkey, "relation": "disease_gene", "weight": 1, "label": ""})

        for row in cg_pairs.iter_rows(named=True):
            ckey = f"chemical:{row['CHEM_ID']}"
            gkey = f"gene:{row['GENE_ID']}"
            if ckey in g and gkey in g:
                g.add_edge(ckey, gkey, relation="chem_gene", weight=float(row["weight"]))
                actions = row["actions"] or []
                edge_rows.append({
                    "source": ckey,
                    "target": gkey,
                    "relation": "chem_gene",
                    "weight": float(row["weight"]),
                    "label": ", ".join(actions[:2]),
                })

    node_df = pl.DataFrame(node_rows).unique(subset=["node_key"], keep="first") if node_rows else pl.DataFrame()
    edge_df = pl.DataFrame(edge_rows) if edge_rows else pl.DataFrame()
    return g, labels, node_df, edge_df


def _layout_graph(g: nx.Graph, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    disease_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "disease"]
    n_d = len(disease_nodes)
    if n_d == 0:
        return {}

    radius = 4.0
    fixed_pos: Dict[str, Tuple[float, float]] = {}
    for i, node in enumerate(sorted(disease_nodes)):
        theta = (2.0 * math.pi * i) / n_d
        fixed_pos[node] = (radius * math.cos(theta), radius * math.sin(theta))

    rng = np.random.default_rng(seed)
    initial_pos = dict(fixed_pos)
    for node, attrs in g.nodes(data=True):
        if node in fixed_pos:
            continue
        disease_neighbors = [nbr for nbr in g.neighbors(node) if g.nodes[nbr].get("node_type") == "disease"]
        if disease_neighbors:
            xs = [fixed_pos[d][0] for d in disease_neighbors]
            ys = [fixed_pos[d][1] for d in disease_neighbors]
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
        else:
            cx, cy = 0.0, 0.0
        jitter = rng.normal(loc=0.0, scale=0.9, size=2)
        initial_pos[node] = (cx + float(jitter[0]), cy + float(jitter[1]))

    return nx.spring_layout(
        g,
        pos=initial_pos,
        fixed=disease_nodes,
        seed=seed,
        iterations=300,
        k=1.35 / max(np.sqrt(max(g.number_of_nodes(), 1)), 1.0),
    )


def _draw_graph(g: nx.Graph, labels: Dict[str, str], pos: Dict[str, Tuple[float, float]], output_png: Path, output_svg: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 12), dpi=180)
    ax.set_facecolor("#f8fafc")

    for rel, color in EDGE_COLORS.items():
        edgelist = [(u, v) for u, v, d in g.edges(data=True) if d.get("relation") == rel]
        if not edgelist:
            continue
        width = 0.9 if rel == "chem_gene" else 1.8
        alpha = 0.38 if rel == "chem_gene" else 0.75
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=edgelist,
            edge_color=color,
            width=width,
            alpha=alpha,
            ax=ax,
        )

    for node_type, color in NODE_COLORS.items():
        nodelist = [n for n, d in g.nodes(data=True) if d.get("node_type") == node_type]
        if not nodelist:
            continue
        if node_type == "disease":
            size = 3600
            lw = 1.8
            alpha = 0.98
        elif node_type == "chemical":
            size = 1600
            lw = 1.0
            alpha = 0.94
        else:
            size = 1400
            lw = 1.0
            alpha = 0.94

        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=nodelist,
            node_color=color,
            node_size=size,
            edgecolors="#0f172a",
            linewidths=lw,
            alpha=alpha,
            ax=ax,
        )

    disease_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "disease"]
    disease_labels = {n: labels[n] for n in disease_nodes if n in labels}
    nx.draw_networkx_labels(
        g,
        pos,
        labels=disease_labels,
        font_size=9.2,
        font_weight="bold",
        font_color="#111827",
        ax=ax,
    )

    other_nodes = [n for n in g.nodes if n not in disease_nodes]
    other_labels = {n: labels[n] for n in other_nodes if n in labels}
    nx.draw_networkx_labels(
        g,
        pos,
        labels=other_labels,
        font_size=7.2,
        font_weight="bold",
        font_color="#1f2937",
        ax=ax,
    )

    legend_lines = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE_COLORS["disease"], markeredgecolor="#0f172a", markersize=12, label="Disease"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE_COLORS["chemical"], markeredgecolor="#0f172a", markersize=10, label="Chemical"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=NODE_COLORS["gene"], markeredgecolor="#0f172a", markersize=10, label="Gene"),
        plt.Line2D([0], [0], color=EDGE_COLORS["chem_disease"], lw=2, label="chemical-disease"),
        plt.Line2D([0], [0], color=EDGE_COLORS["disease_gene"], lw=2, label="disease-gene"),
        plt.Line2D([0], [0], color=EDGE_COLORS["chem_gene"], lw=2, label="chemical-gene"),
    ]
    ax.legend(handles=legend_lines, loc="upper center", ncol=6, frameon=True, fontsize=10)

    node_types = nx.get_node_attributes(g, "node_type")
    n_d = sum(1 for t in node_types.values() if t == "disease")
    n_c = sum(1 for t in node_types.values() if t == "chemical")
    n_g = sum(1 for t in node_types.values() if t == "gene")
    ax.set_title(
        "Heterogeneous Graph Structure Snapshot\n"
        f"{n_d} diseases, {n_c} chemicals, {n_g} genes, {g.number_of_edges()} edges",
        fontsize=18,
        fontweight="bold",
        pad=16,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a larger multi-disease graph visualization for presentation.")
    parser.add_argument("--processed-dir", type=str, default="./data/processed", help="Processed data directory")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--output-name", type=str, default="graph_structure_multidisease", help="Base output filename")
    parser.add_argument("--disease-count", type=int, default=4, help="How many diseases to include (2-5)")
    parser.add_argument("--disease-ids", type=str, default="", help="Comma-separated disease IDs (optional)")
    parser.add_argument("--num-chems-per-disease", type=int, default=6, help="Chemicals per disease")
    parser.add_argument("--num-genes-per-disease", type=int, default=6, help="Genes per disease")
    parser.add_argument("--max-chem-gene-edges-per-disease", type=int, default=18, help="Max chem-gene links per disease")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.disease_count < 2 or args.disease_count > 5:
        raise ValueError("--disease-count must be between 2 and 5")

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(processed_dir)
    selected_diseases = _pick_diseases(tables, args.disease_count, args.disease_ids)

    g, labels, node_df, edge_df = _build_graph(
        tables=tables,
        selected_diseases=selected_diseases,
        num_chems_per_disease=args.num_chems_per_disease,
        num_genes_per_disease=args.num_genes_per_disease,
        max_chem_gene_edges_per_disease=args.max_chem_gene_edges_per_disease,
    )

    if g.number_of_nodes() == 0:
        raise RuntimeError("No nodes were selected for visualization")

    pos = _layout_graph(g)

    output_png = output_dir / f"{args.output_name}.png"
    output_svg = output_dir / f"{args.output_name}.svg"
    output_nodes_csv = output_dir / f"{args.output_name}_nodes.csv"
    output_edges_csv = output_dir / f"{args.output_name}_edges.csv"

    _draw_graph(g=g, labels=labels, pos=pos, output_png=output_png, output_svg=output_svg)
    node_df.write_csv(output_nodes_csv)
    edge_df.write_csv(output_edges_csv)

    disease_names = selected_diseases.select(["DS_NAME", "DS_OMIM_MESH_ID"]).iter_rows(named=True)
    print("Saved graph PNG:", output_png)
    print("Saved graph SVG:", output_svg)
    print("Saved node table:", output_nodes_csv)
    print("Saved edge table:", output_edges_csv)
    print("Selected diseases:")
    for row in disease_names:
        print(f"- {row['DS_NAME']} ({row['DS_OMIM_MESH_ID']})")
    print(f"Graph stats: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")


if __name__ == "__main__":
    main()
