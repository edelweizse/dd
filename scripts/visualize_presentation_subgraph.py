"""
Create a small, labeled graph subset for presentation slides.

The figure focuses on one disease and shows a compact tri-partite neighborhood:
- disease node (center)
- top connected chemicals (left)
- top connected disease genes (right)

Usage:
    python -m scripts.visualize_presentation_subgraph
    python -m scripts.visualize_presentation_subgraph --disease-id MESH:D001943 --num-chemicals 5 --num-genes 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl


NODE_COLORS = {
    "disease": "#ff7f0e",
    "chemical": "#1f77b4",
    "gene": "#2ca02c",
}

EDGE_COLORS = {
    "chem_disease": "#6b7280",
    "disease_gene": "#2563eb",
    "chem_gene": "#059669",
}


def _shorten(text: str, max_len: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def _load_tables(processed_dir: Path) -> Dict[str, pl.DataFrame]:
    return {
        "chemicals": pl.read_parquet(processed_dir / "chemicals_nodes.parquet"),
        "diseases": pl.read_parquet(processed_dir / "diseases_nodes.parquet"),
        "genes": pl.read_parquet(processed_dir / "genes_nodes.parquet"),
        "chem_disease": pl.read_parquet(processed_dir / "chem_disease_edges.parquet"),
        "disease_gene": pl.read_parquet(processed_dir / "disease_gene_edges.parquet"),
        "chem_gene": pl.read_parquet(processed_dir / "chem_gene_edges.parquet"),
    }


def _select_subset(
    tables: Dict[str, pl.DataFrame],
    disease_id: str,
    num_chemicals: int,
    num_genes: int,
) -> Tuple[Dict, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    diseases = tables["diseases"]
    chem_disease = tables["chem_disease"]
    disease_gene = tables["disease_gene"]
    chem_gene = tables["chem_gene"]

    disease_row = diseases.filter(pl.col("DS_OMIM_MESH_ID") == disease_id)
    if disease_row.height == 0:
        available = diseases.select(["DS_OMIM_MESH_ID", "DS_NAME"]).head(10)
        raise ValueError(
            f"Disease '{disease_id}' not found. Example available IDs:\n{available}"
        )

    ds_id = int(disease_row["DS_ID"][0])
    disease_info = {
        "DS_ID": ds_id,
        "DS_OMIM_MESH_ID": disease_row["DS_OMIM_MESH_ID"][0],
        "DS_NAME": disease_row["DS_NAME"][0],
    }

    disease_chems = chem_disease.filter(pl.col("DS_ID") == ds_id).select(["CHEM_ID"]).unique()
    disease_genes = disease_gene.filter(pl.col("DS_ID") == ds_id).select(["GENE_ID"]).unique()

    if disease_chems.height == 0 or disease_genes.height == 0:
        raise ValueError(
            f"Disease {disease_id} has insufficient links "
            f"(chemicals={disease_chems.height}, genes={disease_genes.height})."
        )

    bridge = (
        chem_gene
        .join(disease_chems, on="CHEM_ID", how="inner")
        .join(disease_genes, on="GENE_ID", how="inner")
    )

    # Chemical ranking: prefer chemicals linked to many disease genes.
    chem_rank = (
        bridge.group_by("CHEM_ID")
        .agg(
            pl.col("GENE_ID").n_unique().alias("shared_gene_count"),
            pl.len().alias("interaction_count"),
        )
        .sort(["shared_gene_count", "interaction_count"], descending=[True, True])
        .head(num_chemicals)
    )

    selected_chems = chem_rank.select(["CHEM_ID"])

    selected_gene_rank = (
        bridge.join(selected_chems, on="CHEM_ID", how="inner")
        .group_by("GENE_ID")
        .agg(
            pl.col("CHEM_ID").n_unique().alias("linked_chemicals"),
            pl.len().alias("interaction_count"),
        )
        .sort(["linked_chemicals", "interaction_count"], descending=[True, True])
        .head(num_genes)
    )

    selected_genes = selected_gene_rank.select(["GENE_ID"])

    selected_chem_gene = (
        chem_gene.join(selected_chems, on="CHEM_ID", how="inner")
        .join(selected_genes, on="GENE_ID", how="inner")
    )

    return disease_info, selected_chems, selected_genes, selected_chem_gene


def _build_graph(
    tables: Dict[str, pl.DataFrame],
    disease_info: Dict,
    selected_chems: pl.DataFrame,
    selected_genes: pl.DataFrame,
    selected_chem_gene: pl.DataFrame,
    max_chem_gene_edges: int,
) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]], Dict[Tuple[str, str], str]]:
    chemicals = tables["chemicals"]
    genes = tables["genes"]

    chem_nodes = (
        selected_chems.join(
            chemicals.select(["CHEM_ID", "CHEM_NAME", "CHEM_MESH_ID"]),
            on="CHEM_ID",
            how="left",
        )
        .sort("CHEM_NAME")
    )

    gene_nodes = (
        selected_genes.join(
            genes.select(["GENE_ID", "GENE_SYMBOL", "GENE_NCBI_ID"]),
            on="GENE_ID",
            how="left",
        )
        .sort("GENE_SYMBOL")
    )

    graph = nx.Graph()

    disease_key = f"disease:{disease_info['DS_ID']}"
    graph.add_node(
        disease_key,
        node_type="disease",
        label=f"{_shorten(disease_info['DS_NAME'], 32)}\n({disease_info['DS_OMIM_MESH_ID']})",
    )

    chem_keys: List[str] = []
    for row in chem_nodes.iter_rows(named=True):
        key = f"chemical:{row['CHEM_ID']}"
        chem_keys.append(key)
        graph.add_node(
            key,
            node_type="chemical",
            label=f"{_shorten(row['CHEM_NAME'], 24)}\n({row['CHEM_MESH_ID']})",
        )
        graph.add_edge(key, disease_key, relation="chem_disease")

    gene_keys: List[str] = []
    for row in gene_nodes.iter_rows(named=True):
        key = f"gene:{row['GENE_ID']}"
        gene_keys.append(key)
        graph.add_node(
            key,
            node_type="gene",
            label=f"{row['GENE_SYMBOL']}\n(NCBI:{row['GENE_NCBI_ID']})",
        )
        graph.add_edge(disease_key, key, relation="disease_gene")

    action_labels: Dict[Tuple[str, str], str] = {}
    grouped = (
        selected_chem_gene
        .group_by(["CHEM_ID", "GENE_ID"])
        .agg(
            pl.len().alias("interaction_count"),
            pl.col("ACTION_TYPE").drop_nulls().unique().sort().alias("ACTION_TYPE"),
        )
        .sort("interaction_count", descending=True)
        .head(max_chem_gene_edges)
    )

    for row in grouped.iter_rows(named=True):
        chem_key = f"chemical:{row['CHEM_ID']}"
        gene_key = f"gene:{row['GENE_ID']}"
        if chem_key in graph and gene_key in graph:
            graph.add_edge(chem_key, gene_key, relation="chem_gene")
            actions = row["ACTION_TYPE"] if row["ACTION_TYPE"] is not None else []
            if actions:
                action_labels[(chem_key, gene_key)] = ", ".join(actions[:2])

    # Fixed tri-partite layout for slide-friendly readability.
    pos: Dict[str, Tuple[float, float]] = {disease_key: (0.0, 0.0)}

    if chem_keys:
        chem_span = max(len(chem_keys) - 1, 1)
        for idx, key in enumerate(chem_keys):
            y = 2.8 - (5.6 * idx / chem_span)
            pos[key] = (-3.2, y)

    if gene_keys:
        gene_span = max(len(gene_keys) - 1, 1)
        for idx, key in enumerate(gene_keys):
            y = 2.8 - (5.6 * idx / gene_span)
            pos[key] = (3.2, y)

    return graph, pos, action_labels


def _plot_graph(
    graph: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    action_labels: Dict[Tuple[str, str], str],
    disease_label: str,
    output_png: Path,
    output_svg: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    ax.set_facecolor("#f8fafc")

    node_types = nx.get_node_attributes(graph, "node_type")
    labels = nx.get_node_attributes(graph, "label")

    for node_type, color in NODE_COLORS.items():
        nodelist = [n for n, t in node_types.items() if t == node_type]
        if not nodelist:
            continue
        size = 4600 if node_type == "disease" else 2500
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodelist,
            node_color=color,
            node_size=size,
            linewidths=1.2,
            edgecolors="#0f172a",
            alpha=0.95,
            ax=ax,
        )

    for rel_key, color in EDGE_COLORS.items():
        edgelist = [(u, v) for u, v, d in graph.edges(data=True) if d.get("relation") == rel_key]
        if not edgelist:
            continue
        width = 2.0 if rel_key == "chem_gene" else 1.6
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=edgelist,
            edge_color=color,
            width=width,
            alpha=0.82,
            ax=ax,
        )

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=8.6,
        font_weight="bold",
        font_color="#111827",
        ax=ax,
    )

    if action_labels:
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=action_labels,
            font_size=6.8,
            font_color="#065f46",
            rotate=False,
            bbox={"alpha": 0.6, "color": "#ecfeff", "pad": 0.2},
            ax=ax,
        )

    legend_lines = [
        plt.Line2D([0], [0], color=EDGE_COLORS["chem_disease"], lw=2, label="chemical-disease"),
        plt.Line2D([0], [0], color=EDGE_COLORS["disease_gene"], lw=2, label="disease-gene"),
        plt.Line2D([0], [0], color=EDGE_COLORS["chem_gene"], lw=2, label="chemical-gene"),
    ]
    ax.legend(handles=legend_lines, loc="upper center", ncol=3, frameon=True, fontsize=10)

    ax.set_title(
        f"Labeled Mini-Subgraph for Presentation\nFocus Disease: {disease_label}",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def _write_selection_table(
    tables: Dict[str, pl.DataFrame],
    disease_info: Dict,
    selected_chems: pl.DataFrame,
    selected_genes: pl.DataFrame,
    output_csv: Path,
) -> None:
    chemicals = tables["chemicals"].select(["CHEM_ID", "CHEM_MESH_ID", "CHEM_NAME"])
    genes = tables["genes"].select(["GENE_ID", "GENE_NCBI_ID", "GENE_SYMBOL"])

    chem_tbl = (
        selected_chems.join(chemicals, on="CHEM_ID", how="left")
        .with_columns(
            pl.lit("chemical").alias("node_type"),
            pl.col("CHEM_ID").cast(pl.Int64),
            pl.col("CHEM_MESH_ID").alias("external_id"),
            pl.col("CHEM_NAME").alias("label"),
        )
        .select(["node_type", "CHEM_ID", "external_id", "label"])
        .rename({"CHEM_ID": "internal_id"})
    )

    gene_tbl = (
        selected_genes.join(genes, on="GENE_ID", how="left")
        .with_columns(
            pl.lit("gene").alias("node_type"),
            pl.col("GENE_ID").cast(pl.Int64),
            pl.col("GENE_NCBI_ID").cast(pl.String).alias("external_id"),
            pl.col("GENE_SYMBOL").alias("label"),
        )
        .select(["node_type", "GENE_ID", "external_id", "label"])
        .rename({"GENE_ID": "internal_id"})
    )

    disease_tbl = pl.DataFrame(
        {
            "node_type": ["disease"],
            "internal_id": [int(disease_info["DS_ID"])],
            "external_id": [disease_info["DS_OMIM_MESH_ID"]],
            "label": [disease_info["DS_NAME"]],
        }
    ).with_columns(pl.col("internal_id").cast(pl.Int64))

    pl.concat([disease_tbl, chem_tbl, gene_tbl], how="vertical").write_csv(output_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a labeled mini-subgraph figure for slides.")
    parser.add_argument("--processed-dir", type=str, default="./data/processed", help="Processed data directory")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument(
        "--disease-id",
        type=str,
        default="MESH:D003920",
        help="Disease ID in DS_OMIM_MESH_ID format (default: Diabetes Mellitus)",
    )
    parser.add_argument("--num-chemicals", type=int, default=4, help="Number of chemical nodes")
    parser.add_argument("--num-genes", type=int, default=5, help="Number of gene nodes")
    parser.add_argument(
        "--max-chem-gene-edges",
        type=int,
        default=12,
        help="Maximum number of chemical-gene edges to draw",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="presentation_subgraph",
        help="Base output filename (without extension)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(processed_dir)
    disease_info, selected_chems, selected_genes, selected_chem_gene = _select_subset(
        tables=tables,
        disease_id=args.disease_id,
        num_chemicals=args.num_chemicals,
        num_genes=args.num_genes,
    )

    graph, pos, action_labels = _build_graph(
        tables=tables,
        disease_info=disease_info,
        selected_chems=selected_chems,
        selected_genes=selected_genes,
        selected_chem_gene=selected_chem_gene,
        max_chem_gene_edges=args.max_chem_gene_edges,
    )

    output_png = output_dir / f"{args.output_name}.png"
    output_svg = output_dir / f"{args.output_name}.svg"
    output_csv = output_dir / f"{args.output_name}_nodes.csv"

    _plot_graph(
        graph=graph,
        pos=pos,
        action_labels=action_labels,
        disease_label=f"{disease_info['DS_NAME']} ({disease_info['DS_OMIM_MESH_ID']})",
        output_png=output_png,
        output_svg=output_svg,
    )
    _write_selection_table(
        tables=tables,
        disease_info=disease_info,
        selected_chems=selected_chems,
        selected_genes=selected_genes,
        output_csv=output_csv,
    )

    print("Saved presentation graph:", output_png)
    print("Saved presentation graph (vector):", output_svg)
    print("Saved selected node table:", output_csv)
    print(
        "Subset stats - "
        f"nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}, "
        f"chemicals: {selected_chems.height}, genes: {selected_genes.height}"
    )


if __name__ == "__main__":
    main()
