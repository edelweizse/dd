"""
Create a very simple model architecture diagram for presentations.

Output:
  - evaluation_results/model_architecture_simple.png
  - evaluation_results/model_architecture_simple.svg
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(ax, x, y, w, h, text, fc, ec="#0f172a", fontsize=12, weight="bold"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)


def add_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#334155", shrinkA=4, shrinkB=4),
    )


def main():
    out_dir = Path("evaluation_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#f8fafc")

    add_box(
        ax,
        0.04,
        0.35,
        0.2,
        0.3,
        "Input\nHeterogeneous Graph\n(chemicals, diseases,\ngenes, pathways, GO)",
        fc="#dbeafe",
        fontsize=11,
    )
    add_box(
        ax,
        0.31,
        0.35,
        0.2,
        0.3,
        "HGT Encoder\n(2 message-passing\nlayers)",
        fc="#dcfce7",
        fontsize=12,
    )
    add_box(
        ax,
        0.58,
        0.35,
        0.2,
        0.3,
        "Node Embeddings\nfor Chemical & Disease",
        fc="#fef3c7",
        fontsize=12,
    )
    add_box(
        ax,
        0.82,
        0.35,
        0.14,
        0.3,
        "Bilinear Decoder\nscore = c^T W d\n+ Sigmoid",
        fc="#fee2e2",
        fontsize=11,
    )

    add_arrow(ax, 0.24, 0.5, 0.31, 0.5)
    add_arrow(ax, 0.51, 0.5, 0.58, 0.5)
    add_arrow(ax, 0.78, 0.5, 0.82, 0.5)

    ax.text(0.89, 0.26, "Output: link probability\nfor chemical-disease pair", ha="center", va="center", fontsize=11, color="#1f2937")

    ax.set_title("Simple Model Architecture (HGT for Chemical-Disease Link Prediction)", fontsize=16, fontweight="bold", pad=14)
    fig.tight_layout()

    png_path = out_dir / "model_architecture_simple.png"
    svg_path = out_dir / "model_architecture_simple.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", svg_path)


if __name__ == "__main__":
    main()
