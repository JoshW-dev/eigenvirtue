"""Render a rotating 3D view of the difference vectors as a GIF.

The vectors live in 768 dimensions; this projects them onto the top three
singular directions of the difference matrix and spins the camera. Output:
docs/figures/fig7_rotation.gif
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eigenvirtue import LocalEmbedder, load_pairs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "fig7_rotation.gif"

GOOD = "#2c6fbb"
NEUT = "#9aa0a8"
INK = "#2b2f36"
MUTED = "#6b7078"


def main():
    embedder = LocalEmbedder("all-mpnet-base-v2")
    pairs = load_pairs(ROOT / "data" / "virtue_pairs.json")
    pos = embedder.encode([p.positive for p in pairs])
    neg = embedder.encode([p.negative for p in pairs])
    diffs = pos - neg
    _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    e1 = vt[0] if vt[0] @ diffs.mean(axis=0) > 0 else -vt[0]
    basis = np.stack([e1, vt[1], vt[2]])
    coords = diffs @ basis.T  # (n, 3)

    fig = plt.figure(figsize=(7, 6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
        pane.line.set_color((0.85, 0.85, 0.84))
    ax.grid(False)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_xlabel("$v_1$", color=MUTED, labelpad=-8)
    ax.set_ylabel("$v_2$", color=MUTED, labelpad=-8)
    ax.set_zlabel("$v_3$", color=MUTED, labelpad=-8)

    norms = np.linalg.norm(diffs, axis=1)
    align = coords[:, 0] / norms  # cos(d_i, v1)
    for (x, y, z), a in zip(coords, align):
        strong = a > 0.4
        ax.quiver(0, 0, 0, x, y, z,
                  color=GOOD if strong else NEUT, alpha=0.8 if strong else 0.45,
                  linewidth=1.3 if strong else 1.0, arrow_length_ratio=0.06)
    xmax = float(np.max(coords[:, 0]))
    ax.quiver(0, 0, 0, xmax * 1.15, 0, 0, color=INK,
              linewidth=3.2, arrow_length_ratio=0.05)
    ax.text(xmax * 1.3, 0, 0, "eigenvirtue $v_1$", color=INK, fontsize=11)

    ylim = float(np.max(np.abs(coords[:, 1]))) * 0.9
    zlim = float(np.max(np.abs(coords[:, 2]))) * 0.9
    ax.set_xlim(-xmax * 0.35, xmax * 1.3)
    ax.set_ylim(-ylim, ylim)
    ax.set_zlim(-zlim, zlim)
    ax.set_box_aspect((1.5, 1, 1))
    ax.set_title("61 virtue$-$vice difference vectors, projected from 768 to 3 dimensions",
                 fontsize=10.5, color=INK, pad=0)
    fig.text(0.5, 0.04,
             "one arrow per sentence pair; blue arrows align with $v_1$ at cosine > 0.4",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout()

    def update(frame):
        ax.view_init(elev=14, azim=-80 + frame * 4)
        return []

    anim = FuncAnimation(fig, update, frames=90, blit=False)
    anim.save(OUT, writer=PillowWriter(fps=18), dpi=95)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
