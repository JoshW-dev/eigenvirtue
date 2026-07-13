"""Render the dark hero GIF for the top of the README: the 61 difference
vectors rotating in the top three contrast dimensions, no axes, no chrome.
Output: docs/figures/hero_rotation.gif
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
OUT = ROOT / "docs" / "figures" / "hero_rotation.gif"

BG = "#ffffff"
ALIGNED = "#2c6fbb"
OTHER = "#c3c8ce"
AXIS = "#2b2f36"


def main():
    embedder = LocalEmbedder("all-mpnet-base-v2")
    pairs = load_pairs(ROOT / "data" / "virtue_pairs.json")
    pos = embedder.encode([p.positive for p in pairs])
    neg = embedder.encode([p.negative for p in pairs])
    diffs = pos - neg
    _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    e1 = vt[0] if vt[0] @ diffs.mean(axis=0) > 0 else -vt[0]
    coords = diffs @ np.stack([e1, vt[1], vt[2]]).T

    fig = plt.figure(figsize=(6.2, 5.0), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    ax.set_axis_off()

    align = coords[:, 0] / np.linalg.norm(diffs, axis=1)
    for (x, y, z), a in zip(coords, align):
        strong = a > 0.4
        ax.quiver(0, 0, 0, x, y, z,
                  color=ALIGNED if strong else OTHER, alpha=0.9 if strong else 0.55,
                  linewidth=1.4 if strong else 1.0, arrow_length_ratio=0.06)
    xmax = float(np.max(coords[:, 0]))
    ax.quiver(0, 0, 0, xmax * 1.18, 0, 0, color=AXIS,
              linewidth=2.6, arrow_length_ratio=0.05)

    ylim = float(np.max(np.abs(coords[:, 1]))) * 0.85
    zlim = float(np.max(np.abs(coords[:, 2]))) * 0.85
    ax.set_xlim(-xmax * 0.3, xmax * 1.25)
    ax.set_ylim(-ylim, ylim)
    ax.set_zlim(-zlim, zlim)
    ax.set_box_aspect((1.7, 1, 1))
    fig.subplots_adjust(left=0, right=1, top=1.12, bottom=-0.12)

    def update(frame):
        ax.view_init(elev=12, azim=-80 + frame * 2)
        return []

    anim = FuncAnimation(fig, update, frames=180, blit=False)
    anim.save(OUT, writer=PillowWriter(fps=18), dpi=100)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
