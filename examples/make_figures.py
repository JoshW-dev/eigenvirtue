"""Generate all README figures into docs/figures/.

Figures 1 and 6 explain the method conceptually (real data for fig 1);
figures 2-5 present results, reading from results/summary.json so no
re-embedding is needed except for fig 1 (local mpnet).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eigenvirtue import LocalEmbedder, load_pairs

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

GOOD = "#2c6fbb"   # moral-good pole
BAD = "#c8442c"    # moral-bad pole
NEUT = "#8a8f98"   # diverging midpoint / nonmoral
INK = "#2b2f36"
MUTED = "#6b7078"
GRID = "#e8e8e6"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

MORAL_COLOR = {
    "virtuous": GOOD, "virtuous_unpleasant": GOOD,
    "vicious": BAD, "vicious_pleasant": BAD,
    "neutral": NEUT, "pleasant_nonmoral": NEUT, "unpleasant_nonmoral": NEUT,
}
SENTIMENT_SUFFIX = {
    "virtuous_unpleasant": "  · unpleasant",
    "vicious_pleasant": "  · pleasant",
    "pleasant_nonmoral": "  · pleasant, nonmoral",
    "unpleasant_nonmoral": "  · unpleasant, nonmoral",
}


def fig1_arrows():
    """Real data: every contrast pair is an arrow in embedding space; the
    eigenvirtue is their shared direction. Plane spanned by top two singular
    vectors of the difference matrix (all-mpnet-base-v2)."""
    embedder = LocalEmbedder("all-mpnet-base-v2")
    pairs = load_pairs(ROOT / "data" / "virtue_pairs.json")
    pos = embedder.encode([p.positive for p in pairs])
    neg = embedder.encode([p.negative for p in pairs])
    diffs = pos - neg
    _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    e1 = vt[0] if vt[0] @ diffs.mean(axis=0) > 0 else -vt[0]
    e2 = vt[1]

    # Two panels: pairs scattered in space (left), then every difference
    # arrow translated to a common origin (right); direction agreement
    # becomes visible as a rightward fan around v1.
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 5.4))

    for a, b in zip(neg, pos):
        axl.plot([a @ e1, b @ e1], [a @ e2, b @ e2], "-", color=GRID, linewidth=0.7, zorder=1)
        axl.plot([a @ e1], [a @ e2], "o", color=BAD, markersize=4, alpha=0.8, zorder=2)
        axl.plot([b @ e1], [b @ e2], "o", color=GOOD, markersize=4, alpha=0.8, zorder=2)
    axl.set_xlabel("component along $v_1$")
    axl.set_ylabel("component along $v_2$")
    axl.set_title("61 sentence pairs, scattered across embedding space", fontsize=10.5, color=INK)
    handles = [Line2D([], [], color=GOOD, marker="o", linestyle="", markersize=5, label="virtue sentence"),
               Line2D([], [], color=BAD, marker="o", linestyle="", markersize=5, label="its matched vice sentence")]
    axl.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False)

    labeled_groups = {"courage vs cowardice": "courage", "honesty vs dishonesty": "honesty",
                      "compassion vs cruelty": "compassion", "justice vs injustice": "justice",
                      "patience vs irascibility": "patience", "generosity vs stinginess": "generosity",
                      "humility vs arrogance": "humility", "loyalty vs betrayal": "loyalty",
                      "temperance vs self-indulgence": "temperance", "gratitude vs ingratitude": "gratitude"}
    label_nudge = {"compassion": (0, 10), "loyalty": (0, -8), "patience": (0, -12),
                   "courage": (0, -1), "justice": (0, 9), "humility": (0, 8),
                   "honesty": (0, 2), "gratitude": (0, 6), "temperance": (0, -8)}
    seen = set()
    for p, d in zip(pairs, diffs):
        x, y = d @ e1, d @ e2
        labeled = p.group in labeled_groups and p.group not in seen
        axr.add_patch(FancyArrowPatch(
            (0, 0), (x, y), arrowstyle="-|>", mutation_scale=10,
            color=GOOD if labeled else NEUT, linewidth=1.5 if labeled else 0.8,
            alpha=0.95 if labeled else 0.35, zorder=3 if labeled else 2,
        ))
        if labeled:
            seen.add(p.group)
            name = labeled_groups[p.group]
            dx, dy = label_nudge.get(name, (0, 0))
            axr.annotate(name, (x * 1.06, y * 1.06), fontsize=8, color=INK,
                         xytext=(4 + dx, dy), textcoords="offset points",
                         ha="left" if x >= 0 else "right", zorder=4)
    proj1, proj2 = diffs @ e1, diffs @ e2
    xmax = float(np.max(proj1))
    ymax = float(np.max(np.abs(proj2)))
    axr.set_xlim(min(0, float(np.min(proj1))) - 0.05, xmax * 1.55)
    axr.set_ylim(-ymax * 1.15, ymax * 1.15)
    axr.add_patch(FancyArrowPatch((0, 0), (xmax * 1.05, 0), arrowstyle="-|>",
                                  mutation_scale=18, color=INK, linewidth=2.8, zorder=5))
    axr.annotate("eigenvirtue $v_1$", (xmax * 1.09, -0.008), fontsize=10, color=INK,
                 ha="left", va="center", zorder=5)
    axr.set_xlabel("component along $v_1$")
    axr.set_title("the same pairs as difference arrows, moved to one origin:\nvice $\\rightarrow$ virtue points one way", fontsize=10.5, color=INK)
    axr.set_aspect("equal")

    fig.suptitle("Each virtue/vice sentence pair is an arrow, and the arrows agree", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "fig1_arrows.png", dpi=200)
    plt.close(fig)


def fig2_spectrum(summary):
    """Small multiples: energy spectrum per model, with the uniform-noise floor."""
    models = summary["models"]
    n_pairs = 61
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharey=True)
    for ax, m in zip(axes.flat, models):
        energy = [e * 100 for e in m["energy"]]
        ax.bar(range(1, len(energy) + 1), energy, color=GOOD, width=0.72)
        ax.axhline(100 / n_pairs, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(m["model"], fontsize=9.5, color=INK)
        ax.set_xticks(range(1, len(energy) + 1))
        ax.tick_params(labelsize=8)
    axes[0, 0].annotate("uniform-noise floor (1/61)", (10.4, 100 / 61 + 0.5),
                        fontsize=8, color=MUTED, ha="right")
    for ax in axes[1]:
        ax.set_xlabel("principal component of the contrast", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of energy (%)", fontsize=9)
    fig.suptitle("One component dominates the spectrum of the virtue contrast",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "fig2_spectrum.png", dpi=200)
    plt.close(fig)


def fig3_ranking(summary):
    """Unseen everyday actions ranked along the sentiment-removed virtue axis
    (strongest model). Color = moral polarity; annotations carry sentiment."""
    m = next(r for r in summary["models"] if "text-embedding-3-large" in r["model"])
    probes = summary["probes"]
    scores = m["probe_pure_scores"]
    order = np.argsort(scores)

    import textwrap

    fig, ax = plt.subplots(figsize=(10.2, 0.3 * len(probes) + 1.6))
    ax.grid(axis="y", visible=False)
    for rank, idx in enumerate(order):
        p, s = probes[idx], scores[idx]
        label = textwrap.fill(p["text"] + SENTIMENT_SUFFIX.get(p["category"], ""), 46)
        ax.barh(rank, s, color=MORAL_COLOR[p["category"]], height=0.72)
        if s < 0:
            ax.text(0.004, rank, label, va="center", ha="left", fontsize=8, color=INK)
        else:
            ax.text(-0.004, rank, label, va="center", ha="right", fontsize=8, color=INK)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlim(-0.33, 0.33)
    ax.set_yticks([])
    ax.set_xlabel("projection onto the eigenvirtue, sentiment removed (cosine)")
    ax.set_title("31 unseen actions on the virtue axis (openai/text-embedding-3-large)",
                 fontsize=12, color=INK)
    handles = [Rectangle((0, 0), 1, 1, color=GOOD), Rectangle((0, 0), 1, 1, color=NEUT),
               Rectangle((0, 0), 1, 1, color=BAD)]
    ax.legend(handles, ["virtuous", "nonmoral", "vicious"], loc="upper right",
              fontsize=8.5, frameon=False, title="ground-truth label", title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_ranking.png", dpi=200)
    plt.close(fig)


def fig4_scatter(summary):
    """Virtue (sentiment removed) vs sentiment. Color = moral polarity,
    marker = pleasantness. The off-diagonal quadrants are the point."""
    m = next(r for r in summary["models"] if "text-embedding-3-large" in r["model"])
    probes = summary["probes"]
    xs, ys = m["probe_sentiment_scores"], m["probe_pure_scores"]

    def marker(cat):
        if "pleasant_nonmoral" == cat or cat == "vicious_pleasant":
            return "^"
        if cat in ("unpleasant_nonmoral", "virtuous_unpleasant"):
            return "v"
        return "o"

    short = {
        "spent the night at the hospital comforting a dying neighbor": "comforted a dying neighbor all night",
        "apologized publicly for a mistake and accepted the embarrassment": "apologized publicly, took the embarrassment",
        "reported her own team's safety violation, knowing it would cost them the contract": "blew the whistle on her own team",
        "told her best friend a hard truth he didn't want to hear": "told a friend a hard truth",
        "charmed the whole party with flattery he didn't mean": "insincere flattery, charmed everyone",
        "enjoyed a luxurious vacation paid for with embezzled money": "luxury vacation on embezzled money",
        "won the tournament by quietly cheating and celebrated all night": "cheated, won, celebrated",
        "got upgraded to first class for free": "free first-class upgrade",
        "got caught in the rain without an umbrella": "caught in the rain",
        "returned a lost wallet with all the cash still inside": "returned a lost wallet, cash intact",
        "kept a promise even after it became inconvenient": "kept an inconvenient promise",
        "mocked a beginner struggling at the gym": "mocked a beginner at the gym",
        "cut in line and pretended not to notice": "cut in line",
    }

    left_side = {"luxury vacation on embezzled money", "cheated, won, celebrated",
                 "insincere flattery, charmed everyone", "free first-class upgrade",
                 "returned a lost wallet, cash intact"}
    fig, ax = plt.subplots(figsize=(9.8, 7))
    for p, x, y in zip(probes, xs, ys):
        ax.scatter(x, y, color=MORAL_COLOR[p["category"]], marker=marker(p["category"]),
                   s=64, zorder=3, edgecolors="white", linewidths=1.2)
        if p["text"] in short:
            name = short[p["text"]]
            if name in left_side:
                ax.annotate(name, (x, y), fontsize=7.5, color=INK,
                            xytext=(-7, 4), textcoords="offset points", ha="right", zorder=4)
            else:
                ax.annotate(name, (x, y), fontsize=7.5, color=INK,
                            xytext=(6, 5), textcoords="offset points", zorder=4)
    ax.axhline(0, color=MUTED, linewidth=0.7)
    ax.axvline(0, color=MUTED, linewidth=0.7)
    pad = 0.02
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corner = dict(fontsize=9, color=MUTED, style="italic", zorder=1)
    ax.text(x0 + pad, y1 - pad, "unpleasant but virtuous", ha="left", va="top", **corner)
    ax.text(x1 - pad, y1 - pad, "pleasant and virtuous", ha="right", va="top", **corner)
    ax.text(x0 + pad, y0 + pad, "unpleasant and vicious", ha="left", va="bottom", **corner)
    ax.text(x1 - pad, y0 + pad, "pleasant but vicious", ha="right", va="bottom", **corner)
    ax.set_xlabel("sentiment axis (pleasant →)")
    ax.set_ylabel("virtue axis, sentiment removed (virtuous →)")
    ax.set_title("Virtue is not pleasantness (openai/text-embedding-3-large)", fontsize=12, color=INK)
    handles = [
        Line2D([], [], color=GOOD, marker="s", linestyle="", markersize=7, label="virtuous"),
        Line2D([], [], color=NEUT, marker="s", linestyle="", markersize=7, label="nonmoral"),
        Line2D([], [], color=BAD, marker="s", linestyle="", markersize=7, label="vicious"),
        Line2D([], [], color=INK, marker="^", linestyle="", markersize=6, label="pleasant surface"),
        Line2D([], [], color=INK, marker="v", linestyle="", markersize=6, label="unpleasant surface"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_scatter.png", dpi=200)
    plt.close(fig)


def fig5_capability(summary):
    """Good-vs-bad AUC on unseen probes, per model: reading virtue takes capability."""
    models = sorted(summary["models"], key=lambda m: m["probe_auc_pure"])
    names = [m["model"] for m in models]
    aucs = [m["probe_auc_pure"] for m in models]
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.grid(axis="y", visible=False)
    bars = ax.barh(names, aucs, color=GOOD, height=0.62)
    for bar, auc in zip(bars, aucs):
        ax.text(auc - 0.012, bar.get_y() + bar.get_height() / 2, f"{auc:.2f}",
                va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.503, -0.55, "chance", fontsize=8, color=MUTED)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("good-vs-bad AUC on 31 unseen actions (sentiment removed)")
    ax.set_title("Reading virtue takes capability", fontsize=12, color=INK)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_capability.png", dpi=200)
    plt.close(fig)


def fig6_pipeline():
    """Four-panel schematic of the method."""
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Panel 1: a matched pair
    ax = axes[0]
    ax.add_patch(FancyBboxPatch((0.06, 0.60), 0.88, 0.24, boxstyle="round,pad=0.02",
                                facecolor="#eaf1f9", edgecolor=GOOD, linewidth=1.2))
    ax.text(0.5, 0.72, '"She spoke up despite\nher shaking hands."', ha="center", va="center", fontsize=8, color=INK)
    ax.add_patch(FancyBboxPatch((0.06, 0.16), 0.88, 0.24, boxstyle="round,pad=0.02",
                                facecolor="#f9edea", edgecolor=BAD, linewidth=1.2))
    ax.text(0.5, 0.28, '"She stayed silent\nbecause she was afraid."', ha="center", va="center", fontsize=8, color=INK)
    ax.set_title("1 · write a matched pair", fontsize=9.5, color=INK)

    # Panel 2: embed both sides
    ax = axes[1]
    ax.plot([0.68], [0.72], "o", color=GOOD, markersize=9)
    ax.plot([0.28], [0.30], "o", color=BAD, markersize=9)
    ax.annotate("$\\varphi(\\mathrm{virtue})$", (0.68, 0.72), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9, color=GOOD)
    ax.annotate("$\\varphi(\\mathrm{vice})$", (0.28, 0.30), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=9, color=BAD)
    for spine_x in np.linspace(0.1, 0.9, 5):
        ax.plot([spine_x], [0.5], ".", color=GRID, markersize=2)
    ax.set_title("2 · embed both sides", fontsize=9.5, color=INK)

    # Panel 3: every pair is an arrow
    ax = axes[2]
    rng = np.random.default_rng(3)
    for _ in range(9):
        x0, y0 = rng.uniform(0.1, 0.55), rng.uniform(0.1, 0.75)
        dx, dy = 0.3 + rng.normal(0, 0.04), 0.12 + rng.normal(0, 0.07)
        ax.add_patch(FancyArrowPatch((x0, y0), (x0 + dx, y0 + dy), arrowstyle="-|>",
                                     mutation_scale=9, color=NEUT, linewidth=1.1, alpha=0.75))
    ax.set_title("3 · every pair is an arrow\n$d_i = \\varphi(x_i^{+}) - \\varphi(x_i^{-})$", fontsize=9.5, color=INK)

    # Panel 4: SVD finds the shared direction
    ax = axes[3]
    ax.add_patch(FancyArrowPatch((0.12, 0.40), (0.88, 0.68), arrowstyle="-|>",
                                 mutation_scale=16, color=INK, linewidth=2.6))
    ax.text(0.5, 0.66, "eigenvirtue $v_1$", ha="center", fontsize=9.5, color=INK, rotation=13)
    for i, h in enumerate([0.26, 0.10, 0.07, 0.05, 0.04]):
        ax.add_patch(Rectangle((0.16 + i * 0.14, 0.06), 0.09, h, color=GOOD))
    ax.text(0.5, 0.0, "singular-value spectrum", ha="center", fontsize=7.5, color=MUTED)
    ax.set_title("4 · SVD finds the shared\ndirection and its spectrum", fontsize=9.5, color=INK)

    fig.tight_layout()
    fig.savefig(FIGS / "fig6_pipeline.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    fig6_pipeline()
    fig2_spectrum(summary)
    fig3_ranking(summary)
    fig4_scatter(summary)
    fig5_capability(summary)
    fig1_arrows()  # last: loads the embedding model
    print("figures written to", FIGS)
