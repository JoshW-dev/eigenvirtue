"""The responsible eigenslur: fit a denigration axis from slur-free
respect/contempt pairs and measure how it relates to the virtue and
sentiment axes. Companion analysis for docs/eigenslur.md.

Usage: python examples/run_eigenslur.py [model ...]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eigenvirtue import ConceptAxis, embed_actions, get_embedder, load_pairs, pairwise_accuracy

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "docs" / "figures"

GOOD = "#2c6fbb"
INK = "#2b2f36"
MUTED = "#6b7078"


def split_pairs(pairs, test_fraction=0.25, seed=7):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * test_fraction)
    return shuffled[cut:], shuffled[:cut]


def main():
    models = sys.argv[1:] or ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    probes = json.loads((ROOT / "data" / "probe_actions.json").read_text())["probes"]
    summary = []

    for spec in models:
        embedder = get_embedder(spec)
        print(f"\n=== {embedder.name} ===")

        den_pairs = load_pairs(ROOT / "data" / "denigration_pairs.json")
        train, test = split_pairs(den_pairs)
        heldout = pairwise_accuracy(ConceptAxis.fit(train, embedder), test, embedder)

        den_axis = ConceptAxis.fit(den_pairs, embedder)
        virtue_axis = ConceptAxis.fit(load_pairs(ROOT / "data" / "virtue_pairs.json"), embedder)
        sent_axis = ConceptAxis.fit(load_pairs(ROOT / "data" / "sentiment_pairs.json"), embedder)

        cos_virtue = float(den_axis.direction @ virtue_axis.direction)
        cos_sent = float(den_axis.direction @ sent_axis.direction)

        probe_emb = embed_actions([p["text"] for p in probes], embedder)
        scores = probe_emb @ den_axis.direction
        order = np.argsort(scores)[::-1]

        print(f"PC1 energy share: {den_axis.energy[0]:.1%}  (PC2 {den_axis.energy[1]:.1%})")
        print(f"held-out pairwise accuracy: {heldout:.1%}")
        print(f"cos(denigration axis, virtue axis): {cos_virtue:.3f}")
        print(f"cos(denigration axis, sentiment axis): {cos_sent:.3f}")
        print("most denigration-flavored unseen actions:")
        for i in order[:3]:
            print(f"  {scores[i]:+.3f}  {probes[i]['text']}")
        print("least:")
        for i in order[-3:]:
            print(f"  {scores[i]:+.3f}  {probes[i]['text']}")

        summary.append({
            "model": embedder.name,
            "energy": den_axis.energy[:10].tolist(),
            "heldout_accuracy": heldout,
            "cos_vs_virtue": cos_virtue,
            "cos_vs_sentiment": cos_sent,
            "probe_scores": {probes[i]["text"]: float(scores[i]) for i in order},
        })

        if "mpnet" in embedder.name:
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.bar(range(1, 11), [e * 100 for e in den_axis.energy[:10]], color=GOOD, width=0.72)
            ax.axhline(100 / len(den_pairs), color=MUTED, linewidth=1, linestyle="--")
            ax.annotate(f"uniform-noise floor (1/{len(den_pairs)})", (10.3, 100 / len(den_pairs) + 0.8),
                        fontsize=8, color=MUTED, ha="right")
            ax.set_xticks(range(1, 11))
            ax.set_xlabel("principal component of the contempt contrast")
            ax.set_ylabel("share of energy (%)")
            ax.set_title(f"The denigration spectrum ({embedder.name})", fontsize=11.5, color=INK)
            fig.tight_layout()
            fig.savefig(FIGS / "eigenslur_spectrum.png", dpi=200)
            plt.close(fig)

    (RESULTS / "eigenslur_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved eigenslur_summary.json and eigenslur_spectrum.png")


if __name__ == "__main__":
    main()
