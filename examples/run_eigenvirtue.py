"""Fit the eigenvirtue and run the full battery:

  1. Energy spectrum of the virtue contrast — is virtue one direction or many?
  2. Held-out validation: random-split pairwise accuracy and leave-one-group-out
     (does an axis learned from 24 virtue/vice contrasts classify the 25th, unseen one?)
  3. Naive eigenslur-style baseline (PC1 of virtue sentences alone) for contrast.
  4. Probe ranking: unseen everyday actions scored along the axis, with the
     sentiment direction projected out (virtue minus pleasantness).
  5. Virtue vs sentiment scatter: the axis must not reduce to pleasantness.
  6. Cross-model agreement on probe rankings.

Usage: python examples/run_eigenvirtue.py [model ...]
Default models: all-MiniLM-L6-v2 all-mpnet-base-v2 BAAI/bge-base-en-v1.5
Add openai:text-embedding-3-large if OPENAI_API_KEY is set.
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eigenvirtue import (
    ConceptAxis,
    embed_actions,
    get_embedder,
    leave_one_group_out,
    load_pairs,
    naive_axis,
    orthogonalize,
    pairwise_accuracy,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DEFAULT_MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"]

CATEGORY_COLORS = {
    "virtuous": "#2e7d32",
    "virtuous_unpleasant": "#66bb6a",
    "neutral": "#9e9e9e",
    "pleasant_nonmoral": "#42a5f5",
    "unpleasant_nonmoral": "#7e57c2",
    "vicious_pleasant": "#ef6c00",
    "vicious": "#c62828",
}


def split_pairs(pairs, test_fraction=0.25, seed=7):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * test_fraction)
    return shuffled[cut:], shuffled[:cut]


def naive_pairwise_accuracy(pairs_train, pairs_test, embedder):
    axis = naive_axis(pairs_train, embedder)
    pos = embedder.encode([p.positive for p in pairs_test]) @ axis
    neg = embedder.encode([p.negative for p in pairs_test]) @ axis
    accuracy = float(np.mean(pos > neg))
    return max(accuracy, 1 - accuracy)  # PC1 sign is arbitrary; give it the benefit


def good_vs_bad_auc(scores, probes) -> float:
    good = [s for s, p in zip(scores, probes) if p["category"].startswith("virtuous")]
    bad = [s for s, p in zip(scores, probes) if p["category"].startswith("vicious")]
    return float(np.mean([[g > b for b in bad] for g in good]))


def run_model(spec: str, probes: list[dict]) -> dict:
    embedder = get_embedder(spec)
    print(f"\n=== {embedder.name} ===")

    virtue_pairs = load_pairs(ROOT / "data" / "virtue_pairs.json")
    sentiment_pairs = load_pairs(ROOT / "data" / "sentiment_pairs.json")

    train, test = split_pairs(virtue_pairs)
    axis = ConceptAxis.fit(train, embedder)
    heldout_acc = pairwise_accuracy(axis, test, embedder)
    naive_acc = naive_pairwise_accuracy(train, test, embedder)
    logo = leave_one_group_out(virtue_pairs, embedder)

    full_axis = ConceptAxis.fit(virtue_pairs, embedder)  # final axis uses all pairs
    sentiment_axis = ConceptAxis.fit(sentiment_pairs, embedder)
    pure_axis = orthogonalize(full_axis.direction, sentiment_axis.direction)

    probe_emb = embed_actions([p["text"] for p in probes], embedder)
    virtue_scores = probe_emb @ full_axis.direction
    pure_scores = probe_emb @ pure_axis
    sentiment_scores = probe_emb @ sentiment_axis.direction

    auc_raw = good_vs_bad_auc(virtue_scores, probes)
    auc_pure = good_vs_bad_auc(pure_scores, probes)

    print(f"PC1 energy share: {full_axis.energy[0]:.1%}  (PC2 {full_axis.energy[1]:.1%}, PC3 {full_axis.energy[2]:.1%})")
    print(f"cos(PC1, mean-diff): {full_axis.cos_pc1_mean:.3f}")
    print(f"held-out pairwise accuracy: {heldout_acc:.1%}  |  naive PC1 baseline: {naive_acc:.1%}")
    print(f"leave-one-group-out mean: {np.mean(list(logo.values())):.1%}  min: {min(logo.values()):.1%} ({min(logo, key=logo.get)})")
    print(f"cos(virtue axis, sentiment axis): {float(full_axis.direction @ sentiment_axis.direction):.3f}")
    print(f"probe good-vs-bad AUC: raw {auc_raw:.2f}  |  sentiment removed {auc_pure:.2f}")

    return {
        "model": embedder.name,
        "energy": full_axis.energy[:10].tolist(),
        "cos_pc1_mean": full_axis.cos_pc1_mean,
        "heldout_accuracy": heldout_acc,
        "naive_baseline_accuracy": naive_acc,
        "leave_one_group_out": logo,
        "group_alignment": full_axis.group_alignment,
        "axis_vs_sentiment_cos": float(full_axis.direction @ sentiment_axis.direction),
        "probe_auc_raw": auc_raw,
        "probe_auc_pure": auc_pure,
        "probe_virtue_scores": virtue_scores.tolist(),
        "probe_pure_scores": pure_scores.tolist(),
        "probe_sentiment_scores": sentiment_scores.tolist(),
    }


def plot_spectrum(result: dict, tag: str):
    energy = result["energy"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(energy) + 1), [e * 100 for e in energy], color="#2e7d32")
    ax.set_xlabel("Principal component of the virtue contrast")
    ax.set_ylabel("Share of contrast energy (%)")
    ax.set_title(f"Is virtue one direction? — {result['model']}")
    ax.set_xticks(range(1, len(energy) + 1))
    fig.tight_layout()
    fig.savefig(RESULTS / f"spectrum_{tag}.png", dpi=150)
    plt.close(fig)


def plot_ranking(result: dict, probes: list[dict], tag: str):
    scores = result["probe_pure_scores"]
    order = np.argsort(scores)
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(probes) + 1.5))
    for rank, idx in enumerate(order):
        probe = probes[idx]
        ax.barh(rank, scores[idx], color=CATEGORY_COLORS[probe["category"]])
        if scores[idx] < 0:
            ax.text(0.004, rank, probe["text"], va="center", ha="left", fontsize=8)
        else:
            ax.text(-0.004, rank, probe["text"], va="center", ha="right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Projection onto the eigenvirtue, sentiment removed (cosine)")
    ax.set_title(f"Everyday actions on the virtue axis — {result['model']}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CATEGORY_COLORS.values()]
    ax.legend(handles, CATEGORY_COLORS.keys(), loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / f"ranking_{tag}.png", dpi=150)
    plt.close(fig)


def plot_virtue_vs_sentiment(result: dict, probes: list[dict], tag: str):
    fig, ax = plt.subplots(figsize=(8, 7))
    xs = result["probe_sentiment_scores"]
    ys = result["probe_pure_scores"]
    for probe, x, y in zip(probes, xs, ys):
        ax.scatter(x, y, color=CATEGORY_COLORS[probe["category"]], s=60, zorder=3)
        ax.annotate(probe["text"], (x, y), fontsize=6.5, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("Sentiment axis (pleasant →)")
    ax.set_ylabel("Virtue axis, sentiment removed (virtuous →)")
    ax.set_title(f"Virtue is not pleasantness — {result['model']}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"virtue_vs_sentiment_{tag}.png", dpi=150)
    plt.close(fig)


def main():
    models = sys.argv[1:] or DEFAULT_MODELS
    probes = json.loads((ROOT / "data" / "probe_actions.json").read_text())["probes"]

    results = []
    for spec in models:
        result = run_model(spec, probes)
        tag = result["model"].replace("/", "_").replace(":", "_")
        plot_spectrum(result, tag)
        plot_ranking(result, probes, tag)
        plot_virtue_vs_sentiment(result, probes, tag)
        results.append(result)

    if len(results) > 1:
        print("\n=== cross-model agreement on probe ranking (Spearman) ===")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                rho = spearmanr(results[i]["probe_pure_scores"], results[j]["probe_pure_scores"]).statistic
                print(f"{results[i]['model']}  vs  {results[j]['model']}: {rho:.3f}")

    summary = {"probes": probes, "models": results}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved plots and summary.json to {RESULTS}")


if __name__ == "__main__":
    main()
