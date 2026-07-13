"""Compare probe-scoring variants: raw cosine, template-normalized,
sentiment-orthogonalized, and both. Prints top/bottom probes per variant."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eigenvirtue import ConceptAxis, get_embedder, load_pairs

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ["Someone {a}.", "Yesterday she {a}.", "He {a} and went home."]


def orthogonalize(direction, other):
    residual = direction - (direction @ other) * other
    return residual / np.linalg.norm(residual)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "all-mpnet-base-v2"
    embedder = get_embedder(model)
    virtue = ConceptAxis.fit(load_pairs(ROOT / "data" / "virtue_pairs.json"), embedder)
    sentiment = ConceptAxis.fit(load_pairs(ROOT / "data" / "sentiment_pairs.json"), embedder)
    ortho = orthogonalize(virtue.direction, sentiment.direction)

    probes = json.loads((ROOT / "data" / "probe_actions.json").read_text())["probes"]
    texts = [p["text"] for p in probes]

    raw_emb = embedder.encode(texts)
    templated = np.mean(
        [embedder.encode([t.format(a=x) for x in texts]) for t in TEMPLATES], axis=0
    )
    templated /= np.linalg.norm(templated, axis=1, keepdims=True)

    variants = {
        "raw": raw_emb @ virtue.direction,
        "templated": templated @ virtue.direction,
        "ortho": raw_emb @ ortho,
        "templated+ortho": templated @ ortho,
    }

    for name, scores in variants.items():
        order = np.argsort(scores)[::-1]
        print(f"\n--- {name} ---")
        for i in list(order[:6]) + ["..."] + list(order[-6:]):
            if i == "...":
                print("   ...")
                continue
            print(f"  {scores[i]:+.3f}  [{probes[i]['category']:>20}]  {probes[i]['text']}")
        # sanity metric: mean score by coarse moral label
        good = [s for s, p in zip(scores, probes) if p["category"].startswith("virtuous")]
        bad = [s for s, p in zip(scores, probes) if p["category"].startswith("vicious")]
        rest = [s for s, p in zip(scores, probes) if not p["category"].startswith(("virtuous", "vicious"))]
        # rank-based separation: fraction of good/bad pairs correctly ordered (AUC)
        auc = np.mean([[g > b for b in bad] for g in good])
        print(f"  mean good {np.mean(good):+.3f} | nonmoral {np.mean(rest):+.3f} | bad {np.mean(bad):+.3f} | good-vs-bad AUC {auc:.2f}")


if __name__ == "__main__":
    main()
