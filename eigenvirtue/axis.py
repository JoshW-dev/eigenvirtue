"""Find the dominant direction ("eigenvector") of an abstract concept in an
embedding space, from contrast pairs.

Method: embed both sides of each (concept, anti-concept) pair, take the
difference vectors, and run an uncentered SVD on the stack of differences.
The top right-singular vector is the concept axis — for virtue, the
eigenvirtue. Differencing cancels everything the two sides share (register,
topic, formality), isolating the contrast itself. The singular-value spectrum
says whether the contrast is one direction or many.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ContrastPair:
    positive: str
    negative: str
    group: str  # e.g. "courage vs cowardice"


def load_pairs(path: str | Path) -> list[ContrastPair]:
    data = json.loads(Path(path).read_text())
    pairs = []
    for group in data["groups"]:
        label = f"{group['virtue']} vs {group['vice']}"
        for positive, negative in group["pairs"]:
            pairs.append(ContrastPair(positive, negative, label))
    return pairs


@dataclass
class ConceptAxis:
    direction: np.ndarray            # unit vector: the concept axis (top singular direction)
    mean_diff: np.ndarray            # unit vector: mean of the difference vectors
    singular_values: np.ndarray
    energy: np.ndarray               # fraction of total contrast energy per component
    cos_pc1_mean: float              # agreement between the two candidate axes
    group_alignment: dict[str, float] = field(default_factory=dict)  # mean cos(diff, axis) per group

    @classmethod
    def fit(cls, pairs: list[ContrastPair], embedder) -> "ConceptAxis":
        pos = embedder.encode([p.positive for p in pairs])
        neg = embedder.encode([p.negative for p in pairs])
        diffs = pos - neg

        # Uncentered SVD: components are ranked by how much of the total
        # contrast energy they carry, including the shared mean direction.
        _, s, vt = np.linalg.svd(diffs, full_matrices=False)
        direction = vt[0]

        mean_diff = diffs.mean(axis=0)
        mean_diff = mean_diff / np.linalg.norm(mean_diff)

        # SVD sign is arbitrary; orient the axis toward the concept side.
        if float(direction @ mean_diff) < 0:
            direction = -direction

        energy = s**2 / np.sum(s**2)

        unit_diffs = diffs / np.linalg.norm(diffs, axis=1, keepdims=True)
        alignment: dict[str, list[float]] = {}
        for pair, ud in zip(pairs, unit_diffs):
            alignment.setdefault(pair.group, []).append(float(ud @ direction))
        group_alignment = {g: float(np.mean(v)) for g, v in alignment.items()}

        return cls(
            direction=direction,
            mean_diff=mean_diff,
            singular_values=s,
            energy=energy,
            cos_pc1_mean=float(direction @ mean_diff),
            group_alignment=group_alignment,
        )

    def score(self, texts: list[str], embedder) -> np.ndarray:
        """Cosine of each text with the axis: positive = toward the concept."""
        return embedder.encode(texts) @ self.direction


def pairwise_accuracy(axis: ConceptAxis, pairs: list[ContrastPair], embedder) -> float:
    """Fraction of held-out pairs where the concept side outscores the anti-concept side."""
    pos = axis.score([p.positive for p in pairs], embedder)
    neg = axis.score([p.negative for p in pairs], embedder)
    return float(np.mean(pos > neg))


def leave_one_group_out(pairs: list[ContrastPair], embedder) -> dict[str, float]:
    """Fit the axis without one virtue group, test on that unseen group.
    High accuracy means the axis generalizes to virtues it never saw —
    evidence the pairs share one underlying direction."""
    groups = sorted({p.group for p in pairs})
    results = {}
    for held_out in groups:
        train = [p for p in pairs if p.group != held_out]
        test = [p for p in pairs if p.group == held_out]
        axis = ConceptAxis.fit(train, embedder)
        results[held_out] = pairwise_accuracy(axis, test, embedder)
    return results


def orthogonalize(direction: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """Project a nuisance direction (e.g. sentiment) out of a concept axis.
    What remains of the virtue axis after removing pleasantness is the part
    that scores 'told a hard truth' above 'flattered the whole party'."""
    residual = direction - (direction @ nuisance) * nuisance
    return residual / np.linalg.norm(residual)


TEMPLATES = ["Someone {a}.", "Yesterday she {a}.", "He {a} and went home."]


def embed_actions(actions: list[str], embedder, templates: list[str] = TEMPLATES) -> np.ndarray:
    """Embed short action phrases inside several sentence templates and average,
    so scores reflect the action rather than sentence format."""
    stacked = np.mean(
        [embedder.encode([t.format(a=a) for a in actions]) for t in templates], axis=0
    )
    return stacked / np.linalg.norm(stacked, axis=1, keepdims=True)


def naive_axis(pairs: list[ContrastPair], embedder) -> np.ndarray:
    """The 'eigenslur-style' baseline: PC1 of the concept-side embeddings
    alone, no contrast. Kept for comparison — it tends to capture register
    and topic rather than the concept."""
    pos = embedder.encode([p.positive for p in pairs])
    centered = pos - pos.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]
