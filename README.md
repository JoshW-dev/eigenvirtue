# eigenvirtue

**The dominant direction of virtue in a language model's embedding space — computed, not imagined.**

In Plato's *Meno*, Socrates asks for a definition of virtue and gets a list of virtues instead: justice, charity, courage. "You are giving me examples of virtues. What *is* virtue?" Twenty-four centuries later, embedding models give us a strange new way to ask his question: put many examples of virtue and vice into a high-dimensional space, subtract, and see whether one direction underlies them all.

This repo finds that direction. It fits an axis from virtue/vice contrast pairs — most of them straight out of Aristotle's table of virtues in the *Nicomachean Ethics* — and then tests whether the axis is real: does it generalize to virtues it never saw, does it rank unseen everyday actions sensibly, and is it distinct from mere pleasantness?

The method is general. Point it at any concept with contrast pairs and it returns that concept's axis. Virtue is the flagship example.

## Method

1. **Contrast pairs.** Each pair holds structure constant and flips only the moral content: *"She spoke up in the meeting even though her hands were shaking"* vs *"She stayed silent in the meeting because she was afraid to speak."* 69 pairs across 25 virtue/vice contrasts — Aristotle's virtues each paired against their vices of excess and deficiency (courage vs both cowardice *and* recklessness), plus broader virtues (honesty, compassion, justice, ...).
2. **Difference vectors.** Embed both sides, subtract. Differencing cancels everything the two sides share — register, topic, formality — isolating the moral contrast.
3. **SVD.** The top singular vector of the stacked differences is the **eigenvirtue**. The singular-value spectrum answers Socrates: if one component dominates, virtue is (geometrically) one thing.
4. **Sentiment removal.** A control axis fit the same way on non-moral pleasant/unpleasant pairs is projected out, so scores reflect virtue rather than pleasantness.

## Results

| model | held-out pair accuracy | unseen-virtue accuracy (LOGO) | probe AUC (good vs bad) |
|---|---|---|---|
| all-MiniLM-L6-v2 | 100% | 82% | 0.53 |
| all-mpnet-base-v2 | 100% | 91% | 0.81 |
| BAAI/bge-base-en-v1.5 | 93% | 85% | 0.73 |
| openai/text-embedding-3-large | 100% | **100%** | **1.00** |

- **The axis generalizes.** For the strongest model, an axis fit on 24 virtue/vice contrasts classifies every pair of the 25th, held-out virtue — for all 25 virtues. Whatever courage-vs-cowardice, honesty-vs-deceit, and compassion-vs-cruelty have in common, it is enough to recognize a virtue the axis never saw.
- **Virtue is not pleasantness.** With sentiment projected out, the top-ranked unseen actions are *unpleasant* virtuous ones (comforting a dying neighbor all night, accepting public embarrassment for a mistake), and the bottom is vice regardless of how pleasant it feels (insincere flattery, cheating and celebrating, vacationing on embezzled money).
- **Reading virtue takes capability.** Small local models mostly read surface valence — they rank "took credit for a colleague's work" as fine because the words are pleasant. The frontier embedding model separates every virtuous probe from every vicious one (AUC 1.00). Moral structure is legible to these systems, and more legible the more capable they get.
- **Honest failure:** every model, including the strongest, scores the whistleblower probe ("reported her own team's safety violation, knowing it would cost them the contract") slightly negative. Make of that what you will — humans argue about it too.
- **The naive "eigenslur-style" baseline fails.** PC1 of the virtue sentences alone (no contrast) classifies held-out pairs at 53–87% depending on the model, versus ~100% for the contrast method. The first principal component of a pile of virtue-talk captures register and topic, not virtue.

Plots and raw numbers are in [`results/`](results/) — the eigenvalue spectrum, the ranked actions, and the virtue-vs-sentiment scatter, per model.

## Run it

```bash
python3.11 -m venv .venv && .venv/bin/pip install sentence-transformers numpy scipy matplotlib
.venv/bin/python examples/run_eigenvirtue.py                      # three local models, CPU/MPS friendly
OPENAI_API_KEY=... .venv/bin/python examples/run_eigenvirtue.py openai:text-embedding-3-large
```

Or use the library on your own concept:

```python
from eigenvirtue import ConceptAxis, LocalEmbedder, load_pairs

embedder = LocalEmbedder("all-mpnet-base-v2")
axis = ConceptAxis.fit(load_pairs("data/virtue_pairs.json"), embedder)
print(axis.energy[:3])                       # is the concept one direction?
print(axis.score(["returned a lost wallet with all the cash still inside"], embedder))
```

## Caveats

This measures how models trained on human text organize the *concept* of virtue — a corpus-average of human virtue-talk, seeded by which contrast pairs we chose. It is not a discovery of what virtue objectively is, and a direction in embedding space is not a value a system acts on. It does show that virtue is geometrically legible to language models, which is a more hopeful fact than it first sounds.

## Inspiration & prior work

- The eigenslur thought experiment — [eigenslur.org](https://eigenslur.org/) — and [this video](https://www.tiktok.com/@zach.zeta.s/video/7640238795803151646) by @zach.zeta.s asking whether the same trick could point toward human flourishing instead: *"maybe virtue is the eigenvalue of the direction of human flourishing."* This repo is an attempt at the "more rigorous undertaking" the video asks for.
- Schramowski et al., [*Large pre-trained language models contain human-like biases of what is right and wrong to do*](https://www.nature.com/articles/s42256-022-00458-8), Nature Machine Intelligence (2022) — computed a "moral direction" in BERT via PCA.
- Bolukbasi et al. (2016) and the steering-vector / representation-engineering literature (Zou et al. 2023), which extract concept directions from model internals via contrasts.

MIT licensed. Built for an essay on whether the machines we are about to hand the world to can at least see the direction we mean when we say "good."
