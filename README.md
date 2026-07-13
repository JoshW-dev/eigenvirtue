# eigenvirtue

**The dominant direction of virtue in a language model's embedding space — computed, not imagined.**

*Part linear algebra, part computer science, part philosophy. A small, complete, reproducible study — and a general tool for extracting the axis of any abstract concept.*

---

## Abstract

In Plato's *Meno*, Socrates asks what virtue is and receives a list of virtues instead — justice, charity, courage. The question of whether some single thing underlies the list has stayed open for twenty-four centuries. Text embedding models offer a strange new instrument for asking it: represent many matched examples of virtue and vice as vectors, subtract each pair, and ask whether the resulting difference vectors share one direction. We construct 61 matched virtue/vice sentence pairs across 26 contrasts, most drawn directly from Aristotle's table of virtues and vices in the *Nicomachean Ethics*, and take the top singular vector of the stacked differences — the **eigenvirtue**. Across four embedding models, this single direction (i) carries 8–14× more of the contrast energy than a noise baseline, (ii) classifies held-out virtue/vice pairs at 93–100% accuracy, and (iii) generalizes to *virtues it never saw*: for the strongest model, an axis fit on 25 contrasts correctly orders every pair of the held-out 26th, for all 26 contrasts. After projecting out an independently-fit sentiment axis, the eigenvirtue cleanly separates unseen virtuous from vicious everyday actions (AUC 1.00 for the strongest model), ranking *unpleasant* virtue (comforting the dying, accepting public embarrassment) above pleasant vice (flattery, cheating, embezzled luxury). Separation improves sharply with model capability (AUC 0.53 → 1.00), suggesting moral structure is not merely present in text embeddings but becomes *more* legible as models improve. The method is fully general: the code extracts the dominant direction of any concept expressible as contrast pairs.

---

## 1. Introduction

This project began as a joke inverted. The [eigenslur](https://eigenslur.org/) thought experiment proposed that since offensiveness is a direction in embedding space, there must exist a token that projects furthest along it — "the principal component of offense." The site never ran the computation. A [follow-on video](https://www.tiktok.com/@zach.zeta.s/video/7640238795803151646) by @zach.zeta.s asked the hopeful mirror-image question: could the same construction capture *"the mathematical essence of actions that aim towards human flourishing"* — an eigenvalue of virtue? It ended with an open invitation to attempt the idea rigorously. This repository is that attempt.

The philosophical stakes are older than the math. In the *Meno*, Socrates rejects every definition of virtue offered by example: naming virtues is not defining virtue. Aristotle's answer in the *Nicomachean Ethics* was structural — every virtue is a mean between a vice of deficiency and a vice of excess (courage sits between cowardice and recklessness; generosity between stinginess and wastefulness). That table of oppositions, written ~350 BC, turns out to be exactly the shape of data that modern concept-extraction methods want: **matched contrast pairs**. Aristotle, unknowingly, wrote training data.

Our question is deliberately narrower than "what is virtue?" It is: *when a language model compresses humanity's text into geometry, does the virtue/vice contrast occupy one direction or many?* Three findings:

1. **One direction dominates.** The first principal component of the virtue−vice difference vectors carries far more energy than any other, and the axes fit by two very different construction methods (top singular vector vs. mean difference) agree at cosine 0.93–0.99 (§4.2).
2. **The direction is virtue, not vocabulary.** It transfers to virtues held out of fitting entirely (§4.1), and after removing an independently-fit pleasantness axis, it ranks unseen actions by moral character rather than surface valence (§4.3, §4.4).
3. **Legibility scales with capability.** Small embedding models read mostly sentiment; the strongest model tested separates every virtuous probe from every vicious one (§4.5).

Everything runs on a laptop; the strongest model costs a few cents of API calls and is optional.

![Method overview](docs/figures/fig6_pipeline.png)

*Figure 1 — the whole method in four steps: write matched pairs, embed both sides, treat each pair as an arrow, and let SVD find the arrows' shared direction.*

## 2. Method

### 2.1 The estimator

Let $\varphi : \text{text} \to \mathbb{R}^d$ be an embedding model, with outputs normalized to unit length. Given $n$ contrast pairs $(x_i^{+}, x_i^{-})$ — a virtue sentence and its matched vice sentence — form difference vectors

$$d_i = \varphi(x_i^{+}) - \varphi(x_i^{-}), \qquad D = \begin{bmatrix} d_1^\top \\ \vdots \\ d_n^\top \end{bmatrix} \in \mathbb{R}^{n \times d}.$$

The **eigenvirtue** is the top right-singular vector of $D$:

$$v_1 = \underset{\lVert v \rVert = 1}{\arg\max} \; \sum_{i=1}^{n} (d_i \cdot v)^2,$$

i.e. the unit direction that captures the most energy of the contrasts — equivalently the leading eigenvector of the (uncentered) second-moment matrix $D^\top D$. We deliberately do **not** center $D$: the shared mean of the differences *is* the signal, and the singular spectrum $\sigma_1 \ge \sigma_2 \ge \dots$ then directly answers the unity question. We report the energy share $\eta_k = \sigma_k^2 / \sum_j \sigma_j^2$; if the virtue/vice contrast were $m$ unrelated directions, energy would spread over $m$ components, and pure noise would spread it near the uniform floor $1/n$.

Differencing is what makes this work. A virtue sentence and its matched vice sentence share topic, register, length, and syntax; subtraction cancels all of it, leaving (mostly) the moral contrast. Section 4.6 shows what happens without the contrast — the popular "PC1 of a pile of concept words" recipe fails.

The sign of $v_1$ is arbitrary under SVD; we orient it so $v_1 \cdot \bar d > 0$, pointing toward virtue. A text $t$ is then scored by projection, $\mathrm{score}(t) = \varphi(t) \cdot v_1$.

### 2.2 Sentiment ablation

Virtue-talk is pleasant-talk, and any virtue axis fit from text inherits that correlation (we measure cosine 0.46–0.68 between the two axes, per model). To separate the concepts, we fit a **sentiment axis** $s$ by the identical procedure on 14 pleasant/unpleasant contrast pairs with no moral content ("The soup was delicious" / "The soup was inedible"), then remove it by orthogonal projection:

$$\tilde v = \frac{v_1 - (v_1 \cdot s)\, s}{\lVert v_1 - (v_1 \cdot s)\, s \rVert}.$$

$\tilde v$ is the part of the virtue direction that is *not* pleasantness. The interesting empirical fact is that this residual is large and does the moral work (§4.4).

### 2.3 Scoring protocol for short actions

Probe actions are short phrases ("returned a lost wallet…"), which differ in format from the training sentences. To score an action we embed it inside three neutral sentence templates ("Someone {a}.", "Yesterday she {a}.", "He {a} and went home."), average the embeddings, and renormalize — a cheap way to cancel format variance, in the spirit of the question-template averaging of Schramowski et al. (2022).

### 2.4 Validation design

Three tests, in increasing order of difficulty:

- **Held-out pairs**: fit on a random 75% of pairs, test whether $\mathrm{score}(x^{+}) > \mathrm{score}(x^{-})$ on the remaining 25%.
- **Leave-one-group-out (LOGO)**: fit with an entire virtue/vice contrast removed (all courage-vs-cowardice pairs, say), then test on the removed group. Passing requires the axis to encode something shared across virtues, not memorized per-virtue vocabulary.
- **Unseen probes**: 31 everyday actions never used in fitting, hand-labeled virtuous / vicious / nonmoral, with deliberate *dissociation probes* where moral value and pleasantness disagree (whistleblowing; insincere flattery). We report the AUC of good-vs-bad separation.

## 3. Data

All data is in [`data/`](data/) as human-readable JSON, and all of it is small enough to read in one sitting — by design, since the seed pairs *are* the operational definition of virtue being tested.

- **[`virtue_pairs.json`](data/virtue_pairs.json)** — 61 sentence pairs across 26 virtue/vice contrasts. Fifteen contrasts instantiate Aristotle's table: each virtue appears against its vice of deficiency and/or excess — courage vs. cowardice *and* courage vs. recklessness, generosity vs. stinginess *and* vs. wastefulness, friendliness vs. quarrelsomeness *and* vs. obsequiousness, through to righteous indignation vs. envy. Eleven further contrasts cover broader traditions: honesty, compassion, justice, humility, gratitude, loyalty, diligence, forgiveness, integrity, kindness, prudence. Each pair holds structure constant and flips only the moral content: *"She spoke up in the meeting even though her hands were shaking"* / *"She stayed silent in the meeting because she was afraid to speak."* The excess-side vices matter: recklessness and flattery are *pleasant-adjacent* vices, and their pairs teach the axis that virtue is not enthusiasm.
- **[`sentiment_pairs.json`](data/sentiment_pairs.json)** — 14 pleasant/unpleasant pairs with no moral content, for the control axis.
- **[`probe_actions.json`](data/probe_actions.json)** — 31 unseen everyday actions in seven cells: virtuous, vicious, neutral, pleasant-nonmoral, unpleasant-nonmoral, and the two dissociation cells (virtuous-but-unpleasant, vicious-but-pleasant).

Models: three local sentence-transformers models spanning a capability range (`all-MiniLM-L6-v2`, 22M params; `BAAI/bge-base-en-v1.5`, 109M; `all-mpnet-base-v2`, 110M) plus `openai/text-embedding-3-large` as a frontier reference. The local models run on CPU/Apple-silicon in seconds.

## 4. Results

### 4.1 The axis exists and generalizes to unseen virtues

![Difference arrows share a direction](docs/figures/fig1_arrows.png)

*Figure 2 — left: the 61 pairs (all-mpnet-base-v2), projected onto the top two singular directions of the differences; virtue sentences (blue) and vice sentences (red) form separated clouds. Right: the same pairs as difference arrows moved to a common origin — vice → virtue points one way. The eigenvirtue is that shared direction.*

| model | held-out pair acc. | LOGO mean | LOGO min | probe AUC (raw) | probe AUC (⊥ sentiment) |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 100% | 82% | 0% | 0.39 | 0.53 |
| BAAI/bge-base-en-v1.5 | 93% | 85% | 0% | 0.62 | 0.73 |
| all-mpnet-base-v2 | 100% | 91% | 0% | 0.73 | 0.81 |
| **openai/text-embedding-3-large** | **100%** | **100%** | **100%** | **1.00** | **1.00** |

The leave-one-group-out result is the one we consider the headline. For `text-embedding-3-large`, an axis fit with *courage entirely removed* still orders every courage pair correctly — and the same holds for all 26 contrasts. Whatever courage-vs-cowardice, honesty-vs-deceit, and compassion-vs-cruelty share, it is enough to recognize a virtue the axis never saw. That is a nontrivial, falsifiable version of the claim that the virtues have something in common — Socrates' demand in the *Meno* — and the models increasingly pass it. (The smaller models fail it for one or two specific groups each — MiniLM on integrity-vs-corruption, mpnet on righteous-indignation-vs-envy — which is exactly where their LOGO minimum of 0% comes from.)

### 4.2 The spectrum: one direction, plus echoes

![Energy spectra](docs/figures/fig2_spectrum.png)

*Figure 3 — share of contrast energy by principal component, per model. Dashed line: the uniform-noise floor (1/61 ≈ 1.6%).*

The first component carries 11–20% of total contrast energy — 8–14× the noise floor and 2.3–2.6× the second component — and the top singular vector nearly coincides with the plain mean of the differences (cosine 0.93–0.99), so the "eigen" construction and the simpler averaging construction find the same object. A sharper summary of the spectrum's shape: dominant first direction, a handful of meaningful minor components (plausibly families of virtue — self-regarding vs. other-regarding, restraint vs. action), then a long noise tail. Geometrically, virtue behaves like *one thing plus structured residue* — which, incidentally, is roughly Aristotle's position: the virtues are unified by a common form (the mean relative to us, chosen rightly) while remaining genuinely plural.

### 4.3 Unseen actions land where they should

![Ranking of unseen actions](docs/figures/fig3_ranking.png)

*Figure 4 — 31 unseen everyday actions projected on the sentiment-removed eigenvirtue (strongest model). Bar color is the ground-truth label; annotations mark the dissociation probes.*

Every virtuous probe outranks every vicious probe (AUC 1.00). The top of the list is dominated by *costly* virtue — comforting a dying neighbor through the night, apologizing publicly and eating the embarrassment, keeping a promise that became inconvenient — and the bottom by vice regardless of its pleasantness. Neutral actions cluster near zero. Two imperfections are visible and worth keeping: "reorganized the spreadsheet by date" floats implausibly high (residual conscientiousness/orderliness signal, we suspect), and the whistleblower probe sits below zero — see §6.

### 4.4 Virtue is not pleasantness

![Virtue vs sentiment](docs/figures/fig4_scatter.png)

*Figure 5 — each unseen action placed by its sentiment score (x) and sentiment-removed virtue score (y). Marker shape gives the surface valence; color gives the moral label. The off-diagonal quadrants — unpleasant-but-virtuous, pleasant-but-vicious — are populated exactly as they should be.*

This is the figure that answers the obvious objection ("you've just found the niceness axis"). The raw virtue and sentiment axes are correlated (cosine 0.46–0.68 depending on model) — of course they are; text about virtue is pleasant text. But the residual after projecting sentiment out is what does the moral work: insincere flattery *charms everyone* and still lands at the bottom; the hard truth told to a friend is *unpleasant for everybody* and lands above the flattery by a wide margin. A lottery win — maximally pleasant, morally empty — sits at zero. Pleasantness explains the part of virtue-talk that smiles; it does not explain the part that ranks costly honesty above comfortable deceit.

### 4.5 Reading virtue takes capability

![AUC by model](docs/figures/fig5_capability.png)

*Figure 6 — good-vs-bad AUC on the unseen probes, per model.*

The 22M-parameter model is at chance on unseen actions (0.53): it can separate matched pairs (100% held-out accuracy!) but its single-sentence geometry is dominated by surface valence — it scores "took credit for a colleague's work" as *fine*, because the words are workplace-pleasant, and dislikes the whistleblower sentence because "violation" appears in it. Two 110M models do markedly better (0.73–0.81). The frontier embedding model separates the categories perfectly. Rank agreement between models tells the same story: the small models correlate with the frontier model at Spearman ρ = 0.29–0.69, agreeing on the easy cases and diverging exactly on the dissociation probes. The moral structure of text appears to be a *capability-contingent* signal: present in miniature models only as sentiment, resolved into something recognizably like virtue as representation quality improves.

### 4.6 The naive "eigenslur recipe" fails, and the failure is instructive

The construction the original meme implies — stack embeddings of virtue sentences alone, take PC1 — classifies held-out pairs at 53–87% depending on the model, barely beating chance for two of the four (vs. 93–100% for the contrast method, same data, same models). The first principal component of a pile of virtue-talk is not virtue; it is whatever dominates the *variance* of that pile — topic, register, sentence form. The eigenslur, computed as advertised, would mostly have been an axis of *how slurs are phrased*. Concepts live in contrasts, not in piles.

## 5. Discussion

**What was actually found.** Not virtue itself — a fixed point of how humanity *writes about* virtue, compressed into geometry by models trained on our text, and probed through seed pairs we chose. The axis is an average over the moral testimony of a civilization's corpus, seeded by one ancient Greek's taxonomy. Its philosophical standing is roughly that of a very careful opinion poll of everything ever written.

**Why it is still interesting.** The result did not have to come out this way. The contrasts could have been geometrically incoherent — 26 unrelated directions, LOGO at chance, spectrum flat. Instead, the unity that Socrates demanded and Meno couldn't articulate shows up as a measurable, transferable, dominant direction, and it sharpens with model capability. At minimum this means *the concept of virtue is legible to language models* — they represent it coherently enough that a 61-example probe recovers it, distinguishes it from pleasantness, and applies it to novel cases.

**The speculative coda.** Humanoid robots and agentic systems built on language models are about to exist in very large numbers, and their behavior will be shaped by what their underlying models represent. Nothing here shows that a direction in embedding space *motivates* anything — a representation is not a value, and we would resist the strong reading of our own result. But alignment requires, as a precondition, that the target be expressible in the system's internal language. What this small study suggests is that "the direction of human flourishing" is not poetry: it is an estimable vector, it transfers across unseen cases, and the better the model, the more sharply it resolves. If the machines are ever to be pointed toward the good, it is at least now demonstrable that they can see which way it lies.

## 6. Limitations

- **The seed pairs are the definition.** LOGO mitigates but cannot eliminate this: all 26 contrasts come from one broadly Western, English-language moral vocabulary. A Buddhist, Confucian, or adversarially-chosen seed set might find a different axis, or several. Cross-cultural seed sets are the most obvious extension.
- **The whistleblower fails everywhere.** "Reported her own team's safety violation, knowing it would cost them the contract" scores mildly *negative* on every model, including the strongest. Loyalty-betraying virtue — praiseworthy acts whose local texture is harm — appears to sit off-axis. Humans litigate whistleblowing too, but the honest reading is that one linear direction cannot represent virtues that require weighing a violated norm against a higher one. (One neutral probe also misbehaves: alphabetizing a spreadsheet is not, in fact, a virtue.)
- **Single-sentence scoring is noisy.** Projections of individual sentences carry topic residue; the method is far more reliable on contrasts than on absolutes. Our template-averaging mitigates but does not remove this.
- **English only, four models, one run.** Seeds are fixed for reproducibility; we did not bootstrap confidence intervals. The pattern (unity, transfer, capability scaling) is consistent across all four models, but the numbers should be read as a demonstration, not a benchmark.
- **A direction is not a disposition.** Nothing here bears on whether models *act* well — only on whether they *represent* the distinction.

## 7. Related work

- **Schramowski, Turan, Andersen, Rothkopf & Kersting.** [*Large pre-trained language models contain human-like biases of what is right and wrong to do*](https://www.nature.com/articles/s42256-022-00458-8). Nature Machine Intelligence, 2022. The closest prior work: a "moral direction" computed by PCA over question templates in BERT-era models, validated against human normativity ratings. We differ in the Aristotelian contrast-pair construction, the unity-of-virtue framing (LOGO transfer and spectrum), the explicit sentiment ablation, and the capability-scaling comparison.
- **Bolukbasi, Chang, Zou, Saligrama & Kalai.** [*Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings*](https://arxiv.org/abs/1607.06520). NeurIPS 2016. The origin of concept-direction-by-contrast in word embeddings (the gender direction), including the projection-removal operation we borrow for sentiment.
- **Zou et al.** [*Representation Engineering: A Top-Down Approach to AI Transparency*](https://arxiv.org/abs/2310.01405). 2023. Extracts concept directions (honesty, harmlessness, emotion) from LLM *activations* using contrast prompts, and steers behavior with them. Our study is the embedding-space, laptop-scale relative.
- **Plato**, *Meno*; **Aristotle**, *Nicomachean Ethics* II–IV. The problem statement, and the dataset schema, respectively.
- **Provenance of the question:** [eigenslur.org](https://eigenslur.org/) (the thought experiment, unexecuted) and [@zach.zeta.s's video](https://www.tiktok.com/@zach.zeta.s/video/7640238795803151646) proposing the flourishing-direction inversion that this repo implements.

## 8. Reproducing

```bash
git clone https://github.com/JoshW-dev/eigenvirtue && cd eigenvirtue
python3.11 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python examples/run_eigenvirtue.py            # 3 local models; laptop-friendly
OPENAI_API_KEY=... .venv/bin/python examples/run_eigenvirtue.py openai:text-embedding-3-large
.venv/bin/python examples/make_figures.py               # regenerate all figures
```

`run_eigenvirtue.py` prints every number in §4 and writes plots plus `results/summary.json`. Randomness is seeded; local-model results reproduce exactly.

### Using the library on your own concept

The estimator is concept-agnostic — the virtue framing lives entirely in the data files. Write contrast pairs for any abstract concept (curiosity, freedom, professionalism, camp) and:

```python
from eigenvirtue import ConceptAxis, LocalEmbedder, load_pairs, orthogonalize

embedder = LocalEmbedder("all-mpnet-base-v2")
axis = ConceptAxis.fit(load_pairs("data/your_concept_pairs.json"), embedder)

axis.energy[:5]        # is your concept one direction or many?
axis.score(["some new text"], embedder)   # project anything onto it
```

## License

MIT.
