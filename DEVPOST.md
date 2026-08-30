# Shopping Copilot — a conversational product-search agent that asks before it guesses

**Track 4 · Shopping Copilot: AI Conversational Search and Recommendations**
Solo entry — Aman Khanna

- **Repository:** https://github.com/Khanna-Aman/techjam-2026-shopping-copilot
- **Demo video (3 min):** <YOUTUBE_URL>

---

## The one-line version

A stateful conversational shopping agent that finds a hidden target product inside a
frozen 50,000-item Amazon catalog in **1.93 turns instead of 9.81**, scoring **8.49× the
official baseline** — on the pure Python standard library, with **zero tokens, no API key,
no network access, and no GPU**.

```
                     official baseline        this agent
  Hit Rate@10              0.125       ->       0.995
  MRR                      0.068034    ->       0.757502
  MTTC (turns)             9.81        ->       1.930
  TechnicalScore           0.10671     ->       0.906151      (8.49x)
```

`TechnicalScore = 0.50·HitRate@10 + 0.30·MRR + 0.20·clip((11−MTTC)/10, 0, 1)`

Measured with the **unmodified** official evaluator over all 200 public sessions.
One command reproduces it: `python -m evaluator.local_evaluator`.

Two caveats I would rather state than have you find. The supplied baseline is explicitly
named `weak_bm25`, so the 8.49× multiple flatters me — the ablation table below is the
honest version of that claim. And 200 sessions put a 95% confidence interval of
**[0.8891, 0.9216]** around that score, so the defensible number is **0.91 ± 0.02**, not
0.906151 (`python -m tools.bootstrap`).

---

## How the solution addresses the problem statement

### The insight the whole system is built on

The supplied BM25 baseline scores 0.107 and converges at turn **9.81 out of 10**. My first
assumption was that this was a retrieval-quality problem, so I started building a better
ranker. It wasn't. Look at the baseline's response shape:

```python
return {"message": "...", "ask_attribute": None, ...}   # starter/baseline_agent.py
```

`ask_attribute` is always `None`. In this protocol the customer only discloses information
**in response to a question** — so with no question asked, every turn after the first
returns the same non-answer: *"Those options are not quite right yet. Ask me about one
specific attribute."* The baseline gains zero new information on every subsequent turn and
re-runs one OR-query ten times. It is negotiating with itself.

And the protocol lets a single response carry a clarification question **and** a ranked
list. There is no ask-versus-recommend trade-off to balance — the correct policy is always
to do both. Half the available channel was simply going unused.

That one change is worth **+0.418** of the **+0.799** total improvement. Everything else in
the system is the other 0.38.

### Mapping to the four required pillars

| Pillar | What I built | Evidence |
|---|---|---|
| **I. Intent routing & hybrid pipeline** | Two-tier message parsing classifies Buying / Browsing / Intent Override / Boundary, then a **category lock** cuts 50,000 products to a ~180-item pool, ranked by weighted BM25 + typed constraint satisfaction + popularity prior. I deliberately did **not** fork into two separate retrieval stacks. | Intent detection drives *dialogue policy*, not two rankers — the unified constraint-driven ranker already reaches **Hit@10 = 1.000 on both** Buying and Browsing, so a second stack had nothing left to win. **Vector similarity is implemented** (`copilot/dense.py`, offline fp16 + `mmap`, stdlib only) and measured at three encoder tiers up to `bge-base` (768-dim, GPU-built): no gain at any weight on any encoder, so it ships switched off behind a flag rather than omitted. |
| **II. Multi-turn scenario evolution** | A slot state machine accumulates typed constraints (set-membership, phrase-containment, numeric), handles retraction on Intent Override, marks attributes exhausted so a spent question is never re-asked, and proactively clarifies on every turn. | Ablation: removing state tracking costs **−0.340**; removing clarification costs **−0.418**. |
| **III. Dynamic context programming** | Constraint accumulation with per-constraint confidence and decay; the question policy re-plans every turn from the live pool, escalating from an open-ended prompt to a specific attribute once the open channel is exhausted; the anonymised user profile is distilled in at cold start. | The policy **derives** that the open question is optimal rather than hardcoding it (finding #2 below). Profile prior: **+0.015**, cold start only. |
| **IV. Evaluation matrix** | Scored on the organiser's own evaluator, unmodified, plus an ablation harness (11 configurations) and a paraphrase-robustness harness (5 perturbations) that imports the evaluator's own simulator functions. | Hit@10 **0.995**, MRR **0.7575**, MTTC **1.930**. |

### Architecture

```
 customer message
        |
        +--> [ dialogue.py ]  exact templates, then content-driven recovery
        |          |
        |          +--> scenario (buying / browsing / intent override)
        |          +--> category  ------------------+
        |          +--> constraints ---+            |
        |                              v            v
        +--> observed tokens ---> [ slots.py ]  [ catalog.py ]
             (parse-failure net)      |          category buckets
                                      |          weighted BM25 index
                                      |          typed attribute tables
                                      v
                            [ retrieval.py ]  pool -> score -> rank
                                      |
                            [ question.py ]  what is worth asking next
                                      v
                       { message, ask_attribute, recommendations, usage }
```

About 2,500 lines of agent code across ten modules, plus ~1,900 lines of tests
(222 tests, 53 of them adversarial) and seven measurement harnesses.

### Three findings that overturned my first instinct

**1. Don't erase a retracted preference — it is still true.** The obvious reading of Intent
Override is "the customer withdrew that, so drop it." That is wrong, and measurably so. The
simulator draws *both* the withdrawn preference and its replacement from the **same hidden
target product**. The customer changes their mind; the target never does. Full erasure
scored **worst** of every setting tested. The agent retains the withdrawn value at half
confidence.

**2. A question that splits the pool perfectly is worthless if nobody answers it.** My
first question-selection model scored purely on information gain. It kept choosing
**budget** — which partitions the pool beautifully and goes unanswered **99.5%** of the
time. Measuring P(the customer can answer) across all 50,000 products, from the catalog
alone with no session labels:

| attribute | feature | material | colour | style | size | use_case | budget |
|---|---|---|---|---|---|---|---|
| P(yield) | 0.958 | 0.573 | 0.427 | 0.162 | 0.076 | 0.016 | **0.005** |

The fix is an expected-*value* model:
`value(A) = P(customer can answer A) × E[constraints returned] × how well A splits the pool`.
Worth **+0.015** over the entropy-only policy (0.9062 against 0.8911). It is also worth
saying that the policy ties always-asking-open exactly on this set: the escalation branch
earns nothing here, and is kept as insurance for a private set that may exhaust the open
channel more often.

**3. Personalization is a cold-start signal, not a ranking signal.** The anonymised profile
tags ("fit", "comfort", "durability") match most of the catalog. As a global ranking term
they are **actively harmful (−0.039)**. Applied only before any constraint is known — when
they are the only personal signal that exists — they help (**+0.015**). Same feature,
opposite sign, depending entirely on *when* it fires.

### Results

| scenario | n | Hit@10 | MRR | MTTC | baseline Hit@10 |
|---|---:|---:|---:|---:|---:|
| buying | 80 | 1.0000 | 0.6878 | 1.238 | 0.2375 |
| browsing | 80 | 1.0000 | 0.7278 | 1.825 | 0.0250 |
| intent_override | 30 | 0.9667 | 0.9667 | 3.833 | 0.1333 |
| boundary | 10 | 1.0000 | 0.9250 | 2.600 | 0.0000 |
| **overall** | **200** | **0.9950** | **0.7575** | **1.930** | 0.1250 |

**Ablation — every mechanism removed one at a time:**

| configuration | score | Δ | 95% CI on Δ |
|---|---:|---:|---|
| **full system** | **0.9062** | — | — |
| no clarification | 0.4885 | **−0.4176** | [−0.4789, −0.3561] |
| no state tracking | 0.5666 | **−0.3395** | [−0.3955, −0.2838] |
| no popularity prior | 0.8595 | −0.0466 | [−0.0683, −0.0263] |
| no profile prior (cold start) | 0.8909 | −0.0152 | [−0.0255, −0.0054] |
| no constraint scoring | 0.8910 | −0.0151 | [−0.0227, −0.0081] |
| no category lock | 0.8971 | −0.0090 | [−0.0226, +0.0028] *spans zero* |
| no override handling | 0.9052 | −0.0010 | [−0.0030, +0.0000] *spans zero* |
| no observed-token fallback | 0.9054 | −0.0008 | [−0.0022, +0.0000] *spans zero* |
| no top-10 padding | 0.9062 | 0.0000 | [0.0000, 0.0000] *spans zero* |
| no MMR diversity | 0.9062 | 0.0000 | [0.0000, 0.0000] *spans zero* |

An ablation is a **paired** comparison — both configurations answer the same 200 sessions —
so `tools/ablation_ci.py` bootstraps the delta itself rather than comparing two marginal
intervals, which resolves effects an order of magnitude smaller.

It also says something I did not want to hear. **Five of the ten mechanisms are not
distinguishable from sampling noise**, the category lock among them, which I had been
crediting with +0.009. They stay in the default configuration, because an interval spanning
zero means *unresolved at n=200* rather than *absent* — but the claim I make for them is now
the smaller one. I would rather report this than have a judge derive it.

**Robustness — the same sessions, reworded.** The specification warns that the organiser
may paraphrase customer messages, noting only that paraphrasing "cannot decide
correctness." Template-exact parsing was betting the entire score on wording explicitly
declared unstable, so I built a perturbation harness that imports the evaluator's own
simulator functions — customer policy, scoring and scenario mix identical, **only surface
wording changes**. Its control run reproduces the official score exactly.

| perturbation | before hardening | after |
|---|---:|---:|
| control | 0.8219 | 0.9062 |
| lowercase | 0.8219 | 0.9062 |
| punctuation stripped | 0.4101 | 0.8796 |
| light paraphrase | 0.2691 | 0.8860 |
| heavy paraphrase (+filler, +case drift) | 0.2370 | 0.8824 |

Worst case now sits **2.9% below control**, versus **71% below** before hardening.

One disclosure about that table: the *before hardening* column is historical. It measured
the template-exact parser the hardening replaced, and that code is no longer in the
repository, so unlike every other figure here it cannot be regenerated by running anything.
Running the harness today reproduces the *after* column only.

---

## Impact & relevance

### The finding that transfers, independent of this dataset

Two mechanisms account for **0.757 of the 0.799** improvement — clarification (+0.418) and
state tracking (+0.340). **Neither is a model.** Neither needs one. The expensive component
that a conversational-commerce roadmap usually funds first — an LLM semantic ranker over
the catalog — is not what moved the number here.

That is the transferable claim: **in conversational commerce, dialogue policy is worth more
than model quality, and it is orders of magnitude cheaper.** If you are building a shopping
copilot, the first thing to fund is the decision of *what to ask next*, and only then the
quality of the ranker underneath it. My own build order was the wrong way round, and the
ablation table is what told me so.

### Who benefits, concretely

**Shoppers** answer roughly **one question instead of nine**. The problem statement itself
frames MTTC as penalising "unnecessary conversational cognitive load" — 1.93 turns is the
difference between a copilot people finish and one they abandon. The gain is largest
exactly where recommender systems are weakest: the **Browsing** cold-start case, where the
customer opens with no constraints at all and there is no history to lean on
(**0.025 → 1.000**).

**Operators** get a copilot whose marginal cost is CPU time. The arithmetic below is
illustrative — substitute your own prices, the ratio is the point. Only the **0 tokens**
and **8 ms/turn** figures are measured:

| per 1,000,000 shopping sessions | LLM-ranking copilot | this agent |
|---|---:|---:|
| turns (at measured MTTC 1.93) | ~1.93 M | ~1.93 M |
| input tokens (~6 k/turn to show ~40 candidates) | ~12 B | **0** |
| output tokens (~200/turn) | ~0.4 B | **0** |
| inference cost @ $0.30/M in, $1.50/M out | **~$4,200** | **$0** |
| compute (19 ms/turn worst case ≈ 10 CPU-hours @ $0.04/hr) | on top of the above | **~$0.40** |
| added latency, p95 | ~0.5–2 s per turn | **85 ms** |

Three to four orders of magnitude — and the latency figure matters as much as the money.
**85 ms p95 fits inside an existing search-response budget**, so this can ship as an inline
component of the search path rather than as a separate async chat surface the shopper has
to opt into. Driving every session to all ten turns — far past where the scored loop stops —
pushes p95 to 193 ms, which is the figure to plan against if your sessions run long.

**Deployments an API-gated copilot cannot reach.** No key, no network, no GPU, 226 MB
resident, standard library only. It runs on-device, at the edge, in air-gapped or regulated
environments, and in price-sensitive markets where per-request inference cost is the reason
conversational search never shipped. It also means **the shopper's conversation never
leaves the process** — no third-party model provider sees what someone is shopping for. For
a platform under standing scrutiny about where user data goes, "the copilot does not call
out" is a product property, not a footnote.

**A recipe any catalog owner can run.** The P(yield) table above was computed from catalog
statistics alone, with no session labels: count how often each attribute is actually
populated and discriminating in your own inventory, then ask by expected value rather than
by information gain. That method ports to any marketplace with a product catalog and a
question-answering surface, and it is the part of this project I would hand to another team
first.

### Where this fits in a real system

I would not argue that no LLM belongs in a shopping copilot. I would argue this is the
**wrong place to spend it**. The realistic production shape is a hybrid: this agent as the
always-on core doing constraint tracking, question selection and ranking at 8 ms and zero
marginal cost, with an optional language layer spent on *phrasing* the question naturally
and absorbing genuinely open-ended input — the part a model is uniquely good at — while the
scored retrieval path stays offline and deterministic. The repository is already built that
way: any LLM layer is an environment-gated addition that is **off by default**, because the
rules warn that official scoring may disable network access, and an agent that *needs* a
key is a liability rather than a feature.

### The honest counter-argument

The strongest criticism of this project is that it reverse-engineers a simulator rather than
building a shopping assistant. I take that seriously, and my defence is measurement, not
assertion:

- Every mechanism derives from the **published protocol**, not from public-set labels — the
  ask-and-recommend insight, the constraint typing, the retraction semantics and the
  P(yield) table all come from documented behaviour or from the catalog itself. The
  P(yield) prior is the one a sceptic should push on hardest, since it ships as seven
  hardcoded constants, so it is **re-derivable**: `python -m tools.yield_prior` recomputes
  all seven from the catalog using the evaluator's own `intent_card` and
  `classify_constraint`, touching no session labels, and reproduces the shipped values to
  within 0.0003 — which is the rounding in the source file. A test enforces it.
- Each was validated **three ways**: ablation, cross-scenario consistency, and paraphrase
  perturbation. A mechanism fitted to public wording would not survive the third.
- Weights were chosen from **flat regions** of their sweeps rather than sharp argmaxes.
- The one mechanism that genuinely *is* dataset-specific — the popularity prior, which works
  partly because target products come from a 5-core split and therefore carry ≥5 reviews —
  is flagged in the limitations as something **I would not expect to transfer to a live
  catalog**. Naming that voluntarily is what should make the rest credible.

I also kept two mechanisms that score ≈0 on the public set, deliberately: top-10 padding
(0.0000 here, but an empty slot can never hit) and the observed-token fallback (−0.0008 on
clean input, −0.0029 under heavy paraphrase, neither interval excluding zero). Both are kept
as insurance against the 800 private sessions rather than as public-set optimisations — an
argument from cost, since they are cheap and an unresolved interval is not a demonstration
of absence, not an argument from measurement.

---

## Feasibility & practicality

| | |
|---|---|
| Model / API | **none** |
| Token usage | **0** — zero by construction, not by estimate |
| Network access | **none required** — fully offline |
| Monetary cost | **$0** |
| Dependencies | Python standard library only (`pytest` for tests) |
| Index build | ~19 s cold (one time), ~0.3 s warm from cache |
| Per-turn latency | **8 ms median**, 85 ms p95, 213 ms max (scored loop); 19 ms / 193 ms / 495 ms if every session is driven to all ten turns |
| Memory | **226 MB** resident, agent + index only |
| Full 200-session evaluation | ~8 s warm, ~26 s including a cold index build |
| Tests | 222 passing, including 53 adversarial |

Measured on an Intel i5-1340P laptop, CPU only, no GPU. Latency is measured over 600 turns
with constraints accumulating, not just cheap opening turns. Index caching is best-effort
and wrapped in `try/except`, so a read-only judging sandbox simply rebuilds — slower, never
a failure. The system runs unchanged under the CPU, memory, timeout and network
restrictions the submission rules explicitly reserve the right to impose.

---

## Development tools, APIs, libraries and data

**Development tools:** VS Code, Git, Windows 11 / PowerShell, Python 3.12.6, `pytest`,
plus `cProfile` and `tracemalloc` for the latency and memory figures. Claude (Anthropic)
was used as a coding assistant in the same sense as an IDE or a linter; every design
decision and every number reported here was verified by measurement against the official
evaluator in the repository.

**APIs used:** **none.** No LLM API, no external service, no network call at inference
time. Token usage is zero by construction rather than by estimate.

**Libraries and frameworks:** on the scored path, **none** beyond the Python standard
library — no PyTorch, no Transformers, no scikit-learn, no pandas, no FAISS, no vector
database. The BM25 inverted index, the tokeniser, the slot state machine, the question
policy and the fp16 dense reader are all hand-written; `pytest` is a test dependency only.

Two optional extras exist and are never imported by the agent at scoring time, which I
would rather spell out than let the "stdlib only" claim quietly cover them.
`tools/build_vectors.py` — the *offline* script that builds the dense artifact — uses numpy,
scipy and sentence-transformers, and is a build-time tool that never runs at inference.
`copilot/llm.py` lazily imports `anthropic` only when the LLM layer is explicitly enabled by
both a config flag and an environment variable; with the default configuration the import
never executes. `requirements.txt` carries exactly one line — `pytest`, for the test suite.
`pytest` is used for the test suite only and is not required to run the agent.

**Datasets and assets:** the organiser's frozen participant kit — a 50,000-product catalog
and 200 labelled public development sessions derived from **Amazon Reviews 2023** (McAuley
Lab, UCSD). The catalog is downloaded from the official release and SHA256-verified; it is
**not redistributed** in the repository. No additional data was collected, scraped or
labelled, and no organiser-only or private evaluation data is used anywhere.

---

## What I learned

- **The baseline's failure was in its output schema, not its ranker.** I spent the first
  pass improving retrieval and moved the score barely at all. The 8× came from reading the
  protocol closely enough to notice that one field was permanently `None`.
- **Measure the counter-intuitive option before discarding it.** Erasing a retracted
  preference is the obvious behaviour, and it was the worst setting I tested.
- **Build the harness that tries to break your own work.** The paraphrase harness cost a
  day and revealed that a light rewording removed 67% of my score. Nothing in the ablation
  table was ever going to surface that.
- **Report the zeroes.** MMR diversification measured exactly 0.0000 after a real
  implementation effort. It stays in the ablation table because leaving it out would
  misrepresent how this was built.

## Limitations, and what I would do next

- Tuned on 200 public sessions; **800 private sessions decide the result.**
- **Hit Rate is effectively saturated at 0.995**, so the remaining headroom is almost
  entirely MRR (0.7575 of 1.0) and further public-set gains are increasingly likely to be
  noise.
- **Semantic retrieval is implemented but switched off**, because it measured worse
  (below). Matching on the scored path is lexical, so a customer describing a product in
  words absent from its metadata is served poorly — on *this* benchmark that case barely
  occurs, and on a real storefront it would occur constantly.
- The **paraphrase harness is my model of paraphrasing, not the organiser's**; a
  paraphraser that also rewrote product attributes would degrade the agent further.

The first two items on this list are now built rather than planned, and both lost:

**Offline dense retrieval** (`copilot/dense.py`) encodes the catalog once and cosine-reranks
at inference — stdlib only, fp16 via `struct` and `mmap`, ~6.7 ms/turn, still no GPU and no
network at inference. It loses, and the interesting part is *how* it loses across three
encoder tiers built by the same script against the same catalog:

| encoder | dim | best Δ | 95% CI | Δ at w=3.0 | rows resolved |
|---|---:|---:|---|---:|---:|
| truncated SVD (LSA) | 128 | +0.0010 | [−0.0038, +0.0062] | −0.0507 | 6 of 10, all negative |
| `all-MiniLM-L6-v2` | 384 | +0.0036 | [−0.0032, +0.0109] | −0.0348 | 4 of 10, all negative |
| `BAAI/bge-base-en-v1.5` | 768 | +0.0005 | [−0.0047, +0.0060] | −0.0250 | 2 of 10, all negative |

**A better encoder makes it less harmful and never helpful.** The damage at high weight
shrinks monotonically with model quality, but no tier at any weight produces a gain whose
paired interval excludes zero. That locates the ceiling in the task rather than the model:
if the encoder were the binding constraint, tripling the dimension and moving from an
unsupervised SVD to a contrastively-trained retriever would have flipped the sign somewhere.
It moves the magnitude and leaves the sign alone.

The structural reason: **95.6% of the constraint strings the simulator discloses appear
verbatim in their own target product, and 25.9% are unique to a single product in 50,000.**
Dense retrieval exists to close vocabulary mismatch, and this benchmark has almost none by
construction — so what it mostly does is blur an already-exact lexical match. It ships
switched off, behind a flag, so the ablation row reproduces.

**An environment-gated LLM reranking layer** (`copilot/llm.py`) is double-gated — a config
flag *and* `COPILOT_LLM=1` — with a lazy import, silent total failure, permutation-
preserving sanitisation, and honest token accounting. Off by default, because the rules
warn that network access may be withdrawn at judging time and a scored path that depends on
it is a path that can score zero.

Next, in order: **(1) a cross-encoder reranker** trained on protocol-generated sessions and
quantised to int8 ONNX, reranking the top ~30 — the only remaining lever on MRR, and one I
will ship only if it wins on held-out targets rather than on the public set; **(2) question
selection over the true posterior** — keep a distribution over candidate products and choose
the question that minimises expected posterior entropy, rather than using per-attribute
priors.

---

## Built With

`python` · `python-standard-library` · `bm25` · `information-retrieval` ·
`conversational-ai` · `dialogue-state-tracking` · `expected-value-decision-policy` ·
`inverted-index` · `pytest` · `amazon-reviews-2023` · `no-llm` · `offline-first`
