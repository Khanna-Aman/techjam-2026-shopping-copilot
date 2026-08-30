# Shopping Copilot — TikTok TechJam 2026, Track 4

[![CI](https://github.com/Khanna-Aman/techjam-2026-shopping-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Khanna-Aman/techjam-2026-shopping-copilot/actions/workflows/ci.yml)

A stateful conversational shopping agent for the TechJam Conversational E-Commerce Search
Challenge. It finds a hidden target product inside a frozen 50,000-item Amazon catalog by
asking the right questions, remembering the answers, and re-ranking as evidence arrives.

**It runs on the pure Python standard library — no model, no API key, no network access —
and scores 8.5× the official baseline.**

```
                     official baseline        this agent
  Hit Rate@10              0.125       ->       0.995
  MRR                      0.068034    ->       0.757502
  MTTC (turns)             9.81        ->       1.930
  TechnicalScore           0.10671     ->       0.906151      (8.49x)
```

`TechnicalScore = 0.50·HitRate@10 + 0.30·MRR + 0.20·clip((11−MTTC)/10, 0, 1)`

Measured with the **unmodified** official evaluator over the 200 public sessions.
Reproduce with one command: `python -m evaluator.local_evaluator`.

Two caveats stated up front rather than buried: the supplied baseline is explicitly named
`weak_bm25`, so the 8.49× multiple flatters this result — the [ablation](#ablation--one-mechanism-removed-at-a-time)
is the honest version of the claim. And 200 sessions put a 95% interval of
**[0.8891, 0.9216]** around that score, so the defensible number is **0.91 ± 0.02**.

---

## Start here

This document is long because the evidence is the point. If you have five minutes, read
these five things in order — they are the argument, and each one is a link to its section.

| # | Read this | Why it is the interesting part |
|---|---|---|
| 1 | [The insight](#the-insight-the-whole-system-is-built-on) | The baseline never sets `ask_attribute`. In this protocol the customer only discloses when asked, so it re-runs one query ten times. Fixing that is worth **+0.418** of the **+0.799** total. |
| 2 | [The ablation, with paired intervals](#ablation--one-mechanism-removed-at-a-time) | Every mechanism removed one at a time — and **five of the ten cannot be distinguished from noise**, including one this README used to credit with a real gain. |
| 3 | [Dense retrieval, killed three times](#you-only-tried-a-weak-encoder) | The brief asks for vector similarity. It is built, and it loses at 128, 384 and 768 dimensions. A better encoder shrinks the harm without changing its sign, which locates the ceiling in the task rather than the model. |
| 4 | [Talk to it yourself](#reproduce) | `python -m tools.chat`, then type `/why`. The question policy prints its own reasoning: budget splits the candidate pool beautifully and is answered 0.5% of the time. |
| 5 | [What was tried and rejected](#what-was-tried-and-rejected) | MMR at exactly 0.0000. A profile weight adopted and then reverted. A hypothesis about paraphrase, tested and refuted. The failures are reported because reporting only the wins would misrepresent how this was built. |

If you have one minute instead: the score is **0.906 ± 0.02**, it uses **zero tokens and no
network**, and `python -m pytest -q` runs 224 tests that include a suite asserting this
README's own tables against the committed measurements in `results/`.

---

## The insight the whole system is built on

The supplied baseline scores 0.107 and converges at turn **9.81 out of 10**. That is not a
retrieval-quality problem. Look at what it returns:

```python
return {"message": "...", "ask_attribute": None, ...}   # starter/baseline_agent.py
```

`ask_attribute` is always `None`. In the simulator, the customer only discloses
information *in response to a question* — so with no question asked, every follow-up turn
returns the same non-answer:

> "Those options are not quite right yet. Ask me about one specific attribute."

The baseline therefore gains **zero new information on every turn after the first** and
re-runs a single OR-query ten times. It is negotiating with itself.

Crucially, the protocol lets one response carry a clarification question **and** a ranked
list at the same time. There is no ask-versus-recommend trade-off to balance — the correct
policy is always to do both. Half the available channel was simply going unused.

That single change is worth **+0.418** of the +0.799 total improvement.

---

## Architecture

```mermaid
flowchart TB
    MSG["Customer message"] --> P{"Parse: exact tier"}
    P -->|template matches| EXACT["Scenario + category + constraints"]
    P -->|reworded| FUZZY["Fuzzy tier: n-gram category scan,<br/>negation and retraction cues"]
    EXACT --> ST["Slot state machine"]
    FUZZY --> ST
    MSG -.->|"always, even on parse failure"| OBS["Observed-token safety net"]
    OBS --> ST
    ST --> POOL["Category lock<br/>50,000 → ~180 candidates"]
    POOL --> RANK["Weighted BM25<br/>+ typed constraint satisfaction<br/>+ popularity prior<br/>+ cold-start personalization"]
    RANK --> TOP["Top-10, padded, deduplicated"]
    ST --> Q["Question policy:<br/>expected value, not entropy"]
    Q --> OUT["ask_attribute"]
    TOP --> OUT2["recommendations"]
```

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

| Module | Responsibility |
|---|---|
| `copilot/text.py` | Normalisation, tokenisation, the simulator's constraint vocabulary |
| `copilot/catalog.py` | In-memory weighted-BM25 inverted index, category buckets, typed attribute tables |
| `copilot/dialogue.py` | Two-tier message parsing (exact templates → content-driven recovery) |
| `copilot/slots.py` | Constraint accumulation, typing, and override handling |
| `copilot/question.py` | Expected-value clarification policy |
| `copilot/retrieval.py` | Candidate generation, constraint satisfaction, ranking |
| `copilot/agent.py` | Orchestration and the required `Agent` contract |
| `starter/agent.py` | Submission entry point the official harness imports |

Two design choices worth calling out. The index is a **hand-rolled weighted-BM25 inverted
index rather than SQLite FTS5**, because ranking here is dominated by precision inside a
~180-item pool and needs constraint-satisfaction and prior terms that `bm25()` cannot
express. And **field weights are folded into the stored term frequency** — one postings
list carrying a weighted tf, instead of six per-field indexes, for the same ranking effect
at a sixth of the memory.

---

## Five findings that shaped the design

Each was measured, and **three overturned the intuitive answer**.

### 1. The distinctive information is the information you have to ask for

The hidden intent card is built as `[material, colour, feature₀, feature₁]`, where
`hard_constraints` are the first two and `soft_preferences` the rest. Material and colour
are *weak* — thousands of products are "cotton" and "black". The near-identifying strings
are the feature sentences, and those sit in `soft_preferences`, which are **only ever
revealed in response to a question**. This is why Browsing goes from 0.025 to 1.000 once
the agent starts asking.

### 2. Don't erase a retracted preference — it is still true

The obvious reading of Intent Override is "the customer withdrew that, so drop it". That
is wrong, and measurably so. The simulator draws **both** the withdrawn preference and its
replacement from the *same hidden target product*:

```python
old_value = soft[-1]      # from the target
new_value = hard[0]       # also from the target
```

The customer changes their mind; **the target never changes**. The withdrawn value stays
diagnostic. Full erasure scored **worst** of every setting tested. The agent retains it at
half confidence (`override_decay = 0.5`).

### 3. A question that splits the pool perfectly is worthless if nobody answers it

The first information-gain model scored questions purely on how well an answer would
partition the candidate pool. It kept choosing **budget** — which partitions beautifully
and goes unanswered 99.5% of the time.

Measuring P(the customer can answer) across all 50,000 products, using the catalog only
and no session labels — `python -m tools.yield_prior` re-derives this table by running the
evaluator's own `intent_card` and `classify_constraint` over every product, and a test
asserts the constants shipped in `copilot/question.py` against it:

| attribute | feature | material | colour | style | size | use_case | budget |
|---|---|---|---|---|---|---|---|
| P(yield) | 0.958 | 0.573 | 0.427 | 0.162 | 0.076 | 0.016 | **0.005** |

Folding that in gives an expected-*value* model:

```
value(A) = P(customer can answer A) × E[constraints returned] × how well they split the pool
```

The policy now **derives** that the open-ended question is optimal rather than having it
hardcoded, and yields to a specific question once the open channel is exhausted. Worth
**+0.015** over the entropy-only policy (0.9062 against 0.8911) — and unlike half the
ablation table, this one survives the paired test: 95% CI [+0.0052, +0.0255], reproduced by
`python -m tools.ablation_ci --mode strategy`. Two attributes —
`category` and `brand` — are excluded outright, because the simulator's classifier provably
never emits them, so asking can never pay.

These seven numbers are hardcoded in `copilot/question.py` for speed — deriving them costs a
full catalog pass at import — but **they are not free parameters, and that is checkable
rather than merely asserted.** `tools/yield_prior.py` reproduces all seven from the catalog
alone, and the largest disagreement with the shipped values is 0.0003, which is the rounding
in the source. This matters because a hardcoded prior in a submission scored on a hidden set
invites a fair suspicion that it was fitted to the 200 public sessions. It was not, and the
harness is the evidence.

Reproduce with `python -m tools.sweep --mode strategy`, and note what it also shows: the
hybrid policy scores **exactly** what always-asking-open scores, to six decimal places. On
this set the escalation branch never earns anything. It is kept for the same reason as the
padding and the observed-token fallback — it is insurance for a private set that may
exhaust the open channel more often — but it is not a public-set win, and reporting it as
one would be dishonest.

| clarification strategy | score |
|---|---:|
| none (the baseline's behaviour) | 0.4885 |
| entropy-only (`infogain`) | 0.8911 |
| always open | 0.9062 |
| **expected value (`hybrid`, default)** | **0.9062** |

### 4. Personalization is a cold-start signal, not a ranking signal

The anonymised profile offers generic preference tags ("fit", "comfort", "durability")
that match most of the catalog. As a global ranking term they are **actively harmful**
(−0.039, at the same weight). Applied *only before any constraint is known* — when they are
the sole personal signal available — they help (**+0.015**). Same feature, opposite sign,
depending entirely on when it is applied.

| `w_profile` = 1.0 | score | vs profile off |
|---|---:|---:|
| profile off | 0.8909 | — |
| applied always | 0.8521 | **−0.0388** |
| applied at cold start only | 0.9062 | **+0.0152** |

`python -m tools.sweep --mode profile` carries both arms, because the finding is not
"personalization helps" but that its sign flips with timing, and a grid holding only the
cold-start arm cannot show that.

### 5. Paraphrase resistance was worth more than any ranking tweak

The specification warns that the organiser may add natural-language paraphrasing, noting
only that it "cannot decide correctness". Template-exact parsing was betting the entire
score on wording explicitly declared unstable. A perturbation harness confirmed the
exposure, and the fix — content-driven parsing plus a token-level safety net — moved the
worst case from **0.237 to 0.882**.

---

## Reproduce

Python **3.10+**. **No third-party dependencies** — the agent is standard library only.
(`pytest` is needed for the test suite, nothing else.)

```bash
# 1. catalog (18 MB download, expands to 58 MB, 50,000 products)
gh release download participant-kit \
  --repo TechJam2026/techjam-conversational-search \
  --pattern "catalog.jsonl.gz" --pattern "SHA256SUMS"
sha256sum -c SHA256SUMS --ignore-missing     # verify before trusting it
gzip -dkc catalog.jsonl.gz > data/catalog.jsonl

# 2. official score  (~26 s first run incl. index build, ~8 s afterwards)
python -m evaluator.local_evaluator

# 3. everything else
python -m pytest -q                                 # 222 tests
python -m tools.demo --scenario intent_override --index 1
python -m tools.sweep --mode ablation
python -m tools.robustness
python -m tools.proxy_private                       # held-out generalisation
python -m tools.sweep --mode dense                  # the dense-retrieval rejection

# optional: rebuild the dense artifact (needs numpy/scipy; ~35 s, deterministic)
pip install numpy scipy scikit-learn && python -m tools.build_vectors
```

The first run builds an index and caches it under `artifacts/` (~19 s, one time). Caching
is best-effort and wrapped in `try/except`: a read-only judging environment simply rebuilds
each run, which is slower but never a failure.

You do not have to take the score on trust. [CI](.github/workflows/ci.yml) runs the test
suite on Linux, macOS and Windows across Python 3.10 and 3.12, then downloads the frozen
catalog from the organiser's own release, runs the unmodified evaluator, and **fails the
build if the result disagrees with [`results/official_evaluation.json`](results/official_evaluation.json)**
on hit rate, MRR, MTTC or the composite score — or if token usage is anything but zero. The
number below is checked by machine on every push. A separate step walks the history back to
the participant kit and fails if any commit touched `evaluator/` or `data/public_set.jsonl`.

See [`DEMO_WALKTHROUGH.md`](DEMO_WALKTHROUGH.md) for a scripted three-minute tour.

---

## Results

### Per scenario

| scenario | n | Hit@10 | MRR | MTTC | baseline Hit@10 |
|---|---:|---:|---:|---:|---:|
| buying | 80 | 1.0000 | 0.6878 | 1.238 | 0.2375 |
| browsing | 80 | 1.0000 | 0.7278 | 1.825 | 0.0250 |
| intent_override | 30 | 0.9667 | 0.9667 | 3.833 | 0.1333 |
| boundary | 10 | 1.0000 | 0.9250 | 2.600 | 0.0000 |
| **overall** | **200** | **0.9950** | **0.7575** | **1.930** | 0.1250 |

Intent Override carries the highest MTTC by construction: a hit only counts *after* the
customer revises their intent on turn 3 or 4, so ~3.5 is close to the structural floor.

### How precise is 0.906151?

Not that precise. Six figures is a fact about floating-point arithmetic; the score is a mean
over 200 sessions, and a different 200 would land elsewhere. `tools/bootstrap.py` resamples
the committed per-session records 20,000 times:

| metric | point | 95% CI | std err |
|---|---:|---|---:|
| TechnicalScore | 0.9062 | [0.8891, 0.9216] | ±0.0082 |
| Hit@10 | 0.9950 | [0.9850, 1.0000] | ±0.0050 |
| MRR | 0.7575 | [0.7085, 0.8046] | ±0.0244 |
| MTTC | 1.9300 | [1.7750, 2.1000] | ±0.0828 |

So the defensible claim is **≈0.91 ± 0.02**, and MRR — the metric with the most headroom
left — is also the least certain of the four. Two consequences I try to hold to elsewhere in
this document: a tuning result below roughly ±0.016 on the composite is not a result, and
the gap between this and the generalisation numbers below is well inside the interval.

**What this interval does not cover.** A bootstrap describes what happens if you redraw
sessions *from the same distribution*. The private set may not be that distribution —
different target popularity, possibly paraphrased turns. That is distribution shift, not
sampling noise, and it is measured separately by `tools/proxy_private.py` and
`tools/robustness.py`. Quoting ±0.02 as a bound on private-set performance would be an
overclaim.

### Ablation — one mechanism removed at a time

The Δ column is a **paired** comparison: both configurations answer the same 200 sessions in
the same order, so the variance they share cancels in the difference. `tools/ablation_ci.py`
bootstraps that difference directly, which resolves effects far smaller than the ±0.008
standard error on the score itself would suggest.

| configuration | score | Δ | 95% CI on Δ |
|---|---:|---:|---|
| **full system** | **0.9062** | — | — |
| no clarification | 0.4885 | **−0.4176** | [−0.4789, −0.3561] |
| no state tracking | 0.5666 | **−0.3395** | [−0.3955, −0.2838] |
| no popularity prior | 0.8595 | −0.0466 | [−0.0683, −0.0263] |
| no profile prior (cold start) | 0.8909 | −0.0152 | [−0.0255, −0.0054] |
| no constraint scoring | 0.8910 | −0.0151 | [−0.0227, −0.0081] |
| no category lock | 0.8971 | −0.0090 | [−0.0226, +0.0028] *spans zero* |
| no override erasure | 0.9052 | −0.0010 | [−0.0030, +0.0000] *spans zero* |
| no observed fallback | 0.9054 | −0.0008 | [−0.0022, +0.0000] *spans zero* |
| no top-10 padding | 0.9062 | 0.0000 | [0.0000, 0.0000] *spans zero* |
| no MMR diversity | 0.9062 | 0.0000 | [0.0000, 0.0000] *spans zero* |

**Five of the ten mechanisms are not distinguishable from sampling noise on the public
set.** That includes the category lock, which I had assumed was carrying real weight. The
two zero rows are exactly zero because removing them changes no session's outcome at all.

#### Does paraphrase rescue the unresolved rows? No.

My hypothesis was that those five are cheap guards whose value shows up on harder input than
the clean public set, and that a heavy-paraphrase run would show them carrying real weight.
That is a comfortable story, so it is worth testing rather than asserting.
`python -m tools.ablation_ci --perturbation heavy` runs the identical paired comparison with
every customer message reworded (default configuration under heavy: 0.8824):

| configuration | Δ clean | Δ heavy | resolved? |
|---|---:|---:|---|
| no clarification | −0.4176 | −0.5201 | both |
| no state tracking | −0.3395 | −0.3135 | both |
| no popularity prior | −0.0466 | −0.0323 | both |
| no profile prior (cold start) | −0.0152 | −0.0195 | clean only |
| no constraint scoring | −0.0151 | **+0.0010** | clean only |
| no category lock | −0.0090 | −0.0253 | neither |
| no override erasure | −0.0010 | +0.0000 | neither |
| no observed fallback | −0.0008 | −0.0029 | neither |
| no top-10 padding | 0.0000 | +0.0000 | neither |
| no MMR diversity | 0.0000 | +0.0000 | neither |

**The hypothesis is not supported.** Every row unresolved on the clean set is still
unresolved under heavy paraphrase. The category lock roughly triples its point estimate
(−0.009 → −0.025) but its interval widens with it and still spans zero, and constraint
scoring — which *is* resolved on the clean set — flips sign and becomes unresolved.
Paraphrase adds variance faster than it adds signal, so **fewer** mechanisms are resolvable
under it, not more: three of ten rather than five.

I am still leaving all five in the default configuration, but on narrower grounds than I
started with. An interval spanning zero means *unresolved at n=200*, not absent; the five
cost nothing measurable in latency; and the private set may be hard along axes this
perturbation does not model. That is a weaker argument than "they are insurance, and here is
the proof", and I would rather make the weaker one, because it is the one I can support.

What changes most is the claim, not the configuration. I no longer describe the category
lock as contributing +0.009 — the honest statement is that its contribution is smaller than
this benchmark can measure, on either input.

### Robustness — the same sessions, reworded

`tools/robustness.py` imports the evaluator's own simulator functions, so the customer
policy, scoring and scenario mix are identical and **only surface wording changes**. Its
control run reproduces the official score exactly.

| perturbation | before hardening † | after |
|---|---:|---:|
| control | 0.8219 | 0.9062 |
| lowercase | 0.8219 | 0.9062 |
| punctuation stripped | 0.4101 | 0.8796 |
| light paraphrase | 0.2691 | 0.8860 |
| heavy paraphrase (+filler, +case drift) | 0.2370 | 0.8824 |

Worst case sits **2.9% below control** (0.8796 vs 0.9062), versus 71% below before
hardening.

† **The "before hardening" column is historical and cannot be regenerated from this
repository.** It measured the template-exact parser that the hardening replaced, and that
code no longer exists here. Every other number in this document is checked against a
committed artifact in `results/` by `tests/test_documentation.py`; these four are the
exception, and the "71% below" claim rests on them. Running `python -m tools.robustness`
today reproduces the **after** column only. I am keeping the column because the comparison
is the reason the work was done, and flagging it because an unverifiable number sitting in a
table of verifiable ones is exactly the kind of thing I would want pointed out to me.

### Generalisation — held-out targets the agent was never tuned on

The score above comes from the 200 public sessions. The 800 private ones decide the result,
and "some overfitting risk is unavoidable" is a weaker thing to say than a number.

The public samples carry **no hidden fields** — only a target `parent_asin`, a scenario
label and an anonymised profile. The intent card and the customer's behaviour are
materialised from the catalog product at evaluation time by the organiser's own
`materialize_hidden_fields`. So a session built from a *different* target product is not an
approximation of a real one: it is structurally identical, and `tools/proxy_private.py`
drives it through the **unmodified official `evaluate()` loop**. Only the target differs.

Sampling turned out to be the entire problem. Public targets are nowhere near uniform draws
from the catalog:

| | public targets | whole catalog |
|---|---:|---:|
| median `rating_number` | 6,614 | 12 |
| has `features` | 100% | 90% |
| has `details` | 100% | 97% |
| has a price | 89% | 21% |

Sampling uniformly would have measured a *harder task*, not a generalisation gap. So there
are two regimes:

| regime | n | score | 95% CI | vs public |
|---|---:|---:|---|---:|
| public (official) | 200 | **0.9062** | — | — |
| matched — popularity decile-matched | 110 | 0.8869 | [0.8604, 0.9091] | −0.019 |
| uniform — all eligible targets | 1000 | 0.8648 | [0.8528, 0.8763] | −0.041 |

**The public score sits inside the matched interval**, so on held-out targets of comparable
popularity there is no statistically detectable gap. It sits *outside* the uniform interval,
and that 0.041 is the price of the popularity regularity — the dataset artefact flagged
below, which now has a measurement attached rather than a disclaimer.

The matched regime is capped at 110 sessions because the public 200 have already consumed
most of the catalog's high-review tail, which is why its interval is wide. The match is
close on popularity (median 6,845 against 6,614) but not exact on everything: matched
targets carry a price 55% of the time against 89%. Budget is disclosed in roughly 0.5% of
turns so the residual should be immaterial, but it is a real difference and the tool reports
it rather than reporting only the axis that matched well.

### Feasibility

| | |
|---|---|
| Model / API | **none** on the scored path |
| Token usage | **0** — zero by construction, not by estimate |
| Network access | **none required** — fully offline |
| Monetary cost | **$0** |
| Dependencies | Python standard library only |
| Optional LLM reranking | Implemented, **off by default** — see below |
| Index build | ~19 s cold (one time), ~0.3 s warm from cache |
| Per-turn latency | **8 ms median**, 85 ms p95, 213 ms max (scored loop); 19 ms / 193 ms / 495 ms if every session is driven to all ten turns |
| Memory | **226 MB** resident, agent + index only |
| Full 200-session evaluation | ~8 s warm, ~26 s including a cold index build |

Measured on an Intel i5-1340P laptop, CPU only, no GPU. Latency is over 600 turns with
constraints accumulating, not just cheap opening turns: cost rises with the number of
confirmed constraints, because each is tested against every candidate in the pool.

**On the optional LLM layer.** [`copilot/llm.py`](copilot/llm.py) implements a semantic
reranking stage over the top candidates. It is disabled by default and gated a second time
behind `COPILOT_LLM=1`, so neither switch alone can start a billed run. This is a
deliberate choice rather than a missing piece: the rules state that official scoring may
disable network access, which makes an agent that *needs* a key a liability. The offline
path is the scored one, and the layer exists so that comparison is measured rather than
assumed — run `python -m tools.sweep --mode llm`.

The dependency is imported lazily *inside* the call, so with the layer off the agent stays
standard-library only; CI asserts this by importing the agent with nothing installed. Every
failure mode — missing key, missing package, rate limit, timeout, refusal, malformed
response — returns the offline ordering unchanged, and a bad response can never shorten the
recommendation list, because a dropped slot can never hit. Token usage is reported honestly
in both modes: zero by construction when off, and the counts the API actually charged when
on.

---

## What was tried and rejected

Reporting only what worked would misrepresent how this was built.

| Idea | Outcome |
|---|---|
| **MMR diversification** of an uncertain top-10 | Sounded right; measured **0.0000** and dominated latency. Off by default, kept behind a flag so its ablation row stays reproducible. |
| **Erasing** the retracted override value | The intuitive reading; the **worst** setting tested. See finding #2. |
| **Entropy-only** question selection | Chose `budget`, which is answered 0.5% of the time. See finding #3. |
| **Global** profile personalization | −0.039. Helps only at cold start. See finding #4. |
| Tuning `w_constraint` from 1.8 → 6.0 | **No effect at all** — constraint satisfaction already dominates ordering, so the weight is inert across that range. Left at its default rather than reported as a tuned win. |
| Raising `w_profile` from 1.0 → 5.0 | Worked on every metric I checked first, then failed the one I checked last. See below. |
| **Dense retrieval** — offline cosine, at three encoder tiers up to `bge-base` (768-dim, GPU-built) | Built it, measured it, **no gain at any weight on any encoder**. A better encoder shrinks the harm without changing its sign, which locates the ceiling in the task rather than the model. See below. |

Two mechanisms are kept despite scoring ≈0 on the public set, deliberately. **Top-10
padding** never triggers here but prevents a short list, and an empty slot can never hit.
The **observed-token fallback** costs −0.0008 on clean input and −0.0029 under heavy
paraphrase; neither interval excludes zero, so on the evidence available it is worth nothing
measurable on either input. Both are kept as insurance against the private set rather than
as public-set optimisations, and that is an argument from cost, not from measurement: they
are cheap, and an interval spanning zero is not a demonstration of absence.

*An earlier version of this paragraph credited the observed-token fallback with "+0.65 under
heavy paraphrase". That number is the improvement of the **whole** paraphrase-hardening
effort (0.2370 → 0.8824), not of this one mechanism, and attributing it here was wrong. The
paired ablation above is the correct measurement.*

### Dense retrieval makes this task worse, and the reason is the task

The brief asks for vector similarity, so I built it: [`tools/build_vectors.py`](tools/build_vectors.py)
takes a truncated SVD of the BM25-weighted document-term matrix and ships two fp16 arrays
(21 MB, `k=128`); [`copilot/dense.py`](copilot/dense.py) reads them back with `struct` and
`mmap` and scores cosine similarity over the candidate pool. Inference stays **standard
library only** — numpy and scipy are needed to *build* the artifact, never to use it — and
it costs about 6.7 ms per turn, because the category lock means scoring ~180 rows rather
than 50,000.

It does not work. Not marginally: at every weight tested.

| `w_dense` | 0 (off) | 0.15 | 0.30 | 0.60 | 1.00 | 1.80 | 3.00 |
|---|---:|---:|---:|---:|---:|---:|---:|
| score | **0.9062** | 0.8971 | 0.8917 | 0.8878 | 0.8746 | 0.8606 | 0.8555 |

Applied only at cold start — the trick that rescued the profile prior in finding #4 — it
reaches 0.9072, **+0.0010**, whose paired interval is [−0.0038, +0.0062]. It spans zero, so
it is not a result I would ship on.

#### "You only tried a weak encoder"

That is the obvious objection to the table above, and it deserved an answer rather than a
rebuttal. So the same artifact was rebuilt with two real sentence encoders on a GPU —
`all-MiniLM-L6-v2` at 384 dimensions and `BAAI/bge-base-en-v1.5` at 768 — using the *same*
`tools/build_vectors.py`, against a catalog whose digest each artifact records, so the three
tiers are comparable by construction rather than by assertion.

| encoder | dim | best row | Δ | 95% CI on Δ | Δ at `w_dense`=3.0 | rows resolved |
|---|---:|---|---:|---|---:|---:|
| truncated SVD (LSA) | 128 | cold start `w`=0.6 | +0.0010 | [−0.0038, +0.0062] | −0.0507 | 6 of 10, all negative |
| `all-MiniLM-L6-v2` | 384 | cold start `w`=1.2 | +0.0036 | [−0.0032, +0.0109] | −0.0348 | 4 of 10, all negative |
| `BAAI/bge-base-en-v1.5` | 768 | cold start `w`=1.2 | +0.0005 | [−0.0047, +0.0060] | −0.0250 | 2 of 10, all negative |

**A better encoder makes dense reranking less harmful, and never helpful.** The damage at
high weight shrinks monotonically with model quality — −0.051 to −0.035 to −0.025 — and the
number of rows a paired bootstrap can resolve as harmful falls from six to two. But no tier,
at any weight, produces a gain whose interval excludes zero. The best row in the whole
experiment is +0.0036, and it spans zero comfortably.

That is a more useful finding than "it loses", because it locates the ceiling. If the
encoder were the binding constraint, tripling the dimension and moving from an unsupervised
SVD to a contrastively-trained retriever would have changed the sign somewhere. It moves the
magnitude and leaves the sign alone, which says the constraint is the task: when 95.6% of
the disclosed constraint strings appear verbatim in their target, there is almost no
vocabulary gap left for a semantic model to close, and what it mostly does is blur an
already-exact lexical match.

Reproduce with `python -m tools.build_vectors --mode transformer --model BAAI/bge-base-en-v1.5`
(GPU recipe in [tools/KAGGLE.md](tools/KAGGLE.md)), then
`python -m tools.ablation_ci --mode dense --base '{"dense_path": "artifacts/dense-bge"}'`.

The reason is a property of the task, not of LSA. Measuring the constraint strings the
simulator actually discloses:

- **95.6%** of them appear **verbatim** in their own target product's text.
- **25.9%** are unique to exactly one product in the entire 50,000-item catalog.
- **32.6%** narrow the catalog to ten products or fewer.

A quarter of the time, one disclosed constraint *is* the answer, by exact string match. This
is what dense retrieval exists to fix — vocabulary mismatch, where the shopper's words and
the product's words differ — and by construction this task has almost none. Semantic
similarity is a *smoothing* operator: it deliberately blurs exact matches to surface related
items. Smoothing a signal that is already exact can only add noise to the ordering, and the
table above is what that looks like.

That generalises past LSA. Any bi-encoder faces the same structural mismatch here, however
good its embeddings, which is why I did not spend a day on a GPU to reproduce the finding
with better vectors. It also does *not* generalise past this benchmark: a real shopper
describing a product in their own words is exactly the vocabulary-mismatch case, and there
dense retrieval would earn its place. The mechanism is kept behind
`use_dense_rerank` so the numbers above stay reproducible.

### The other rejection worth reading: `w_profile` 1.0 → 5.0

The profile sweep says raise it. The score climbs from 0.9062 to 0.9083 and then sits on a
flat plateau out to at least w=15, so it is not a fragile argmax. Held-out targets agree:
+0.0019 on the popularity-matched proxy set and +0.0013 on the uniform one, all three moving
the same direction. By the standards applied everywhere else in this repository — measured,
reproduced, validated off the tuning set, taken from a flat region — it is a real
improvement, and I adopted it.

Then I ran the paraphrase harness.

| `w_profile` | control | punctuation | light | heavy | **worst case** |
|---|---:|---:|---:|---:|---:|
| **1.0 (shipped)** | 0.9062 | 0.8796 | 0.8860 | 0.8824 | **0.8796** |
| 1.4 | 0.9074 | 0.8796 | 0.8873 | 0.8799 | 0.8796 |
| 5.0 | 0.9083 | 0.8796 | 0.8901 | **0.8711** | 0.8711 |

Raising the weight makes the agent lean harder on a cold-start prior, and that prior is
exactly what a reworded opener disturbs. The clean score rises by 0.0021; the heavy-paraphrase
score falls by 0.0113, and the worst case slips from 2.9% to 4.1% below control. Setting
`P` as the chance the organiser paraphrases, the change pays only while `P < 16%` — and the
specification says paraphrasing may be added without saying how often, while "heavy" here is
*my* model of it rather than theirs.

So it is not shipped. This is finding #5 restated as a decision rather than an observation:
paraphrase resistance was worth more than any ranking tweak, and that has to keep being true
when the ranking tweak is one I already talked myself into.

---

## Limitations and honest caveats

- **Tuned on 200 public sessions; 800 private sessions decide the result.** Weights were
  chosen from flat regions of their sweeps rather than sharp argmaxes, and every mechanism
  derives from the *published protocol* rather than from quirks of the public labels. This
  is now measured rather than argued: on popularity-matched held-out targets the score is
  0.8869, and the public 0.9062 falls inside its 95% interval. Some overfitting risk
  remains — 110 sessions is a wide interval — but it is bounded.
- **Hit Rate is effectively saturated at 0.995** on the public set. Remaining headroom is
  almost entirely MRR (0.7575 of a possible 1.0), so further public-set gains are small
  and increasingly likely to be noise.
- **The popularity prior works partly because of how sessions are sampled.** Targets come
  from a 5-core split, so they carry ≥5 reviews — and in practice far more: their median
  `rating_number` is 6,614 against a catalog median of 12. Abandoning that regularity costs
  **0.041** (uniform 0.8648 vs public 0.9062). It is a dataset regularity rather than
  shopper behaviour, and I would not expect it to transfer to a live catalog. The private
  set shares the sampling, so it should transfer *there*.
- **The paraphrase harness is my model of paraphrasing, not the organiser's.** It rewords
  chrome while preserving payload. A paraphraser that also rewrites *product attributes*
  would degrade the agent further, and I have not simulated that.
- **No semantic retrieval in the shipped configuration** — but not for want of trying. It
  is implemented and measured, and it loses at every weight, because the benchmark's
  constraints are mined verbatim from the target and leave almost no vocabulary mismatch to
  close. A customer describing a product in words that never appear in its metadata is
  still served poorly, and that is a real limitation for a real deployment; it is simply
  not one this benchmark can reward fixing.

## What I would do next

1. **Semantic retrieval for a real catalog, not this one.** The offline dense tier is built
   and measured, and it loses here for a structural reason (see above). The version worth
   building is one aimed at genuine vocabulary mismatch — a shopper describing a product in
   words the listing never uses — which this benchmark does not contain and therefore cannot
   reward. That is a change of task, not of weight.
2. **An optional LLM reranking layer**, environment-gated and **off by default**, with the
   offline path remaining the scored one — the rules warn that official scoring may disable
   network access, so an agent that *needs* a key is a liability, not a feature.
3. **Question selection over the true posterior** rather than per-attribute priors: keep a
   distribution over candidate products and choose the question that minimises expected
   posterior entropy.

---

## Submission compliance

| Requirement | Status |
|---|---|
| Exports `Agent` with `reset` / `respond` | `starter/agent.py` (and `copilot/agent.py`) |
| Official evaluator unmodified | Yes — enforced by CI, which fails if any commit touches `evaluator/` |
| Public labels and docs unmodified | Yes — same CI check covers `data/public_set.jsonl` |
| Requires network access | **No.** Fully offline; declared explicitly |
| Offline fallback | Not applicable — offline *is* the primary path |
| Model choice, cost, token usage, latency disclosed | Yes — see Feasibility. Zero tokens, $0, 8 ms median on the scored path |
| Optional external service | `copilot/llm.py`, **disabled by default** and gated behind `COPILOT_LLM=1`. Never used for scoring; absent it, the agent is unchanged |
| Secrets in repo | None. No API keys, no credentials, no `.env` |
| Python version | 3.10+ — CI covers 3.10 and 3.12 on Linux, macOS and Windows |
| Dependencies | Standard library only; `pytest` for tests, `anthropic` only for the optional layer |
| Catalog redistributed | **No** — `data/catalog.jsonl` is git-ignored, downloaded from the official release and checksum-verified |
| Reads private or organizer-only data | No. The agent opens only the frozen catalog |

**Ablations never edit the evaluator.** Configuration is injected through `AgentConfig`,
or via the `COPILOT_CONFIG` environment variable (a JSON object of field overrides) on the
harness path. With that variable unset, the agent runs its documented defaults.

## Contributions

Solo entry — Aman Khanna. Architecture, implementation, benchmarking and documentation.
Built with assistance from Claude (Anthropic) as a coding tool, in the same sense as an IDE
or a linter; all design decisions and findings above were verified by measurement against
the official evaluator in this repository.

## Data attribution

Catalog and sessions derive from **Amazon Reviews 2023** (McAuley Lab, UCSD), packaged and
frozen by the TechJam organisers. See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md). The
catalog is not redistributed in this repository. The organiser's original participant-kit
README is preserved at [`docs/PARTICIPANT_KIT_README.md`](docs/PARTICIPANT_KIT_README.md).
