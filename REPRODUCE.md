# Reproducing the results

Every claim in this repository comes from a command you can run. This file lists them in
order, with the output each one produces, so any figure in the README or the Devpost write-up
can be traced back to the thing that generated it.

All commands run from the repository root. Nothing takes longer than six seconds except the
first index build and the four long harnesses listed at the end.

## Setup

```bash
python -m evaluator.local_evaluator     # builds and caches the index (~12 s once)
```

Everything afterwards reads that cache, including `tools.chat`, which shares it. Caching is
best-effort and wrapped in `try/except`, so a read-only environment simply rebuilds each run
— slower, never a failure.

Recommended terminal: 100 columns by 40 rows, UTF-8 (`PYTHONIOENCODING=utf-8` on Windows).

---

## 1. The baseline never asks

```bash
grep -n "ask_attribute" starter/baseline_agent.py
```

```
99:            "ask_attribute": None,
```

The supplied baseline hardcodes `ask_attribute` to `None`. In this protocol the customer
discloses information only in response to a question, so from turn two onward the baseline
re-runs the same query against the same catalog and learns nothing. This is the observation
the whole system is built on.

## 2. What each mechanism is worth

```bash
sed -n '509,513p' README.md | tr -d '*' | column -t -s'|'
```

```
full system            0.9633    -          -
no clarification       0.5685    -0.3949    [-0.4511, -0.3384]
no state tracking      0.6448    -0.3186    [-0.3735, -0.2647]
no confidence gate     0.9054    -0.0579    [-0.0703, -0.0457]
no popularity prior    0.9170    -0.0463    [-0.0654, -0.0300]
```

The full eleven-row table is in the README. Regenerate it with
`python -m tools.sweep --mode ablation` and the intervals with `python -m tools.ablation_ci`.
The intervals are paired bootstraps: both configurations answer the same 200 sessions, so
the delta itself is resampled rather than two marginal intervals being compared.

## 3. A browsing session, from zero constraints

```bash
python -m tools.demo --scenario browsing --index 2
```

The customer opens with a category and nothing else — the hardest of the four scenarios, and
the one the baseline scores 0.025 on. Turn one locks the category to 665 in-category
candidates out of 50,000. Turn two returns two constraints, typed: `[material] leather` is a
set-membership test, `[phrase] 100% leather` is phrase containment. Turn three adds a
near-identifying platform-measurement string that exists only because the agent asked for it.

```
 HIT at turn 3, rank 1   (reciprocal rank 1.000)
```

Turns 1 and 2 list one product each; turn 3 lists a full page. That is the confidence gate:
it withholds a wide list until four constraints are known, because the evaluator locks the
rank at the first hit. (`tools/demo.py` prints at most three per turn, so turn 3's three is
the display cap rather than the agent's list, which is ten there.)

## 4. The question policy, inspected

```bash
python -m tools.chat
```

Type a request in your own words rather than the simulator's phrasing — for instance
`I need leather loafers for women` — then `/why`:

```
 pool of 400 candidates
 question            value  splits pool   P(answered)
 other (open)       1.5000                (any type)
 feature            1.2933        0.900   0.958
 color              0.2780        0.434   0.427
 size               0.0882        0.774   0.076
 style              0.0843        0.347   0.162
 use_case           0.0184        0.765   0.016
 material           0.0096        0.056   0.573
 budget             0.0018        0.242   0.005
```

Size splits the pool almost twice as well as colour — 0.774 against 0.434 — and is worth
roughly a third as much, because colour is answered 42.7% of the time and size 7.6%. Value
is a product: how well a question divides the pool, times how likely the customer is to
answer it, times how much they disclose when they do. Budget is the same effect at its
extreme, answered half a percent of the time.

`/state` prints the agent's current beliefs. `/quit` exits.

## 5. Intent Override: a retraction is still evidence

```bash
python -m tools.demo --scenario intent_override --index 1
```

At turn 4 the customer withdraws their opening preference. The memory line shows it marked
`x0.5` rather than deleted:

```
   memory     > [phrase x0.5] leather loafers women:can be bend and curle...
                 [material] leather
                 [phrase] rubber sole
                 [phrase] cowhide leather,rubber sole,hand-sewn loafe...
```

The simulator draws both the withdrawn preference and its replacement from the same hidden
target product, so the customer changes their mind but the target does not. Full erasure was
the worst setting tested; the agent retains the value at half confidence.

```
 HIT at turn 4, rank 1   (reciprocal rank 1.000)
```

## 6. Precision, generalisation and paraphrase

```bash
python -m tools.bootstrap
```

```
TechnicalScore     0.9633   [0.9547, 0.9711]   +/-0.0042
MRR                0.9567   [0.9322, 0.9787]   +/-0.0119
```

The score is a mean over 200 sessions, so the defensible figure is 0.96 +/- 0.01. Five of the
eleven ablation rows cannot be distinguished from sampling noise, and the README table says
so rather than leaving it to be derived.

```bash
python -m tools.proxy_private --report
```

```
regime                   n    w=0.55   w=1.20    change   95% CI on the change
matched                800    0.9354   0.9416   +0.0062   [+0.0029, +0.0102]
uniform  (stress)     1000    0.9135   0.9009   -0.0126   [-0.0180, -0.0078]
```

Held-out targets disjoint from the public set, under two sampling regimes. `matched`
reproduces the organiser's popularity distribution; `uniform` is an out-of-distribution
stress test. `--report` reads the committed artifact — a live run is 1,800 sessions.

```bash
python -m tools.robustness --report
```

```
perturbation        score   vs control
control            0.9633        +0.0%
casing             0.9633        +0.0%
punctuation        0.9515        -1.2%
light              0.9547        -0.9%
heavy              0.9342        -3.0%

worst case 0.9342, 3.0% below control
```

Every customer message reworded and re-scored. The harness imports the evaluator's own
simulator functions, so customer policy, scoring and scenario mix are identical and only
surface wording changes. Its control run reproduces the official score exactly.

## 7. The official result

```bash
python -m evaluator.local_evaluator
```

```
  hit_rate_at_10 1.0        mrr 0.956742      mttc 2.185
  recommended_technical_score 0.963323
  reported_token_usage {"total_tokens": 0}
```

About 6 s warm, ~18 s including a cold index build. Zero tokens, no API key, no network, no
GPU, nothing outside the Python standard library. 6 ms median per turn, 226 MB resident.

---

## Verification

| Question | Command |
|---|---|
| Does it hold up under test? | `python -m pytest -q` — 280 tests, 53 adversarial |
| Is the evaluator unmodified? | `git log --oneline -- evaluator/` |
| Which ablation rows are resolved? | `python -m tools.ablation_ci` |
| Why not rank 1 more often? | `python -m tools.diagnose_rank` — 13 sessions left, none lost to ties |
| Why no dense retrieval? | `python -m tools.constraint_stats` — 99.2% of mined constraints are verbatim |
| Was dense retrieval tried? | `python -m tools.sweep --mode dense` — built, measured, rejected |
| Was an LLM layer tried? | `copilot/llm.py` — built, double-gated off; `requirements-llm.txt` |
| Another scenario? | `python -m tools.demo --scenario buying` |
| How were the weights chosen? | `python -m tools.sweep --mode pop` / `--mode profile` / `--mode prior` |
| Is the yield prior fitted? | `python -m tools.yield_prior` — re-derives all seven from the catalog |

## Runtimes

| Command | Runtime |
|---|---|
| `tools.demo` (one session) | ~1 s |
| `tools.chat` (warm index) | instant to start |
| `tools.bootstrap` | ~3 s |
| `tools.proxy_private --report` | instant |
| `tools.robustness --report` | instant |
| `evaluator.local_evaluator` | ~6 s warm, ~18 s cold |
| `pytest -q` | ~4 s (280 tests) |
| `tools.sweep --mode ablation` | ~2 min (11 configurations) |
| `tools.ablation_ci` | ~4 min (11 configurations plus bootstrap) |
| `tools.robustness` (live) | ~5 min (5 perturbations) |
| `tools.proxy_private` (live) | ~20 min (1,800 sessions) |

The last four write artifacts into `results/`. The copies committed there were produced by
exactly those commands.
