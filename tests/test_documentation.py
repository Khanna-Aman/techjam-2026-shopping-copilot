"""The README's numbers must match the committed measurements.

This exists because they had drifted. `results/ablation.json` was generated before a commit
that changed how the search index is built, and was never regenerated; the README's "no
category lock" row carried the pre-change figure for several commits. Separately, the
question-policy finding claimed +0.044 -- a number that no committed artifact supported and
that turned out to be roughly three times the real effect.

Neither was caught by review, because verifying a documented number means re-running a
several-minute harness and comparing by eye. That is exactly the kind of check a machine
should be doing, so these tests parse the README's tables and assert them against
`results/*.json`.

If one of these fails, the fix is almost always to re-run the harness and commit the
regenerated result file alongside the documentation change -- not to edit the number until
the test passes.
"""

from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_README = (_ROOT / "README.md").read_text(encoding="utf-8")

#: README renders negatives with a Unicode minus and marks emphasis with asterisks.
_MINUS = "−"


def _load(name: str) -> dict:
    path = _ROOT / "results" / name
    if not path.exists():  # pragma: no cover - results are committed
        pytest.skip(f"results/{name} not present")
    return json.loads(path.read_text(encoding="utf-8"))


def _cells(row: str) -> list[str]:
    return [cell.strip().replace("*", "") for cell in row.strip().strip("|").split("|")]


def _round4(value: float) -> float:
    """Round half away from zero, deterministically.

    `round()` uses banker's rounding on the binary representation, so two float values that
    print identically can round to different fourth decimals -- which is exactly what
    happened to the override-erasure delta at -0.000850, where the ablation file and the
    bootstrap file disagreed on whether it was -0.0008 or -0.0009. A documented number
    should not depend on which artifact it was copied from.
    """
    return float(Decimal(repr(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _number(text: str) -> float:
    return float(text.replace(_MINUS, "-").replace(",", ""))


def _tables(heading: str) -> list[list[list[str]]]:
    """Every markdown table between ``heading`` and the next heading, body rows only.

    A section can hold more than one table -- the generalisation section carries a
    target-popularity comparison before the results themselves -- so callers pick by index
    rather than getting whichever came first.
    """
    start = _README.index(heading)
    body = _README[start:].splitlines()[1:]
    tables: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    header_pending = False

    for line in body:
        stripped = line.strip()
        if stripped.startswith("#"):
            break
        if stripped.startswith("|"):
            if set(stripped) <= set("|-: "):
                # The separator row: everything captured for this table so far was header.
                current = []
                header_pending = False
                tables.append(current)
                continue
            if current is None or header_pending:
                header_pending = True
                continue
            current.append(_cells(stripped))
        else:
            current = None
            header_pending = False
    return [table for table in tables if table]


def _table_rows(heading: str, index: int = 0) -> list[list[str]]:
    tables = _tables(heading)
    assert len(tables) > index, f"no table #{index} under {heading!r}"
    return tables[index]


# ------------------------------------------------------------------- headline figures
def test_headline_block_matches_the_official_result():
    official = _load("official_evaluation.json")
    block = _README[_README.index("official baseline") : _README.index("`TechnicalScore =")]

    def _stated(label: str) -> float:
        match = re.search(rf"{re.escape(label)}\s+[\d.]+\s+->\s+([\d.]+)", block)
        assert match, f"{label!r} row not found in the headline block"
        return float(match.group(1))

    assert _stated("Hit Rate@10") == official["hit_rate_at_10"]
    assert _stated("MRR") == official["mrr"]
    assert _stated("MTTC (turns)") == pytest.approx(official["mttc"])
    assert _stated("TechnicalScore") == official["recommended_technical_score"]


def test_headline_multiplier_is_arithmetically_true():
    """8.49x is a claim about two numbers that are both in the file; check the division."""
    official = _load("official_evaluation.json")
    baseline = json.loads((_ROOT / "docs" / "baseline_results.json").read_text(encoding="utf-8"))
    stated = float(re.search(r"\(([\d.]+)x\)", _README).group(1))
    actual = official["recommended_technical_score"] / baseline["technical_score"]
    assert round(actual, 2) == stated


# --------------------------------------------------------------------------- tables
def test_per_scenario_table_matches_the_official_result():
    official = _load("official_evaluation.json")
    rows = _table_rows("### Per scenario")
    assert rows, "per-scenario table not found"

    for row in rows:
        name, count, hit, mrr, mttc = row[0], row[1], row[2], row[3], row[4]
        source = (
            official if name == "overall" else official["scenario_metrics"][name]
        )
        key = "sample_count"
        assert int(count) == source[key], f"{name}: sample count"
        assert _number(hit) == pytest.approx(source["hit_rate_at_10"], abs=5e-5), f"{name}: hit rate"
        assert _number(mrr) == pytest.approx(source["mrr"], abs=5e-5), f"{name}: MRR"
        assert _number(mttc) == pytest.approx(source["mttc"], abs=5e-4), f"{name}: MTTC"


#: "[-0.0226, +0.0028] spans zero" -- the emphasis markers are stripped by `_cells`.
_INTERVAL = re.compile(r"\[\s*([^,\]]+),\s*([^\]]+?)\s*\]")


def _interval(text: str) -> tuple[float, float]:
    match = _INTERVAL.search(text)
    assert match, f"no interval found in {text!r}"
    return _number(match.group(1)), _number(match.group(2))


def test_precision_table_matches_the_committed_bootstrap():
    bootstrap = _load("bootstrap.json")
    rows = _table_rows("### How precise is")
    assert rows, "precision table not found"

    keys = {
        "TechnicalScore": "technical_score",
        "Hit@10": "hit_rate_at_10",
        "MRR": "mrr",
        "MTTC": "mttc",
    }
    documented = set()
    for label, point, interval, std_error in ((r[0], r[1], r[2], r[3]) for r in rows):
        key = keys[label]
        documented.add(key)
        stats = bootstrap["overall"]["metrics"][key]
        assert _number(point) == pytest.approx(stats["point"], abs=5e-5), f"{label}: point"
        low, high = _interval(interval)
        assert low == pytest.approx(stats["ci95_low"], abs=5e-5), f"{label}: CI low"
        assert high == pytest.approx(stats["ci95_high"], abs=5e-5), f"{label}: CI high"
        stated = _number(std_error.replace("±", ""))
        assert stated == pytest.approx(stats["std_error"], abs=5e-5), f"{label}: std err"

    assert documented == set(keys.values()), "the precision table omits a metric"


def test_the_bootstrap_was_taken_from_the_current_official_result():
    """A stale bootstrap.json would put a CI around a score nobody reports any more."""
    bootstrap = _load("bootstrap.json")
    official = _load("official_evaluation.json")
    assert bootstrap["source_technical_score"] == official["recommended_technical_score"]


def test_the_baseline_comparison_column_is_measured_not_quoted():
    """The per-scenario baseline column, against a run of the organiser's own baseline.

    `docs/baseline_results.json` publishes only the overall figure, so the per-scenario
    column used to be the one part of that table with no artifact behind it.
    `results/baseline_by_scenario.json` is a full run of `starter.baseline_agent` through
    the unmodified evaluator; its overall score reproduces the organiser's 0.10671 exactly,
    which is what makes the per-scenario breakdown trustworthy.
    """
    measured = _load("baseline_by_scenario.json")
    published = json.loads(
        (_ROOT / "docs" / "baseline_results.json").read_text(encoding="utf-8")
    )
    assert measured["technical_score"] == pytest.approx(
        published["technical_score"], abs=5e-6
    ), "our baseline run no longer reproduces the organiser's published score"

    rows = _table_rows("### Per scenario")
    for row in rows:
        name, baseline_hit = row[0], row[-1]
        if name == "overall":
            assert _number(baseline_hit) == pytest.approx(
                measured["hit_rate_at_10"], abs=5e-5
            )
            continue
        assert _number(baseline_hit) == pytest.approx(
            measured["scenario_metrics"][name]["hit_rate_at_10"], abs=5e-5
        ), f"{name}: baseline hit rate"


def test_ablation_table_matches_the_committed_ablation():
    ablation = _load("ablation.json")
    rows = _table_rows("### Ablation")
    assert rows, "ablation table not found"

    full = ablation["full system"]["technical_score"]
    documented = set()
    for label, score, delta in ((r[0], r[1], r[2]) for r in rows):
        # The README sometimes annotates a row, e.g. "no profile prior (cold start)".
        name = label if label in ablation else re.sub(r"\s*\(.*\)$", "", label)
        assert name in ablation, f"README documents {label!r}, which the harness does not run"
        documented.add(name)
        measured = ablation[name]["technical_score"]
        assert _number(score) == pytest.approx(measured, abs=5e-5), f"{name}: score"
        if delta != "—":
            assert _number(delta) == pytest.approx(measured - full, abs=5e-5), f"{name}: delta"

    missing = set(ablation) - documented
    assert not missing, f"the harness measures {sorted(missing)}, which the README omits"


def test_ablation_intervals_match_the_paired_bootstrap():
    """The CI column, and the "spans zero" labels, against the harness that produced them.

    The annotation is the load-bearing part: it is what tells a reader that half the table
    is unresolved at n=200. A row whose label disagrees with its own interval would be worse
    than no label, so the flag is asserted rather than assumed.
    """
    intervals = _load("ablation_ci.json")["ablations"]
    rows = _table_rows("### Ablation")
    assert rows, "ablation table not found"

    documented = set()
    for label, delta, interval in ((r[0], r[2], r[3]) for r in rows):
        if delta == "—":  # the reference row carries no delta and no interval
            assert interval == "—", "the reference row should not claim an interval"
            continue
        name = label if label in intervals else re.sub(r"\s*\(.*\)$", "", label)
        assert name in intervals, f"README documents {label!r}, which the harness does not run"
        documented.add(name)
        row = intervals[name]

        # The README shows four decimals, so the invariant is exact equality *after*
        # rounding -- a tolerance here would sit right on the boundary for a delta like
        # -0.00075 and would have to be loosened until it could hide real drift.
        assert _number(delta) == _round4(row["delta"]), f"{name}: delta"
        low, high = _interval(interval)
        assert low == _round4(row["ci95_low"]), f"{name}: CI low"
        assert high == _round4(row["ci95_high"]), f"{name}: CI high"

        labelled = "spans zero" in interval
        assert labelled != row["significant"], (
            f"{name}: the README {'labels' if labelled else 'does not label'} this row "
            f"'spans zero', but the bootstrap calls it "
            f"{'significant' if row['significant'] else 'unresolved'}"
        )

    missing = set(intervals) - documented
    assert not missing, f"the harness measures {sorted(missing)}, which the README omits"


def test_the_paraphrase_comparison_matches_both_bootstraps():
    """The clean/heavy table, including which rows each run actually resolves.

    This table is the one that reports a hypothesis failing, so its "resolved?" column is
    the claim -- get that backwards and the section argues the opposite of its own data.
    """
    clean = _load("ablation_ci.json")["ablations"]
    heavy = _load("ablation_ci_heavy.json")["ablations"]
    rows = _table_rows("#### Does paraphrase rescue")
    assert rows, "the clean/heavy comparison table not found"

    seen = set()
    for label, clean_delta, heavy_delta, resolved in (
        (r[0], r[1], r[2], r[3]) for r in rows
    ):
        name = label if label in clean else re.sub(r"\s*\(.*\)$", "", label)
        assert name in clean and name in heavy, f"unknown configuration {label!r}"
        seen.add(name)
        assert _number(clean_delta) == _round4(clean[name]["delta"]), f"{name}: clean"
        assert _number(heavy_delta) == _round4(heavy[name]["delta"]), f"{name}: heavy"

        expected = {
            (True, True): "both",
            (True, False): "clean only",
            (False, True): "heavy only",
            (False, False): "neither",
        }[(clean[name]["significant"], heavy[name]["significant"])]
        assert resolved == expected, (
            f"{name}: README says {resolved!r}, the bootstraps say {expected!r}"
        )

    assert seen == set(clean) == set(heavy), "the comparison omits a configuration"


def test_the_paraphrase_hypothesis_is_still_refuted():
    """The section's headline is that heavy paraphrase resolves *fewer* rows, not more.

    If a future change made paraphrase resolve more mechanisms, that prose would silently
    become false while every individual number in the table stayed correct.
    """
    clean = _load("ablation_ci.json")["ablations"]
    heavy = _load("ablation_ci_heavy.json")["ablations"]
    clean_resolved = sum(1 for row in clean.values() if row["significant"])
    heavy_resolved = sum(1 for row in heavy.values() if row["significant"])
    assert heavy_resolved < clean_resolved, "the README claims paraphrase resolves fewer rows"

    stated = re.search(
        r"(\w+) of \w+ rather than (\w+)\.", _README
    )
    assert stated, "the resolved-row counts are no longer phrased as expected"
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
    assert words[stated.group(1).lower()] == heavy_resolved
    assert words[stated.group(2).lower()] == clean_resolved

    # Every row unresolved on the clean set must still be unresolved under paraphrase --
    # the sentence "every row unresolved on the clean set is still unresolved" says so.
    for name, row in clean.items():
        if not row["significant"]:
            assert not heavy[name]["significant"], f"{name} became resolved under paraphrase"


def test_the_encoder_tier_table_matches_the_three_dense_bootstraps():
    """The claim is that a better encoder never flips the sign. Check each tier's numbers.

    This table exists to answer "you only tried a weak encoder", so every figure in it is
    load-bearing: the best row per tier, its interval, the harm at maximum weight, and how
    many rows each tier resolves. All four come straight from the committed artifacts.
    """
    tiers = {
        "truncated SVD (LSA)": _load("dense_ci_lsa.json")["ablations"],
        "`all-MiniLM-L6-v2`": _load("dense_ci_mini.json")["ablations"],
        "`BAAI/bge-base-en-v1.5`": _load("dense_ci_bge.json")["ablations"],
    }
    rows = _table_rows('#### "You only tried a weak encoder"')
    assert len(rows) == 3, f"expected three encoder tiers, found {len(rows)}"

    for encoder, _dim, best_label, delta, interval, worst, resolved in (
        (r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows
    ):
        ablations = tiers[encoder]

        # The row the README calls best must actually be the highest-scoring one measured.
        best_measured = max(ablations.values(), key=lambda row: row["delta"])
        assert _number(delta) == _round4(best_measured["delta"]), f"{encoder}: best delta"
        low, high = _interval(interval)
        assert low == _round4(best_measured["ci95_low"]), f"{encoder}: CI low"
        assert high == _round4(best_measured["ci95_high"]), f"{encoder}: CI high"

        assert _number(worst) == _round4(ablations["dense w=3.0"]["delta"]), f"{encoder}: w=3.0"

        count = sum(1 for row in ablations.values() if row["significant"])
        stated_count = 0 if resolved.split()[0] == "none" else int(resolved.split()[0])
        assert stated_count == count, f"{encoder}: resolved count"
        assert all(
            row["delta"] < 0 for row in ablations.values() if row["significant"]
        ), f"{encoder}: no resolved row may be a gain"


def test_no_dense_configuration_beats_the_offline_default():
    """The section's headline claim, stated as a property of all three artifacts."""
    for name in ("dense_ci_lsa.json", "dense_ci_mini.json", "dense_ci_bge.json"):
        for label, row in _load(name)["ablations"].items():
            assert not (row["significant"] and row["delta"] > 0), (
                f"{name}: {label} is a resolved *gain*, which would refute the section"
            )


def test_the_noise_floor_claim_counts_the_unresolved_rows():
    """"Five of the ten" is a count, and counts drift when a mechanism is added."""
    intervals = _load("ablation_ci.json")["ablations"]
    unresolved = sum(1 for row in intervals.values() if not row["significant"])
    stated = re.search(r"\*\*(\w+) of the (\w+) mechanisms are not distinguishable", _README)
    assert stated, "the noise-floor claim is no longer phrased as expected"
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11}
    assert words[stated.group(1).lower()] == unresolved
    assert words[stated.group(2).lower()] == len(intervals)


def test_robustness_table_matches_the_committed_robustness():
    robustness = _load("robustness.json")
    rows = _table_rows("### Robustness")
    assert rows, "robustness table not found"

    # The README labels the runs in prose; map them onto the harness's keys.
    labels = {
        "control": "control",
        "lowercase": "casing",
        "punctuation stripped": "punctuation",
        "light paraphrase": "light",
        "heavy paraphrase (+filler, +case drift)": "heavy",
    }
    for label, _before, after in ((r[0], r[1], r[2]) for r in rows):
        key = labels[label]
        assert _number(after) == pytest.approx(
            robustness[key]["technical_score"], abs=5e-5
        ), f"{label}: post-hardening score"


def test_generalisation_table_matches_the_proxy_private_run():
    proxy = _load("proxy_private.json")
    official = _load("official_evaluation.json")
    # The section leads with a target-popularity comparison; the results are the second.
    rows = _table_rows("### Generalisation", index=1)

    for row in rows:
        label, count, score = row[0], row[1], row[2]
        if label.startswith("public"):
            assert int(count) == official["sample_count"]
            assert _number(score) == pytest.approx(
                official["recommended_technical_score"], abs=5e-5
            )
            continue
        key = "matched" if label.startswith("matched") else "uniform"
        assert int(count) == proxy[key]["sample_count"], f"{key}: sample count"
        assert _number(score) == pytest.approx(
            proxy[key]["technical_score"], abs=5e-5
        ), f"{key}: score"
        low, high = row[3].strip("[]").split(",")
        assert _number(low) == pytest.approx(proxy[key]["ci95_low"], abs=5e-5)
        assert _number(high) == pytest.approx(proxy[key]["ci95_high"], abs=5e-5)


def test_feasibility_table_matches_the_latency_harness():
    """Latency and memory, against `results/latency.json` rather than a fresh measurement.

    Comparing against the committed artifact rather than re-timing is deliberate: these
    figures are machine-dependent, so a test that measured would fail on CI for reasons
    that have nothing to do with the documentation being wrong. What must hold is that the
    README quotes the run it committed.
    """
    latency = _load("latency.json")
    rows = {row[0]: row[1] for row in _table_rows("### Feasibility")}

    scored = latency["per_turn_ms"]["scored_loop"]
    exhaustive = latency["per_turn_ms"]["all_ten_turns"]
    stated = rows["Per-turn latency"]
    numbers = [float(n) for n in re.findall(r"([\d.]+) ms", stated)]
    assert len(numbers) >= 6, f"expected six latency figures, parsed {numbers}"
    expected = [
        round(scored["median"]), round(scored["p95"]), round(scored["max"]),
        round(exhaustive["median"]), round(exhaustive["p95"]), round(exhaustive["max"]),
    ]
    assert [round(n) for n in numbers[:6]] == expected, (
        f"README quotes {numbers[:6]}, the harness measured {expected}"
    )

    memory = rows["Memory"]
    stated_mb = float(re.search(r"([\d.]+) MB", memory).group(1))
    assert round(stated_mb) == round(latency["memory_mb"]["agent_only_resident"])

    build = latency["index_build_seconds"]
    stated_build = [float(n) for n in re.findall(r"([\d.]+) s", rows["Index build"])]
    assert len(stated_build) == 2, f"expected cold and warm build times, got {stated_build}"
    assert round(stated_build[0]) == round(build["cold"])
    assert abs(stated_build[1] - build["warm_from_cache"]) < 0.5


# ------------------------------------------------------- claims made in the prose
def test_the_shipped_yield_prior_is_the_one_derived_from_the_catalog():
    """The seven constants in `copilot/question.py` must be the catalog's numbers.

    This is the answer to "you fitted your priors to the public set". The constants are
    hardcoded for speed -- deriving them costs a full catalog pass at import time -- but
    they are not free parameters, and this asserts it: each must equal the value
    `tools/yield_prior.py` measures, to the three decimals the source rounds to.
    """
    from copilot.question import YIELD_PROBABILITY

    report = _load("yield_prior.json")
    assert report["products"] == 50000, "the prior must come from the whole catalog"

    for attribute, measured in report["measured"].items():
        shipped = YIELD_PROBABILITY[attribute]
        assert abs(measured - shipped) <= 5e-4, (
            f"{attribute}: ships {shipped}, catalog gives {measured}"
        )
    assert set(report["measured"]) == set(YIELD_PROBABILITY)


def test_the_yield_table_in_the_readme_matches_the_derivation():
    report = _load("yield_prior.json")
    rows = _table_rows("### 3. A question that splits the pool")
    assert rows, "the P(yield) table was not found"

    header = _README[_README.index("### 3. A question that splits the pool"):]
    header_row = next(l for l in header.splitlines() if l.strip().startswith("| attribute"))
    attributes = _cells(header_row)[1:]
    values = [_number(cell) for cell in rows[0][1:]]
    assert len(attributes) == len(values), "the P(yield) table is malformed"

    for attribute, stated in zip(attributes, values):
        key = "color" if attribute == "colour" else attribute
        assert stated == pytest.approx(report["measured"][key], abs=5e-4), attribute


def test_the_question_policy_claim_matches_the_strategy_sweep():
    """Finding 3's headline number is hybrid minus infogain, and must stay that.

    The comparison is named carefully. `infogain` is not an entropy-only policy -- it scores
    with the same expected-value product the default uses and merely refuses to ask the open
    question (see `copilot/question.py`), so describing it as "entropy-only" overstated what
    the ablation isolates. The README says "specific-questions-only", and this pins the
    phrasing so it cannot drift back.
    """
    strategies = _load("clarification_strategies.json")
    hybrid = strategies["strategy: hybrid"]["technical_score"]
    infogain = strategies["strategy: infogain"]["technical_score"]
    stated = re.search(
        r"Worth\s+\*\*\+([\d.]+)\*\*\s+over the specific-questions-only policy", _README
    )
    assert stated, "the question-policy claim is no longer phrased as expected"
    assert float(stated.group(1)) == pytest.approx(hybrid - infogain, abs=5e-4)
    assert "entropy-only" not in _README, (
        "`infogain` is not an entropy-only policy; it uses the same expected-value product "
        "and only declines the open question"
    )


def test_the_question_policy_interval_matches_the_paired_bootstrap():
    """The README quotes this interval with the sign flipped, deliberately.

    `tools/ablation_ci.py` always measures `variant - default`, so the infogain row is
    negative: the specific-questions-only policy is *worse* than the shipped one. Finding 3 states the
    same fact the other way up -- what the expected-value policy is worth *over* infogain --
    so the documented bounds are the negated, swapped tool output. That flip is easy to get
    backwards in an edit, which is why it is pinned here rather than left to review.
    """
    row = _load("strategy_ci.json")["ablations"]["strategy: infogain"]
    assert row["delta"] < 0, "infogain is expected to score below the shipped default"

    stated = re.search(
        r"95% CI \[\+([\d.]+), \+([\d.]+)\], reproduced by", _README
    )
    assert stated, "the question-policy interval is no longer phrased as expected"
    low, high = float(stated.group(1)), float(stated.group(2))
    assert low < high, "the README interval is inverted"
    assert low == pytest.approx(-row["ci95_high"], abs=5e-5)
    assert high == pytest.approx(-row["ci95_low"], abs=5e-5)
    assert row["significant"], "the README calls this interval decisive; the data must agree"


def test_the_hybrid_ties_open_claim_is_still_true():
    """The README says the escalation branch earns nothing here. If that changes, say so."""
    strategies = _load("clarification_strategies.json")
    assert (
        strategies["strategy: hybrid"]["technical_score"]
        == strategies["strategy: open"]["technical_score"]
    ), "hybrid no longer ties open -- the README paragraph saying it does needs rewriting"


def test_the_profile_timing_table_matches_the_profile_sweep():
    profile = _load("sweep_profile.json")
    off = profile["profile off"]["technical_score"]
    rows = {r[0]: r for r in _table_rows("### 4. Personalization")}

    assert _number(rows["profile off"][1]) == pytest.approx(off, abs=5e-5)

    always = profile["always     w=1.0"]["technical_score"]
    cold = profile["cold-start w=1.0"]["technical_score"]
    assert _number(rows["applied always"][1]) == pytest.approx(always, abs=5e-5)
    assert _number(rows["applied always"][2]) == pytest.approx(always - off, abs=5e-5)
    assert _number(rows["applied at cold start only"][1]) == pytest.approx(cold, abs=5e-5)
    assert _number(rows["applied at cold start only"][2]) == pytest.approx(cold - off, abs=5e-5)


def test_the_sign_of_the_personalization_finding_holds():
    """Global personalization must stay clearly harmful; cold start must stay ~neutral.

    The original finding was that the sign flips with timing -- harmful applied globally,
    helpful applied at cold start. The confidence gate removed the second half: the
    cold-start prior is now worth about -0.0008, inside the noise floor. The half that
    still holds is asserted strictly, and the half that changed is asserted as the weaker
    claim the README now actually makes.
    """
    profile = _load("sweep_profile.json")
    off = profile["profile off"]["technical_score"]
    always = profile["always     w=1.0"]["technical_score"]
    cold = profile["cold-start w=1.0"]["technical_score"]

    assert always < off - 0.02, "global personalization is no longer clearly harmful"
    assert abs(cold - off) < 0.005, (
        "the cold-start prior has moved out of the noise floor; finding 4 needs rewriting"
    )
    assert cold > always, "timing must still matter"


def test_the_popularity_skew_figures_are_true_of_the_actual_catalog():
    """The recalibration argument rests on four numbers about the data, not about the agent.

    "Targets sit at the 99.4th percentile", "86.5% are in the top decile", "uniform sampling
    draws a sub-100-rating target 81% of the time" and "the real split does so 5% of the
    time" are quoted in the README, in DEVPOST, in the demo narration and in the comment
    justifying `w_popularity` in `copilot/config.py`. They are claims about `data/catalog.jsonl`
    and `data/public_set.jsonl`, so they are checkable, and if they drift the whole argument
    for the weight goes with them.
    """
    import bisect

    catalog = _ROOT / "data" / "catalog.jsonl"
    if not catalog.exists():
        pytest.skip("data/catalog.jsonl not present")

    from evaluator.local_evaluator import catalog_index, load_jsonl

    _ids, _cats, products = catalog_index(str(catalog))
    targets = [
        str(row["ground_truth"]["parent_asin"])
        for row in load_jsonl(str(_ROOT / "data" / "public_set.jsonl"))
    ]

    def ratings(product: dict) -> int:
        try:
            return int(product.get("rating_number", 0) or 0)
        except (TypeError, ValueError):
            return 0

    catalog_counts = sorted(ratings(p) for p in products.values())
    target_counts = [ratings(products[a]) for a in targets if a in products]

    percentiles = [
        100.0 * bisect.bisect_left(catalog_counts, c) / len(catalog_counts)
        for c in target_counts
    ]
    percentiles.sort()
    median_percentile = percentiles[len(percentiles) // 2]
    top_decile = 100.0 * sum(1 for p in percentiles if p > 90) / len(percentiles)

    assert median_percentile > 98.0, (
        f"targets sit at the {median_percentile:.1f}th percentile, not ~99.4th; the "
        "README's popularity-skew argument no longer holds"
    )
    assert top_decile > 80.0, f"only {top_decile:.1f}% of targets are in the top decile"

    catalog_obscure = 100.0 * sum(1 for c in catalog_counts if c < 100) / len(catalog_counts)
    target_obscure = 100.0 * sum(1 for c in target_counts if c < 100) / len(target_counts)
    assert catalog_obscure > 75.0, (
        f"uniform catalog sampling now draws a sub-100-rating target {catalog_obscure:.0f}% "
        "of the time; the README says ~81%"
    )
    assert target_obscure < 10.0, (
        f"the real split now draws one {target_obscure:.0f}% of the time; README says ~5%"
    )


def test_the_popularity_recalibration_is_supported_by_the_corrected_proxy():
    """`w_popularity` 0.55 -> 1.2 was rejected, then adopted. The evidence must be committed.

    The comparison lives inside one artifact rather than two, and that is deliberate: both
    configurations are run over the *same* synthesised sessions and the difference is
    bootstrapped pairwise. Two separate runs would draw different targets, and comparing
    their marginal intervals is the weaker test -- those intervals overlap here while the
    paired one does not, which is exactly the trap the original rejection fell into.

    It also asserts the part that argues against us. The uniform regime still prefers the
    old weight, and that is the honest content of the finding: popularity is evidence *for
    this sampling scheme* and a liability outside it.
    """
    proxy = _load("proxy_private.json")
    sweep = _load("sweep_popularity.json")

    assert sweep["w_pop=1.2"]["technical_score"] > sweep["w_pop=0.55"]["technical_score"], (
        "the adopted weight is supposed to score better on the public set"
    )

    matched = proxy["matched"]["paired_comparison"]
    assert matched["against"] == {"w_popularity": 0.55}, (
        "the committed comparison is no longer against the previous weight"
    )
    assert matched["ci95_low"] > 0, (
        f"the paired gain on popularity-matched held-out targets no longer excludes zero "
        f"({matched['delta']:+.4f} [{matched['ci95_low']:+.4f}, {matched['ci95_high']:+.4f}]); "
        "the README rests the reversal on this interval"
    )
    assert matched["improved"] > matched["regressed"] * 2, (
        "the gain is no longer broad: it should improve far more sessions than it breaks, "
        "or it is the kind of concentrated effect that signals overfitting"
    )

    uniform = proxy["uniform"]["paired_comparison"]
    assert uniform["ci95_high"] < 0, (
        "the uniform stress regime is supposed to still prefer the old weight; if that "
        "stopped being true the README overstates the trade-off it discloses"
    )


def test_the_matched_proxy_is_no_longer_sample_capped():
    """The bug that made the original rejection unfalsifiable must not come back.

    `sample_matched` used to take an equal count from each of ten popularity deciles, so it
    was capped at ten times the smallest decile -- 110 targets, with a 95% interval 0.046
    wide, being asked to resolve an effect of eight thousandths. Any future change that
    reintroduces a per-decile cap will drop this number back to ~110 and fail here.
    """
    proxy = _load("proxy_private.json")
    assert proxy["matched"]["sample_count"] > 110, (
        f"the matched regime is back down to {proxy['matched']['sample_count']} samples; "
        "the decile cap that made w_popularity=1.2 look like a non-result has returned"
    )


def test_the_shipped_popularity_weight_is_still_the_documented_one():
    from copilot.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.w_popularity == 1.20, (
        "the README documents the recalibration to 1.20"
    )


def test_the_residual_is_saturated_ties_and_not_ranking_headroom():
    """The README argues the remaining MRR gap cannot be taken. That is a measurement.

    If a product ever outranks a target while satisfying *fewer* of the disclosed
    constraints, that is a genuine scoring defect and the claim "no reranker can fix this"
    becomes false. The number to watch is `rivals_mis_scored`: it must stay at zero, and
    every rival above a target must be in a saturated tie.
    """
    summary = _load("coverage_ceiling.json")["summary"]

    assert summary["rivals_mis_scored"] == 0, (
        f"{summary['rivals_mis_scored']} products outrank a target while matching fewer "
        "constraints; that is fixable headroom and the ceiling argument needs rewriting"
    )
    assert summary["rivals_in_saturated_ties"] == summary["rivals_above_targets"], (
        "not every rival above a target is a saturated tie any more; finding the residual "
        "unreachable is no longer supported by the measurement"
    )


def test_the_gate_setting_sits_on_a_flat_region():
    """Finding 5 says the gate thresholds were "chosen from a flat region, not an argmax".

    That is a claim about a grid, so it needs the grid. The README quotes the spread and
    concedes the shipped setting is not the top of it; both must stay true, and the spread
    must stay under the standard error on the score, or "flat" is the wrong word.
    """
    plateau = _load("sweep_gate_plateau.json")
    bootstrap = _load("bootstrap.json")["overall"]["metrics"]["technical_score"]
    scores = {k: v["technical_score"] for k, v in plateau.items()}
    assert len(scores) == 9, f"expected a 3x3 grid, found {len(scores)} rows"

    spread = max(scores.values()) - min(scores.values())
    assert spread < bootstrap["std_error"], (
        f"the gate grid spans {spread:.4f}, wider than the {bootstrap['std_error']:.4f} "
        "standard error; it is no longer a flat region and finding 5 needs rewriting"
    )
    shipped = scores["min_c=4 max_turn=3"]
    assert max(scores.values()) - shipped < 0.005, (
        "the shipped gate setting has fallen measurably behind the best cell in the grid"
    )
    assert "the shipped setting is not the" in _README, (
        "the README no longer concedes that the shipped gate setting is not the argmax"
    )


def test_the_confidence_gate_defaults_match_what_the_readme_documents():
    """Finding 5 names specific settings; the shipped config must be those settings."""
    from copilot.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.use_confidence_gate is True
    assert DEFAULT_CONFIG.gate_list_size == 1, "the README argues one beats none"
    assert DEFAULT_CONFIG.gate_max_turn > 0, "an uncapped gate scores 0.0000"
    assert 4 <= DEFAULT_CONFIG.gate_min_constraints <= 6, "outside the documented plateau"


def test_the_override_retention_finding_holds():
    """Full erasure must remain the worst setting, or finding 2 is no longer true."""
    sweep = _load("sweep_override_decay.json")
    scores = {k: v["technical_score"] for k, v in sweep.items()}
    assert scores["override_decay=0.0"] == min(scores.values()), "erasure is no longer worst"

    # The finding is "erasure is worst", not "0.5 is the argmax" -- and since the popularity
    # recalibration 0.5 is no longer the argmax, though it trails by 0.0001. Asserting the
    # argmax here would demand chasing a difference this document calls noise everywhere
    # else. What must stay true is that the shipped value is not meaningfully behind the
    # best one, and that the README says so rather than implying 0.5 wins.
    shipped = scores["override_decay=0.5"]
    assert max(scores.values()) - shipped < 0.005, (
        "override_decay=0.5 has fallen measurably behind the best setting; finding 2 needs "
        "revisiting rather than a comment"
    )
    assert "0.5 is not" in _README, (
        "the README no longer discloses that 0.5 is not the argmax"
    )


# ------------------------------------ guards added after the second audit (30 Aug, pm)
#
# Each of these binds a claim that the first audit's tests did not reach. Every one of them
# fails against the repository as it stood before that audit, which is the only evidence
# that a regression test is worth committing.


def test_every_sweep_row_at_the_shipped_default_reproduces_the_official_score():
    """A sweep row labelled "(default)" has to actually be the default.

    `_pop_grid` used to pin `w_profile` to 0.0 on every row, so the row the README quoted as
    the default's score read 0.9643 while the shipped agent scores 0.9633. The existing
    popularity test only asserted an ordering (1.2 beats 0.55), which the mis-configured
    sweep satisfied, so nothing caught it for two sessions.

    The check is structural rather than a hard-coded list of numbers: find every sweep row
    whose overrides are *all* equal to the shipped configuration, and demand it reproduce
    the official evaluation exactly. A row that sets only defaults and scores something else
    is measuring a configuration it does not name.
    """
    from copilot.config import DEFAULT_CONFIG

    official = _load("official_evaluation.json")["recommended_technical_score"]
    missing = object()
    checked: list[str] = []

    for path in sorted((_ROOT / "results").glob("sweep_*.json")) + [
        _ROOT / "results" / "clarification_strategies.json"
    ]:
        sweep = json.loads(path.read_text(encoding="utf-8"))
        for label, row in sweep.items():
            overrides = row.get("overrides") if isinstance(row, dict) else None
            if not isinstance(overrides, dict) or not overrides:
                continue
            if any(getattr(DEFAULT_CONFIG, key, missing) != value
                   for key, value in overrides.items()):
                continue
            checked.append(f"{path.name}:{label}")
            assert row["technical_score"] == pytest.approx(official, abs=5e-7), (
                f"{path.name} row {label!r} overrides only shipped defaults ({overrides}) "
                f"but scores {row['technical_score']} rather than the official {official}. "
                "The sweep is not being run at the configuration it claims to describe, so "
                "every row in it is measuring something the README does not say it is."
            )

    assert len(checked) >= 5, (
        f"expected the sweeps to contain several default-equivalent rows, found {checked}"
    )


def test_no_sweep_silently_pins_a_parameter_it_does_not_claim_to_vary():
    """The other half of the same bug, and the half the check above cannot see.

    `_pop_grid` set ``w_profile=0.0`` on all eight rows. That did not make any row score
    wrongly *for its own overrides* -- it made every row describe a configuration nobody
    ships, including the one the README labels "(default)". A row carrying a non-default
    pin is simply skipped by the default-equivalence test above, so that test passes
    against the bug; this one is what fails.

    The rule: within one sweep, a parameter held at the same non-default value on *every*
    row is a hidden constant. A sweep varies what it names and inherits everything else.
    """
    from copilot.config import DEFAULT_CONFIG

    missing = object()
    absent = "<absent>"
    offenders: list[str] = []

    for path in sorted((_ROOT / "results").glob("sweep_*.json")) + [
        _ROOT / "results" / "clarification_strategies.json",
        _ROOT / "results" / "ablation.json",
    ]:
        if not path.exists():  # pragma: no cover - results are committed
            continue
        sweep = json.loads(path.read_text(encoding="utf-8"))
        rows = [
            row["overrides"] for row in sweep.values()
            if isinstance(row, dict) and isinstance(row.get("overrides"), dict)
        ]
        if len(rows) < 2:
            continue
        for key in sorted(set().union(*(set(row) for row in rows))):
            values = {json.dumps(row.get(key, absent), sort_keys=True) for row in rows}
            if len(values) != 1:
                continue  # the sweep varies it, which is the point
            value = json.loads(values.pop())
            if value == absent or getattr(DEFAULT_CONFIG, key, missing) == value:
                continue
            offenders.append(f"{path.name}: {key}={value!r} on all {len(rows)} rows")

    assert not offenders, (
        "a sweep holds a parameter at a non-default value on every row, so the whole "
        "grid describes a configuration that is never shipped: "
        + "; ".join(offenders)
        + ". This is how the popularity sweep came to report the default as 0.9643 "
          "when the agent scores 0.9633."
    )


def test_the_quoted_public_target_median_matches_the_artifact_that_produces_it():
    """6,846 is quoted in the README, in `config.py` and in the harness's own docstring.

    `popularity_summary` returned the lower of the two middle values on an even-sized
    sample, so `results/proxy_private.json` said 6,614 while every document said 6,846.
    Session 4 corrected the prose and not the tool, which left the artifact refuting the
    documents it exists to support. Whatever the number is, the file and the prose have to
    agree on it.
    """
    proxy = _load("proxy_private.json")
    median = proxy["public_reference"]["target_popularity"]["median_rating_number"]
    rendered = f"{median:,}"

    sources = {
        "README.md": _README,
        "copilot/config.py": (_ROOT / "copilot" / "config.py").read_text(encoding="utf-8"),
        "tools/proxy_private.py": (
            _ROOT / "tools" / "proxy_private.py"
        ).read_text(encoding="utf-8"),
    }
    for name, text in sources.items():
        assert rendered in text, (
            f"{name} does not quote the public-target median {rendered} that "
            f"results/proxy_private.json reports; the artifact and the prose have drifted "
            "apart again"
        )


def test_the_paraphrase_worst_case_in_finding_6_matches_the_robustness_run():
    """Finding 6's headline sentence carried 0.882 while its own table said 0.9342.

    The sentence is the first statement of the result a reader meets, 200 lines above the
    table that contradicts it, and no test reached it because it is prose rather than a
    row.
    """
    robustness = _load("robustness.json")
    before, after = re.search(
        r"worst case from \*\*([\d.]+) to ([\d.]+)\*\*", _README
    ).groups()

    heavy = robustness["heavy"]["technical_score"]
    assert float(after) == pytest.approx(heavy, abs=5e-4), (
        f"finding 6 says the worst case is now {after}; `results/robustness.json` measures "
        f"heavy paraphrase at {heavy:.4f}"
    )

    # The "before" number is the historical column, which cannot be regenerated -- but it
    # is tabulated in this document, so the sentence must at least match the table.
    row = {r[0]: r for r in _table_rows("### Robustness")}
    tabulated = _number(row["heavy paraphrase (+filler, +case drift)"][1])
    assert float(before) == pytest.approx(tabulated, abs=5e-4), (
        f"finding 6 says the worst case was {before}; the robustness table says {tabulated}"
    )


def test_the_prose_index_build_time_agrees_with_the_feasibility_table():
    """The reproduce section said ~19 s where the compliance table says ~25 s.

    Both describe one number that `results/latency.json` measures, and session 4 corrected
    the table without noticing the prose.
    """
    cold = _load("latency.json")["index_build_seconds"]["cold"]
    stated = re.search(
        r"builds an index and caches it under `artifacts/` \(~(\d+) s, one time\)", _README
    )
    assert stated is not None, "the reproduce section no longer states an index build time"
    assert int(stated.group(1)) == round(cold), (
        f"the reproduce section says ~{stated.group(1)} s to build the index; the harness "
        f"measured {cold:.1f} s and the feasibility table quotes ~{round(cold)} s"
    )


def test_the_intent_override_floor_is_derived_from_the_data_not_assumed():
    """A hit cannot register in an override session until the customer revises their intent.

    `evaluate()` opens an override session with `override_applied = False` and gates the hit
    check on it, so a correct answer at turn 1 or 2 is discarded outright. That puts a hard
    floor under this scenario's MTTC, equal to the mean override turn in the data. The
    README used to call it "~3.5", which is the floor of a 50/50 turn-3/turn-4 split; the
    actual split here is 12/18, so the floor is 3.600 and the claim was loose in the
    direction that flattered us less.

    Derived from `data/public_set.jsonl` through the organiser's own `materialize_hidden_fields`,
    so it tracks the data rather than a number typed into a document.
    """
    catalog = _ROOT / "data" / "catalog.jsonl"
    if not catalog.exists():
        pytest.skip("data/catalog.jsonl not present")

    from evaluator.local_evaluator import (
        catalog_index,
        load_jsonl,
        materialize_hidden_fields,
    )

    _ids, _cats, products = catalog_index(str(catalog))
    turns = [
        int(materialize_hidden_fields(row, products)[1]["override"]["turn"])
        for row in load_jsonl(str(_ROOT / "data" / "public_set.jsonl"))
        if row["scenario_type"] == "intent_override"
    ]
    assert turns, "no intent_override sessions found in the public set"
    floor = sum(turns) / len(turns)

    assert f"{floor:.3f}" in _README, (
        f"the README no longer states the intent-override MTTC floor as {floor:.3f}; it is "
        "the mean override turn across the public set and bounds what any agent can score"
    )

    official = _load("official_evaluation.json")
    measured = official["scenario_metrics"]["intent_override"]["mttc"]
    assert measured >= floor, (
        f"intent_override MTTC {measured} is below the structural floor {floor:.3f}, which "
        "is impossible — either the evaluator changed or the floor is being derived wrongly"
    )
    assert f"{measured:.3f}" in _README, (
        f"the README no longer states the measured intent-override MTTC {measured:.3f} "
        "alongside the floor it is being compared against"
    )


def test_no_override_session_scores_before_the_customer_revises():
    """The floor above is only real if the evaluator actually enforces it. Check the records.

    If a committed session shows an override hit at turn 1 or 2, then `override_applied` is
    not gating what the README says it gates, and the whole paragraph is wrong.
    """
    official = _load("official_evaluation.json")
    early = [
        s for s in official["sessions"]
        if s["scenario_type"] == "intent_override"
        and s["first_hit_turn"] is not None
        and s["first_hit_turn"] < 3
    ]
    assert not early, (
        f"{len(early)} intent_override sessions recorded a hit before turn 3, which the "
        "evaluator's override gate is supposed to make impossible"
    )
