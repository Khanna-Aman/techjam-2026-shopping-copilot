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
        assert _number(delta) == round(row["delta"], 4), f"{name}: delta"
        low, high = _interval(interval)
        assert low == round(row["ci95_low"], 4), f"{name}: CI low"
        assert high == round(row["ci95_high"], 4), f"{name}: CI high"

        labelled = "spans zero" in interval
        assert labelled != row["significant"], (
            f"{name}: the README {'labels' if labelled else 'does not label'} this row "
            f"'spans zero', but the bootstrap calls it "
            f"{'significant' if row['significant'] else 'unresolved'}"
        )

    missing = set(intervals) - documented
    assert not missing, f"the harness measures {sorted(missing)}, which the README omits"


def test_the_noise_floor_claim_counts_the_unresolved_rows():
    """"Five of the ten" is a count, and counts drift when a mechanism is added."""
    intervals = _load("ablation_ci.json")["ablations"]
    unresolved = sum(1 for row in intervals.values() if not row["significant"])
    stated = re.search(r"\*\*(\w+) of the (\w+) mechanisms are not distinguishable", _README)
    assert stated, "the noise-floor claim is no longer phrased as expected"
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10}
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


# ------------------------------------------------------- claims made in the prose
def test_the_question_policy_claim_matches_the_strategy_sweep():
    """The +0.015 in finding 3 is hybrid minus infogain, and must stay that."""
    strategies = _load("clarification_strategies.json")
    hybrid = strategies["strategy: hybrid"]["technical_score"]
    infogain = strategies["strategy: infogain"]["technical_score"]
    stated = re.search(r"Worth\s+\*\*\+([\d.]+)\*\*\s+over the entropy-only policy", _README)
    assert stated, "the question-policy claim is no longer phrased as expected"
    assert float(stated.group(1)) == pytest.approx(hybrid - infogain, abs=5e-4)


def test_the_question_policy_interval_matches_the_paired_bootstrap():
    """The README quotes this interval with the sign flipped, deliberately.

    `tools/ablation_ci.py` always measures `variant - default`, so the infogain row is
    negative: the entropy-only policy is *worse* than the shipped one. Finding 3 states the
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
    """The finding is that the sign flips with timing. If it stops flipping, it is wrong."""
    profile = _load("sweep_profile.json")
    off = profile["profile off"]["technical_score"]
    assert profile["always     w=1.0"]["technical_score"] < off
    assert profile["cold-start w=1.0"]["technical_score"] > off


def test_the_override_retention_finding_holds():
    """Full erasure must remain the worst setting, or finding 2 is no longer true."""
    sweep = _load("sweep_override_decay.json")
    scores = {k: v["technical_score"] for k, v in sweep.items()}
    assert scores["override_decay=0.0"] == min(scores.values()), "erasure is no longer worst"
    best = max(scores, key=lambda key: scores[key])
    assert best == "override_decay=0.5", f"the tuned default is no longer best: {best}"
