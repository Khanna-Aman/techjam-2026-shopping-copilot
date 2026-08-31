"""The judge-facing documents must carry the same numbers as the committed measurements.

`tests/test_documentation.py` guards the README and has caught three hand-copied errors.
`DEVPOST.md` and `DEMO_WALKTHROUGH.md` had no guard at all, which is the more dangerous gap:
the README is read by people who can also run the code, whereas the Devpost text and the
demo narration are read and *heard* by judges who cannot. A stale number there is a claim
nobody can check and I cannot retract once the video is uploaded.

These tests parse the headline blocks out of both documents and assert them against
`results/official_evaluation.json`. They are deliberately shallow -- headline metrics and the
baseline multiple, not every table -- because those are the numbers spoken aloud.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results"


def _official() -> dict:
    path = _RESULTS / "official_evaluation.json"
    if not path.exists():
        pytest.skip("results/official_evaluation.json not present")
    return json.loads(path.read_text(encoding="utf-8"))


def _doc(name: str) -> str:
    path = _ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return path.read_text(encoding="utf-8")


def _numbers_after(text: str, label: str) -> list[float]:
    """Every number on the first line that starts with ``label``."""
    for line in text.splitlines():
        if line.strip().startswith(label):
            return [float(value) for value in re.findall(r"\d+\.\d+|\d+", line)]
    raise AssertionError(f"no line starting with {label!r}")


def test_devpost_headline_metrics_match_the_official_result():
    official = _official()
    devpost = _doc("DEVPOST.md")

    # Each line reads "<label> <baseline> -> <ours>", so ours is the last number, except
    # TechnicalScore which also carries the multiple in parentheses.
    assert _numbers_after(devpost, "Hit Rate@10")[-1] == pytest.approx(
        official["hit_rate_at_10"], abs=5e-4
    ), "DEVPOST Hit@10 disagrees with results/official_evaluation.json"

    assert _numbers_after(devpost, "MRR")[-1] == pytest.approx(
        official["mrr"], abs=5e-6
    ), "DEVPOST MRR disagrees with results/official_evaluation.json"

    assert _numbers_after(devpost, "MTTC")[-1] == pytest.approx(
        official["mttc"], abs=5e-4
    ), "DEVPOST MTTC disagrees with results/official_evaluation.json"


def test_devpost_technical_score_and_multiple_are_arithmetically_true():
    official = _official()
    devpost = _doc("DEVPOST.md")
    values = _numbers_after(devpost, "TechnicalScore")
    baseline, ours, multiple = values[0], values[1], values[-1]

    assert ours == pytest.approx(official["recommended_technical_score"], abs=5e-6), (
        "DEVPOST TechnicalScore disagrees with results/official_evaluation.json"
    )
    assert multiple == pytest.approx(ours / baseline, abs=0.02), (
        f"DEVPOST claims {multiple}x over {baseline}, but {ours}/{baseline} "
        f"is {ours / baseline:.2f}x"
    )


def test_the_demo_script_quotes_the_current_score():
    """The narration is spoken on camera, so a stale figure here cannot be corrected."""
    official = _official()
    walkthrough = _doc("DEMO_WALKTHROUGH.md")
    score = official["recommended_technical_score"]

    assert f"{score:.6f}" in walkthrough, (
        f"DEMO_WALKTHROUGH.md does not mention the current score {score:.6f}; "
        "the narration would state a number the repository no longer produces"
    )
    assert f"{official['mrr']:.6f}" in walkthrough, (
        f"DEMO_WALKTHROUGH.md does not mention the current MRR {official['mrr']:.6f}"
    )


#: Full-precision scores this submission has shipped and retired. Add to this list when a
#: score is superseded; never remove from it.
_SUPERSEDED_SCORES = ("0.906151", "0.954756")

#: Retired figures that are not scores, with what replaced them. These are what the second
#: audit found: ablation deltas, totals and an MTTC that no test reached because they live
#: in prose rather than in a parsed table, and that contradicted their own documents.
_RETIRED_FIGURES = {
    "+0.478": "the clarification delta, now +0.395",
    "+0.848": "the total improvement, now +0.857",
    "0.852 of": "the two-mechanism share, now 0.713 of 0.857",
    "-0.340": "the state-tracking delta, now -0.319",
    "-0.418": "the clarification delta, now -0.395",
    "1.93 turns": "a superseded MTTC; it is 2.19",
    "-0.0502": "the heavy-paraphrase gate delta, now -0.0497",
    "-0.0128": "the uniform held-out delta, now -0.0126",
    "+0.0088": "the public popularity gain, now +0.0086",
}


def test_no_submission_document_still_quotes_a_superseded_score():
    """A retired score must not appear in a judge-facing document at all.

    The previous version of this test could not fail. It fired only when a stale score
    appeared *without* the current one -- but `test_the_demo_script_quotes_the_current_score`
    already asserts the current score is present, so the inner assertion was satisfied
    unconditionally. Appending both retired scores to both documents left all four tests
    green, which is how it was found.

    The rule is now absence. If a document ever needs to discuss a retired number on
    purpose, the honest fix is to name it here with its reason, not to loosen the test.
    """
    current = f"{_official()['recommended_technical_score']:.6f}"

    for name in ("DEVPOST.md", "DEMO_WALKTHROUGH.md"):
        text = _doc(name)
        for stale in _SUPERSEDED_SCORES:
            if stale == current:
                continue
            assert stale not in text, (
                f"{name} still quotes the superseded score {stale}. The current score is "
                f"{current}. If the mention is deliberate, allowlist it here with a reason "
                "rather than weakening the assertion."
            )


def test_no_submission_document_still_quotes_a_retired_figure():
    """The same rule for the non-score figures the second audit found stale.

    DEVPOST carried +0.478 / +0.848 for the clarification finding, -0.340 / -0.418 in the
    compliance table, "0.852 of the 0.848" in the transfer section and an MTTC of 1.93 --
    each contradicted by that document's own headline block, and none of them reachable by
    a test that parses only the headline block.
    """
    for name in ("README.md", "DEVPOST.md", "DEMO_WALKTHROUGH.md"):
        text = _doc(name).replace("−", "-")
        for stale, reason in _RETIRED_FIGURES.items():
            assert stale not in text, f"{name} still quotes {stale} -- {reason}"


def test_every_document_agrees_with_the_actual_test_count(request):
    """Three documents said 259, one said 249, and the demo script said both 259 and 262.

    The count is read off the live pytest session rather than hard-coded, so it cannot go
    stale by itself. It matters because the demo narration points a judge at
    `python -m pytest -q`, whose output is about to be on camera.
    """
    collected = request.session.testscollected
    if collected < 200:
        pytest.skip(f"partial run ({collected} collected); this asserts the full suite size")

    for name in ("README.md", "DEVPOST.md", "DEMO_WALKTHROUGH.md"):
        text = _doc(name)
        claimed = {int(n) for n in re.findall(r"(\d{3})\s+(?:tests|passing)", text)}
        assert claimed, f"{name} no longer states a test count"
        assert claimed == {collected}, (
            f"{name} claims {sorted(claimed)} tests; the suite collects {collected}"
        )


def test_devpost_ablation_prose_matches_the_committed_ablation():
    """DEVPOST states the two largest ablation deltas in prose, twice, in two forms.

    Both were stale in both places and disagreed with the ablation table in the same
    document. The figures are derived here rather than listed, so regenerating the ablation
    fails this test until the prose is updated alongside it.
    """
    path = _RESULTS / "ablation.json"
    if not path.exists():
        pytest.skip("results/ablation.json not present")
    ablation = json.loads(path.read_text(encoding="utf-8"))
    full = ablation["full system"]["technical_score"]
    devpost = _doc("DEVPOST.md").replace("−", "-")

    for label, key in (
        ("clarification", "no clarification"),
        ("state tracking", "no state tracking"),
    ):
        delta = full - ablation[key]["technical_score"]
        assert f"-{delta:.3f}" in devpost, (
            f"DEVPOST does not state the {label} ablation as -{delta:.3f}"
        )
        assert f"+{delta:.3f}" in devpost, (
            f"DEVPOST does not state the {label} contribution as +{delta:.3f}"
        )

    baseline_path = _RESULTS / "baseline_by_scenario.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["technical_score"]
        total = _official()["recommended_technical_score"] - baseline
        assert f"+{total:.3f}" in devpost, (
            f"DEVPOST does not state the total improvement as +{total:.3f}"
        )


def test_the_demo_crib_sheet_matches_the_diagnostic_it_names():
    """The Q&A sheet said "16 sessions left" beside the command that prints 13.

    Anything on that sheet is something a judge may run while the author is on camera, so
    it is held to the same standard as the spoken narration.
    """
    path = _RESULTS / "rank_diagnosis.json"
    if not path.exists():
        pytest.skip("results/rank_diagnosis.json not present")
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    demo = _doc("DEMO_WALKTHROUGH.md")

    assert f"{summary['not_rank_1']} sessions left" in demo, (
        f"the crib sheet does not say {summary['not_rank_1']} sessions left; "
        "`python -m tools.diagnose_rank` reports that number"
    )
    assert summary["lost_to_ties"] == 0, (
        "the crib sheet claims none were lost to ties, but the diagnostic now finds "
        f"{summary['lost_to_ties']}"
    )


#: Multiplier expressions that are not baseline multiples. `100×` is the recommended
#: terminal size (100x40) in the demo setup notes.
_NON_MULTIPLE_TIMES = {"100"}


def test_every_baseline_multiple_claim_matches_the_arithmetic():
    """One document said "the 8×" long after the multiple became 9.03×.

    The headline multiple is asserted elsewhere, but it is also referred to in passing --
    in a "what I learned" bullet, in a framing note, in the README's opening line -- and
    those mentions drifted. Every `N×` in a judge-facing document is checked here against
    the arithmetic, so a passing reference cannot rot on its own.
    """
    official = _official()["recommended_technical_score"]
    baseline_path = _RESULTS / "baseline_by_scenario.json"
    if not baseline_path.exists():
        pytest.skip("results/baseline_by_scenario.json not present")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["technical_score"]
    true = official / baseline
    allowed = {f"{true:.2f}", f"{true:.1f}", str(int(round(true)))}

    for name in ("README.md", "DEVPOST.md", "DEMO_WALKTHROUGH.md"):
        for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*×", _doc(name)):
            value = match.group(1)
            if value in _NON_MULTIPLE_TIMES:
                continue
            assert value in allowed, (
                f"{name} claims {value}× over the baseline; {official}/{baseline} is "
                f"{true:.4f}×, so the acceptable renderings are {sorted(allowed)}"
            )
