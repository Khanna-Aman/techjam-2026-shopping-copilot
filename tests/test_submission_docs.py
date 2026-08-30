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


def test_no_submission_document_still_quotes_a_superseded_score():
    """Guard against a half-finished update leaving both numbers in the same document.

    Every previous shipped score is listed here as it is retired. A document may of course
    discuss an old number deliberately -- the confidence-gate and popularity sections both
    do -- so this only fires when the stale value appears *without* the current one, which
    is the signature of an edit that stopped halfway.
    """
    official = _official()
    current = f"{official['recommended_technical_score']:.6f}"
    superseded = ["0.906151", "0.954756"]

    for name in ("DEVPOST.md", "DEMO_WALKTHROUGH.md"):
        text = _doc(name)
        for stale in superseded:
            if stale == current:
                continue
            if stale in text:
                assert current in text, (
                    f"{name} quotes the superseded score {stale} but never the current "
                    f"{current}; the update looks half-applied"
                )
