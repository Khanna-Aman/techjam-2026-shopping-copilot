"""Per-turn latency and memory, measured rather than asserted.

The feasibility argument is the strongest part of this submission -- zero tokens, no
network, no GPU, and a per-turn cost small enough to sit inside an existing search-response
budget. It was also, until this harness existed, the *least* reproducible part: the README
quoted "37 ms median, 67 ms p95, 225 MB resident" from an ad-hoc profiling session that
nobody else could re-run. Every other number in the repository is backed by a committed
harness and a committed result file. These now are too.

What it measures, and how honestly:

* **Latency** is wall-clock time around `agent.respond()` only. Index construction is
  excluded and reported separately, because it happens once per process rather than once
  per turn, and folding it in would flatter or damn the per-turn figure depending purely on
  session count. `time.perf_counter` is the clock.
* Turns are driven by the **official evaluator's own simulator functions**, so the
  distribution of work is the real one: later turns carry accumulated constraints and cost
  more than opening turns, and a benchmark of nothing but first turns would understate the
  tail. This is the same reason `tools/robustness.py` imports the evaluator rather than
  reimplementing it.
* **Memory** is resident set size -- what an operator actually pays for. The headline
  figure comes from a **clean subprocess holding only the agent and its index**, because
  this harness additionally holds the evaluator's catalog dictionary and briefly a second,
  cache-disabled index; reporting its own total would overstate a deployment roughly
  threefold. The harness's own figures are reported alongside, labelled as such. Reading it needs a per-platform call, and where that is unavailable the
  harness reports `null` rather than guessing. `tracemalloc` is deliberately *not* used:
  it roughly doubles the measured per-turn time, and a latency harness that perturbs the
  latency it reports is worse than none.

Scores are not produced here; this harness answers "what does it cost to run?", not "how
good is it?".

Usage:
    python -m tools.latency                  # 200 sessions, writes results/latency.json
    python -m tools.latency --limit 40       # quicker, for a smoke check
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.catalog import CatalogIndex  # noqa: E402


def resident_bytes() -> int | None:
    """Resident set size, or None where this platform cannot be read with the stdlib.

    Returning None is deliberate. A plausible-looking wrong number in the feasibility table
    would be worse than an honest gap, and every caller here prints the gap.
    """
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            return None
        return None
    if sys.platform == "darwin":
        try:
            import resource

            # macOS reports ru_maxrss in bytes; Linux would be kibibytes, but that branch
            # is handled above.
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            return None
    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            handle = ctypes.c_void_p(kernel32.GetCurrentProcess())

            # psapi on older Windows, kernel32's K32 alias on newer. Try both rather than
            # assuming; the pseudo-handle must be passed as a pointer, not a truncated int.
            for name in ("psapi", "kernel32"):
                library = ctypes.WinDLL(name)
                function = getattr(library, "GetProcessMemoryInfo", None) or getattr(
                    library, "K32GetProcessMemoryInfo", None
                )
                if function is None:
                    continue
                counters = _Counters()
                counters.cb = ctypes.sizeof(_Counters)
                if function(handle, ctypes.byref(counters), counters.cb):
                    return int(counters.WorkingSetSize)
            return None
        except Exception:
            return None
    return None


def _probe_agent_resident(catalog: str) -> float | None:
    """RSS of a process holding *only* the agent's index, measured in a clean subprocess.

    Measuring this in-process would be wrong by a wide margin. This harness also holds the
    evaluator's `products` dictionary and, briefly, a second index built with the cache
    disabled -- neither of which a deployment pays for. Reporting that total as the agent's
    footprint would overstate it roughly threefold, which is the sort of error that turns a
    feasibility claim into a liability.
    """
    code = (
        "import json,sys;"
        "sys.path.insert(0, %r);"
        "from copilot.catalog import CatalogIndex;"
        "from tools.latency import resident_bytes;"
        "index = CatalogIndex(%r);"
        "print(json.dumps({'rss': resident_bytes(), 'terms': len(getattr(index, '_postings', ()) or ())}))"
    ) % (str(_ROOT), catalog)
    try:
        finished = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
        if finished.returncode != 0:
            return None
        return _megabytes(json.loads(finished.stdout.strip().splitlines()[-1])["rss"])
    except Exception:
        return None


def _megabytes(value: int | None) -> float | None:
    return None if value is None else round(value / (1024 * 1024), 1)


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Explicit because `statistics.quantiles` interpolates."""
    if not ordered:
        return 0.0
    rank = max(1, min(len(ordered), int(fraction * len(ordered) + 0.5)))
    return ordered[rank - 1]


def _time_turns(agent, samples, catalog_ids, categories, products,
                early_stop: bool) -> list[float]:
    """Wall-clock ms per `respond()` call across every session.

    `early_stop` decides which question is being answered, and they are different ones.
    True mirrors the scored loop: it stops once the target is found, so opening turns
    dominate and the median describes a typical session. False runs all ten turns
    regardless, so constraints keep accumulating and the candidate pool keeps being
    rescored -- the conservative figure, and the one to quote when claiming that a latency
    budget is safe.
    """
    timings: list[float] = []
    for sample in samples:
        session_id = f"latency_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = initial_message(
            effective, coarse_category(categories.get(target, [])), disclosed
        )

        for turn in range(1, MAX_TURNS + 1):
            started = time.perf_counter()
            response = agent.respond(session_id, message, turn, TOP_K)
            timings.append((time.perf_counter() - started) * 1000.0)

            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if early_stop and override_applied and target in ranked:
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
            else:
                message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )
    return timings


def _distribution(timings: list[float]) -> dict:
    ordered = sorted(timings)
    return {
        "turns": len(ordered),
        "median": round(statistics.median(ordered), 3),
        "mean": round(statistics.fmean(ordered), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "p99": round(_percentile(ordered, 0.99), 3),
        "max": round(ordered[-1], 3),
    }


def _median_of_runs(runs: list[dict]) -> dict:
    """Median of each statistic across repeats.

    A single pass on a laptop is not a publishable latency figure: across three runs the
    p95 of the exhaustive regime moved by more than 20%, because thermal state and
    background load vary. Taking the median across repeats does not remove that noise, but
    it stops one unlucky pass from becoming the documented number, and `runs` keeps every
    individual pass so the spread stays visible rather than averaged away.
    """
    keys = ("median", "mean", "p95", "p99", "max")
    summary = {"turns": runs[0]["turns"]}
    summary.update({key: round(statistics.median(r[key] for r in runs), 3) for key in keys})
    return summary


def collect(catalog: str, dataset: str, limit: int | None, repeats: int = 1) -> dict:
    samples = load_jsonl(dataset)
    if limit is not None:
        samples = samples[:limit]
    catalog_ids, categories, products = catalog_index(catalog)

    # Warm the cache first, untimed, so the "warm" figure below really is a cache load
    # rather than whatever state this machine happened to be left in.
    CatalogIndex(catalog)

    # `use_cache=False` forces a genuine build from the catalog file. It also skips writing,
    # so measuring the cold path does not disturb the cache the warm path then reads.
    cold_start = time.perf_counter()
    CatalogIndex(catalog, use_cache=False)
    cold_seconds = time.perf_counter() - cold_start

    warm_start = time.perf_counter()
    index = CatalogIndex(catalog)
    warm_seconds = time.perf_counter() - warm_start

    agent = ShoppingCopilot(catalog, config=None, index=index)
    resident_after_index = resident_bytes()

    scored_runs, exhaustive_runs = [], []
    for _ in range(repeats):
        scored_runs.append(
            _distribution(
                _time_turns(agent, samples, catalog_ids, categories, products, early_stop=True)
            )
        )
        exhaustive_runs.append(
            _distribution(
                _time_turns(agent, samples, catalog_ids, categories, products, early_stop=False)
            )
        )

    resident_after_run = resident_bytes()
    return {
        "sessions": len(samples),
        "repeats": repeats,
        "per_turn_ms": {
            "scored_loop": _median_of_runs(scored_runs),
            "all_ten_turns": _median_of_runs(exhaustive_runs),
        },
        "per_turn_ms_runs": {
            "scored_loop": scored_runs,
            "all_ten_turns": exhaustive_runs,
        },
        "index_build_seconds": {
            "cold": round(cold_seconds, 3),
            "warm_from_cache": round(warm_seconds, 3),
        },
        "memory_mb": {
            "agent_only_resident": _probe_agent_resident(catalog),
            "harness_after_index_load": _megabytes(resident_after_index),
            "harness_after_run": _megabytes(resident_after_run),
        },
        "platform": sys.platform,
        "python": sys.version.split()[0],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N sessions")
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="timed passes; reported statistics are the median across them",
    )
    parser.add_argument("--out", default=str(_ROOT / "results" / "latency.json"))
    args = parser.parse_args(argv)

    report = collect(args.catalog, args.dataset, args.limit, args.repeats)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{report['sessions']} sessions, {report['repeats']} timed passes\n")
    print(
        f"{'':<13}{'turns':>7}{'median':>9}{'mean':>9}{'p95':>9}{'p99':>9}{'max':>9}"
    )
    for label, key in (("scored loop", "scored_loop"), ("all 10 turns", "all_ten_turns")):
        row = report["per_turn_ms"][key]
        print(
            f"{label:<13}{row['turns']:>7}{row['median']:>9.1f}{row['mean']:>9.1f}"
            f"{row['p95']:>9.1f}{row['p99']:>9.1f}{row['max']:>9.1f}"
        )
    print("  milliseconds; 'all 10 turns' is the constraint-accumulating worst case")
    build = report["index_build_seconds"]
    print(f"\n  index    {build['cold']:>8.1f} s cold, {build['warm_from_cache']:.1f} s warm")
    memory = report["memory_mb"]
    agent_only = memory["agent_only_resident"]
    if agent_only is None:
        print("  memory   resident set size unavailable on this platform")
    else:
        print(
            f"  memory   {agent_only:>8.1f} MB resident, agent + index only"
            f" (this harness itself peaks at {memory['harness_after_run']:.0f} MB,"
            f" holding the simulator's catalog too)"
        )
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
