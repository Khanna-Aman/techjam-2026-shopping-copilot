"""Submission entry point for the official evaluator.

The harness imports ``Agent`` from this module and constructs it with the catalog path,
so this file stays a thin adapter over :mod:`copilot.agent`.

Behaviour can be overridden through the ``COPILOT_CONFIG`` environment variable, which
accepts a JSON object of :class:`copilot.config.AgentConfig` fields. That exists so the
ablation harness can vary the agent *without editing the evaluator*, which the submission
rules forbid. With the variable unset the agent runs its full default configuration.

The organiser's original weak baseline is preserved verbatim in
``starter/baseline_agent.py`` as the ablation zero point.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

# Allow ``python -m evaluator.local_evaluator`` from the repository root to import the
# copilot package regardless of how the harness sets up sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from copilot.agent import ShoppingCopilot  # noqa: E402
from copilot.config import DEFAULT_CONFIG, AgentConfig  # noqa: E402

_ENV_KEY = "COPILOT_CONFIG"


def _load_config() -> AgentConfig:
    raw = os.environ.get(_ENV_KEY)
    if not raw:
        return DEFAULT_CONFIG
    try:
        overrides = json.loads(raw)
    except (TypeError, ValueError):
        return DEFAULT_CONFIG
    if not isinstance(overrides, dict):
        return DEFAULT_CONFIG
    allowed = set(AgentConfig.__dataclass_fields__)
    clean = {key: value for key, value in overrides.items() if key in allowed}
    try:
        return replace(DEFAULT_CONFIG, **clean)
    except (TypeError, ValueError):
        return DEFAULT_CONFIG


class Agent(ShoppingCopilot):
    """Thin adapter matching the harness constructor signature ``Agent(catalog_path)``."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        super().__init__(catalog_path, config=_load_config())


__all__ = ["Agent"]
