"""Shopping Copilot: a stateful, offline-first conversational retrieval agent.

Public entry point is :class:`copilot.agent.ShoppingCopilot`, which implements the
TechJam Agent contract (``reset`` / ``respond``).
"""

__all__ = ["ShoppingCopilot", "AgentConfig"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name == "ShoppingCopilot":
        from copilot.agent import ShoppingCopilot

        return ShoppingCopilot
    if name == "AgentConfig":
        from copilot.config import AgentConfig

        return AgentConfig
    raise AttributeError(name)
