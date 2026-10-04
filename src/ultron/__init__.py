"""Ultron: a cost-optimized, safe, autonomous tool-forging agent harness.

Phase 1 (this branch) ships a *safe executor* plus an eval gate:

    goal -> cheap router -> deterministic plan -> policy gate -> Docker sandbox
         -> verification -> memory -> answer

Public surface::

    from ultron import Agent, Registry, PolicyGate, Sandbox, Verifier

Everything is dependency-injectable, so the whole harness runs offline with
deterministic stubs (see ``eval/run.py`` and ``tests/``).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:  # installed package
    __version__ = _version("ultron")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentResult",
    "BreakerVerifier",
    "Cache",
    "Charter",
    "FailureLedger",
    "ForgeEngine",
    "Memory",
    "Planner",
    "PolicyGate",
    "ProvenanceEnvelope",
    "PyPIHarvester",
    "Registry",
    "RepairEngine",
    "RepairTicket",
    "Router",
    "Sandbox",
    "Scavenger",
    "TelegramBotClient",
    "TelegramCockpit",
    "TelegramPrompter",
    "ToolRefinery",
    "Verifier",
    "__version__",
]

_LAZY = {
    "Agent": ("ultron.agent", "Agent"),
    "AgentResult": ("ultron.agent", "AgentResult"),
    "BreakerVerifier": ("ultron.breaker", "BreakerVerifier"),
    "Cache": ("ultron.cache", "Cache"),
    "Charter": ("ultron.charter", "Charter"),
    "FailureLedger": ("ultron.ledger", "FailureLedger"),
    "ForgeEngine": ("ultron.forge", "ForgeEngine"),
    "Memory": ("ultron.memory", "Memory"),
    "PyPIHarvester": ("ultron.harvester", "PyPIHarvester"),
    "RepairEngine": ("ultron.repair", "RepairEngine"),
    "RepairTicket": ("ultron.repair", "RepairTicket"),
    "Planner": ("ultron.planner", "Planner"),
    "PolicyGate": ("ultron.policy", "PolicyGate"),
    "ProvenanceEnvelope": ("ultron.provenance", "ProvenanceEnvelope"),
    "Registry": ("ultron.registry", "Registry"),
    "Router": ("ultron.router", "Router"),
    "Scavenger": ("ultron.scavenger", "Scavenger"),
    "Sandbox": ("ultron.sandbox", "Sandbox"),
    "TelegramBotClient": ("ultron.interfaces.telegram", "TelegramBotClient"),
    "TelegramCockpit": ("ultron.interfaces.telegram", "TelegramCockpit"),
    "TelegramPrompter": ("ultron.interfaces.telegram", "TelegramPrompter"),
    "ToolRefinery": ("ultron.refinery", "ToolRefinery"),
    "Verifier": ("ultron.verifier", "Verifier"),
}


def __getattr__(name: str):  # PEP 562 lazy imports keep `import ultron` cheap
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module 'ultron' has no attribute {name!r}")
