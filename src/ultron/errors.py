"""Typed errors. Fail-closed by default: every one of these stops the run."""

from __future__ import annotations


class UltronError(Exception):
    """Base class for all harness errors."""


class ManifestError(UltronError):
    """A tool manifest on disk is invalid."""


class ToolNotFound(UltronError):
    """No manifest matched the requested tool name/version."""


class PolicyDenied(UltronError):
    """The policy gate refused an action."""

    def __init__(self, message: str, *, tool: str = "", risk: str = "") -> None:
        super().__init__(message)
        self.tool = tool
        self.risk = risk


class HumanApprovalRequired(UltronError):
    """A MEDIUM/HIGH risk action needs a human and none was available."""

    def __init__(self, message: str, *, tool: str = "", risk: str = "") -> None:
        super().__init__(message)
        self.tool = tool
        self.risk = risk


class SandboxUnavailable(UltronError):
    """The requested sandbox backend cannot run here (e.g. no Docker daemon)."""


class SandboxError(UltronError):
    """The sandbox ran but failed (non-zero exit, timeout, bad envelope)."""


class BudgetExceeded(UltronError):
    """The cost/time/step kill-switch fired."""

    def __init__(self, message: str, *, limit: str = "", spent: float = 0.0) -> None:
        super().__init__(message)
        self.limit = limit
        self.spent = spent


class LLMUnavailable(UltronError):
    """Live LLM mode was requested but no provider credentials are configured."""


class VerificationError(UltronError):
    """A tool produced output that failed verification."""
