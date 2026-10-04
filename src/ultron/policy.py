"""Policy gate: the single choke point every tool execution must pass.

Model
-----
    LOW     auto-approve, but still sandboxed with no network unless declared
    MEDIUM  ask a human; a stored, input-bound approval can satisfy the ask
    HIGH    deny unless an *explicitly pinned* approval exists. Never auto.

Approvals are bound to ``(tool, version, manifest content_hash, input digest)``
and expire. So approving "web_research for query X" can never authorise
"web_research for query Y" after someone edits the manifest or the tool code.

Fail-closed behaviours implemented here:
  * unknown input fields            -> deny
  * missing/wrong-typed inputs      -> deny
  * oversized inputs                -> deny
  * input values that look like secrets (sk-…, ghp_…, AKIA…, PEM blobs) -> deny,
    so a model can never smuggle a credential into a sandbox
  * ``secrets:*`` permission        -> deny in Phase 1 unconditionally
  * ``fs:write:*`` (host writes)    -> deny in Phase 1 (mount is read-only)
  * network requested but policy is 'deny' -> deny
  * LOW risk **and** network requested -> escalated to MEDIUM (ask) unless
    ``ULTRON_POLICY_NETWORK_LOW_AUTO=1``. Phase 3 auto-discovers manifests, and
    "LOW + network runs unattended" is a landmine.
  * non-interactive terminal + MEDIUM/HIGH -> deny (never silently assume yes)
  * a stored approval is *claimed* (consumed) as part of the decision in
    :meth:`PolicyGate.check`, so a decision can never be replayed by a caller
    that runs it without going through :meth:`PolicyGate.evaluate`

Every decision is appended to a JSONL audit log with the *digest* of the inputs,
not the raw inputs, so the log itself is not a data-leak vector.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from .cache import canonical_json, make_key
from .charter import Charter
from .config import Settings, get_settings
from .errors import HumanApprovalRequired, PolicyDenied
from .registry import PHASE1_DENIED_SCOPES, TYPE_MAP, RiskTier, ToolManifest

Decision = Literal["allow", "ask", "deny"]
ApprovalVerdict = Literal["granted", "denied", "no_human", "not_required"]

MAX_INPUT_BYTES = 64_000
DEFAULT_APPROVAL_TTL_S = 86_400

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}")),
    ("github_token", re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,})")),
    # Fine-grained PATs (`github_pat_...`) do not match the classic `gh[pousr]_`
    # shape, so the token most likely to be pasted into a prompt slipped through.
    ("github_pat_fg", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("stripe_live", re.compile(r"\b(sk|rk)_live_[A-Za-z0-9]{20,}")),
    ("aws_access_key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}")),
    ("slack_token", re.compile(r"\bxox[abprs]-[0-9A-Za-z\-]{10,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}")),
)


# --------------------------------------------------------------------- scanning
def scan_for_secrets(value: Any, *, path: str = "") -> list[str]:
    """Return ``'field:pattern_name'`` hits for anything that smells like a credential."""
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            hits += scan_for_secrets(item, path=f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            hits += scan_for_secrets(item, path=f"{path}[{idx}]")
    elif isinstance(value, str):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(value):
                hits.append(f"{path or '<root>'}:{label}")
    return hits


def input_digest(inputs: dict[str, Any]) -> str:
    return make_key("inputs", inputs)[:32]


def goal_digest(goal: str) -> str:
    return hashlib.sha256(goal.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ data types
@dataclass(slots=True)
class ApprovalPrompt:
    """What a human is shown before a MEDIUM/HIGH risk action runs."""

    tool: str
    version: str
    risk: str
    reason: str
    run_id: str
    goal: str
    content_hash: str
    input_digest: str
    inputs_preview: dict[str, Any]
    network: str
    permissions: list[str]

    def render(self) -> str:
        preview = json.dumps(self.inputs_preview, indent=2, default=str)
        if len(preview) > 1200:
            preview = preview[:1200] + "\n  ... (truncated)"
        return (
            f"Tool     : {self.tool}@{self.version}  (risk={self.risk})\n"
            f"Reason   : {self.reason}\n"
            f"Network  : {self.network}\n"
            f"Perms    : {', '.join(self.permissions) or 'none'}\n"
            f"Run      : {self.run_id}\n"
            f"Goal     : {self.goal[:200]}\n"
            f"Bind     : hash={self.content_hash[:12]} inputs={self.input_digest}\n"
            f"Inputs   : {preview}\n"
        )


class Prompter(Protocol):
    """Human-in-the-loop hook. Return True/False, or None when no human is available."""

    def __call__(self, prompt: ApprovalPrompt) -> bool | None: ...


class DenyAllPrompter:
    """Default for non-interactive contexts (CI, eval): never assume consent."""

    def __call__(self, prompt: ApprovalPrompt) -> bool | None:
        return None


class ScriptedPrompter:
    """Deterministic prompter for tests/eval: answers from a fixed script."""

    def __init__(self, answers: list[bool] | None = None, default: bool | None = None) -> None:
        self.answers = list(answers or [])
        self.default = default
        self.prompts: list[ApprovalPrompt] = []

    def __call__(self, prompt: ApprovalPrompt) -> bool | None:
        self.prompts.append(prompt)
        if self.answers:
            return self.answers.pop(0)
        return self.default


class AutoApprovePrompter:
    """Dev-only: says yes to MEDIUM. Refuses HIGH (the gate refuses it anyway)."""

    def __call__(self, prompt: ApprovalPrompt) -> bool | None:
        return prompt.risk != "high"  # HIGH is refused by the gate regardless


class ConsolePrompter:
    """Interactive rich prompt. No TTY -> ``None`` -> deny."""

    def __call__(self, prompt: ApprovalPrompt) -> bool | None:
        if not sys.stdin.isatty():
            return None
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Confirm

        console = Console(stderr=True)
        colour = {"low": "green", "medium": "yellow", "high": "red"}.get(prompt.risk, "white")
        console.print(Panel(prompt.render(), title=f"[{colour}]approval required[/{colour}]"))
        try:
            return bool(Confirm.ask("Allow this action?", console=console, default=False))
        except (EOFError, KeyboardInterrupt):
            return False


def default_prompter(settings: Settings) -> Prompter:
    if settings.policy_assume_yes:
        return AutoApprovePrompter()
    return ConsolePrompter()


@dataclass(slots=True)
class PolicyRequest:
    tool: ToolManifest
    inputs: dict[str, Any]
    run_id: str = "run"
    goal: str = ""
    step: int = 0
    action_type: str = "tool_execution"
    allow_network: bool | None = None  # None -> derive from manifest + settings


@dataclass(slots=True)
class PolicyDecision:
    action: Decision
    reason: str
    risk: str
    tool: str
    version: str
    network: str = "none"
    granted: bool = False
    verdict: ApprovalVerdict = "not_required"
    approval_id: str | None = None
    limits: dict[str, Any] = field(default_factory=dict)
    secrets_found: list[str] = field(default_factory=list)
    #: Risk the manifest declares, before any escalation (see ``escalated``).
    declared_risk: str = ""
    #: True when a LOW-risk tool that asks for network was escalated to MEDIUM.
    escalated: bool = False

    @property
    def allowed(self) -> bool:
        return self.action == "allow" and self.granted

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "risk": self.risk,
            "declared_risk": self.declared_risk,
            "escalated": self.escalated,
            "tool": self.tool,
            "version": self.version,
            "network": self.network,
            "verdict": self.verdict,
            "approval_id": self.approval_id,
            "limits": self.limits,
        }


# ---------------------------------------------------------------- audit + store
class AuditLog:
    """Append-only JSONL trail of every gate decision."""

    def __init__(self, path: Path, *, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def append(self, event: str, **payload: Any) -> None:
        record = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str, sort_keys=True) + "\n")

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(line) for line in lines if line.strip()]


class ApprovalStore:
    """File-backed store of human approvals, pinned to content + inputs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"approvals": []})

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"approvals": []}

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def grant(
        self,
        request: PolicyRequest,
        *,
        approver: str,
        ttl_s: int = DEFAULT_APPROVAL_TTL_S,
        max_uses: int = 1,
        note: str = "",
    ) -> dict[str, Any]:
        data = self._read()
        approval = {
            "id": make_key(
                "approval",
                request.tool.key,
                request.tool.content_hash,
                input_digest(request.inputs),
            )[:16],
            "tool": request.tool.name,
            "version": request.tool.version,
            "content_hash": request.tool.content_hash,
            "input_digest": input_digest(request.inputs),
            "risk": request.tool.risk.value,
            "goal": request.goal[:300],
            "approver": approver,
            "note": note,
            "created_at": time.time(),
            "expires_at": time.time() + ttl_s,
            "max_uses": max_uses,
            "uses": 0,
        }
        data["approvals"] = [
            a
            for a in data.get("approvals", [])
            if not (
                a.get("tool") == approval["tool"]
                and a.get("content_hash") == approval["content_hash"]
                and a.get("input_digest") == approval["input_digest"]
            )
        ]
        data["approvals"].append(approval)
        self._write(data)
        return approval

    @staticmethod
    def _select(data: dict[str, Any], request: PolicyRequest) -> dict[str, Any] | None:
        """Pick the approval that authorises this exact action, if any."""
        now = time.time()
        digest = input_digest(request.inputs)
        for approval in data.get("approvals", []):
            if approval.get("tool") != request.tool.name:
                continue
            if approval.get("content_hash") != request.tool.content_hash:
                continue
            if approval.get("input_digest") != digest:
                continue
            if approval.get("expires_at", 0) < now:
                continue
            if approval.get("uses", 0) >= approval.get("max_uses", 1):
                continue
            return approval
        return None

    def find(self, request: PolicyRequest) -> dict[str, Any] | None:
        """Return a valid approval pinned to this exact action, *without* spending it.

        Read-only, for inspection paths (UI, :meth:`PolicyGate.peek`). Anything
        that grants access must use :meth:`claim` instead.
        """
        return self._select(self._read(), request)

    def claim(self, request: PolicyRequest) -> dict[str, Any] | None:
        """Atomically find **and** spend a valid approval for this exact action.

        This is the only way the gate grants access from a stored approval.
        Previously the gate did ``find()`` and then ``consume()`` as two separate
        read-modify-write passes, so a caller that used ``check()`` directly
        (instead of ``evaluate()``) could replay one approval for its whole TTL:
        the decision stayed valid, and every later run re-found the same grant.

        Doing it in one locked read-modify-write closes the replay window and also
        makes it safe for two concurrent agent processes to share an approval
        store — only one of them can claim a given grant.
        """
        with self._locked():
            data = self._read()
            approval = self._select(data, request)
            if approval is None:
                return None
            approval["uses"] = approval.get("uses", 0) + 1
            approval["last_used_at"] = time.time()
            self._write(data)
            return approval

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Exclusive cross-process lock around an approval read-modify-write.

        ``fcntl.flock`` on a sidecar lock file (POSIX). On platforms without
        fcntl the lock degrades to a no-op, so atomicity would then be
        per-process only — acceptable for a single-operator Phase 1 harness,
        and documented rather than silently assumed.
        """
        lock_path = self.path.with_name(self.path.name + ".lock")
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX dev machines
            yield
            return
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def consume(self, approval_id: str) -> None:
        data = self._read()
        for approval in data.get("approvals", []):
            if approval.get("id") == approval_id:
                approval["uses"] = approval.get("uses", 0) + 1
                approval["last_used_at"] = time.time()
        self._write(data)

    def revoke_all(self) -> int:
        data = self._read()
        count = len(data.get("approvals", []))
        self._write({"approvals": []})
        return count


# ------------------------------------------------------------------------ gate
class PolicyGate:
    """Evaluates :class:`PolicyRequest` values against the risk model."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        prompter: Prompter | None = None,
        store: ApprovalStore | None = None,
        audit: AuditLog | None = None,
        charter: Charter | None = None,
        run_id: str = "run",
    ) -> None:
        self.settings = settings or get_settings()
        self.run_id = run_id
        self.prompter: Prompter = prompter or default_prompter(self.settings)
        self.store = store or ApprovalStore(self.settings.approvals_file)
        self.audit = audit or AuditLog(self.settings.audit_log, run_id=run_id)
        self.charter = charter or Charter(self.settings.state_dir)

    # ------------------------------------------------------------ entry points
    def check(self, request: PolicyRequest, *, consume: bool = True) -> PolicyDecision:
        """Decide, and by default *spend* any stored approval the decision relies on.

        Never prompts a human. It is deliberately not a pure read: when a stored
        approval authorises the action, that approval is **claimed** (one locked
        read-modify-write) as part of producing the decision. A caller that runs
        the returned decision directly — bypassing :meth:`evaluate` — therefore
        cannot replay it; the next ``check()`` for the same action has no grant
        left and returns ``ask`` (MEDIUM) or ``deny`` (HIGH).

        Use ``consume=False`` or :meth:`peek` for pure inspection (reporting, UI).
        """
        manifest = request.tool
        limits = {
            "timeout_s": min(
                float(manifest.timeout_s or self.settings.sandbox_timeout_s),
                float(self.settings.sandbox_timeout_s),
            ),
            "memory": self.settings.sandbox_memory,
            "cpus": self.settings.sandbox_cpus,
            "pids": self.settings.sandbox_pids,
        }
        base = {
            "risk": manifest.risk.value,
            "tool": manifest.name,
            "version": manifest.version,
            "limits": limits,
        }

        # 1. Input contract ------------------------------------------------
        problem = self._validate_inputs(manifest, request.inputs)
        if problem:
            return self._deny(request, problem, **base)

        secrets_found = scan_for_secrets(request.inputs)
        if secrets_found:
            return self._deny(
                request,
                "inputs appear to contain secrets; refusing to move credentials "
                f"across the sandbox boundary ({', '.join(sorted(set(secrets_found)))})",
                secrets_found=sorted(set(secrets_found)),
                **base,
            )

        # 2. Permission model ---------------------------------------------
        for perm in manifest.parsed_permissions:
            if perm.scope in PHASE1_DENIED_SCOPES:
                return self._deny(
                    request,
                    f"permission {perm.raw!r} is not grantable in Phase 1 "
                    "(no secret injection into sandboxes, by design)",
                    **base,
                )
            if (
                perm.scope == "fs"
                and perm.detail.startswith(("write", "rw"))
                and perm.detail != "write:tmp"
            ):
                return self._deny(
                    request,
                    f"permission {perm.raw!r} needs a writable host path; Phase 1 mounts "
                    "the repo read-only (use 'fs:write:tmp' for scratch space)",
                    **base,
                )

        wants_network = manifest.wants_network
        network_ok = self._network_allowed(request, wants_network)
        # A scavenged wrapper can be tested against its explicit mock response
        # while retaining its network declaration for real execution. The
        # decision remains network=none; this narrow exception never applies to
        # ordinary tool execution or to a forge test without a mock input.
        offline_forge_mock = (
            request.action_type == "forge_test"
            and request.allow_network is False
            and "mock_response" in request.inputs
        )
        network = manifest.network_detail if (wants_network and network_ok) else "none"
        if wants_network and not network_ok and not offline_forge_mock:
            return self._deny(
                request, "tool requests network egress but policy denies it", network="none", **base
            )

        # 3. Effective risk tier -------------------------------------------
        # A LOW-risk tool that asks for network access is escalated to MEDIUM by
        # default. Phase 3 auto-discovers manifests, and "declared LOW" is not a
        # property we can trust from a discovered tool: LOW + egress running
        # unattended is a landmine, so a human looks at it unless the operator
        # has explicitly opted in with ULTRON_POLICY_NETWORK_LOW_AUTO=1.
        declared_risk = manifest.risk
        escalated = (
            declared_risk is RiskTier.LOW
            and wants_network
            and not self.settings.policy_network_low_auto
        )
        risk = RiskTier.MEDIUM if escalated else declared_risk
        base.update(risk=risk.value, declared_risk=declared_risk.value, escalated=escalated)

        if risk is RiskTier.LOW:
            if consume:
                # Burn any leftover grant for this exact action so it cannot be
                # replayed later (it does not affect the decision: LOW auto-runs).
                self.store.claim(request)
            return self._allow(
                request, "low risk: auto-approved (sandboxed)", network=network, **base
            )

        # MEDIUM / HIGH: a stored approval is *claimed* here, not merely found.
        approval = self.store.claim(request) if consume else self.store.find(request)
        if risk is RiskTier.HIGH:
            if approval is None:
                return self._deny(
                    request,
                    "high risk: denied. Requires an explicit, pinned approval written to "
                    f"{self.settings.approvals_file} (never auto-granted)",
                    network=network,
                    **base,
                )
            return self._allow(
                request,
                "high risk allowed by pinned approval",
                network=network,
                verdict="granted",
                approval_id=approval["id"],
                **base,
            )

        # MEDIUM
        if approval is not None:
            return self._allow(
                request,
                "medium risk allowed by stored approval",
                network=network,
                verdict="granted",
                approval_id=approval["id"],
                **base,
            )
        reason = "medium risk: human approval required"
        if escalated:
            reason = (
                "low risk tool requests network access: escalated to human approval "
                "(set ULTRON_POLICY_NETWORK_LOW_AUTO=1 to auto-run it)"
            )
        return self._ask(request, reason, network=network, **base)

    def peek(self, request: PolicyRequest) -> PolicyDecision:
        """Non-consuming preview of a decision (reporting/UI paths)."""
        return self.check(request, consume=False)

    def evaluate(self, request: PolicyRequest, *, interactive: bool = True) -> PolicyDecision:
        """``check`` then apply the charter before any human prompt."""
        tier = self.charter.evaluate(request.action_type)
        if tier == "red":
            self.charter.log(request.action_type, tier=tier, detail="halted")
            raise PolicyDenied(
                f"autonomy charter RED action: {request.action_type}",
                tool=request.tool.name,
                risk=request.tool.risk.value,
            )
        if tier == "yellow":
            self.charter.log(
                request.action_type, tier=tier, detail="notify-after; policy continues"
            )

        decision = self.check(request)
        if (
            tier == "green"
            and request.action_type in {"forge_test", "sandbox_execution"}
            and not request.tool.wants_network
            and decision.action == "ask"
        ):
            # A green charter bypass is only for an offline sandbox. A network
            # declaration follows the ordinary policy path and therefore needs a
            # separate grant/prompt; the charter never creates egress.
            decision.action = "allow"
            decision.granted = True
            decision.verdict = "not_required"
            decision.reason = "charter GREEN: sandbox test auto-approved"
            self.audit.append(
                "charter_green", action_type=request.action_type, tool=request.tool.key
            )
        if decision.action == "deny":
            raise PolicyDenied(decision.reason, tool=request.tool.name, risk=decision.risk)
        if decision.action == "allow":
            # check() already claimed (consumed) any stored approval it relied on;
            # consuming again here would burn a second use of the same grant.
            return decision

        # action == "ask"
        if not interactive:
            self.audit.append(
                "ask_skipped",
                tool=request.tool.key,
                risk=decision.risk,
                reason="non-interactive",
                goal_digest=goal_digest(request.goal),
            )
            raise HumanApprovalRequired(
                f"{request.tool.key} is {decision.risk} risk and needs human approval "
                "(non-interactive run: no human available, refusing to proceed)",
                tool=request.tool.name,
                risk=decision.risk,
            )

        prompt = ApprovalPrompt(
            tool=request.tool.name,
            version=request.tool.version,
            risk=decision.risk,
            reason=decision.reason,
            run_id=request.run_id,
            goal=request.goal,
            content_hash=request.tool.content_hash,
            input_digest=input_digest(request.inputs),
            inputs_preview=_redact(request.inputs),
            network=decision.network,
            permissions=request.tool.permissions,
        )
        granted = self.prompter(prompt)
        if granted is None:
            self.audit.append(
                "ask_unavailable",
                tool=request.tool.key,
                risk=decision.risk,
                goal_digest=goal_digest(request.goal),
            )
            raise HumanApprovalRequired(
                f"no human available to approve {request.tool.key} ({decision.risk} risk)",
                tool=request.tool.name,
                risk=decision.risk,
            )
        if not granted:
            self.audit.append(
                "human_denied",
                tool=request.tool.key,
                risk=decision.risk,
                goal_digest=goal_digest(request.goal),
            )
            raise PolicyDenied(
                f"human denied {request.tool.key}", tool=request.tool.name, risk=decision.risk
            )

        approval = self.store.grant(request, approver="human", note="interactive approval")
        self.store.consume(approval["id"])
        self.audit.append(
            "human_approved",
            tool=request.tool.key,
            risk=decision.risk,
            approval_id=approval["id"],
            goal_digest=goal_digest(request.goal),
        )
        decision.action = "allow"
        decision.granted = True
        decision.verdict = "granted"
        decision.approval_id = approval["id"]
        decision.reason = "human approved"
        return decision

    # ----------------------------------------------------------------- helpers
    def _validate_inputs(self, manifest: ToolManifest, inputs: dict[str, Any]) -> str | None:
        if not isinstance(inputs, dict):
            return "inputs must be a JSON object"
        unknown = sorted(set(inputs) - set(manifest.inputs))
        if unknown:
            return f"undeclared input fields: {', '.join(unknown)}"
        missing = sorted(set(manifest.inputs) - set(inputs))
        if missing:
            return f"missing required inputs: {', '.join(missing)}"
        for name, typ in manifest.inputs.items():
            expected = TYPE_MAP[typ]
            value = inputs[name]
            if expected is int and isinstance(value, bool):
                return f"input {name!r} must be int, got bool"
            if expected is float and isinstance(value, bool):
                return f"input {name!r} must be float, got bool"
            if expected is tuple and isinstance(value, bool):
                return f"input {name!r} must be numeric, got bool"
            if expected is not object and not isinstance(value, expected):
                return f"input {name!r} must be {typ}, got {type(value).__name__}"
        encoded = canonical_json(inputs).encode("utf-8")
        if len(encoded) > MAX_INPUT_BYTES:
            return f"inputs exceed {MAX_INPUT_BYTES} bytes ({len(encoded)}); split the request"
        return None

    def _network_allowed(self, request: PolicyRequest, wants_network: bool) -> bool:
        if not wants_network:
            return True
        if self.settings.policy_network == "deny":
            return False
        if request.allow_network is not None:
            return bool(request.allow_network)
        return True

    def _allow(self, request: PolicyRequest, reason: str, **kw: Any) -> PolicyDecision:
        decision = PolicyDecision(action="allow", granted=True, reason=reason, **kw)
        self._log(request, decision)
        return decision

    def _ask(self, request: PolicyRequest, reason: str, **kw: Any) -> PolicyDecision:
        decision = PolicyDecision(action="ask", granted=False, reason=reason, **kw)
        self._log(request, decision)
        return decision

    def _deny(self, request: PolicyRequest, reason: str, **kw: Any) -> PolicyDecision:
        decision = PolicyDecision(action="deny", granted=False, reason=reason, **kw)
        self._log(request, decision)
        return decision

    def _log(self, request: PolicyRequest, decision: PolicyDecision) -> None:
        self.audit.append(
            f"policy_{decision.action}",
            tool=request.tool.key,
            risk=decision.risk,
            reason=decision.reason,
            network=decision.network,
            declared_risk=decision.declared_risk or decision.risk,
            escalated=decision.escalated or None,
            inputs_digest=input_digest(request.inputs),
            input_fields=sorted(request.inputs),
            goal_digest=goal_digest(request.goal),
            step=request.step,
            approval_id=decision.approval_id,
            secrets_found=decision.secrets_found or None,
        )


def _redact(inputs: dict[str, Any], limit: int = 240) -> dict[str, Any]:
    """Preview inputs to a human without echoing anything credential-shaped."""
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, str):
            redacted = value
            for _, pattern in SECRET_PATTERNS:
                redacted = pattern.sub("[REDACTED]", redacted)
            out[key] = redacted if len(redacted) <= limit else redacted[:limit] + "…"
        elif isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = f"<{type(value).__name__}>"
    return out


def scrub_env(env: dict[str, str], allowlist: list[str]) -> dict[str, str]:
    """Keep only explicitly allowlisted, non-secret-looking env vars."""
    from .config import SECRET_ENV_MARKERS

    kept: dict[str, str] = {}
    for name, value in env.items():
        if name not in allowlist:
            continue
        upper = name.upper()
        if any(marker in upper for marker in SECRET_ENV_MARKERS):
            continue
        kept[name] = value
    return kept


__all__ = [
    "ApprovalPrompt",
    "ApprovalStore",
    "AuditLog",
    "AutoApprovePrompter",
    "ConsolePrompter",
    "DenyAllPrompter",
    "PolicyDecision",
    "PolicyGate",
    "PolicyRequest",
    "ScriptedPrompter",
    "goal_digest",
    "input_digest",
    "scan_for_secrets",
    "scrub_env",
]
