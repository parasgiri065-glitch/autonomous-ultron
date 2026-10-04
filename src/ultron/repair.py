"""Bounded, non-mutating self-repair validation."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .charter import Charter
from .config import Settings, get_settings
from .errors import UltronError
from .sandbox import BoundedRun, run_bounded


class RepairError(UltronError):
    """A repair ticket or candidate patch could not be validated."""


@dataclass(slots=True)
class RepairTicket:
    ticket_id: str
    component: str
    error: str
    traceback_str: str
    context: dict[str, Any] = field(default_factory=dict)
    status: str = "open"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    patch_diff: str = ""
    validation: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""

    @property
    def id(self) -> str:
        return self.ticket_id

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def validated_patch(self) -> str:
        return self.patch_diff if self.status == "validated" else ""

    @property
    def ready_for_review(self) -> bool:
        return self.status == "validated"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["id"] = self.ticket_id
        data["traceback"] = self.traceback_str
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepairTicket:
        return cls(
            ticket_id=str(data.get("ticket_id") or data.get("id") or ""),
            component=str(data.get("component", "")),
            error=str(data.get("error", "")),
            traceback_str=str(data.get("traceback_str", data.get("traceback", ""))),
            context=dict(data.get("context") or {}),
            status=str(data.get("status", "open")),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            patch_diff=str(data.get("patch_diff", "")),
            validation=dict(data.get("validation") or {}),
            failure_reason=str(data.get("failure_reason", "")),
        )


class RepairEngine:
    """Validate patches in a temporary copy; never mutate the checkout."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        charter: Charter | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.charter = charter or Charter(self.settings.state_dir)
        self.path = self.settings.state_dir / "repair_tickets.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_fault(
        self,
        component: str,
        error: str,
        traceback_str: str,
        context: dict[str, Any],
    ) -> RepairTicket:
        ticket = RepairTicket(
            ticket_id=f"repair-{uuid.uuid4().hex[:12]}",
            component=component,
            error=error,
            traceback_str=traceback_str,
            context=dict(context),
        )
        self._append(ticket)
        return ticket

    def list_tickets(self, *, open_only: bool = False) -> list[RepairTicket]:
        latest: dict[str, RepairTicket] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    ticket = RepairTicket.from_dict(json.loads(line))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if ticket.ticket_id:
                    latest[ticket.ticket_id] = ticket
        tickets = list(latest.values())
        if open_only:
            tickets = [ticket for ticket in tickets if ticket.is_open]
        return sorted(tickets, key=lambda ticket: ticket.created_at, reverse=True)

    def open_tickets(self) -> list[RepairTicket]:
        return self.list_tickets(open_only=True)

    def validate_patch(self, ticket: RepairTicket, patch_diff: str) -> bool:
        """Apply and test a candidate diff in an isolated workspace."""
        if not patch_diff.strip():
            return self._reject(ticket, patch_diff, "empty patch")
        workspace: Path | None = None
        validation: dict[str, Any] = {}
        try:
            workspace = Path(tempfile.mkdtemp(prefix="ultron-repair-")) / "repo"
            shutil.copytree(
                self.settings.repo_root,
                workspace,
                ignore=shutil.ignore_patterns(
                    ".git", ".venv", ".ultron", ".env", ".env.*", "__pycache__", "*.pyc"
                ),
            )
            applied = self._apply_patch(workspace, patch_diff)
            validation["patch_applied"] = applied
            if not applied:
                return self._reject(ticket, patch_diff, "candidate diff did not apply", validation)

            pytest_result = self._run_command(
                [sys.executable, "-m", "pytest", "-q"], workspace, timeout=900
            )
            validation["pytest"] = _command_summary(pytest_result)
            if pytest_result.exit_code != 0 or pytest_result.timed_out or pytest_result.capped:
                return self._reject(ticket, patch_diff, "pytest failed", validation)

            eval_result = self._run_command(
                [sys.executable, "eval/run.py", "--passes", "3"], workspace, timeout=900
            )
            validation["eval"] = _command_summary(eval_result)
            if eval_result.exit_code != 0 or eval_result.timed_out or eval_result.capped:
                return self._reject(ticket, patch_diff, "eval regression", validation)

            target_ok, target_detail = self._target_fault_check(ticket, workspace)
            validation["target_fault"] = {"ok": target_ok, "detail": target_detail}
            if not target_ok:
                return self._reject(ticket, patch_diff, target_detail, validation)

            ticket.status = "validated"
            ticket.patch_diff = patch_diff
            ticket.validation = validation
            ticket.failure_reason = ""
            ticket.updated_at = time.time()
            self._append(ticket)
            # Patch approval/PR creation is deliberately a Charter YELLOW action;
            # this method validates and records readiness, but never commits/pushes.
            self.charter.log(
                "repair_patch_validation",
                tier="yellow",
                detail=f"{ticket.ticket_id}: validated patch ready for review",
            )
            return True
        except (OSError, subprocess.SubprocessError, RepairError) as exc:
            return self._reject(
                ticket, patch_diff, f"repair validation error: {type(exc).__name__}", validation
            )
        finally:
            if workspace is not None:
                shutil.rmtree(workspace.parent, ignore_errors=True)

    def _apply_patch(self, workspace: Path, patch_diff: str) -> bool:
        patch_file = workspace.parent / "candidate.patch"
        patch_file.write_text(patch_diff, encoding="utf-8")
        env = _test_env(workspace)
        check = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", str(patch_file)],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if check.returncode != 0:
            return False
        applied = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_file)],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return applied.returncode == 0

    def _run_command(self, command: list[str], workspace: Path, *, timeout: float) -> BoundedRun:
        return run_bounded(
            command,
            cwd=str(workspace),
            env=_test_env(workspace),
            timeout_s=timeout,
            limit=1_000_000,
            start_new_session=True,
        )

    def _target_fault_check(self, ticket: RepairTicket, workspace: Path) -> tuple[bool, str]:
        context = ticket.context
        command = (
            context.get("fault_test")
            or context.get("test_command")
            or context.get("fault_check")
            or context.get("target_test")
            or context.get("command")
        )
        if command:
            if isinstance(command, str):
                try:
                    command = shlex.split(command)
                except ValueError:
                    return False, "invalid target fault check"
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                return False, "invalid target fault check"
            result = self._run_command(list(command), workspace, timeout=300)
            return (
                result.exit_code == 0 and not result.timed_out and not result.capped,
                "target fault check passed"
                if result.exit_code == 0 and not result.timed_out and not result.capped
                else "target fault check failed",
            )

        test_path = context.get("test_path") or context.get("test_file")
        test_name = context.get("test_name")
        if test_path:
            target = str(test_path)
            if test_name:
                target += f"::{test_name}"
            result = self._run_command(
                [sys.executable, "-m", "pytest", "-q", target], workspace, timeout=300
            )
            return (
                result.exit_code == 0 and not result.timed_out and not result.capped,
                "target test passed"
                if result.exit_code == 0 and not result.timed_out and not result.capped
                else "target test failed",
            )

        # A full green suite/eval is the only available evidence when the fault
        # was reported without a reproducible target command.
        return True, "full pytest and eval passed; no separate target check supplied"

    def _reject(
        self,
        ticket: RepairTicket,
        patch_diff: str,
        reason: str,
        validation: dict[str, Any] | None = None,
    ) -> bool:
        ticket.status = "rejected"
        ticket.patch_diff = patch_diff
        ticket.validation = validation or {}
        ticket.failure_reason = reason
        ticket.updated_at = time.time()
        self._append(ticket)
        return False

    def _append(self, ticket: RepairTicket) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ticket.as_dict(), sort_keys=True, default=str) + "\n")


def _test_env(workspace: Path) -> dict[str, str]:
    """Minimal environment: no ambient credentials enter candidate tests."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(workspace / ".home"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(workspace),
        "ULTRON_REPAIR_VALIDATION": "1",
    }


def _command_summary(result: BoundedRun) -> dict[str, Any]:
    # Do not persist command output: test tools can accidentally print secrets.
    return {
        "ok": result.exit_code == 0 and not result.timed_out and not result.capped,
        "returncode": result.exit_code,
        "timed_out": result.timed_out,
        "output_capped": result.capped,
    }
