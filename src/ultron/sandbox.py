"""Docker sandbox executor. The only thing in the harness that runs tool code.

Container hard guarantees (all asserted by ``tests/test_smoke.py``):

* ``--network none`` **unless** the manifest declares ``network:*`` *and* the
  policy gate granted it (the gate's decision is the only input considered here).
* ``--read-only`` root filesystem + a small ``noexec,nosuid`` tmpfs at ``/tmp``.
  The repo is mounted ``:ro``; a tool cannot mutate the harness or itself.
* No secrets: the container env is only ``PYTHON*`` plus explicitly allowlisted
  names, filtered again by :func:`policy.scrub_env` for secret-looking keys. The
  host environment is never inherited (``--env-file`` is not used).
* Dropped capabilities, no new privileges, non-root uid, pid/memory/cpu/file
  limits, and a wall-clock timeout after which the container is killed.
* The entrypoint is executed as **argv without a shell** (manifest validation
  already rejected metacharacters), so nothing can chain commands.

Cache-first: a hit returns before ``docker run`` is even composed, so a cached
step costs zero CPU, zero network and zero dollars.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .cache import Cache, make_key
from .config import Settings, get_settings
from .errors import SandboxError, SandboxUnavailable
from .policy import PolicyDecision, scrub_env
from .registry import ToolManifest

Backend = Literal["docker", "local"]
ALLOWED_PROGRAMS = {"python", "python3", "uv", "node", "npm", "deno"}
MAX_STDOUT_BYTES = 4_000_000

#: Permission details that mean "this tool may reach the network". A granted
#: `network:<detail>` maps to the default bridge network, which has egress.
EGRESS_DETAILS = frozenset({"http", "https", "any", "dns", "tcp"})


def docker_network_mode(grant: str) -> str:
    """Map a *policy* network grant to a Docker network **mode**.

    ``--network`` takes a network name (``none``/``bridge``/``host``/a custom
    name), never a protocol. Passing the permission detail straight through made
    Docker look for a network literally called ``http`` and fail with
    ``network http not found`` — caught by the Docker CI job.

    Fail-closed for anything unrecognised: an unknown detail grants no egress
    rather than silently joining the bridge.
    """
    detail = (grant or "none").strip().lower()
    if detail in {"", "none"}:
        return "none"
    if detail in EGRESS_DETAILS:
        return "bridge"
    return "none"


@dataclass(slots=True)
class SandboxResult:
    """Outcome of one tool execution attempt."""

    tool: str
    version: str
    ok: bool
    exit_code: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_s: float = 0.0
    cached: bool = False
    timed_out: bool = False
    backend: str = "docker"
    #: Semantic network grant from the policy gate ("none" or, e.g., "http").
    network: str = "none"
    #: The Docker network mode actually applied ("none" | "bridge").
    docker_network: str = "none"
    stdout: str = ""
    stderr: str = ""
    image: str = ""
    image_digest: str = ""
    run_id: str = ""
    container: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cacheable(self) -> bool:
        return self.ok and self.result is not None

    def summary(self) -> str:
        if self.cached:
            return f"{self.tool}@{self.version}: cache hit ({self.duration_s:.3f}s, $0.000000)"
        status = "ok" if self.ok else f"FAILED({self.error or self.exit_code})"
        return (
            f"{self.tool}@{self.version}: {status} in {self.duration_s:.2f}s "
            f"[net={self.network}/docker:{self.docker_network} backend={self.backend}]"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "version": self.version,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "result": self.result,
            "error": self.error,
            "duration_s": round(self.duration_s, 4),
            "cached": self.cached,
            "timed_out": self.timed_out,
            "backend": self.backend,
            "network": self.network,
            "docker_network": self.docker_network,
            "meta": self.meta,
        }


def parse_envelope(stdout: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract the tool's JSON envelope from stdout (tolerates stray log lines)."""
    if not stdout.strip():
        return None, "tool produced no stdout"
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "ok" in payload:
            return payload, None
    return None, "no valid envelope found on stdout (expected a JSON object with an 'ok' field)"


class Sandbox:
    """Executes a validated, policy-approved tool invocation."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backend: Backend | None = None,
        cache: Cache | None = None,
        run_id: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or Cache(self.settings, run_id=run_id)
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.backend: Backend = backend or self.settings.sandbox_backend  # type: ignore[assignment]
        self._docker_ok: bool | None = None

    # ------------------------------------------------------------------ backend
    def docker_available(self) -> bool:
        if self._docker_ok is not None:
            return self._docker_ok
        if shutil.which(self.settings.docker_bin) is None:
            self._docker_ok = False
            return False
        try:
            proc = subprocess.run(
                [self.settings.docker_bin, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self._docker_ok = proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            self._docker_ok = False
        return self._docker_ok

    def image_present(self) -> bool:
        if not self.docker_available():
            return False
        proc = subprocess.run(
            [self.settings.docker_bin, "image", "inspect", self.settings.sandbox_image],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.returncode == 0

    def ensure_backend(self) -> Backend:
        """Verify the configured backend is usable; otherwise fail closed."""
        if self.backend == "docker":
            if self.docker_available():
                if not self.image_present():
                    if self.settings.sandbox_build_image:
                        self.build_image()
                    else:
                        raise SandboxUnavailable(
                            f"Docker image {self.settings.sandbox_image!r} not found. "
                            f"Build it: docker build -t {self.settings.sandbox_image} -f docker/Dockerfile ."
                        )
            elif self.settings.allow_local_sandbox:
                return "local"
            else:
                raise SandboxUnavailable(
                    "Docker is not available. Install/start Docker, or set "
                    "ULTRON_SANDBOX=local together with ULTRON_ALLOW_LOCAL_SANDBOX=1 "
                    "(development only: the local backend is NOT a security boundary)."
                )
            return "docker"
        if self.backend == "local" and not self.settings.allow_local_sandbox:
            raise SandboxUnavailable(
                "the local backend requires an explicit double opt-in: set "
                "ULTRON_ALLOW_LOCAL_SANDBOX=1 (or ULTRON_SANDBOX=local together with it). "
                "It provides no isolation and is for tests/CI only."
            )
        return self.backend

    def build_image(self) -> None:
        dockerfile = self.settings.repo_root / "docker" / "Dockerfile"
        if not dockerfile.exists():
            raise SandboxUnavailable(f"missing {dockerfile}")
        proc = subprocess.run(
            [
                self.settings.docker_bin,
                "build",
                "-f",
                str(dockerfile),
                "-t",
                self.settings.sandbox_image,
                str(self.settings.repo_root),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            raise SandboxUnavailable(f"docker build failed: {proc.stderr[-2000:]}")

    # --------------------------------------------------------------------- run
    def run(
        self,
        manifest: ToolManifest,
        inputs: dict[str, Any],
        decision: PolicyDecision,
        *,
        use_cache: bool = True,
    ) -> SandboxResult:
        """Execute ``manifest`` with ``inputs`` under a granted policy decision."""
        if not decision.allowed:
            raise SandboxError(
                f"refusing to execute {manifest.key}: policy decision was {decision.action}"
            )
        backend = self.ensure_backend()

        cache_key = make_key("tool", manifest.content_hash, inputs, decision.network)
        ttl = (
            manifest.cache_ttl_s
            if manifest.cache_ttl_s is not None
            else self.settings.cache_ttl_tool
        )
        if use_cache and ttl > 0:
            entry = self.cache.get("tool", cache_key)
            if entry is not None:
                payload = dict(entry.value)
                return SandboxResult(
                    tool=manifest.name,
                    version=manifest.version,
                    ok=payload.get("ok", True),
                    exit_code=payload.get("exit_code", 0),
                    result=payload.get("result"),
                    duration_s=0.0,
                    cached=True,
                    backend=payload.get("backend", backend),
                    network=decision.network,
                    docker_network=docker_network_mode(decision.network),
                    run_id=self.run_id,
                    meta={"cache_age_s": round(entry.age_s, 2), **payload.get("meta", {})},
                )

        argv = shlex.split(manifest.entrypoint)
        if not argv:
            raise SandboxError(f"{manifest.key}: empty entrypoint")
        program = Path(argv[0]).name
        if program not in ALLOWED_PROGRAMS:
            raise SandboxError(
                f"{manifest.key}: program {program!r} is not in the Phase 1 allowlist "
                f"({', '.join(sorted(ALLOWED_PROGRAMS))})"
            )

        start = time.perf_counter()
        if backend == "docker":
            outcome = self._run_docker(manifest, inputs, argv, decision)
        else:
            outcome = self._run_local(manifest, inputs, argv, decision)
        outcome.duration_s = round(time.perf_counter() - start, 4)

        if use_cache and ttl > 0 and outcome.cacheable:
            self.cache.set(
                "tool",
                cache_key,
                {
                    "ok": outcome.ok,
                    "exit_code": outcome.exit_code,
                    "result": outcome.result,
                    "backend": outcome.backend,
                    "docker_network": outcome.docker_network,
                    "meta": outcome.meta,
                },
                ttl_s=ttl,
            )
        return outcome

    # ------------------------------------------------------------------ docker
    def _docker_argv(
        self,
        manifest: ToolManifest,
        argv: list[str],
        decision: PolicyDecision,
        *,
        container_name: str,
    ) -> tuple[list[str], float]:
        repo = str(self.settings.repo_root)
        limits = decision.limits or {}
        timeout_s = float(limits.get("timeout_s") or self.settings.sandbox_timeout_s)
        memory = str(limits.get("memory") or self.settings.sandbox_memory)
        cpus = str(limits.get("cpus") or self.settings.sandbox_cpus)
        pids = int(limits.get("pids") or self.settings.sandbox_pids)

        args = [
            self.settings.docker_bin,
            "run",
            "--rm",
            "--interactive",
            "--name",
            container_name,
            "--init",
            "--label",
            f"ultron.run={self.run_id}",
            "--label",
            f"ultron.tool={manifest.key}",
            # --- isolation -------------------------------------------------
            # `--network` needs a docker network NAME (none/bridge/...); the
            # policy grant ("http") is semantic and mapped here.
            f"--network={docker_network_mode(decision.network)}",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={self.settings.sandbox_tmpfs_mb}m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--ipc=none",
            # --- resource ceilings -----------------------------------------
            f"--memory={memory}",
            f"--memory-swap={memory}",
            f"--cpus={cpus}",
            f"--pids-limit={pids}",
            "--ulimit",
            "nofile=256:256",
            # --- filesystem / env ------------------------------------------
            "-v",
            f"{repo}:/workspace:ro",
            "-w",
            "/workspace",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "HOME=/tmp",
        ]
        for name, value in scrub_env(dict(os.environ), self.settings.env_allowlist).items():
            args += ["-e", f"{name}={value}"]
        args += [self.settings.sandbox_image, *argv]
        return args, timeout_s

    def _run_docker(
        self,
        manifest: ToolManifest,
        inputs: dict[str, Any],
        argv: list[str],
        decision: PolicyDecision,
    ) -> SandboxResult:
        container = f"ultron-{self.run_id[:12]}-{manifest.name[:20]}-{uuid.uuid4().hex[:6]}"
        docker_argv, timeout_s = self._docker_argv(
            manifest, argv, decision, container_name=container
        )
        image_digest = self._image_digest()

        # TODO(phase-2): `capture_output=True` buffers the tool's *entire* stdout in
        # host memory, and MAX_STDOUT_BYTES is only applied after the process has
        # exited — so it does not bound memory. `--memory` caps the container's RSS,
        # not the host-side pipe buffer, so a tool that floods stdout can exhaust
        # host memory faster than the wall-clock timeout can stop it: a host-side
        # DoS vector for any tool we did not write ourselves. Fix with bounded
        # streaming (read at most MAX_STDOUT_BYTES + slack from a pipe, then kill
        # the container; cap the producer side inside the container too).
        # Tracked as issue #2 "bounded stdout streaming in sandbox".
        timed_out = False
        try:
            proc = subprocess.run(
                docker_argv,
                input=json.dumps(inputs, default=str),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            self._kill(container)
            stdout = (
                (exc.stdout or b"").decode()
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                (exc.stderr or b"").decode()
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            code = None
        except OSError as exc:
            raise SandboxUnavailable(f"failed to invoke docker: {exc}") from exc

        if len(stdout) > MAX_STDOUT_BYTES:
            stdout = stdout[:MAX_STDOUT_BYTES]

        envelope, parse_error = parse_envelope(stdout)
        ok = bool(envelope and envelope.get("ok")) and code == 0 and not timed_out
        result = envelope.get("result") if envelope else None
        error: str | None = None
        if timed_out:
            error = f"timeout after {timeout_s:.0f}s (container killed)"
        elif code != 0:
            error = (envelope or {}).get("error") or f"exit code {code}"
        elif parse_error:
            error = parse_error
        elif envelope and not envelope.get("ok"):
            error = str(envelope.get("error") or "tool reported failure")

        return SandboxResult(
            tool=manifest.name,
            version=manifest.version,
            ok=ok,
            exit_code=code,
            result=result if isinstance(result, dict) else None,
            error=error,
            timed_out=timed_out,
            backend="docker",
            network=decision.network,
            docker_network=docker_network_mode(decision.network),
            stdout=stdout[-4000:],
            stderr=stderr[-4000:],
            image=self.settings.sandbox_image,
            image_digest=image_digest,
            run_id=self.run_id,
            container=container,
            meta={"tool_meta": (envelope or {}).get("meta", {})},
        )

    def _kill(self, container: str) -> None:
        # Best effort: the container is already gone if the daemon is unreachable.
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [self.settings.docker_bin, "kill", container],
                capture_output=True,
                text=True,
                timeout=15,
            )

    def _image_digest(self) -> str:
        try:
            proc = subprocess.run(
                [
                    self.settings.docker_bin,
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    self.settings.sandbox_image,
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return proc.stdout.strip() if proc.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return ""

    # ------------------------------------------------------------------- local
    def _run_local(
        self,
        manifest: ToolManifest,
        inputs: dict[str, Any],
        argv: list[str],
        decision: PolicyDecision,
    ) -> SandboxResult:
        """UNSAFE dev/test backend. No isolation, no network policy, no fs policy.

        It exists so the harness can be tested and CI'd on machines without
        Docker. It refuses to run unless ``ULTRON_ALLOW_LOCAL_SANDBOX=1``.

        Same stdout-buffering caveat as the docker path (see the TODO in
        ``_run_docker`` / issue #2), and no isolation whatsoever.
        """
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": "/tmp",
            "PYTHONPATH": str(self.settings.repo_root),
            **scrub_env(dict(os.environ), self.settings.env_allowlist),
        }
        resolved = [sys.executable if p in {"python", "python3"} else p for p in argv]
        timeout_s = float(decision.limits.get("timeout_s") or self.settings.sandbox_timeout_s)
        timed_out = False
        try:
            proc = subprocess.run(
                resolved,
                input=json.dumps(inputs, default=str),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(self.settings.repo_root),
                env=env,
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            code = None

        envelope, parse_error = parse_envelope(stdout or "")
        ok = bool(envelope and envelope.get("ok")) and code == 0 and not timed_out
        error = None
        if timed_out:
            error = f"timeout after {timeout_s:.0f}s"
        elif code != 0:
            error = (envelope or {}).get("error") or f"exit code {code}"
        elif parse_error:
            error = f"{parse_error} (local backend)"
        return SandboxResult(
            tool=manifest.name,
            version=manifest.version,
            ok=ok,
            exit_code=code,
            result=(envelope or {}).get("result") if envelope else None,
            error=error,
            timed_out=timed_out,
            backend="local",
            network="host(unenforced)",
            stdout=(stdout or "")[-4000:],
            stderr=(stderr or "")[-4000:],
            run_id=self.run_id,
            meta={
                "tool_meta": (envelope or {}).get("meta", {}),
                "warning": "local backend: not a sandbox",
            },
        )

    # ------------------------------------------------------------------ helpers
    def docker_command_preview(self, manifest: ToolManifest, decision: PolicyDecision) -> list[str]:
        """The exact ``docker run`` argv that *would* be used (used by tests/docs)."""
        argv, _ = self._docker_argv(
            manifest,
            shlex.split(manifest.entrypoint),
            decision,
            container_name="ultron-preview",
        )
        return argv
