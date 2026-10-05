"""Docker sandbox executor. The only thing in the harness that runs tool code.

Container hard guarantees (all asserted by ``tests/test_smoke.py``):

* ``--network none`` **unless** the manifest declares ``network:*`` *and* the
  policy gate granted it (the gate's decision is the only input considered here).
* ``--read-only`` root filesystem + a small ``noexec,nosuid`` tmpfs at ``/tmp``.
  The repo is mounted ``:ro``; a tool cannot mutate the harness or itself.
* No secrets: the container env is only ``PYTHON*`` plus explicitly allowlisted
  names plus the tool's own ``web_mock``/``eval_live`` values, filtered again by
  :func:`policy.scrub_env` for secret-looking keys. The host environment is never
  inherited (``--env-file`` is not used).
* Exactly one mount, read-only (the repo at ``/workspace:ro``), and the only
  writable path inside the container is the noexec tmpfs. No mode adds another --
  not even live network mode.
* Dropped capabilities, no new privileges, non-root uid, pid/memory/cpu/file
  limits, and a wall-clock timeout after which the container is killed.
* The entrypoint is executed as **argv without a shell** (manifest validation
  already rejected metacharacters), so nothing can chain commands.

Cache-first: a hit returns before ``docker run`` is even composed, so a cached
step costs zero CPU, zero network and zero dollars.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .cache import Cache, make_key
from .config import Settings, get_settings
from .errors import SandboxError, SandboxUnavailable
from .policy import PolicyDecision, scrub_env
from .provenance import (
    ProvenanceEnvelope,
    deserialize_envelopes,
    serialize_envelopes,
)
from .registry import ToolManifest

Backend = Literal["docker", "local"]
ALLOWED_PROGRAMS = {"python", "python3", "uv", "node", "npm", "deno"}

#: Per-stream ceiling on what the *host* keeps from a tool. Applied while
#: reading, not after: a tool that floods stdout must not be able to make the
#: harness buffer its output (see :func:`run_bounded`).
MAX_STDOUT_BYTES = 4_000_000
#: Pipe read size. Host memory for a run is bounded by ``limit + one chunk``
#: per stream, no matter how much the tool writes.
READ_CHUNK_BYTES = 64 * 1024
#: How long a killed producer gets to actually die before we SIGKILL it.
REAP_GRACE_S = 5.0

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
    provenance: list[ProvenanceEnvelope] = field(default_factory=list)
    # Populated for deferred writes; the Agent commits only after verification.
    cache_key: str = ""
    cache_ttl_s: float | None = None

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
            "provenance": serialize_envelopes(self.provenance),
        }


@dataclass(slots=True)
class StreamCapture:
    """One pipe's bounded capture: a sliding *tail* plus byte accounting.

    The tail is what we keep because a tool's JSON envelope is the **last** line
    it prints: when a flood forces us to drop bytes, the bytes worth keeping are
    at the end, not the start.

    ``dropped`` is what was evicted from *our* window, which is a lower bound on
    what the tool emitted: reading stops at the cap (that is the point), so the
    harness never learns the tool's true total. The marker says exactly that
    rather than inventing a number.
    """

    kept: bytearray = field(default_factory=bytearray)
    total: int = 0
    dropped: int = 0

    @property
    def truncated(self) -> bool:
        return self.dropped > 0

    def text(self) -> str:
        body = bytes(self.kept).decode("utf-8", "replace")
        if self.dropped:
            return f"[TRUNCATED {self.dropped} bytes]\n{body}"
        return body


@dataclass(slots=True)
class BoundedRun:
    """Result of :func:`run_bounded`."""

    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    capped: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_dropped: int = 0
    stderr_dropped: int = 0
    spawned: bool = True
    spawn_error: str | None = None

    def as_meta(self) -> dict[str, Any]:
        return {
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "stdout_dropped": self.stdout_dropped,
            "stderr_dropped": self.stderr_dropped,
            "output_capped": self.capped,
        }


class _KillOnce:
    """Runs the kill hook at most once, from whichever thread sees the problem."""

    def __init__(self, hook: Callable[[subprocess.Popen], None] | None) -> None:
        self.hook = hook
        self.fired = False
        self._lock = threading.Lock()

    def __call__(self, proc: subprocess.Popen) -> None:
        with self._lock:
            if self.fired:
                return
            self.fired = True
        if self.hook is not None:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                self.hook(proc)


def _pump(
    stream: Any,
    capture: StreamCapture,
    limit: int,
    on_over: _KillOnce,
    proc: subprocess.Popen,
) -> None:
    """Read one pipe to EOF, keeping at most ``limit`` bytes; kill on overflow."""
    try:
        while True:
            chunk = stream.read(READ_CHUNK_BYTES)
            if not chunk:
                return
            capture.total += len(chunk)
            capture.kept += chunk
            overflow = len(capture.kept) - limit
            if overflow > 0:
                del capture.kept[:overflow]
                capture.dropped += overflow
                # The producer is flooding: kill it and STOP reading. Nothing is
                # left to read for anyway -- the process is about to die and we
                # already hold a full window of its output.
                on_over(proc)
                return
    except (OSError, ValueError):
        return
    finally:
        with contextlib.suppress(OSError, ValueError):
            stream.close()


def _feed_stdin(stdin: Any, payload: bytes) -> None:
    """Write the tool's input and close stdin, from its own thread.

    A separate thread because the tool may print megabytes before it reads a
    single byte of stdin; writing inline would block the host on a full pipe.
    """
    try:
        if payload:
            stdin.write(payload)
        stdin.close()
    except (OSError, ValueError):
        with contextlib.suppress(OSError, ValueError):
            stdin.close()


def _start_waiter(proc: subprocess.Popen) -> tuple[threading.Event, threading.Thread]:
    """Blocking ``wait()`` on its own thread, so the main thread never polls.

    ``Popen.wait(timeout=...)`` is not a blocking wait: it polls ``waitpid`` with
    exponentially growing sleeps (0.5 ms -> 1 -> 2 ... -> 50 ms). For a tool that
    runs for tens of milliseconds that quantisation shows up as several ms of
    pure overhead per call. A thread parked in a blocking ``wait()`` plus an
    ``Event`` gives the same semantics with no polling.
    """
    done = threading.Event()

    def _wait() -> None:
        try:
            proc.wait()
        finally:
            done.set()

    thread = threading.Thread(target=_wait, daemon=True, name="ultron-reaper")
    thread.start()
    return done, thread


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill the child *and its children* (local backend runs real processes)."""
    with contextlib.suppress(OSError, ProcessLookupError):
        if proc.pid:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        proc.kill()


def run_bounded(
    argv: list[str],
    *,
    input_text: str = "",
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float = 60.0,
    limit: int = MAX_STDOUT_BYTES,
    on_limit: Callable[[subprocess.Popen], None] | None = None,
    start_new_session: bool = False,
) -> BoundedRun:
    """Run ``argv`` with stdout/stderr streamed into a bounded window.

    This replaces ``subprocess.run(capture_output=True)``, which buffered the
    tool's entire output in host memory and only truncated *afterwards* -- the
    container's ``--memory`` cap does not cover the host-side pipe buffer, so a
    tool that floods stdout could exhaust host memory well before the wall-clock
    timeout stopped it (issue #2).

    Guarantees:

    * host memory per stream never exceeds ``limit`` (+ one read chunk);
    * the first time either stream overflows, ``on_limit(proc)`` fires exactly
      once (docker: ``docker kill``; local: process-group SIGKILL) and reading
      stops;
    * what is kept is the **tail**, with a visible ``[TRUNCATED n bytes]`` marker
      in front, so envelope parsing still works;
    * the child is always reaped, even if the hook fails.
    """
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=start_new_session,
            bufsize=0,
        )
    except OSError as exc:
        return BoundedRun(spawned=False, spawn_error=str(exc))

    stdout, stderr = StreamCapture(), StreamCapture()
    default_kill: Callable[[subprocess.Popen], None] = (
        _kill_process_tree if start_new_session else (lambda p: p.kill())
    )
    kill = _KillOnce(on_limit if on_limit is not None else default_kill)

    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, stdout, limit, kill, proc), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, stderr, limit, kill, proc), daemon=True),
        threading.Thread(
            target=_feed_stdin,
            args=(proc.stdin, input_text.encode("utf-8")),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    done, waiter = _start_waiter(proc)
    timed_out = False
    if not done.wait(timeout=timeout_s):
        timed_out = True
        kill(proc)
        if not done.wait(timeout=REAP_GRACE_S):
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                proc.kill()
            done.wait(timeout=REAP_GRACE_S)
    code = proc.returncode

    threads.append(waiter)
    for thread in threads:
        # The readers hit EOF as soon as the process (and any child holding the
        # pipe) is gone; a bounded join keeps a pathological grandchild from
        # stalling the harness -- the fds are closed right after.
        thread.join(timeout=2.0)
    for stream in (proc.stdout, proc.stderr, proc.stdin):
        with contextlib.suppress(OSError, ValueError):
            stream.close()

    return BoundedRun(
        exit_code=code,
        stdout=stdout.text(),
        stderr=stderr.text(),
        timed_out=timed_out,
        capped=stdout.truncated or stderr.truncated,
        stdout_bytes=stdout.total,
        stderr_bytes=stderr.total,
        stdout_dropped=stdout.dropped,
        stderr_dropped=stderr.dropped,
    )


def tail_excerpt(text: str, dropped: int, n: int = 4000) -> str:
    """Last ``n`` characters of a stream for the result, marker included.

    Truncation is a property of the *run*, so it must survive into the result's
    short excerpt -- otherwise ``result.stdout`` looks like a healthy wall of
    bytes and the marker is only visible in a 4 MB field nobody prints.
    """
    head = f"[TRUNCATED {dropped} bytes]\n" if dropped else ""
    return head + text[-n:]


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


def _wheel_dependencies(manifest: ToolManifest, settings: Settings) -> list[dict[str, str]]:
    """Validate sealed wheel attestations and return safe read-only mounts."""
    dependencies = manifest.meta.get("dependencies", {}) if manifest.meta else {}
    raw_items = dependencies.get("wheels", []) if isinstance(dependencies, dict) else []
    if not isinstance(raw_items, list):
        raise SandboxError(f"{manifest.key}: wheel dependencies must be a list")
    wheel_root = (Path(settings.state_dir) / "wheels").resolve()
    validated: list[dict[str, str]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise SandboxError(f"{manifest.key}: malformed wheel dependency")
        package = str(raw.get("package") or "").strip()
        filename = Path(str(raw.get("filename") or "")).name
        path = Path(str(raw.get("path") or wheel_root / filename)).resolve()
        expected = str(raw.get("sha256") or "").lower()
        if not package or not filename.endswith((".whl", ".tar.gz", ".zip")) or not expected:
            raise SandboxError(
                f"{manifest.key}: wheel dependency lacks package, filename, or sha256"
            )
        try:
            path.relative_to(wheel_root)
        except ValueError as exc:
            raise SandboxError(
                f"{manifest.key}: wheel path escapes sealed wheel directory"
            ) from exc
        if not path.is_file():
            raise SandboxError(f"{manifest.key}: wheel is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SandboxError(
                f"{manifest.key}: wheel sha256 mismatch for {filename}: expected {expected}, got {digest}"
            )
        validated.append(
            {
                "package": package,
                "version": str(raw.get("version") or ""),
                "filename": filename,
                "path": str(path),
            }
        )
    return validated


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
        defer_cache_write: bool = False,
    ) -> SandboxResult:
        """Execute ``manifest`` with ``inputs`` under a granted policy decision."""
        if not decision.allowed:
            raise SandboxError(
                f"refusing to execute {manifest.key}: policy decision was {decision.action}"
            )
        backend = self.ensure_backend()

        cache_key = make_key("tool", manifest.content_hash, inputs, decision.network)
        input_provenance = ProvenanceEnvelope.user_input(
            inputs,
            source_id=f"input:{manifest.key}:{make_key(inputs)[:32]}",
        )
        ttl = (
            manifest.cache_ttl_s
            if manifest.cache_ttl_s is not None
            else self.settings.cache_ttl_tool
        )
        if use_cache and ttl > 0:
            entry = self.cache.get("tool", cache_key)
            if entry is not None:
                payload = dict(entry.value)
                upstream = deserialize_envelopes(payload.get("provenance"))
                if not upstream:
                    # Pre-provenance cache rows are usable as raw evidence, but
                    # remain unverified until the normal verifier runs again.
                    upstream = [
                        ProvenanceEnvelope.tool_output(
                            manifest.key,
                            manifest.content_hash,
                            payload.get("result"),
                            verified=False,
                            metadata={"ok": payload.get("ok", False)},
                        )
                    ]
                cache_provenance = ProvenanceEnvelope.cache_hit(
                    cache_key,
                    payload.get("result"),
                    verified=False,
                    upstream=upstream,
                )
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
                    provenance=[input_provenance, cache_provenance, *upstream],
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
        output_origin = "web_fetch" if manifest.wants_network else "sandbox_tool"
        output_provenance = ProvenanceEnvelope.tool_output(
            manifest.key,
            manifest.content_hash,
            outcome.result,
            verified=False,
            origin=output_origin,
            metadata={
                "ok": outcome.ok,
                "error": outcome.error,
                "raw_stdout": outcome.stdout,
                "raw_stderr": outcome.stderr,
            },
        )
        outcome.provenance = [input_provenance, output_provenance]
        outcome.cache_key = cache_key
        outcome.cache_ttl_s = float(ttl)

        if use_cache and not defer_cache_write:
            self.commit_cache(outcome)
        return outcome

    def commit_cache(self, outcome: SandboxResult) -> bool:
        """Write a tool result only after its caller has verified it.

        The default low-level Sandbox API keeps its historical eager-cache
        behaviour for compatibility. Agent executions pass ``defer_cache_write``
        and call this method only after Verifier plus Breaker approval.
        """
        if not outcome.cache_key or outcome.cache_ttl_s is None or not outcome.cacheable:
            return False
        self.cache.set(
            "tool",
            outcome.cache_key,
            {
                "ok": outcome.ok,
                "exit_code": outcome.exit_code,
                "result": outcome.result,
                "backend": outcome.backend,
                "docker_network": outcome.docker_network,
                "meta": outcome.meta,
                "provenance": serialize_envelopes(outcome.provenance),
            },
            ttl_s=outcome.cache_ttl_s,
        )
        return True

    # --------------------------------------------------------------- tool env
    def _container_path(self, host_path: str) -> str:
        """Translate a host path under the repo into its container equivalent.

        The repo is mounted read-only at ``/workspace``, so a fixture path that
        is valid for the local backend (``/home/me/repo/eval/fixtures/...``) has
        to become ``/workspace/eval/fixtures/...`` inside the container. Paths
        that are already container paths (or live elsewhere) are passed through.
        """
        try:
            relative = Path(host_path).resolve().relative_to(self.settings.repo_root.resolve())
        except (ValueError, OSError):
            return str(host_path)
        return f"/workspace/{relative.as_posix()}"

    def _tool_env(self, *, container: bool = False) -> dict[str, str]:
        """Environment for the *tool*, sourced from this sandbox's Settings.

        Deliberately not read from the ambient process environment: Settings is
        the single source of truth, so two sandboxes built from two Settings
        objects cannot contaminate each other, and a fixture configured for one
        run cannot leak into an unrelated subprocess or test.
        """
        env: dict[str, str] = {}
        if self.settings.web_mock:
            mock = str(self.settings.web_mock)
            env["ULTRON_WEB_MOCK"] = self._container_path(mock) if container else mock
        if self.settings.eval_live:
            env["ULTRON_EVAL_LIVE"] = "1"
            env["ULTRON_CACHE_TTL_WEB"] = str(self.settings.cache_ttl_web)
            if not container:
                # The URL-keyed page cache is a *local backend* feature. The docker
                # sandbox gets no extra mount for it: the container's root fs stays
                # read-only and its only writable space is the per-run noexec tmpfs,
                # which cannot serve as a cache. Live docker runs fetch without the
                # page cache; the harness-level envelope cache still applies.
                env["ULTRON_WEB_CACHE"] = str(self.settings.web_cache_path)
        return env

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
        # The tool's own env comes from Settings; the ambient allowlist is a
        # separate, operator-controlled channel. Settings wins on a name clash,
        # and exactly one ``-e`` is emitted per name.
        tool_env = self._tool_env(container=True)
        allowed = scrub_env(dict(os.environ), self.settings.env_allowlist)
        allowed.update(tool_env)
        # No writable mount is ever added, in any mode. The only volume attached
        # is the repo at `:ro`; the only writable path inside the container is the
        # noexec/nosuid tmpfs. Live mode does not widen the container's
        # filesystem footprint at all -- it only permits egress.
        for name, value in sorted(allowed.items()):
            args += ["-e", f"{name}={value}"]
        wheel_items = _wheel_dependencies(manifest, self.settings)
        for item in wheel_items:
            args += [
                "-v",
                f"{item['path']}:/wheels/{item['filename']}:ro",
            ]
        if wheel_items:
            packages = [
                f"{item['package']}=={item['version']}" if item.get("version") else item["package"]
                for item in wheel_items
            ]
            setup = (
                "# pip install --no-index --find-links /wheels <pkg>;"
                "import os,subprocess,sys;"
                "subprocess.run([sys.executable,'-m','pip','install','--user','--no-index',"
                "'--find-links','/wheels',"
                + ",".join(repr(package) for package in packages)
                + "],check=True);"
                "os.execvp(sys.argv[1],sys.argv[1:])"
            )
            argv = ["python", "-c", setup, *argv]
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

        # Bounded streaming, not capture_output: the host keeps at most
        # MAX_STDOUT_BYTES per stream and kills the container the moment the tool
        # exceeds it (issue #2). `--memory` bounds the container, not the host
        # pipe buffer, so buffering first and truncating later was a DoS vector.
        run = run_bounded(
            docker_argv,
            input_text=json.dumps(inputs, default=str),
            timeout_s=timeout_s,
            limit=MAX_STDOUT_BYTES,
            on_limit=lambda _proc: self._kill(container),
        )
        if not run.spawned:
            raise SandboxUnavailable(f"failed to invoke docker: {run.spawn_error}")

        stdout, stderr, code = run.stdout, run.stderr, run.exit_code
        timed_out = run.timed_out

        # The kill was requested from the reader thread; make sure it landed
        # before returning a result. `container_stopped` is reported either way,
        # so a leak would be visible in the envelope rather than inferred.
        container_stopped: bool | None = None
        if run.capped or timed_out:
            container_stopped = self._ensure_container_stopped(container)

        envelope, parse_error = parse_envelope(stdout)
        ok = (
            bool(envelope and envelope.get("ok")) and code == 0 and not timed_out and not run.capped
        )
        result = envelope.get("result") if envelope else None
        error: str | None = None
        if timed_out:
            error = f"timeout after {timeout_s:.0f}s (container killed)"
        elif run.capped:
            streams = []
            if run.stdout_dropped:
                streams.append(f"stdout {run.stdout_bytes} B")
            if run.stderr_dropped:
                streams.append(f"stderr {run.stderr_bytes} B")
            error = (
                f"output exceeded {MAX_STDOUT_BYTES} bytes ({', '.join(streams)}); container killed"
            )
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
            stdout=tail_excerpt(stdout, run.stdout_dropped),
            stderr=tail_excerpt(stderr, run.stderr_dropped),
            image=self.settings.sandbox_image,
            image_digest=image_digest,
            run_id=self.run_id,
            container=container,
            meta={
                "tool_meta": (envelope or {}).get("meta", {}),
                **run.as_meta(),
                "container_stopped": container_stopped,
            },
        )

    def _kill(self, container: str) -> bool:
        """``docker kill`` one container; report whether the command succeeded.

        Deliberately not fire-and-forget: a kill that silently fails leaves a
        runaway tool running, which is the failure mode bounded output exists to
        prevent. The caller decides what to do about a ``False``.
        """
        try:
            proc = subprocess.run(
                [self.settings.docker_bin, "kill", container],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        # A container that is already gone is a success from our point of view.
        return proc.returncode == 0 or "No such container" in (proc.stderr or "")

    def _ensure_container_stopped(self, container: str, *, grace_s: float = 8.0) -> bool:
        """Verify a requested kill took effect, retrying until the deadline.

        ``docker kill`` is asynchronous from the host's point of view: it returns
        as soon as the daemon accepts the signal, and the daemon may still report
        the container as running for a while after that (observed in CI: the
        flooder blocked writing into a pipe we had stopped reading, and the first
        kill did not stop it). A sandbox that *requests* a kill and does not check
        leaks the very process the cap was meant to stop, so this confirms the
        state, retries, and returns False if the container refuses to die.
        """
        deadline = time.time() + grace_s
        while True:
            try:
                proc = subprocess.run(
                    [
                        self.settings.docker_bin,
                        "inspect",
                        "--format",
                        "{{.State.Running}}",
                        container,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except (OSError, subprocess.SubprocessError):
                return False
            if proc.returncode != 0 or proc.stdout.strip() == "false":
                return True  # removed by --rm, or stopped: both are "not running"
            if time.time() >= deadline:
                return False
            self._kill(container)
            time.sleep(0.5)

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

        Output is streamed with the same bound as the docker path (issue #2):
        the tool is SIGKILLed, process group and all, the moment it exceeds
        MAX_STDOUT_BYTES on either stream. No isolation whatsoever.
        """
        wheel_items = _wheel_dependencies(manifest, self.settings)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": "/tmp",
            "PYTHONPATH": os.pathsep.join(
                [str(self.settings.repo_root), *(item["path"] for item in wheel_items)]
            ),
            **scrub_env(dict(os.environ), self.settings.env_allowlist),
            **self._tool_env(),
        }
        resolved = [sys.executable if p in {"python", "python3"} else p for p in argv]
        timeout_s = float(decision.limits.get("timeout_s") or self.settings.sandbox_timeout_s)
        run = run_bounded(
            resolved,
            input_text=json.dumps(inputs, default=str),
            cwd=str(self.settings.repo_root),
            env=env,
            timeout_s=timeout_s,
            limit=MAX_STDOUT_BYTES,
            start_new_session=True,  # own process group, so the kill hits children too
        )
        if not run.spawned:
            raise SandboxUnavailable(f"failed to spawn tool: {run.spawn_error}")

        stdout, code = run.stdout, run.exit_code
        timed_out = run.timed_out
        envelope, parse_error = parse_envelope(stdout)
        ok = (
            bool(envelope and envelope.get("ok")) and code == 0 and not timed_out and not run.capped
        )
        error = None
        if timed_out:
            error = f"timeout after {timeout_s:.0f}s"
        elif run.capped:
            error = (
                f"output exceeded {MAX_STDOUT_BYTES} bytes "
                f"(stdout {run.stdout_bytes} B, stderr {run.stderr_bytes} B); "
                "process killed"
            )
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
            stdout=tail_excerpt(stdout, run.stdout_dropped),
            stderr=tail_excerpt(run.stderr, run.stderr_dropped),
            run_id=self.run_id,
            meta={
                "tool_meta": (envelope or {}).get("meta", {}),
                "warning": "local backend: not a sandbox",
                **run.as_meta(),
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
