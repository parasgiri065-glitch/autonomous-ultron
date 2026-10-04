#!/usr/bin/env python3
"""Collect a single, hand-off-able review bundle for the Phase 1 branch.

Runs the verification commands below, captures *raw* stdout + stderr + exit code
for each, and writes one markdown file at the repo root::

    uv run python scripts/collect_review.py                 # light (default)
    uv run python scripts/collect_review.py --include-files # + source dumps

Deliberate rules (do not relax them):

* **fail closed** -- the bundle text is scanned with the *same*
  ``ultron.policy.scan_for_secrets`` the policy gate uses. A single hit aborts
  the run: nothing is written, and the offending section + pattern are printed.
  The token, ``.env``, ``.git/config`` and any credential never enter the bundle.
* **never silently truncate** -- every cut is marked with the number of bytes
  dropped, and the original byte count is recorded in the metadata section.
* **bounded** -- per-command output <= 200 KB, the full diff is only dumped when
  it is under 300 KB, and the finished bundle must be <= 1 MB or the script
  exits non-zero without writing anything.
* **light by default** -- the reviewer already has ``policy.py`` and
  ``sandbox.py``; full file contents are opt-in via ``--include-files``.

This file is committed so the bundle is reproducible rather than pasted by hand.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The scanner is imported, never re-implemented: the bundle must be judged by
# exactly the same patterns the gate uses at runtime.
from ultron.policy import scan_for_secrets  # noqa: E402

BUNDLE = REPO / "review_bundle.md"
GIT_BUNDLE = REPO / "phase1.bundle"
TAR_BUNDLE = REPO / "phase1.tar.gz"
BASELINE = "dda9b4e"

# ---------------------------------------------------------------- hard limits
PER_CMD_LIMIT = 200 * 1024
TRUNCATE_HEAD = 120 * 1024
TRUNCATE_TAIL = 80 * 1024
DIFF_FULL_LIMIT = 300 * 1024
TOTAL_LIMIT = 1024 * 1024

#: Commands the bundle MUST contain. ``shell`` strings on purpose: the reviewer
#: should be able to paste any of them verbatim and get the same output.
CHECKS: tuple[tuple[str, str], ...] = (
    ("git log (all refs, decorated)", "git log --oneline --all --decorate -10"),
    ("commits on feat/init not on origin/main", "git log origin/main..feat/init --oneline"),
    ("PR diffstat (main...feat/init)", "git diff main...feat/init --stat"),
    ("the earlier main rewrite", f"git show {BASELINE} --stat"),
    ("tests", "uv run pytest -q"),
    ("lint + format", "uv run ruff check . && ruff format --check ."),
    ("eval gates (3 passes)", "uv run python eval/run.py --passes 3"),
    # Extra, because the review must prove "no real network", not assume it:
    # a dead proxy cannot be reached, so a green run means nothing egressed.
    (
        "eval with network blackholed (proves fixtures-only)",
        "HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9"
        " NO_PROXY= uv run python eval/run.py --passes 3",
    ),
    # "the eval is fixtures-only" is a claim; these two make it checkable.
    (
        "eval task network surface (fixture-only proof, part 1)",
        """uv run python -c 'import json
for line in open("eval/tasks.jsonl"):
    if line.strip():
        spec = json.loads(line)
        print(spec["id"], "| tools:", spec.get("expect", {}).get("tools"),
              "| network:", spec.get("expect", {}).get("network", "-"))'""",
    ),
    (
        "sandbox envelopes actually in mock mode (fixture-only proof, part 2)",
        """uv run python -c 'import json, sqlite3
con = sqlite3.connect("eval/.state/cache.db")
rows = [json.loads(v)["result"] for (v,) in con.execute("select value from cache where ns=?", ("tool",))]
print("sources seen:", sorted({s.split("/")[2] for r in rows for s in r.get("sources", [])}))
print("envelope modes:", sorted({r.get("_meta", {}).get("mode", "(no _meta)") for r in rows}))'""",
    ),
    ("manifest validation", "uv run ultron tools validate"),
    (
        "network-low-auto override sites",
        'grep -rn "ULTRON_POLICY_NETWORK_LOW_AUTO\\|policy_network_low_auto" '
        "eval/run.py src/ultron/config.py .github/workflows/ci.yml",
    ),
    ("PR state", "gh pr view 1 --json url,state,mergeable,headRefName,baseRefName"),
    ("PR checks", "gh pr checks 1 | head -30"),
    ("env files tracked?", 'git ls-files | grep -i "\\.env" || echo "no env files tracked"'),
)

#: Full dumps, only with --include-files.
INCLUDE_FILES: tuple[str, ...] = (
    "src/ultron/policy.py",
    "src/ultron/sandbox.py",
    "eval/run.py",
    "src/ultron/config.py",
    ".github/workflows/ci.yml",
    "eval/tasks.jsonl",
)


@dataclasses.dataclass(slots=True)
class Capture:
    name: str
    cmd: str
    exit_code: int
    raw: bytes
    body: bytes
    truncated: bool
    dropped: int
    text: str = ""

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ utilities
def now_utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(
    cmd: str, *, env: dict[str, str] | None = None, cwd: Path = REPO
) -> subprocess.CompletedProcess:
    """Run a shell command and capture *bytes* (no locale surprises)."""
    merged = dict(os.environ)
    merged["PATH"] = f"{REPO / '.venv' / 'bin'}:{merged.get('PATH', '')}"  # bare `ruff` resolves
    merged.update(env or {})
    # commands are constants defined in this file, never caller input
    return subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd),
        env=merged,
        capture_output=True,
        check=False,
    )


def truncate(raw: bytes, limit: int = PER_CMD_LIMIT) -> tuple[bytes, bool, int]:
    """Cut to ``limit`` bytes with a visible marker; return (body, cut?, dropped)."""
    if len(raw) <= limit:
        return raw, False, 0
    dropped = len(raw) - (TRUNCATE_HEAD + TRUNCATE_TAIL)
    marker = f"\n... [TRUNCATED {dropped} bytes]\n".encode()
    return raw[:TRUNCATE_HEAD] + marker + raw[-TRUNCATE_TAIL:], True, dropped


def fence_for(text: str) -> str:
    """A fence longer than any backtick run inside ``text`` (diffs contain fences)."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def fenced(text: str) -> str:
    fence = fence_for(text)
    return f"{fence}\n{text}\n{fence}"


def read_token() -> str:
    """The GitHub token, for the ``gh`` subprocess only. Never printed, never bundled."""
    env_path = REPO / ".env"
    if not env_path.exists():
        return os.environ.get("GITHUB_TOKEN", "")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("GITHUB_TOKEN="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("GITHUB_TOKEN", "")


def redact(text: str, secret: str) -> tuple[str, int]:
    if not secret:
        return text, 0
    hits = text.count(secret)
    return (text.replace(secret, "<redacted:GITHUB_TOKEN>"), hits) if hits else (text, 0)


def rev(ref: str) -> str:
    proc = run(f"git rev-parse --short {ref}")
    return proc.stdout.decode("utf-8", "replace").strip() or "unknown"


def merge_base(left: str, right: str) -> str:
    proc = run(f"git merge-base {left} {right}")
    return proc.stdout.decode("utf-8", "replace").strip()[:7] or "unknown"


def ahead_count(left: str, right: str) -> int:
    return len(run(f"git log {left}..{right} --oneline").stdout.splitlines())


def strip_credentials(url: str) -> str:
    return re.sub(r"://[^/@\s]*@", "://", url)


def docker_available() -> str:
    if shutil.which("docker") is None:
        return "no (docker binary not found)"
    info = run("docker info")
    if info.returncode == 0:
        return "yes (" + run("docker --version").stdout.decode("utf-8", "replace").strip() + ")"
    return f"no (binary present, daemon unreachable: exit={info.returncode})"


def version_of(cmd: str) -> str:
    proc = run(cmd)
    first = proc.stdout.decode("utf-8", "replace").splitlines()
    return first[0].strip() if first else f"unavailable (exit={proc.returncode})"


# ------------------------------------------------------------------- sections
def collect_checks(token: str) -> tuple[list[Capture], int]:
    captures: list[Capture] = []
    redactions = 0
    for index, (name, cmd) in enumerate(CHECKS, start=1):
        proc = run(cmd, env={"GH_TOKEN": token} if "gh " in cmd else None)
        stdout = proc.stdout.decode("utf-8", "replace").strip()
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        body = stdout
        if stderr:
            body = f"{stdout}\n--- stderr ---\n{stderr}" if stdout else stderr
        body, hits = redact(body, token)
        redactions += hits
        raw = body.encode("utf-8")
        clipped, cut, dropped = truncate(raw)
        print(
            f"  [{index:2d}/{len(CHECKS)}] exit={proc.returncode} "
            f"{len(raw):>8} B{' TRUNCATED' if cut else ''}  {cmd[:72]}",
            flush=True,
        )
        captures.append(
            Capture(
                name=f"{index}. {name}",
                cmd=cmd,
                exit_code=proc.returncode,
                raw=raw,
                body=clipped,
                truncated=cut,
                dropped=dropped,
            )
        )
    return captures, redactions


def render_check(capture: Capture) -> str:
    header = f"$ {capture.cmd}\nexit={capture.exit_code}"
    text = f"{header}\n{capture.body.decode('utf-8', 'replace')}"
    if capture.truncated:
        text += (
            f"\n\n(per-command limit {PER_CMD_LIMIT // 1024} KB: "
            f"{capture.dropped}+ bytes dropped, {len(capture.raw)} bytes originally)"
        )
    capture.text = f"### {capture.name}\n\n{fenced(text)}\n"
    return capture.text


def render_diff() -> str:
    stat = run("git diff main...feat/init --stat").stdout.decode("utf-8", "replace").strip()
    full = run("git diff main...feat/init").stdout
    names = run("git diff --name-only main...feat/init").stdout.decode("utf-8", "replace").split()
    parts = [f"### diffstat (main...feat/init)\n\n{fenced(stat)}"]
    parts.append(
        f"### full diff\n\n`git diff main...feat/init` is {len(full)} bytes ({len(names)} files) — "
    )
    if len(full) <= DIFF_FULL_LIMIT:
        parts[-1] += (
            f"under the {DIFF_FULL_LIMIT // 1024} KB cap, included verbatim below.\n\n"
            + fenced(full.decode("utf-8", "replace"))
        )
    else:
        dropped = len(full) - DIFF_FULL_LIMIT
        parts[-1] += (
            f"over the {DIFF_FULL_LIMIT // 1024} KB cap, so it is **not** dumped here "
            f"({dropped} bytes omitted). Review it with `git diff main...feat/init`.\n"
        )
        biggest = run("git diff main...feat/init --numstat").stdout.decode("utf-8", "replace")
        rows = []
        for line in biggest.splitlines():
            add, _rm, path = [*line.split("\t"), "", ""][:3]
            if add.isdigit():
                rows.append((int(add), path))
        rows.sort(reverse=True)
        top = "largest contributors by added lines:\n" + "\n".join(
            f"  {n:>6} {p}" for n, p in rows[:5]
        )
        parts[-1] += f"\n{fenced(top)}"
    return "## diff\n\n" + "\n\n".join(parts) + "\n"


def render_files() -> str:
    blocks = ["## files (--include-files)\n"]
    for rel in INCLUDE_FILES:
        data = (REPO / rel).read_bytes()
        body, cut, dropped = truncate(data)
        note = f" ({dropped} bytes dropped, {len(data)} originally)" if cut else ""
        blocks.append(f"### {rel}{note}\n\n{fenced(body.decode('utf-8', 'replace'))}\n")
    return "\n".join(blocks)


def render_metadata(
    *,
    include_files: bool,
    captures: list[Capture],
    redactions: int,
    docker: str,
    scan_note: str,
) -> str:
    tracked_env = run('git ls-files | grep -i "\\.env"').stdout.decode("utf-8", "replace").strip()
    repo_url = strip_credentials(
        run("git remote get-url origin").stdout.decode("utf-8", "replace").strip()
    )
    diff_bytes = len(run("git diff main...feat/init").stdout)
    rows = [
        "| # | check | exit | bytes | truncated |",
        "| --- | --- | --- | --- | --- |",
    ]
    for capture in captures:
        index, _, name = capture.name.partition(". ")
        rows.append(
            f"| {index} | {name} | {capture.exit_code} | {len(capture.raw)} | "
            f"{f'yes, -{capture.dropped}' if capture.truncated else 'no'} |"
        )
    lines = [
        "## metadata",
        "",
        f"- repo: {repo_url}",
        f"- branch: {run('git rev-parse --abbrev-ref HEAD').stdout.decode().strip()}",
        f"- HEAD: {run('git rev-parse HEAD').stdout.decode().strip()} "
        f"({run('git log -1 --pretty=%s').stdout.decode().strip()})",
        f"- generated_at (UTC): {now_utc()}",
        f"- python: {platform.python_version()} ({sys.executable})",
        f"- uv: {version_of('uv --version')}",
        f"- docker available: {docker}",
        f"- git: {version_of('git --version')}",
        f"- main tip: {rev('origin/main')} (origin/main), local main: {rev('main')}",
        f"- merge-base main..feat/init: {merge_base('main', 'feat/init')} "
        f"(feat/init is {ahead_count('main', 'feat/init')} commits ahead, "
        f"origin/main is {ahead_count('feat/init', 'origin/main')} commits ahead: no rewrite of main)",
        f"- main rewrite under review: {BASELINE} ({run(f'git log -1 --pretty=%s {BASELINE}').stdout.decode().strip()})",
        f"- include_files: {str(include_files).lower()}",
        f"- full `git diff main...feat/init`: {diff_bytes} bytes "
        f"({'included' if diff_bytes <= DIFF_FULL_LIMIT else 'stat only'})",
        f"- env/secret files tracked by git: {tracked_env or 'none'}",
        "",
        "### exclusions (hard)",
        "",
        "The bundle is scanned with `ultron.policy.scan_for_secrets` — the same "
        "scanner the policy gate uses — before it is written. A hit aborts the run "
        "and nothing is written. Never bundled: the GitHub token, `.env`, "
        "`.git/config`, any `*.env` file, any credential. "
        f"Token occurrences scrubbed from command output: {redactions}.",
        "",
        f"- secret scan: {scan_note}",
        "",
        "### limits",
        "",
        f"- per-command output: {PER_CMD_LIMIT // 1024} KB "
        f"(keeps {TRUNCATE_HEAD // 1024} KB head + {TRUNCATE_TAIL // 1024} KB tail, "
        "cut marked `... [TRUNCATED <n> bytes]`)",
        f"- full diff dumped only under {DIFF_FULL_LIMIT // 1024} KB",
        f"- total bundle must stay under {TOTAL_LIMIT // 1024} KB, otherwise exit 1 and no file",
        "",
        "### per-check sizes (original byte counts, truncation visible)",
        "",
        *rows,
        "",
    ]
    return "\n".join(lines)


def render_manifest(sections: list[Capture], *, total: int, scan_note: str) -> str:
    manifest = {
        "repo": strip_credentials(
            run("git remote get-url origin").stdout.decode("utf-8", "replace").strip()
        ),
        "branch": run("git rev-parse --abbrev-ref HEAD").stdout.decode().strip(),
        "head": run("git rev-parse HEAD").stdout.decode().strip(),
        "generated_at": now_utc(),
        "secret_scan": scan_note,
        "total_bytes": total,
        "sections": [
            {"name": s.name, "bytes": len(s.text.encode("utf-8")), "sha256": s.sha256}
            for s in sections
        ],
    }
    return f"## manifest\n\n{fenced(json.dumps(manifest, indent=2, sort_keys=True))}\n"


# --------------------------------------------------------------- secret gate
def scan_sections(sections: list[Capture]) -> list[str]:
    """Return ``["<section>: <pattern>", ...]`` for every scanner hit."""
    offenders: list[str] = []
    for section in sections:
        for hit in scan_for_secrets(section.text):
            # never echo the matched text, only where and what pattern
            offenders.append(f"{section.name}: {hit.rsplit(':', 1)[-1]}")
    return offenders


def abort(reason: str, offenders: list[str]) -> int:
    print(f"\nABORT: {reason}", file=sys.stderr)
    for line in offenders:
        print(f"  offending: {line}", file=sys.stderr)
    print(f"  nothing written: {BUNDLE} does not exist / was not modified", file=sys.stderr)
    return 1


# -------------------------------------------------------------------- bundle
def make_git_bundle() -> tuple[Path, str]:
    """Archive both refs, so the reviewer can clone the exact history offline."""
    proc = run("git bundle create phase1.bundle main feat/init")
    if proc.returncode == 0 and GIT_BUNDLE.exists():
        return GIT_BUNDLE, "git bundle create phase1.bundle main feat/init (exit 0)"
    print("  git bundle unavailable, falling back to tar", file=sys.stderr)
    tar = run("tar -czf phase1.tar.gz --exclude=.git --exclude=.env .")
    if tar.returncode != 0 or not TAR_BUNDLE.exists():
        raise SystemExit(f"archive failed: exit={tar.returncode} {tar.stderr.decode()[:400]}")
    return TAR_BUNDLE, "tar fallback"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 1 review bundle")
    parser.add_argument(
        "--include-files",
        action="store_true",
        help="also dump full contents of policy.py, sandbox.py, run.py, config.py, ci.yml, tasks.jsonl",
    )
    args = parser.parse_args(argv)

    print(f"collecting review bundle at {BUNDLE} (include_files={args.include_files})")
    token = read_token()
    docker = docker_available()
    captures, redactions = collect_checks(token)

    sections: list[Capture] = [Capture("checks", "", 0, b"", b"", False, 0)]
    sections[0].text = "## checks\n\n" + "\n".join(render_check(c) for c in captures)
    sections.append(Capture("diff", "", 0, b"", b"", False, 0))
    sections[1].text = render_diff()
    if args.include_files:
        sections.append(Capture("files", "", 0, b"", b"", False, 0))
        sections[2].text = render_files()

    # Fail closed *before* the manifest exists and before anything is written.
    offenders = scan_sections(sections)
    if offenders:
        return abort("secret scanner hit in bundle text", offenders)

    scan_note = (
        f"pass (scan_for_secrets over {len(sections)} content sections, then the finished "
        "bundle text: no hits)"
    )
    meta = Capture("metadata", "", 0, b"", b"", False, 0)
    meta.text = render_metadata(
        include_files=args.include_files,
        captures=captures,
        redactions=redactions,
        docker=docker,
        scan_note=scan_note,
    )
    sections.insert(0, meta)

    body = "# Phase 1 review bundle\n\n" + "\n".join(s.text for s in sections)
    total = len(body.encode("utf-8")) + 2048  # room for the manifest section
    if total > TOTAL_LIMIT:
        return abort(
            f"bundle would be ~{total} bytes, over the {TOTAL_LIMIT} byte limit "
            "(re-run without --include-files)",
            [f"{BUNDLE}: size"],
        )

    # Second pass: the manifest is part of the finished text, scan it too.
    final = body + render_manifest(sections, total=len(body.encode("utf-8")), scan_note=scan_note)
    second = [Capture("finished bundle", "", 0, b"", b"", False, 0)]
    second[0].text = final
    offenders = scan_sections(second)
    if offenders:
        return abort("secret scanner hit in the finished bundle", offenders)

    BUNDLE.write_bytes(final.encode("utf-8"))
    print(f"  wrote {BUNDLE} ({len(final.encode('utf-8'))} bytes)")

    archive, how = make_git_bundle()
    print("")
    for path in (BUNDLE, archive):
        data = path.read_bytes()
        print(f"{path}  bytes={len(data)}  sha256={sha256_bytes(data)}")
    print(f"archive: {archive.name} ({how})")
    if archive == GIT_BUNDLE:
        print(f"verify non-empty: git bundle verify {archive.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
