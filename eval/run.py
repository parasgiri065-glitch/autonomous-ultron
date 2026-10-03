#!/usr/bin/env python3
"""Eval harness: the kill-switch that keeps Ultron cheap *and* correct.

Usage::

    uv run python eval/run.py --limit 3            # quick smoke (used by CI)
    uv run python eval/run.py --passes 1 --json    # machine-readable
    uv run python eval/run.py --update-baseline    # after an intentional change

What it does
------------
1. Loads ``eval/tasks.jsonl`` — each task is a goal plus *checkable* expectations
   (expected status, tools, substrings, source count, policy outcome, cost cap).
2. Runs them through the real ``Agent`` against an isolated state directory
   (``eval/.state``) so eval runs never pollute the operator's cache or memory.
3. Runs the suite **twice**: pass 1 cold (truth about correctness and spend),
   pass 2 warm (truth about the cache). An optional pass 3 measures memory recall.
4. Prints success rate, avg cost, avg latency and cache hit rate, then *gates*:
   exit code 1 if success rate drops below ``ULTRON_EVAL_MIN_SUCCESS_RATE``, if
   avg cost exceeds ``ULTRON_EVAL_MAX_AVG_COST_USD``, if latency exceeds
   ``ULTRON_EVAL_MAX_AVG_LATENCY_S``, or if the run regresses against
   ``eval/baseline.json``.

Network is never required: tools read a fixture corpus via ``ULTRON_WEB_MOCK``.
Policy behaviour is part of the contract — a medium-risk tool that refuses to run
unattended *passes* its task, because refusing is the correct outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ultron.agent import Agent, AgentResult  # noqa: E402
from ultron.config import Settings, load_settings  # noqa: E402
from ultron.registry import Registry  # noqa: E402
from ultron.verifier import Verifier  # noqa: E402

DEFAULT_TASKS = REPO_ROOT / "eval" / "tasks.jsonl"
DEFAULT_BASELINE = REPO_ROOT / "eval" / "baseline.json"
MOCK_FIXTURES = REPO_ROOT / "eval" / "fixtures" / "web_mock.json"
EVAL_STATE = REPO_ROOT / "eval" / ".state"

#: How much worse than the baseline is still acceptable before the gate fails.
SUCCESS_TOLERANCE = 0.01
COST_TOLERANCE = 0.25
LATENCY_TOLERANCE = 0.50


# --------------------------------------------------------------------- task spec
@dataclass(slots=True)
class TaskSpec:
    id: str
    goal: str
    category: str = "general"
    expect: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "goal": self.goal, "category": self.category, "expect": self.expect}


@dataclass(slots=True)
class TaskOutcome:
    spec: TaskSpec
    status: str
    ok: bool
    failures: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    tools_used: list[str] = field(default_factory=list)
    policy_actions: list[str] = field(default_factory=list)
    answer: str = ""
    answer_source: str = "none"
    grounded: bool | None = None
    cost_basis: str = "actual"
    pass_index: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec.id,
            "status": self.status,
            "ok": self.ok,
            "failures": self.failures,
            "cost_usd": round(self.cost_usd, 8),
            "latency_s": round(self.latency_s, 4),
            "tools_used": self.tools_used,
            "policy_actions": self.policy_actions,
            "answer_source": self.answer_source,
            "grounded": self.grounded,
            "answer": self.answer[:600],
        }


# ------------------------------------------------------------------- evaluation
def load_tasks(path: Path | None = None) -> list[TaskSpec]:
    path = Path(path or DEFAULT_TASKS)
    tasks: list[TaskSpec] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        spec = TaskSpec(
            id=str(data["id"]),
            goal=str(data["goal"]),
            category=str(data.get("category", "general")),
            expect=dict(data.get("expect", {})),
            notes=str(data.get("notes", "")),
        )
        tasks.append(spec)
    return tasks


def eval_settings(*, backend: str, fresh: bool) -> Settings:
    """Isolated settings for an eval run (own cache, memory, audit trail)."""
    if fresh and EVAL_STATE.exists():
        shutil.rmtree(EVAL_STATE)
    EVAL_STATE.mkdir(parents=True, exist_ok=True)
    mock = str(MOCK_FIXTURES) if backend != "docker" else "/workspace/eval/fixtures/web_mock.json"
    settings = load_settings(
        state_dir=EVAL_STATE,
        cache_path=EVAL_STATE / "cache.db",
        memory_path=EVAL_STATE / "memory.db",
        approvals_file=EVAL_STATE / "approvals.json",
        audit_log=EVAL_STATE / "audit.jsonl",
        sandbox_backend=backend,
        allow_local_sandbox=(backend == "local"),
        # Deterministic fixtures: no network egress in any environment.
        env_allowlist=["ULTRON_WEB_MOCK"],
        # The eval never spends real money: no judge, no synthesis by default.
        enable_llm_judge=os.environ.get("ULTRON_ENABLE_LLM_JUDGE", "0") == "1",
        llm_mode=os.environ.get("ULTRON_LLM_MODE", "auto"),
    )
    os.environ["ULTRON_WEB_MOCK"] = mock
    return settings


def run_task(
    spec: TaskSpec, settings: Settings, *, pass_index: int, use_recall: bool
) -> tuple[TaskOutcome, AgentResult]:
    agent = Agent(
        settings=settings,
        interactive=False,  # eval is always unattended: no human, no silent approvals
        use_memory_recall=use_recall,
        sandbox_backend=settings.sandbox_backend,
    )
    result = agent.run(spec.goal)
    outcome = TaskOutcome(
        spec=spec,
        status=result.status,
        ok=False,
        cost_usd=result.cost_usd,
        latency_s=result.latency_s,
        cache_hits=result.cache_hits,
        cache_misses=result.cache_misses,
        tools_used=[s.tool for s in result.steps],
        policy_actions=[s.policy_action for s in result.steps],
        answer=result.answer,
        answer_source=result.answer_source,
        cost_basis=result.cost_basis,
        pass_index=pass_index,
    )
    outcome.failures = check_expectations(spec, result, settings)
    outcome.grounded = grounding_check(spec, result, settings)
    if outcome.grounded is False:
        outcome.failures.append("answer is not grounded in the provided sources")
    outcome.ok = not outcome.failures
    return outcome, result


def check_expectations(spec: TaskSpec, result: AgentResult, settings: Settings) -> list[str]:
    """Turn a task's ``expect`` block into concrete pass/fail reasons."""
    expect = spec.expect
    failures: list[str] = []

    if "status" in expect and result.status != expect["status"]:
        failures.append(f"status {result.status!r} != expected {expect['status']!r}")

    expected_tools = expect.get("tools")
    if expected_tools is not None:
        used = {s.tool for s in result.steps}
        missing = [t for t in expected_tools if t not in used]
        if missing:
            failures.append(f"tools not used: {missing} (used: {sorted(used)})")

    for needle in expect.get("answer_contains", []):
        if needle.lower() not in (result.answer or "").lower():
            failures.append(f"answer does not contain {needle!r}")

    for needle in expect.get("answer_not_contains", []):
        if needle.lower() in (result.answer or "").lower():
            failures.append(f"answer unexpectedly contains {needle!r}")

    if "min_sources" in expect:
        sources: list[str] = []
        for step in result.steps:
            if step.result and isinstance(step.result.get("sources"), list):
                sources.extend(step.result["sources"])
        if len(sources) < int(expect["min_sources"]):
            failures.append(f"{len(sources)} sources < required {expect['min_sources']}")

    if "policy" in expect:
        observed = {s.policy_action for s in result.steps} - {"none"}
        want = expect["policy"]
        if want == "ask" and not {"ask", "deny"} & observed:
            failures.append(f"expected an approval request, saw {sorted(observed) or 'no steps'}")
        elif want == "deny" and "deny" not in observed and not result.refused:
            failures.append(f"expected a denial, saw {sorted(observed) or 'no steps'}")
        elif want == "allow" and observed and observed != {"allow"}:
            failures.append(f"expected an auto-approval, saw {sorted(observed)}")

    if "network" in expect:
        granted = [s.network for s in result.steps if s.policy_action == "allow"]
        if expect["network"] not in granted:
            failures.append(f"network grant {granted} did not include {expect['network']!r}")

    if "max_cost_usd" in expect and result.cost_usd > float(expect["max_cost_usd"]) + 1e-9:
        failures.append(f"cost ${result.cost_usd:.6f} > cap ${float(expect['max_cost_usd']):.6f}")

    if "max_latency_s" in expect and result.latency_s > float(expect["max_latency_s"]):
        failures.append(f"latency {result.latency_s:.2f}s > cap {expect['max_latency_s']}s")

    if expect.get("verified") and not result.verified:
        failures.append("run was not verified")

    return failures


def grounding_check(spec: TaskSpec, result: AgentResult, settings: Settings) -> bool | None:
    """Re-verify research answers against the fixture corpus that produced them.

    The agent verifies without evidence (it must not trust tool metadata), so the
    eval supplies the ground truth: every content word of the answer must trace
    back to the source documents the tool actually read.
    """
    if spec.category != "research" or not result.steps:
        return None
    payload = next((s.result for s in result.steps if s.result and "summary" in s.result), None)
    if not payload:
        return None
    sources = set(payload.get("sources") or [])
    if not sources:
        return False
    try:
        corpus = json.loads(MOCK_FIXTURES.read_text(encoding="utf-8"))["documents"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    evidence = [d["text"] for d in corpus if d.get("url") in sources]
    if not evidence:
        return None
    manifest_name = next((s.tool for s in result.steps if s.result is payload), "web_research")
    registry = Registry(settings).load()
    try:
        manifest = registry.get(manifest_name)
    except Exception:
        return None
    verification = Verifier(settings=settings).verify(
        manifest, payload, goal=spec.goal, evidence=evidence
    )
    return verification.ok


# ----------------------------------------------------------------------- report
@dataclass(slots=True)
class GateResult:
    name: str
    ok: bool
    detail: str
    value: float | None = None
    limit: float | None = None


@dataclass(slots=True)
class EvalReport:
    cold: list[TaskOutcome] = field(default_factory=list)
    warm: list[TaskOutcome] = field(default_factory=list)
    recall: list[TaskOutcome] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    duration_s: float = 0.0
    backend: str = "local"
    bases_seen: set[str] = field(default_factory=set)
    tasks_path: str = str(DEFAULT_TASKS)

    @property
    def cost_basis(self) -> str:
        """The most 'real' basis seen across tasks (actual > simulated > stubs)."""
        for basis in ("actual", "simulated", "cache_hit", "offline_stub", "no_llm"):
            if basis in self.bases_seen:
                return basis
        return "no_llm"

    # ------------------------------------------------------------- aggregation
    @staticmethod
    def _rate(outcomes: list[TaskOutcome]) -> float:
        return (sum(1 for o in outcomes if o.ok) / len(outcomes)) if outcomes else 0.0

    @property
    def success_rate(self) -> float:
        return self._rate(self.cold)

    @property
    def avg_cost_usd(self) -> float:
        return (sum(o.cost_usd for o in self.cold) / len(self.cold)) if self.cold else 0.0

    @property
    def avg_latency_s(self) -> float:
        return (sum(o.latency_s for o in self.cold) / len(self.cold)) if self.cold else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(o.cost_usd for o in self.cold)

    @property
    def warm_avg_cost_usd(self) -> float:
        return (sum(o.cost_usd for o in self.warm) / len(self.warm)) if self.warm else 0.0

    @property
    def cache_hit_rate(self) -> float:
        hits = sum(o.cache_hits for o in self.warm)
        misses = sum(o.cache_misses for o in self.warm)
        return (hits / (hits + misses)) if (hits + misses) else 0.0

    @property
    def recall_rate(self) -> float:
        if not self.recall:
            return 0.0
        return sum(1 for o in self.recall if o.answer_source == "memory") / len(self.recall)

    @property
    def deny_rate(self) -> float:
        return (
            (
                sum(1 for o in self.cold if "deny" in o.policy_actions or "ask" in o.policy_actions)
                / len(self.cold)
            )
            if self.cold
            else 0.0
        )

    @property
    def exit_code(self) -> int:
        return 0 if all(g.ok for g in self.gates) else 1

    def by_category(self) -> dict[str, dict[str, float]]:
        categories: dict[str, list[TaskOutcome]] = {}
        for outcome in self.cold:
            categories.setdefault(outcome.spec.category, []).append(outcome)
        return {
            name: {
                "tasks": len(items),
                "success_rate": round(self._rate(items), 4),
                "avg_cost_usd": round(sum(o.cost_usd for o in items) / len(items), 8),
                "avg_latency_s": round(sum(o.latency_s for o in items) / len(items), 4),
            }
            for name, items in sorted(categories.items())
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "tasks_path": self.tasks_path,
            "backend": self.backend,
            "cost_basis": self.cost_basis,
            "duration_s": round(self.duration_s, 3),
            "metrics": {
                "tasks": len(self.cold),
                "success_rate": round(self.success_rate, 4),
                "avg_cost_usd": round(self.avg_cost_usd, 8),
                "total_cost_usd": round(self.total_cost_usd, 8),
                "avg_latency_s": round(self.avg_latency_s, 4),
                "cache_hit_rate": round(self.cache_hit_rate, 4),
                "warm_avg_cost_usd": round(self.warm_avg_cost_usd, 8),
                "memory_recall_rate": round(self.recall_rate, 4),
                "policy_gate_rate": round(self.deny_rate, 4),
                "by_category": self.by_category(),
            },
            "gates": [
                {"name": g.name, "ok": g.ok, "detail": g.detail, "value": g.value, "limit": g.limit}
                for g in self.gates
            ],
            "passed": self.exit_code == 0,
            "outcomes": [o.as_dict() for o in self.cold],
            "warm_outcomes": [o.as_dict() for o in self.warm],
        }

    # ------------------------------------------------------------------ output
    def render(self, *, console: Any | None = None) -> None:
        from rich.console import Console
        from rich.table import Table

        console = console or Console()
        table = Table(title=f"ultron eval — {self.tasks_path}", header_style="bold")
        for column in ("task", "status", "tools", "policy", "cost", "latency", "result"):
            table.add_column(column)
        for outcome in self.cold:
            first_failure = outcome.failures[0] if outcome.failures else ""
            table.add_row(
                outcome.spec.id,
                outcome.status,
                ",".join(outcome.tools_used) or "-",
                ",".join(outcome.policy_actions) or "-",
                f"${outcome.cost_usd:.6f}",
                f"{outcome.latency_s:.2f}s",
                "[green]pass[/green]" if outcome.ok else f"[red]fail[/red] {first_failure[:60]}",
            )
        console.print(table)
        console.print(
            f"[bold]success[/bold] {self.success_rate:.1%}  "
            f"[bold]avg cost[/bold] ${self.avg_cost_usd:.6f}  "
            f"[bold]avg latency[/bold] {self.avg_latency_s:.2f}s  "
            f"[bold]cache hit rate[/bold] {self.cache_hit_rate:.1%}  "
            f"[bold]recall[/bold] {self.recall_rate:.1%}  "
            f"[dim](basis: {self.cost_basis}, {self.duration_s:.1f}s wall)[/dim]"
        )
        for gate in self.gates:
            mark = "[green]PASS[/green]" if gate.ok else "[red]FAIL[/red]"
            console.print(f"  {mark} {gate.name}: {gate.detail}")

    def write(self, path_json: Path, path_md: Path | None = None) -> None:
        path_json.parent.mkdir(parents=True, exist_ok=True)
        path_json.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        if path_md is not None:
            path_md.write_text(self.markdown(), encoding="utf-8")

    def markdown(self) -> str:
        lines = [
            "# Ultron eval report",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| tasks | {len(self.cold)} |",
            f"| success rate | {self.success_rate:.1%} |",
            f"| avg cost/run | ${self.avg_cost_usd:.6f} (basis: {self.cost_basis}) |",
            f"| avg latency | {self.avg_latency_s:.2f}s |",
            f"| cache hit rate (warm pass) | {self.cache_hit_rate:.1%} |",
            f"| memory recall rate | {self.recall_rate:.1%} |",
            "",
            "| task | status | tools | policy | cost | latency | verdict |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for outcome in self.cold:
            lines.append(
                f"| {outcome.spec.id} | {outcome.status} | {','.join(outcome.tools_used) or '-'} | "
                f"{','.join(outcome.policy_actions) or '-'} | ${outcome.cost_usd:.6f} | "
                f"{outcome.latency_s:.2f}s | {'pass' if outcome.ok else 'fail: ' + '; '.join(outcome.failures)[:120]} |"
            )
        lines += ["", "## gates", ""]
        for gate in self.gates:
            lines.append(f"- {'PASS' if gate.ok else 'FAIL'} — {gate.name}: {gate.detail}")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ gates
def evaluate_gates(report: EvalReport, settings: Settings, baseline_path: Path) -> list[GateResult]:
    gates: list[GateResult] = []
    success, cost, latency = report.success_rate, report.avg_cost_usd, report.avg_latency_s

    gates.append(
        GateResult(
            "success_rate",
            success >= settings.eval_min_success_rate,
            f"{success:.1%} (min {settings.eval_min_success_rate:.0%})",
            value=success,
            limit=settings.eval_min_success_rate,
        )
    )
    gates.append(
        GateResult(
            "avg_cost_usd",
            cost <= settings.eval_max_avg_cost_usd + 1e-12,
            f"${cost:.6f} (max ${settings.eval_max_avg_cost_usd:.6f}, basis {report.cost_basis})",
            value=cost,
            limit=settings.eval_max_avg_cost_usd,
        )
    )
    gates.append(
        GateResult(
            "avg_latency_s",
            latency <= settings.eval_max_avg_latency_s,
            f"{latency:.2f}s (max {settings.eval_max_avg_latency_s:.1f}s)",
            value=latency,
            limit=settings.eval_max_avg_latency_s,
        )
    )

    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        base_success = float(baseline.get("success_rate", 0.0))
        base_cost = float(baseline.get("avg_cost_usd", 0.0))
        base_latency = float(baseline.get("avg_latency_s", 0.0))
        gates.append(
            GateResult(
                "no_regression.success_rate",
                success >= base_success - SUCCESS_TOLERANCE,
                f"{success:.1%} vs baseline {base_success:.1%} (tolerance {SUCCESS_TOLERANCE:.0%})",
                value=success,
                limit=base_success - SUCCESS_TOLERANCE,
            )
        )
        cost_limit = (
            base_cost * (1 + COST_TOLERANCE)
            if base_cost > 0
            else max(settings.eval_max_avg_cost_usd, 0.0)
        )
        gates.append(
            GateResult(
                "no_regression.avg_cost_usd",
                cost <= cost_limit + 1e-12,
                f"${cost:.6f} vs baseline ${base_cost:.6f} (+{COST_TOLERANCE:.0%})",
                value=cost,
                limit=cost_limit,
            )
        )
        if base_latency > 0:
            gates.append(
                GateResult(
                    "no_regression.avg_latency_s",
                    latency <= base_latency * (1 + LATENCY_TOLERANCE),
                    f"{latency:.2f}s vs baseline {base_latency:.2f}s (+{LATENCY_TOLERANCE:.0%})",
                    value=latency,
                    limit=base_latency * (1 + LATENCY_TOLERANCE),
                )
            )
    else:
        gates.append(GateResult("baseline", True, f"no baseline at {baseline_path} (first run)"))
    return gates


# ------------------------------------------------------------------------- main
def run_eval(
    *,
    tasks_path: Path | None = None,
    limit: int | None = None,
    backend: str = "local",
    passes: int = 2,
    quiet: bool = False,
    write_report: bool = True,
    fresh: bool = True,
) -> EvalReport:
    started = time.perf_counter()
    tasks = load_tasks(tasks_path)
    if limit:
        tasks = tasks[:limit]
    settings = eval_settings(backend=backend, fresh=fresh)
    report = EvalReport(backend=backend, tasks_path=str(tasks_path or DEFAULT_TASKS))

    for index, spec in enumerate(tasks):
        outcome, result = run_task(spec, settings, pass_index=1, use_recall=False)
        report.cold.append(outcome)
        report.bases_seen.add(result.cost_basis)
        if not quiet:
            status = "pass" if outcome.ok else "FAIL"
            detail = "" if outcome.ok else f" — {outcome.failures[0][:80]}"
            print(f"  [{index + 1}/{len(tasks)}] {spec.id}: {status}{detail}", flush=True)

    if passes >= 2:
        for spec in tasks:
            warm_outcome, _ = run_task(spec, settings, pass_index=2, use_recall=False)
            report.warm.append(warm_outcome)

    if passes >= 3:
        for spec in tasks:
            recall_outcome, _ = run_task(spec, settings, pass_index=3, use_recall=True)
            report.recall.append(recall_outcome)

    report.duration_s = time.perf_counter() - started
    report.gates = evaluate_gates(report, settings, DEFAULT_BASELINE)
    if write_report:
        report.write(REPO_ROOT / "eval" / "last_report.json", REPO_ROOT / "eval" / "last_report.md")
    return report


def update_baseline(report: EvalReport, path: Path = DEFAULT_BASELINE) -> dict[str, Any]:
    baseline = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": len(report.cold),
        "success_rate": round(report.success_rate, 4),
        "avg_cost_usd": round(report.avg_cost_usd, 8),
        "avg_latency_s": round(report.avg_latency_s, 4),
        "cache_hit_rate": round(report.cache_hit_rate, 4),
        "cost_basis": report.cost_basis,
        "by_category": report.by_category(),
    }
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ultron eval harness")
    parser.add_argument("--tasks", default=None, help="path to tasks.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    parser.add_argument("--backend", choices=["docker", "local"], default="local")
    parser.add_argument(
        "--passes", type=int, default=2, help="1 = cold only, 2 = +cache pass, 3 = +recall pass"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-report", action="store_true", help="do not write eval/last_report.*")
    parser.add_argument(
        "--update-baseline", action="store_true", help="write eval/baseline.json from this run"
    )
    parser.add_argument("--no-gate", action="store_true", help="always exit 0 (report only)")
    args = parser.parse_args(argv)

    report = run_eval(
        tasks_path=Path(args.tasks) if args.tasks else None,
        limit=args.limit,
        backend=args.backend,
        passes=args.passes,
        quiet=args.quiet or args.json,
        write_report=not args.no_report,
    )
    if args.update_baseline:
        update_baseline(report)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    elif not args.quiet:
        report.render()
    if args.no_gate:
        return 0
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
