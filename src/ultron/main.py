"""``ultron`` CLI: run the agent, inspect tools/registry/cache/memory, run evals.

    ultron doctor                      # is this environment ready?
    ultron run "calculate 12*(3+4)"    # one goal through the whole loop
    ultron tools list                  # registry contents
    ultron cache stats                 # what we are saving
    ultron memory stats                # success rate, cost, latency
    ultron eval --limit 3              # eval harness

Every command supports ``--json`` for CI consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .agent import Agent
from .cache import Cache
from .config import get_settings
from .domains.intel import IntelError, IntelResearchEngine
from .errors import HumanApprovalRequired, PolicyDenied, SandboxUnavailable, UltronError
from .extractor import ExtractionError, GroundedDataExtractor
from .forge import ForgeEngine
from .harvester import HarvestError, PyPIHarvester
from .interfaces.telegram import TelegramBotClient, TelegramCockpit
from .ledger import FailureLedger
from .llm import providers_configured
from .memory import Memory
from .policy import DenyAllPrompter, PolicyGate, default_prompter
from .refinery import RefineryError, ToolRefinery
from .registry import Registry
from .repair import RepairEngine, RepairError
from .sandbox import Sandbox
from .scavenger import Scavenger


def _console():
    from rich.console import Console

    return Console()


def _print_json(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")


# --------------------------------------------------------------------- commands
def cmd_doctor(args: argparse.Namespace) -> int:
    settings = get_settings()
    registry = Registry(settings)
    registry.load()
    sandbox = Sandbox(settings)
    report: dict[str, Any] = {
        "ultron_version": __version__,
        "python": sys.version.split()[0],
        "repo_root": str(settings.repo_root),
        "settings": {
            "sandbox_backend": settings.sandbox_backend,
            "sandbox_image": settings.sandbox_image,
            "llm_mode": settings.llm_mode,
            "router_model": settings.router_model,
            "policy_network": settings.policy_network,
            "enable_llm_judge": settings.enable_llm_judge,
            "budget_max_usd": settings.budget_max_usd,
            "allow_local_sandbox": settings.allow_local_sandbox,
        },
        "checks": [],
        "tools": {
            "count": len(registry),
            "names": registry.names(),
            "errors": registry.load_errors,
        },
        "paths": {
            "cache": str(settings.cache_path),
            "memory": str(settings.memory_path),
            "audit": str(settings.audit_log),
            "approvals": str(settings.approvals_file),
        },
    }
    checks = report["checks"]
    checks.append(
        {
            "check": "provider credentials",
            "ok": providers_configured(),
            "detail": "live LLM available"
            if providers_configured()
            else "offline mode: deterministic stubs (cost $0)",
            "blocking": False,
        }
    )
    checks.append(
        {
            "check": "docker daemon",
            "ok": sandbox.docker_available(),
            "detail": "available"
            if sandbox.docker_available()
            else "not available (set ULTRON_ALLOW_LOCAL_SANDBOX=1 for dev)",
            "blocking": True,
        }
    )
    checks.append(
        {
            "check": f"image {settings.sandbox_image}",
            "ok": sandbox.image_present(),
            "detail": "present"
            if sandbox.image_present()
            else f"missing: docker build -t {settings.sandbox_image} -f docker/Dockerfile .",
            "blocking": True,
        }
    )
    checks.append(
        {
            "check": "tool manifests",
            "ok": len(registry) > 0 and not registry.load_errors,
            "detail": f"{len(registry)} tool(s)"
            + (f", errors: {registry.load_errors}" if registry.load_errors else ""),
            "blocking": False,
        }
    )
    checks.append(
        {
            "check": "audit log writable",
            "ok": settings.audit_log.parent.exists(),
            "detail": str(settings.audit_log),
            "blocking": False,
        }
    )
    if args.json:
        _print_json(report)
        return 0

    console = _console()
    console.print(f"[bold]ultron[/bold] {__version__} — environment report")
    for check in checks:
        mark = (
            "[green]ok  [/green]"
            if check["ok"]
            else ("[red]FAIL[/red]" if check["blocking"] else "[yellow]warn[/yellow]")
        )
        console.print(f"  {mark} {check['check']}: {check['detail']}")
    console.print(f"  [dim]tools: {', '.join(registry.names()) or 'none'}[/dim]")
    blocking = [c for c in checks if c["blocking"] and not c["ok"]]
    return 0 if not blocking else 1


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract caller-declared fields from one local file or approved URL."""
    try:
        fields = json.loads(args.fields)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"--fields must be valid JSON: {exc.msg}") from exc
    extractor = GroundedDataExtractor(get_settings(), interactive=args.interactive)
    result = extractor.extract(args.source, fields, output_format=args.format, output=args.output)
    if args.output:
        # Keep stdout useful for scripts while making the artifact location clear.
        if args.json:
            _print_json({"status": "ok" if result.ok else "rejected", "output": str(result.output_path), "result": result.as_dict()})
        else:
            sys.stdout.write(f"Wrote {result.output_path}\n")
    else:
        sys.stdout.write(result.render(args.format))
    return 0 if result.ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    interactive = args.interactive and not args.no_interactive
    # Approval policy is wired at construction: non-interactive runs get the
    # DenyAllPrompter, so an unattended run can never silently self-approve.
    prompter = default_prompter(settings) if interactive else DenyAllPrompter()
    gate = PolicyGate(settings, prompter=prompter)
    agent = Agent(
        settings=settings,
        gate=gate,
        interactive=interactive,
        use_memory_recall=not args.no_recall,
        sandbox_backend=args.backend,
    )

    try:
        result = agent.run(args.goal, max_steps=args.max_steps)
    except (PolicyDenied, HumanApprovalRequired) as exc:
        _print_json({"status": "denied", "error": str(exc)})
        return 2
    except SandboxUnavailable as exc:
        if args.json:
            _print_json({"status": "unavailable", "error": str(exc)})
        else:
            _console().print(f"[red]sandbox unavailable[/red]: {exc}")
        return 3
    except UltronError as exc:
        _print_json({"status": "error", "error": str(exc)})
        return 1

    if args.json:
        _print_json(result.as_dict())
    else:
        _render_run(result)
    return 0 if result.ok else 1


def _render_run(result: Any) -> None:
    from rich.panel import Panel
    from rich.table import Table

    console = _console()
    colour = {"ok": "green", "denied": "yellow", "no_plan": "yellow"}.get(result.status, "red")
    console.print(Panel(f"[bold]{result.goal}[/bold]", title="goal", border_style="cyan"))
    if result.steps:
        table = Table(show_header=True, header_style="bold", box=None)
        for column in ("#", "tool", "risk", "policy", "net", "cache", "time", "verify"):
            table.add_column(column)
        for step in result.steps:
            verify = "-"
            if step.verification is not None:
                verify = "ok" if step.verification.ok else "[red]fail[/red]"
            table.add_row(
                str(step.index),
                f"{step.tool}@{step.version}",
                step.risk,
                step.policy_action,
                step.network,
                "hit" if step.cached else "miss",
                f"{step.duration_s:.2f}s",
                verify,
            )
        console.print(table)
    console.print(Panel(result.answer or "(no answer)", title="answer", border_style="green"))
    console.print(
        f"[{colour}]{result.status}[/{colour}] {result.run_id} | "
        f"cost=${result.cost_usd:.6f} ({result.cost_basis}) | {result.latency_s:.2f}s | "
        f"steps={len(result.steps)} | cache {result.cache_hits} hit / {result.cache_misses} miss"
    )
    for note in result.notes:
        console.print(f"  [dim]· {note}[/dim]")


def cmd_tools(args: argparse.Namespace) -> int:
    settings = get_settings()
    registry = Registry(settings).load(strict=args.strict)
    if args.action == "list":
        payload = registry.snapshot()
        if args.json:
            _print_json(payload)
            return 0
        console = _console()
        from rich.table import Table

        table = Table(header_style="bold")
        for column in (
            "name",
            "version",
            "risk",
            "permissions",
            "inputs",
            "outputs",
            "deterministic",
        ):
            table.add_column(column)
        for manifest in registry.latest():
            table.add_row(
                manifest.name,
                manifest.version,
                manifest.risk.value,
                ", ".join(manifest.permissions) or "-",
                ", ".join(manifest.inputs) or "-",
                ", ".join(manifest.outputs) or "-",
                "yes" if manifest.deterministic else "no",
            )
        console.print(table)
        if registry.load_errors:
            console.print("[yellow]manifest errors:[/yellow]")
            for err in registry.load_errors:
                console.print(f"  [red]· {err}[/red]")
        return 0

    if args.action == "show":
        manifest = registry.get(args.name)
        payload = manifest.model_dump(mode="json")
        payload["content_hash"] = manifest.content_hash
        payload["schema_errors"] = registry.validate_against_schema(manifest)
        _print_json(payload)
        return 0

    if args.action == "validate":
        errors: list[str] = list(registry.load_errors)
        targets = [registry.get(args.name)] if args.name else registry.all()
        checked = 0
        for manifest in targets:
            checked += 1
            errors += [
                f"{manifest.key}: {err}" for err in registry.validate_against_schema(manifest)
            ]
        _print_json(
            {
                "tool": args.name or "*",
                "checked": checked,
                "ok": not errors,
                "errors": errors,
            }
        )
        return 0 if not errors else 1
    return 1


def cmd_cache(args: argparse.Namespace) -> int:
    cache = Cache(get_settings())
    if args.action == "stats":
        _print_json(cache.stats().as_dict())
    elif args.action == "clear":
        removed = cache.clear(args.namespace)
        _print_json({"cleared": removed, "namespace": args.namespace or "all"})
    elif args.action == "prune":
        _print_json({"pruned": cache.prune_expired()})
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    memory = Memory(get_settings())
    if args.action == "stats":
        _print_json(memory.stats().as_dict())
    elif args.action == "runs":
        _print_json(memory.recent_runs(limit=args.limit))
    elif args.action == "failures":
        _print_json(memory.failure_modes(limit=args.limit))
    elif args.action == "forget":
        _print_json({"forgotten": memory.forget_runs()})
    return 0


def cmd_gaps(args: argparse.Namespace) -> int:
    ledger = FailureLedger(get_settings().state_dir)
    if args.clear:
        ledger.clear()
        if args.json:
            _print_json({"cleared": True})
        else:
            _console().print("Capability gap ledger cleared.")
        return 0
    gaps = ledger.read()
    if not gaps:
        if args.json:
            _print_json([])
        else:
            _console().print("No capability gaps recorded.")
        return 0
    if args.json:
        _print_json([gap.as_dict() for gap in gaps])
        return 0
    from rich.table import Table

    table = Table(title="Missing capability gaps", header_style="bold")
    for column in ("requests", "goal", "requires", "provides", "last attempt"):
        table.add_column(column)
    for gap in gaps:
        table.add_row(
            str(gap.frequency),
            gap.goal[:80],
            ", ".join(f"{k}:{v}" for k, v in gap.required_inputs.items()) or "-",
            ", ".join(gap.suggested_provides) or "-",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(gap.timestamp)),
        )
    _console().print(table)
    return 0


def cmd_forge(args: argparse.Namespace) -> int:
    engine = ForgeEngine(get_settings())
    forged = engine.auto_forge_from_ledger(top_n=args.limit)
    report = {
        "forged": [manifest.key for manifest in forged],
        **engine.last_report,
    }
    if args.json:
        _print_json(report)
    else:
        _console().print(f"Forged: {', '.join(report['forged']) or 'none'}")
        for label in ("failed", "skipped"):
            for item in report[label]:
                _console().print(f"{label.title()}: {item['goal']} — {item['reason']}")
    return 0 if not report["failed"] else 1


def cmd_scavenge(args: argparse.Namespace) -> int:
    settings = get_settings()
    scavenger = Scavenger(settings, max_candidates=args.max)
    candidates = scavenger.discover()
    forged = (
        scavenger.forge_candidates(candidates, max_tools=args.max)
        if candidates
        else {"forged": [], "failed": [], "skipped": []}
    )
    report = {
        "candidates": [
            {
                "name": item.name,
                "url": item.spec_url,
                "spec_hash": item.spec_hash,
                "method": item.method,
                "endpoint": item.endpoint,
            }
            for item in candidates
        ],
        "forged": [getattr(item, "name", item) for item in forged["forged"]],
        "failed": forged["failed"],
        "skipped": forged["skipped"],
        "rejections": scavenger.rejections,
        "fetch_errors": scavenger.fetch_errors,
    }
    if args.json:
        _print_json(report)
        return 0
    from rich.table import Table

    if not candidates:
        _console().print("No scavenger candidates found (disabled or empty allowlisted sources).")
    else:
        table = Table(title="OpenAPI scavenger candidates", header_style="bold")
        for column in ("name", "method", "endpoint", "spec hash"):
            table.add_column(column)
        for item in candidates:
            table.add_row(item.name, item.method.upper(), item.endpoint, item.spec_hash[:12])
        _console().print(table)
        _console().print(
            f"Forged: {len(report['forged'])}; failed: {len(report['failed'])}; "
            f"rejected: {len(report['rejections'])}"
        )
    return 0 if not report["failed"] else 1


def cmd_harvest(args: argparse.Namespace) -> int:
    settings = get_settings()
    harvester = PyPIHarvester(settings)
    try:
        metadata = harvester.inspect_package(args.package_name)
        output_key = args.output_key
        inputs = json.loads(args.test_input) if args.test_input else {}
        if not isinstance(inputs, dict):
            raise HarvestError("--test-input must be a JSON object")
        code, manifest = harvester.synthesize_wrapper(
            metadata,
            args.function,
            {key: _infer_cli_type(value) for key, value in inputs.items()},
            output_key,
        )
        if args.dry_run:
            payload = {
                "package": metadata.name,
                "version": metadata.version,
                "license": metadata.license,
                "modules": metadata.top_level_modules,
                "manifest": manifest,
                "wrapper_bytes": len(code.encode("utf-8")),
                "dry_run": True,
            }
        else:
            expected = {output_key: "any"}
            tool = harvester.harvest_and_forge(args.package_name, args.function, inputs, expected)
            payload = {
                "package": metadata.name,
                "version": metadata.version,
                "license": metadata.license,
                "tool": tool.key,
                "risk": tool.risk.value,
                "provides": tool.provides,
                "dry_run": False,
            }
    except (HarvestError, json.JSONDecodeError) as exc:
        if args.json:
            _print_json({"ok": False, "error": str(exc)})
        else:
            _console().print(f"[red]harvest failed[/red]: {exc}")
        return 1
    if args.json:
        _print_json({"ok": True, **payload})
    else:
        from rich.table import Table

        table = Table(title="PyPI harvest", header_style="bold")
        table.add_column("field")
        table.add_column("value")
        for key, value in payload.items():
            table.add_row(
                key, json.dumps(value, default=str) if not isinstance(value, str) else value
            )
        _console().print(table)
    return 0


def _infer_cli_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list[string]"
    if isinstance(value, dict):
        return "dict"
    return "string"


def cmd_refine(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        raw_code = Path(args.path).read_text(encoding="utf-8")
        test_input = json.loads(args.test_input) if args.test_input else {}
        expected_schema = (
            json.loads(args.output_schema) if args.output_schema else {"result": "any"}
        )
        if not isinstance(test_input, dict) or not isinstance(expected_schema, dict):
            raise RefineryError("--test-input and --output-schema must be JSON objects")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in expected_schema.items()
        ):
            raise RefineryError("--output-schema values must be manifest type names")
        provides = (
            args.provides or [f"refined.{key}" for key in expected_schema] or ["refined.result"]
        )
        requires = args.requires or []
        refinery = ToolRefinery(settings)
        code, manifest = refinery.refine_code(raw_code, args.function, provides, requires)
        if args.dry_run:
            payload = {
                "path": str(args.path),
                "target": args.function,
                "manifest": manifest,
                "wrapper_bytes": len(code.encode("utf-8")),
                "dry_run": True,
            }
        else:
            tool = refinery.cook_and_register(raw_code, args.function, test_input, expected_schema)
            payload = {
                "path": str(args.path),
                "target": args.function,
                "tool": tool.key,
                "risk": tool.risk.value,
                "provides": tool.provides,
                "requires": tool.requires,
                "dry_run": False,
            }
    except (OSError, RefineryError, json.JSONDecodeError, RepairError) as exc:
        if args.json:
            _print_json({"ok": False, "error": str(exc)})
        else:
            _console().print(f"[red]refine failed[/red]: {exc}")
        return 1
    if args.json:
        _print_json({"ok": True, **payload})
    else:
        from rich.table import Table

        table = Table(title="Tool refinery", header_style="bold")
        table.add_column("field")
        table.add_column("value")
        for key, value in payload.items():
            table.add_row(
                key, json.dumps(value, default=str) if not isinstance(value, str) else value
            )
        _console().print(table)
    return 0


def cmd_repair(args: argparse.Namespace) -> int:
    engine = RepairEngine(get_settings())
    if not args.list:
        if args.json:
            _print_json({"ok": False, "error": "use `ultron repair --list` to view open tickets"})
        else:
            _console().print("Use [bold]ultron repair --list[/bold] to view open repair tickets.")
        return 1
    tickets = engine.open_tickets()
    payload = [
        {
            "id": ticket.ticket_id,
            "component": ticket.component,
            "error": ticket.error,
            "status": ticket.status,
            "created_at": ticket.created_at,
        }
        for ticket in tickets
    ]
    if args.json:
        _print_json({"ok": True, "tickets": payload})
        return 0
    from rich.table import Table

    table = Table(title="Open repair tickets", header_style="bold")
    for column in ("id", "component", "error", "status"):
        table.add_column(column)
    for ticket in payload:
        table.add_row(ticket["id"], ticket["component"], ticket["error"], ticket["status"])
    _console().print(table)
    if not tickets:
        _console().print("No open repair tickets.")
    return 0


def cmd_intel(args: argparse.Namespace) -> int:
    try:
        settings = get_settings()
        output = Path(args.output).expanduser() if args.output else None
        brief = IntelResearchEngine(settings).research(args.topic, depth=args.depth, output=output)
        payload = {
            "topic": brief.topic,
            "depth": args.depth,
            "sources": len(brief.sources),
            "verified_claims": len(brief.verified_facts),
            "markdown": str(brief.markdown_path) if brief.markdown_path else None,
            "json": str(brief.json_path) if brief.json_path else None,
            "report": brief.markdown,
        }
    except (IntelError, OSError) as exc:
        if args.json:
            _print_json({"ok": False, "error": str(exc)})
        else:
            _console().print(f"[red]intel failed[/red]: {exc}")
        return 1
    if args.json:
        _print_json({"ok": True, **payload})
    else:
        from rich.table import Table

        table = Table(title="Autonomous intelligence brief", header_style="bold")
        table.add_column("field")
        table.add_column("value")
        for key in ("topic", "depth", "sources", "verified_claims", "markdown", "json"):
            value = payload[key]
            table.add_row(key, "-" if value is None else str(value))
        _console().print(table)
        _console().print(brief.markdown)
    return 0


def cmd_telegram(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.telegram_bot_token:
        message = (
            "Telegram is not configured. Set TELEGRAM_BOT_TOKEN and "
            "TELEGRAM_ALLOWED_USERS (comma-separated numeric user IDs), then run `ultron telegram`."
        )
        if args.json:
            _print_json({"configured": False, "message": message})
        else:
            _console().print(message)
        return 0
    if not settings.telegram_allowed_user_ids:
        message = (
            "Telegram is fail-closed: TELEGRAM_ALLOWED_USERS is empty. "
            "Set at least one authorized numeric user ID."
        )
        if args.json:
            _print_json({"configured": False, "message": message})
        else:
            _console().print(message)
        return 0
    client = TelegramBotClient(settings=settings, timeout=args.timeout)
    cockpit = TelegramCockpit(client, settings=settings)
    if args.once:
        cockpit.run_polling(timeout=args.timeout, max_cycles=1)
    else:
        cockpit.run_polling(timeout=args.timeout)
    return 0


def _load_eval_module():
    """Import ``eval/run.py`` by path (it deliberately is not a package)."""
    import importlib.util

    path = get_settings().repo_root / "eval" / "run.py"
    spec = importlib.util.spec_from_file_location("ultron_eval_run", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise UltronError(f"cannot load eval runner at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_extraction_eval_module():
    import importlib.util

    path = get_settings().repo_root / "eval" / "extraction_run.py"
    spec = importlib.util.spec_from_file_location("ultron_extraction_eval", path)
    if spec is None or spec.loader is None:
        raise UltronError(f"cannot load extraction eval runner at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_extract_eval(args: argparse.Namespace) -> int:
    report = _load_extraction_eval_module().run_extraction_eval(
        Path(args.tasks) if args.tasks else None, quiet=args.json
    )
    if args.json:
        _print_json(report.as_dict())
    return 0 if report.exact_tasks == report.tasks else 1


def cmd_eval(args: argparse.Namespace) -> int:
    run_eval = _load_eval_module().run_eval

    report = run_eval(
        tasks_path=Path(args.tasks) if args.tasks else None,
        limit=args.limit,
        backend=args.backend,
        quiet=args.json,
        write_report=not args.no_report,
    )
    if args.json:
        _print_json(report.as_dict())
    return report.exit_code


def cmd_version(_: argparse.Namespace) -> int:
    _print_json({"ultron": __version__, "python": sys.version.split()[0]})
    return 0


# ----------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ultron", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_sub(name: str, **kwargs: Any) -> argparse.ArgumentParser:
        """Subparser that accepts --json after the command without clobbering the
        value given before it (SUPPRESS keeps the parent's store_true alive)."""
        child = sub.add_parser(name, **kwargs)
        child.add_argument(
            "--json",
            action="store_true",
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
        return child

    p_extract = add_sub("extract", help="grounded extraction from one URL or local file")
    p_extract.add_argument("source", help="public http(s) URL or local HTML/CSV/JSON/text/PDF path")
    p_extract.add_argument("--fields", required=True, help='JSON schema, e.g. \'{"title":"string","price":"float"}\'')
    p_extract.add_argument("--format", choices=["json", "csv", "md"], default="json")
    p_extract.add_argument("--output", default=None, help="write the rendered artifact to this path")
    p_extract.add_argument("--interactive", action="store_true", help="allow the live-network approval prompt")
    p_extract.set_defaults(func=cmd_extract)

    p_run = add_sub("run", help="run one goal through the agent loop")
    p_run.add_argument("goal")
    p_run.add_argument("--max-steps", type=int, default=None)
    p_run.add_argument("--backend", choices=["docker", "local"], default=None)
    p_run.add_argument("--interactive", action="store_true", help="allow human approval prompts")
    p_run.add_argument("--no-interactive", action="store_true", help="never prompt (default)")
    p_run.add_argument("--no-recall", action="store_true", help="ignore memory recall")
    p_run.set_defaults(func=cmd_run)

    p_tools = add_sub("tools", help="inspect the tool registry")
    p_tools.add_argument("action", choices=["list", "show", "validate"], default="list", nargs="?")
    p_tools.add_argument("name", nargs="?", default=None)
    p_tools.add_argument("--strict", action="store_true", help="fail on any invalid manifest")
    p_tools.set_defaults(func=cmd_tools)

    p_cache = add_sub("cache", help="cache statistics and maintenance")
    p_cache.add_argument("action", choices=["stats", "clear", "prune"], default="stats", nargs="?")
    p_cache.add_argument(
        "--namespace", choices=["router", "plan", "llm", "web", "tool", "judge"], default=None
    )
    p_cache.set_defaults(func=cmd_cache)

    p_mem = add_sub("memory", help="episodic memory")
    p_mem.add_argument(
        "action", choices=["stats", "runs", "failures", "forget"], default="stats", nargs="?"
    )
    p_mem.add_argument("--limit", type=int, default=20)
    p_mem.set_defaults(func=cmd_memory)

    p_extract_eval = add_sub("extract-eval", help="run the offline extractor fixture evaluation")
    p_extract_eval.add_argument("--tasks", default=None)
    p_extract_eval.set_defaults(func=cmd_extract_eval)

    p_eval = add_sub("eval", help="run the eval harness and gate on thresholds")
    p_eval.add_argument("--tasks", default=None)
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--backend", choices=["docker", "local"], default="local")
    p_eval.add_argument("--no-report", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_gaps = add_sub("gaps", help="show or clear recorded capability gaps")
    p_gaps.add_argument("--clear", action="store_true", help="empty the failure ledger")
    p_gaps.set_defaults(func=cmd_gaps)

    p_forge = add_sub("forge", help="process explicit-code capability forge templates")
    p_forge.add_argument("--limit", type=int, default=3, help="maximum gaps to process")
    p_forge.set_defaults(func=cmd_forge)

    p_scavenge = add_sub("scavenge", help="discover and safely wrap allowlisted OpenAPI specs")
    p_scavenge.add_argument("--max", type=int, default=20, dest="max", help="maximum candidates")
    p_scavenge.set_defaults(func=cmd_scavenge)

    p_harvest = add_sub("harvest", help="inspect and optionally forge an allowlisted PyPI package")
    p_harvest.add_argument("package_name")
    p_harvest.add_argument("--function", default="main", help="dotted target function")
    p_harvest.add_argument("--test-input", default="{}", help="JSON object passed to the wrapper")
    p_harvest.add_argument("--output-key", default="result", help="manifest output field")
    p_harvest.add_argument("--dry-run", action="store_true", help="inspect and synthesize only")
    p_harvest.set_defaults(func=cmd_harvest)

    p_refine = add_sub("refine", help="AST-harden and test a raw Python tool")
    p_refine.add_argument("path")
    p_refine.add_argument("--function", default="run", help="target function in the raw file")
    p_refine.add_argument(
        "--test-input", default="{}", help="JSON object passed to the cooked tool"
    )
    p_refine.add_argument("--output-schema", default='{"result":"any"}', help="JSON output schema")
    p_refine.add_argument("--provides", action="append", default=None, help="semantic output tag")
    p_refine.add_argument("--requires", action="append", default=None, help="semantic input tag")
    p_refine.add_argument(
        "--dry-run", action="store_true", help="refine without sandboxing or registration"
    )
    p_refine.set_defaults(func=cmd_refine)

    p_repair = add_sub("repair", help="inspect self-repair tickets")
    p_repair.add_argument("--list", action="store_true", help="show open repair tickets")
    p_repair.set_defaults(func=cmd_repair)

    p_intel = add_sub("intel", help="research a topic from public unauthenticated sources")
    p_intel.add_argument("topic")
    p_intel.add_argument("--depth", choices=["light", "deep"], default="light")
    p_intel.add_argument(
        "--output", default=None, help="path for the Markdown brief (JSON is adjacent)"
    )
    p_intel.set_defaults(func=cmd_intel)

    p_telegram = add_sub("telegram", help="start the Telegram cockpit")
    p_telegram.add_argument("--timeout", type=int, default=20, help="long-poll timeout seconds")
    p_telegram.add_argument(
        "--once", action="store_true", help="process one polling cycle and exit"
    )
    p_telegram.set_defaults(func=cmd_telegram)

    p_doc = add_sub("doctor", help="check the environment")
    p_doc.set_defaults(func=cmd_doctor)

    p_ver = add_sub("version", help="print versions")
    p_ver.set_defaults(func=cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        _console().print("[yellow]interrupted[/yellow]")
        return 130
    except UltronError as exc:
        _console().print(f"[red]{type(exc).__name__}[/red]: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
