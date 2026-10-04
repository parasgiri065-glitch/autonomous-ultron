# Tools

A *tool* is two things that must agree:

1. a **manifest** — JSON in `tools/**/*.json`, validated against `tools/schema.json`
   (the contract read by the registry, policy gate, sandbox and verifier);
2. an **implementation** — a runnable module under `tools/`, invoked as
   `python -m tools.<name>`.

## The stdin/stdout contract

```
echo '{"expression": "12*(3+4)"}' | python -m tools.calc
{"ok": true, "result": {"result": 84.0, "expression": "12*(3+4)", "normalized": "12 * (3 + 4)"}, "meta": {...}}
```

* **stdin**: a single JSON object of inputs (must match `inputs` in the manifest).
* **stdout**: one JSON **envelope**, `{"ok": bool, "result": {...}, "error": str|null}`.
  Stray log lines are tolerated, but only the last JSON object carrying `"ok"` counts.
* **exit code**: `0` on success. Non-zero (or a timeout) fails the step.
* `result` must match `outputs` in the manifest or the verifier rejects it — and
  a rejected result is never cached and never remembered.

Use `tools/_io.py` for all of this: `main_guard(run)` gives you parsing, error
handling and timing for free.

## Adding a tool (Phase 1)

```bash
cp tools/examples/calc.json tools/examples/my_tool.json   # edit name/version/risk/permissions/inputs/outputs
$EDITOR tools/my_tool.py                                  # implement run(payload) -> dict
uv run python -m tools.my_tool < input.json               # smoke test locally
uv run pytest                                             # contract + policy tests
uv run python eval/run.py --update-baseline                # if it changes eval metrics
```

Rules of the road:

* **Declare honestly.** `permissions` is not documentation, it is the argument
  the policy gate and sandbox use. `network:*` means the container is started
  with egress; `secrets:*` and `fs:write:*` are denied in Phase 1.
* **Pick the right risk tier.** LOW = safe to auto-run (still sandboxed, no
  network unless declared). MEDIUM = a human is asked every time (approvals are
  pinned to the exact inputs and are single-use). HIGH = denied unless a pinned
  approval is written to `.ultron/approvals.json`.
* **Be deterministic if you can.** `"deterministic": true` tools cost $0, cache
  perfectly and are preferred by the planner. An LLM inside a tool is a Phase 2
  concern (it needs a budget and its own cache).
* **Keep outputs small.** Results are cached and shown to humans; a 5 MB result
  is a smell (and the sandbox truncates stdout at 4 MB).
* **Never read secrets.** Nothing in the host environment reaches the container
  except explicitly allowlisted, non-secret names.

## Catalogue

| tool | risk | network | what it does |
| --- | --- | --- | --- |
| `calc` | low | none | whitelisted-AST arithmetic (no `eval`), free and offline |
| `web_research` | low | `http` | fetch + **extractive** summary + sources + confidence (no LLM inside the tool) |
| `http_fetch` | medium | `http` | single URL fetch; exists to exercise the human-in-the-loop path |

`tools/examples/*.json` holds the manifests; keep example tools cheap and honest
so they double as documentation.
