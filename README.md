# autonomous-ultron

A cost-optimized, safety-first harness for autonomous **tool-forging** agents.

Ultron takes a goal, routes it cheaply, plans the fewest steps, runs tools inside a
locked-down Docker sandbox, verifies the output, remembers what worked, and refuses
to do anything its policy gate has not approved — all while treating **cost as a
first-class failure mode** with hard budgets and eval gates in CI.

> **Status: Phase 1 (this branch, `feat/init`).** Safe executor + cache + eval gates.
> No autonomous repo installs, no self-modification. Those are Phase 2 and 3, and
> they are gated behind the machinery built here.

---

## The loop

```
        goal
          │
          ▼
   ┌─────────────┐   cache hit ──► return remembered answer ($0, no tools)
   │  memory     │
   └─────────────┘
          │ miss
          ▼
   ┌─────────────┐   rules match ──► $0 classification   ┌──────────────┐
   │  router     │ ─────────────────────────────────────►│  planner     │
   │ (cheap LLM) │   ambiguous ──► small/cheap model      │ deterministic│
   └─────────────┘                                       └──────────────┘
          │                                                      │
          │                                        ┌─────────────┘
          ▼                                        ▼
   ┌────────────────┐   deny/ask ──► stop, explain  ┌──────────────────┐
   │  policy gate   │ ─────────────────────────────►│ human (MEDIUM)   │
   │ LOW/MED/HIGH   │                               │ pinned approvals │
   └────────────────┘                               └──────────────────┘
          │ allow
          ▼
   ┌──────────────────────────────────────────────┐
   │ Docker sandbox                               │
   │  network=none (unless manifest + gate say ok)│
   │  read-only fs · tmpfs /tmp · no secrets      │
   │  dropped caps · non-root · cpu/mem/pid caps  │
   └──────────────────────────────────────────────┘
          │ result envelope
          ▼
   ┌─────────────┐  fail ──► not cached, not remembered, step fails closed
   │  verifier   │
   └─────────────┘
          │ pass
          ▼
   cache + memory  ──►  next step | final answer | eval gate in CI
```

Every arrow is a module you can read in one sitting:

| file | responsibility |
| --- | --- |
| `src/ultron/agent.py` | the loop, budgets/kill-switches, answer composition |
| `src/ultron/router.py` | difficulty → plan depth (rules first, cheap model second) |
| `src/ultron/planner.py` | goal → concrete steps (deterministic, validated LLM opt-in) |
| `src/ultron/registry.py` | JSON manifests: versions, permissions, risk tiers |
| `src/ultron/policy.py` | the choke point: allow / ask / deny, approvals, audit log |
| `src/ultron/sandbox.py` | Docker execution: network off, read-only, no secrets |
| `src/ultron/verifier.py` | schema + sanity + grounding (+ optional LLM judge) |
| `src/ultron/cache.py` | SQLite cache: tool results, web fetches, LLM responses |
| `src/ultron/memory.py` | episodic memory, metrics, recall of verified answers |
| `eval/run.py` | eval harness: success rate, cost, latency, cache hit rate, gates |

---

## Quickstart (5 minutes, no API keys, no Docker)

```bash
git clone <this repo> && cd autonomous-ultron
uv sync --group dev

# 1. Is this machine ready? (checks Docker, image, manifests, keys)
uv run ultron doctor

# 2. Build the sandbox image, then run a goal end-to-end ($0: deterministic tool)
docker build -t ultron-tools:phase1 -f docker/Dockerfile .
uv run ultron run "calculate 12*(3+4)"

# 3. Inspect the safety + cost plumbing
uv run ultron tools list
uv run ultron cache stats
uv run ultron memory stats

# 4. The eval gate (same thing CI runs), incl. a warm-cache and recall pass
uv run python eval/run.py --passes 3
```

No Docker on this machine? The `local` backend exists for development, but it is
**not** a security boundary, so it needs an explicit double opt-in and every result
is labelled `network=host(unenforced)`:

```bash
ULTRON_ALLOW_LOCAL_SANDBOX=1 uv run ultron run "calculate 12*(3+4)" --backend local
ULTRON_ALLOW_LOCAL_SANDBOX=1 ULTRON_SANDBOX=local uv run python eval/run.py --passes 3
```

Want live models? `cp .env.example .env`, set one provider key, and the router/planner/judge
stop using stubs. Everything still works without them — that is the point of `ULTRON_LLM_MODE=auto`.

---

## Phase 1 — safe executor + cache + eval gates

What actually exists and is tested today:

* **Safe executor.** One auditable code path from "policy said yes" to "container ran".
  Network off by default, read-only root filesystem, no secrets in the container,
  dropped capabilities, non-root uid, memory/CPU/PID limits, wall-clock kill.
* **Cache-first economics.** SQLite cache in five namespaces (`router`, `plan`, `llm`,
  `web`, `tool`) plus a judge verdict cache. A hit skips the container *and* the model.
* **Eval gates.** `eval/tasks.jsonl` + `eval/run.py` measure success rate, cost per run,
  latency and cache hit rate against a committed baseline; CI fails the PR on regression.
* **Policy gate.** LOW auto-approves, MEDIUM asks a human, HIGH is denied unless a
  pinned approval exists. Approvals are bound to the manifest hash *and* the exact inputs.
* **Verification before trust.** Schema, types, sanity, source-URL shape, and a
  deterministic grounding check; an optional cheap LLM judge for free-form answers.
* **Memory.** Only *verified* runs are remembered, so a broken tool cannot poison the cache.

### Roadmap

| Phase | Scope | Enabled by Phase 1 |
| --- | --- | --- |
| **2** | **Sandboxed tool discovery/install** — scout GitHub/PyPI/npm for candidate tools, fetch into a scratch sandbox, static-scan, dry-run with `network=none`, then propose a manifest (never auto-enable) | registry + policy gate + sandbox already refuse unknown/permission-heavy tools |
| **3** | **Self-upgrade via PRs** — Ultron proposes changes to its own tools/prompts as pull requests; CI runs the eval gate + cost gate on every PR; humans merge | eval harness + thresholds + baseline are the merge gate |
| **4** | **Domain packs** — research, coding, lawful OSINT, recovery copilot: curated manifest bundles with domain-specific risk tiers and verifiers | manifest `tags`/`risk` and the verifier interface |

The sequencing is deliberate: **the eval gate must exist before anything is allowed
to change the system** — including Ultron itself. Phase 3's "self-upgrade" is only
safe because Phase 1 measures success rate and cost per run on every PR.

---

## Cost architecture

Five mechanisms, in the order they trigger:

1. **Recall before work.** An identical goal against an identical registry returns the
   previously *verified* answer: no tools, no tokens. (`memory.recall`, digest-pinned.)
2. **Deterministic-first planning.** The planner maps goals to tools with regex/keyword
   rules and a small cost model. LLM planning is opt-in (`ULTRON_PLANNER_USE_LLM=1`) and
   its output is validated against the registry, so a hallucinated tool dies before the sandbox.
3. **A cheap router that mostly doesn't run.** Deterministic rules resolve arithmetic,
   single-URL fetches, single-tool matches, chit-chat and multi-step shapes with **zero**
   model calls. Only genuinely ambiguous goals escalate to a small model in JSON mode
   with `max_tokens` capped, and even then the answer is cached and validated.
4. **Aggressive caching.** Tool outputs (keyed by manifest content hash + inputs),
   web fetches, router classifications, plans, LLM responses and judge verdicts all land
   in SQLite with per-namespace TTLs. Cache hits cost nothing and are counted.
5. **Eval kill-switches.** CI fails if avg cost per run exceeds the threshold or if
   success rate regresses against `eval/baseline.json`. Cheap-but-wrong cannot ship.

Model tiers are env-driven and default to small: `ULTRON_ROUTER_MODEL`,
`ULTRON_PLANNER_MODEL`, `ULTRON_JUDGE_MODEL` (all `gpt-4o-mini` by default, ~$0.15/1M in).
`ULTRON_BUDGET_MAX_USD`, `ULTRON_BUDGET_MAX_STEPS`, `ULTRON_BUDGET_MAX_SECONDS` and
`ULTRON_BUDGET_MAX_LLM_CALLS` are **hard stops**, not warnings: breaching one ends the run
with status `budget_exceeded` and a composed partial answer.

Cost is reported honestly: `cost_basis` is `actual`, `simulated` (stub priced with the real
table, used by CI so the gate isn't vacuous), `cache_hit`, `offline_stub` or `no_llm`.

---

## Safety model

**Nothing runs without passing the policy gate.** The gate is the only place that can
authorise execution, and the sandbox only accepts a granted decision.

| tier | policy | notes |
| --- | --- | --- |
| `low` | auto-approve | still sandboxed; network only if the manifest declares it *and* policy allows |
| `medium` | **ask a human** | approval pinned to `(tool, version, manifest hash, input digest)`, single-use, TTL'd |
| `high` | **deny** | requires an explicitly written, pinned approval; never auto-granted, refusable by anyone |

Additional fail-closed behaviours:

* **No secret leakage.** The container env is `PYTHON*` + explicitly allowlisted names,
  filtered again for secret-looking keys (`*KEY*`, `*TOKEN*`, `*SECRET*`, …). Inputs are
  scanned for credential shapes (`sk-…`, `ghp_…`, `AKIA…`, PEM blobs, JWTs, bearer tokens)
  and the gate **refuses the step** rather than moving a credential across the boundary.
  Approval previews are redacted; the audit log stores digests, not raw payloads.
* **No shell.** Manifests with shell metacharacters in `entrypoint` are rejected at load
  time; execution is argv-only; the program must be in a small allowlist.
* **No silently granted network.** `network:http` is granted only when the manifest asks
  *and* `ULTRON_POLICY_NETWORK != deny`; `--network none` otherwise, in every other case.
* **Nothing is trusted before verification.** Unverified results are never cached and
  never remembered, so failures cannot become "facts". Research answers get a
  deterministic grounding check (answer tokens must trace to the sources).
* **Unattended runs cannot self-approve.** No TTY ⇒ `DenyAllPrompter` ⇒ MEDIUM risk is
  refused, not assumed. `ULTRON_POLICY_ASSUME_YES=1` exists for dev and is never set in CI.
* **Phase 1 can't do X, and says so.** `secrets:*` and `fs:write:*` are denied by design,
  not silently ignored.

See [`docs/safety.md`](docs/safety.md) for the threat model and its limits, and
[`docs/cost-model.md`](docs/cost-model.md) for the arithmetic behind the caching claims.

---

## Tool manifests

`tools/schema.json` is the contract; `tools/examples/` has working examples. A manifest is
read by four components: registry (validate), policy gate (risk + permissions), sandbox
(entrypoint + network), verifier (`outputs`).

```json
{
  "name": "web_research",
  "version": "0.1.0",
  "entrypoint": "python -m tools.web_research",
  "permissions": ["network:http"],
  "risk": "low",
  "inputs": {"query": "string", "max_sources": "int"},
  "outputs": {"summary": "string", "sources": "list[string]", "confidence": "float"}
}
```

Optional fields: `description`, `tags`, `deterministic`, `timeout_s`, `cache_ttl_s`,
`price_estimate_usd`, `author`. Several versions may coexist; `registry.get("name@1.2.0")`
pins one, and plans are keyed on the registry fingerprint so an edited manifest cannot
silently re-use a cached plan or an old approval. See [`tools/README.md`](tools/README.md).

---

## CLI

```bash
ultron doctor                       # environment readiness, tool inventory, paths
ultron run "<goal>" [--json] [--backend docker|local] [--interactive] [--no-recall]
ultron tools list|show NAME|validate
ultron cache stats|clear|prune [--namespace tool|llm|web|router|plan|judge]
ultron memory stats|runs|failures|forget [--limit N]
ultron eval [--limit N] [--passes 1|2|3] [--backend local|docker] [--json]
```

`--json` on every subcommand; `python -m ultron.main ...` works without installation.

---

## Eval harness

`eval/tasks.jsonl` — one JSON object per line: a goal plus *checkable* expectations
(expected status, tools, answer substrings, source count, policy outcome, cost cap).
A medium-risk tool that correctly refuses to run unattended **passes** its task: refusing
is the contract.

`eval/run.py` runs the suite three times: cold (truth about correctness/spend), warm
(truth about the cache) and a recall pass (does memory short-circuit the work?).

```bash
uv run python eval/run.py                                  # gate + rich table
uv run python eval/run.py --limit 3 --passes 1 --json      # CI smoke
uv run python eval/run.py --update-baseline                # after an intentional change
```

Gates (all configurable, enforced by exit code):

| gate | default | env |
| --- | --- | --- |
| success rate ≥ | 80% | `ULTRON_EVAL_MIN_SUCCESS_RATE` |
| avg cost / run ≤ | $0.02 | `ULTRON_EVAL_MAX_AVG_COST_USD` |
| avg latency ≤ | 20 s | `ULTRON_EVAL_MAX_AVG_LATENCY_S` |
| no regression vs `eval/baseline.json` | success −1 pt, cost +25%, latency +50% | `ULTRON_EVAL_BASELINE` |

---

## Live network mode (opt-in, local only)

By default every network tool reads the **fixture corpus** (`web_mock`), so tests, CI and
the eval gate never touch the internet. Two things change that, both explicit:

| knob | default | what it does |
| --- | --- | --- |
| `ULTRON_EVAL_LIVE=1` | `0` | `web_research` / `http_fetch` may reach the real network instead of fixtures |
| `ULTRON_WEB_CACHE` | `<state_dir>/webcache/webcache.db` | URL-keyed page cache (local backend) |
| `ULTRON_CACHE_TTL_WEB` | `900` s | page-cache TTL (0 disables the page cache) |

What live mode gives you: a **URL-keyed SQLite cache**, so two different queries that
touch the same page pay for it once (`select key, hits from cache where ns='web'` — the
tools write the same schema the harness uses). Precedence is `web_mock` first, then live,
then plain HTTP, so exporting `ULTRON_EVAL_LIVE=1` can never make an eval run leave the
fixtures: the eval builds its own `Settings` with `web_mock` set and `eval_live=False`.

What live mode does **not** change: the policy gate. `web_research` declares
`network:http`, and with the default `policy_network_low_auto=False` a LOW+network tool is
escalated to a human, so a live run still needs an interactive approval. Live mode is
*never* enabled by CI.

```bash
# manual, local, on purpose -- expect an approval prompt
ULTRON_EVAL_LIVE=1 uv run ultron run "research grid-scale battery storage economics"
```

Docker note: live mode **adds no mount and does not relax the container**. The repo is
still the only volume and still `:ro`; the only writable path inside is the noexec tmpfs.
The URL-keyed page cache is therefore a *local-backend* feature: a docker run over a live
URL fetches without the page cache (the harness-level envelope cache still applies), so
nothing about the sandbox's filesystem guarantees changes when you enable egress.

---

## CI

`.github/workflows/ci.yml`:

1. **lint** — `ruff`, import sanity, every manifest validated against `tools/schema.json`.
2. **pytest** — the whole suite runs offline on Python 3.11 + 3.12, no Docker, no keys.
3. **eval-gate** — full eval with `ULTRON_COST_SIMULATE=1`, writes the report to the job
   summary, uploads it as an artifact, and **fails the PR** on any breached gate.
4. **sandbox-docker** — builds the real image and runs tools through real Docker with
   `--network none` and a read-only filesystem, so the sandbox config can't silently rot.
   It also runs a 10 MB stdout flooder inside the container to prove the sandbox kills a
   runaway tool and reports it (`result.ok == False`), since the kill path is Docker-only.

---

## Configuration

All knobs are `ULTRON_*` env vars (see [`.env.example`](.env.example) for the annotated
list): budget ceilings, model choices, cache TTLs, sandbox backend/image/limits, policy
network mode, approval + audit paths, judge/synthesis toggles and eval thresholds.
Nothing in the environment is forwarded into a sandbox except explicitly allowlisted,
non-secret names.

## Development

```bash
uv sync --group dev
uv run pytest -q                       # offline, hermetic, no docker required
uv run ruff check . && uv run ruff format --check .
uv run python eval/run.py --limit 3    # what CI gates on
```

Layout:

```
src/ultron/     the harness (config, policy, sandbox, cache, verifier, agent, …)
tools/          manifests (schema.json, examples/) + reference tool implementations
eval/           tasks.jsonl, run.py, fixtures/, baseline.json
tests/          smoke tests asserting the safety + cost contract
docker/         the sandbox image (repo is mounted, never baked in)
docs/           safety threat model, cost model
```

## Non-goals (Phase 1)

* No installing or executing third-party repos/packages — that is Phase 2, and it will
  arrive with static scanning and `network=none` dry-runs first.
* No self-modification, no generated tools merged automatically — Phase 3, via PR + eval gate.
* No secret handling of any kind inside tools. Ever (that needs a broker design, not a flag).
* No multi-tenant isolation guarantees: one operator, one host, one Docker daemon.

## License

MIT.
