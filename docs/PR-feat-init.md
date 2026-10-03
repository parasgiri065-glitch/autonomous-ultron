# PR: `feat/init` → `main`

**Title:** `Phase 1: safe executor + cache + eval gates (scaffold)`

Paste the section below as the pull-request description (`gh pr create --body-file docs/PR-feat-init.md`,
or copy it into the GitHub UI). Delete this file after the PR is opened.

---

## What this is

Phase 1 of `autonomous-ultron`: a cost-optimized, safety-first agent harness with a
**safe executor** (Docker + policy gate), a **cache-first** cost architecture and
**eval gates** wired into CI. No autonomous repo installs and no self-modification —
those are Phases 2 and 3, and they are constrained by the machinery in this PR.

## The loop

```
goal → memory recall → cheap router (rules first, small model only if ambiguous)
     → deterministic plan → policy gate → Docker sandbox (network=none by default)
     → verifier → cache + memory → next step | answer
```

## What is implemented

| area | file | notes |
| --- | --- | --- |
| agent loop + budgets | `src/ultron/agent.py` | USD/steps/seconds/LLM-call kill-switches; honest status taxonomy |
| routing | `src/ultron/router.py` | rules resolve most goals at $0; cheap model only on ambiguity, reply validated + cached |
| planning | `src/ultron/planner.py` | deterministic, cost-annotated steps; LLM planner opt-in and registry-validated |
| registry | `src/ultron/registry.py` | JSON manifests, semver coexistence, permissions, risk tiers, deterministic search, `tools/schema.json` validation |
| policy gate | `src/ultron/policy.py` | LOW auto / MEDIUM ask / HIGH deny-unless-pinned; approvals bound to manifest hash + input digest; audit JSONL; secret scanning |
| sandbox | `src/ultron/sandbox.py` | `--network none`, `--read-only`, tmpfs, `--cap-drop ALL`, `no-new-privileges`, non-root, cpu/mem/pid caps, timeout kill, no host env |
| cache | `src/ultron/cache.py` | SQLite, 6 namespaces, TTLs, hit/miss accounting; hits skip execution *and* LLM calls |
| verifier | `src/ultron/verifier.py` | envelope → schema → types → sanity → sources → grounding (+ optional priced LLM judge) |
| memory | `src/ultron/memory.py` | episodic runs/steps/tool stats; recall of *verified* answers only |
| eval harness | `eval/run.py`, `eval/tasks.jsonl` | 3 passes (cold/warm/recall), success rate, avg cost, latency, cache hit rate, baseline regression gates |
| CI | `.github/workflows/ci.yml` | lint, pytest (3.11+3.12), eval gate, real Docker sandbox job |
| tools | `tools/` | manifest schema + 3 reference tools (`calc`, `web_research`, `http_fetch`) + stdin/stdout contract |

## Evidence

* `uv run pytest -q` → **47 passed** (no network, no Docker, no API keys)
* `uv run python eval/run.py --passes 3` → **100% success · $0.000005 avg cost · 0.03 s avg
  latency · 100% warm cache hit rate** (all gates and the committed baseline pass)
* `uv run ruff check . && uv run ruff format --check .` → clean

## Cost design

Deterministic-first (rules beat models), content-addressed SQLite caching in six
namespaces, memory recall before any work, temperature 0 so cache keys are stable, and
hard budget kill-switches. CI runs the eval in `ULTRON_LLM_MODE=stub` so the cost gate
measures the real accounting path (priced stubs) rather than a vacuous `$0`.

## Safety design

Every execution passes the policy gate; the sandbox only accepts a granted decision.
No secrets cross the boundary (allowlisted env + input secret scanning + redacted
approvals + digest-only audit log). No shell. No unattended self-approval. Unverified
output is never cached or remembered. See `docs/safety.md` for the threat model and its
explicit limits (Docker is the boundary, not a VM; `network:*` is not domain-scoped yet).

## Reviewer notes / open questions

1. **MEDIUM risk UX**: approvals are single-use and input-pinned; is that the right
   default for interactive use, or should there be a scoped "approve this tool for N
   minutes" grant?
2. **`network:http` is not domain-scoped.** Phase 2 should add an egress proxy with a
   domain allowlist before discovery lands.
3. **Grounding check is lexical** (first-5-char stems). It catches invented entities and
   numbers; subtle misattribution needs the judge or Phase 4 domain verifiers.
4. **`eval/tasks.jsonl` is small (7 tasks).** Propose growing it to ~25 before Phase 2,
   including a hallucination-bait task and a timeout task.

## Deliberately out of scope

Autonomous tool discovery/install (Phase 2), self-upgrade PRs (Phase 3), domain packs
(Phase 4), secret brokering, multi-tenant isolation.
