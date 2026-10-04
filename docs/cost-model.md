# Cost model

Ultron treats spend as a hard constraint with an audit trail, not a dashboard.
This document explains where money can be spent, how it is prevented, and how the
eval gate keeps the claim honest.

## The only places money is spent

| stage | model | when it runs | cached? |
| --- | --- | --- | --- |
| router escalation | `ULTRON_ROUTER_MODEL` (small, JSON, `max_tokens=120`) | only ambiguous goals | yes (`llm` + `router` namespaces) |
| planner upgrade | `ULTRON_PLANNER_MODEL` | only if `ULTRON_PLANNER_USE_LLM=1` **and** depth ≥ 2 | yes (`plan`) |
| verifier judge | `ULTRON_JUDGE_MODEL` | only if `ULTRON_ENABLE_LLM_JUDGE=1` | yes (`judge`) |
| answer synthesis | `ULTRON_SYNTH_MODEL` | only if `ULTRON_ENABLE_SYNTHESIS=1` | yes (`llm`) |

Everything else — routing rules, planning rules, registry search, tool execution,
schema/sanity/grounding verification, memory recall, answer composition — is
deterministic and costs **$0**. Tools in Phase 1 contain no models at all: the
research summarizer is extractive, the calculator is arithmetic.

## Why a "cheap router" is actually cheap

The router is not "a small model on every request"; it is **rules first, model
only on ambiguity**:

1. identical goal + identical registry → SQLite lookup (`$0`, microseconds);
2. arithmetic, single-URL, chit-chat, single-registry-match, and multi-step-shape
   goals are classified by regexes (`$0`);
3. only genuinely ambiguous goals reach the small model, and the result is cached
   forever.

`RouteDecision.cheap_path` records when a goal was served without any model at
all, and `RouteDecision.cost_usd` records the price when one was used. In the eval
suite, 6 of 7 tasks are rule-classified; the ambiguous task is the only priced call
(≈$0.00003 at `gpt-4o-mini` table prices).

## Cache economics

Values are content-addressed (`sha256` of canonical JSON) and TTL'd per namespace:

| namespace | key | default TTL | what a hit saves |
| --- | --- | --- | --- |
| `tool` | manifest content hash + inputs + network grant | `ULTRON_CACHE_TTL_TOOL` (1 h) | the whole container run |
| `web` | URL + fetch params | `ULTRON_CACHE_TTL_WEB` (15 min) | an HTTP round trip (and rate-limit budget) |
| `llm` | model + messages + temperature + max_tokens | `ULTRON_CACHE_TTL_LLM` (24 h) | tokens |
| `router` | goal + registry fingerprint | 24 h | the classification call |
| `plan` | goal + depth + registry fingerprint | 24 h | planning (and any planner tokens) |
| `judge` | goal + answer + evidence + judge model | 24 h | the judge call |

Consequences that matter:

* **Temperature defaults to 0**, so the same prompt really is the same key. (Cached
  non-zero-temperature sampling would be a correctness bug, not a saving.)
* **A cache hit skips the container *and* the model.** `SandboxResult.cached=True`
  and `duration_s=0.0`; `LLMResponse.cached=True` and `cost_usd=0.0`.
* **Failures are never cached.** A step that did not verify is not stored, so a
  flaky tool cannot serve stale "success" to later runs.
* **Memory recall is the outer cache.** `Memory.recall` returns a previously
  *verified* answer for an identical goal + registry fingerprint: no tools, no
  tokens, no container. Recall is measured (`memory_recall_rate`) rather than
  mixed into the cache-hit metric.
* **Registry fingerprints invalidate plans.** Editing a manifest or adding a tool
  changes the fingerprint, so cached plans (and approvals) cannot silently persist
  across a tool change.

## Budgets are kill-switches

`Budget` is checked before and after every stage: USD, wall-clock seconds, steps
and LLM calls. Breaching any of them raises `BudgetExceeded`, which the agent turns
into `status="budget_exceeded"` plus a partial answer composed from whatever
already verified. Defaults are deliberately tiny (`$0.05`, 6 steps, 120 s, 12 LLM
calls) because the failure mode we care about is a runaway loop, not a slightly
expensive run.

## Measuring cost honestly

`AgentResult.cost_basis` tells you how to read `cost_usd`:

| basis | meaning |
| --- | --- |
| `actual` | provider-reported or price-table cost of real calls |
| `cache_hit` | every call was served from cache: the number is really $0 |
| `simulated` | deterministic stub output priced with the real price table (CI) |
| `offline_stub` | stubs, unpriced: $0 by construction |
| `no_llm` | the run never touched a model call site at all |

CI sets `ULTRON_LLM_MODE=stub` + `ULTRON_COST_SIMULATE=1` so the cost gate measures
the *accounting path* (pricing, caching, budget charging) instead of a vacuous
zero. Without that, every CI run would trivially pass a "$0 spent" gate.

## The gate

`eval/run.py` exits non-zero if any of these fail:

* `success_rate` < `ULTRON_EVAL_MIN_SUCCESS_RATE` (default 0.80)
* `avg_cost_usd` > `ULTRON_EVAL_MAX_AVG_COST_USD` (default $0.02)
* `avg_latency_s` > `ULTRON_EVAL_MAX_AVG_LATENCY_S` (default 20 s)
* regression vs `eval/baseline.json`: success −1 pt, cost +25 %, latency +50 %

Current baseline (stub mode, 7 tasks, offline fixtures):

| metric | value |
| --- | --- |
| success rate | 100 % |
| avg cost / run | $0.0000045 |
| avg latency | 0.03 s |
| cache hit rate (warm pass) | 100 % |
| memory recall rate | 57 % (the 2 non-success outcomes are, correctly, not remembered) |

Change a prompt or a tool and CI tells you — in money and in success rate —
whether it was worth it. That is the mechanism Phase 3 depends on: Ultron may
propose upgrades to itself, but it cannot merge a cost regression.
