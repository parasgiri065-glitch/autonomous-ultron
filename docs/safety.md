# Safety model (Phase 1)

This document is the threat model for the *safe executor*. It states what is
enforced, where it is enforced, and — importantly — what it does **not** protect
against yet.

## Non-negotiables

1. **No execution without a granted policy decision.** `Sandbox.run()` raises if
   `decision.allowed` is false. There is no alternate path from the agent to the
   container.
2. **No secrets in the sandbox.** Container env = `PYTHON*` + explicitly
   allowlisted names, minus anything matching secret markers. The host environment
   is never inherited, no `.env` is mounted, no `--env-file` is used.
3. **No network unless asked for and granted.** `--network none` is the default;
   `network:http` in the manifest plus a granting decision is the only way egress
   turns on. `ULTRON_POLICY_NETWORK=deny` disables it globally.
4. **No shell.** `entrypoint` is validated (shell metacharacters rejected at
   manifest load) and executed as argv with a program allowlist.
5. **No unattended self-approval.** Without a TTY the prompter is `DenyAllPrompter`;
   MEDIUM risk is refused, not assumed. `ULTRON_POLICY_ASSUME_YES=1` is dev-only and
   is not set in CI.
6. **Nothing is trusted before verification.** Outputs are schema-checked, type-checked,
   sanity-checked and (for research answers) grounding-checked. Unverified output is
   never cached and never written to memory as a success.

## Enforcement points

| control | implemented in | failure mode if bypassed |
| --- | --- | --- |
| input contract (fields/types/size) | `policy.PolicyGate._validate_inputs` | step denied, nothing runs |
| secret scanning of inputs | `policy.scan_for_secrets` | step denied with the matched pattern classes |
| permission model | `registry.Permission` + `policy.PolicyGate.check` | unknown scopes are manifest errors; `secrets:*`/`fs:write:*` denied |
| risk tiers | `policy.PolicyGate` | LOW allow · MEDIUM ask · HIGH deny-unless-pinned |
| approval binding | `policy.ApprovalStore` | approvals keyed to `(tool, version, manifest hash, input digest)`, TTL'd, single-use |
| container hardening | `sandbox.Sandbox._docker_argv` | network mode (`none`/`bridge`), ro-fs, caps, uid, limits, timeout |
| output trust | `verifier.Verifier` | fail = not cached, not remembered, run stops |
| audit trail | `policy.AuditLog` | JSONL of every decision, with digests rather than payloads |

## Container properties (asserted in `tests/test_smoke.py`)

```
--network=none | bridge(when granted)   --read-only      --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m
--cap-drop ALL                 --security-opt no-new-privileges      --user 65534:65534
--ipc=none                     --memory=512m --memory-swap=512m      --cpus=1 --pids-limit=128
--ulimit nofile=256:256        -v <repo>:/workspace:ro               -w /workspace
-e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp      [+ allowlisted, non-secret vars]
```

On timeout the container is killed by name; the result records `timed_out=True`
and the tool's partial stdout is *not* treated as a result.

## Trust boundaries

* **The model is untrusted.** Its router/planner/judge output is parsed as JSON,
  range-checked, and validated against the registry. Unknown tool names, HIGH-risk
  steps, wrong input shapes and out-of-enum values are dropped, and the
  deterministic plan is used instead.
* **Tool output is untrusted.** A tool that returns a refusal, an empty source
  list, a fabricated-looking summary or a type mismatch fails verification.
* **Inputs are untrusted.** A prompt that tries to smuggle `sk-…`, `ghp_…`, an
  AWS key, a JWT or a PEM blob into a tool call is refused before the sandbox is
  even composed; approval previews are redacted.
* **The manifest is trusted-but-hashed.** Every approval and plan is pinned to the
  manifest's content hash, so editing a tool invalidates both.
* **The host is trusted.** Docker and the operator are inside the TCB.

## Known limits (deliberate, Phase 1)

* **Docker is the boundary, not a VM.** A kernel LPE escapes it. Use rootless
  Docker/gVisor/microVM if you need stronger isolation.
* **The `local` backend is not a sandbox.** It exists for tests/CI, requires
  `ULTRON_ALLOW_LOCAL_SANDBOX=1`, and labels every result
  `network=host(unenforced)`.
* **Network is binary.** A granted `network:http` maps to Docker mode `bridge`, i.e. a
  tool can reach *any* host (unknown permission details fail closed to `none`). Domain
  allowlisting needs an egress proxy (planned with Phase 2 discovery).
* **Resource limits are per-container**, not per-run: the run budget caps cost,
  time, steps and LLM calls, but a tool can still use its full CPU/memory slice
  for up to `timeout_s`.
* **No multi-tenant isolation.** One operator, one host, one daemon; the audit log
  and approval store are not protected against a malicious local user.
* **Grounding check is lexical.** It catches invented entities/numbers, not subtle
  misattribution. The optional LLM judge (off by default, priced, cached) is the
  next layer, and Phase 4 domain verifiers are the third.
* **Cached results are trusted for their TTL.** A cached web page is up to
  `ULTRON_CACHE_TTL_WEB` old — that is a correctness tradeoff, not a safety one.
* **Phase 2/3 are not implemented.** No repo/package installation, no
  self-modification, no code generation merging. When they land, they inherit this
  gate: fetch → static scan → `network=none` dry run → manifest proposal → human PR.

## Operational checklist before running unattended

```bash
uv run ultron doctor                       # docker present, image built, manifests valid
echo $ULTRON_ALLOW_LOCAL_SANDBOX           # must be unset/0
echo $ULTRON_POLICY_ASSUME_YES             # must be unset/0
uv run python eval/run.py --limit 3        # gates still pass
tail -n 20 .ultron/audit.jsonl             # every decision is on the record
```
