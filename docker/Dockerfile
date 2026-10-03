# Tool sandbox image for Ultron Phase 1.
#
# Design notes:
#   * The repo is NOT baked in. It is mounted read-only at /workspace by
#     `ultron.sandbox`, so the image stays generic and a tool can never modify
#     the harness. Rebuild only when the *runtime dependencies* change.
#   * No credentials, no build secrets, no network config: the sandbox runs with
#     `--network none` unless the policy gate grants egress.
#   * Runs as an unprivileged uid (the sandbox also passes --user 65534:65534).
#
#   docker build -t ultron-tools:phase1 -f docker/Dockerfile .
#
# Keep it small: every extra package is attack surface inside the sandbox.

FROM python:3.11-slim

# Deterministic, quiet, no bytecode writing (root fs is read-only at runtime).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp

WORKDIR /workspace

# Phase 1 tool runtimes. Pinned to minor versions so CI is reproducible.
RUN pip install --no-cache-dir "httpx>=0.27,<1"

# Drop privileges. uid/gid 65534 = nobody/nogroup in Debian.
USER 65534:65534

# Neutral default: a manifest must declare an explicit entrypoint.
CMD ["python", "-c", "import json,sys; sys.stdout.write(json.dumps({'ok': False, 'error': 'no entrypoint passed'}))"]
