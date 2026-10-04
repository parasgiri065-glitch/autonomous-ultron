"""Typed, env-driven configuration for the Ultron harness.

Design rules encoded here:
  * Every knob is overridable by env var (ULTRON_*) so CI can pin budgets.
  * Defaults are *safe*: Docker backend, no network, judge off, budget capped.
  * Nothing in here is ever handed to a tool sandbox (see ``sandbox.scrub_env``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}

#: Env var names that must never cross into a sandbox, matched case-insensitively
#: as substrings. The policy gate and the sandbox both refuse these.
SECRET_ENV_MARKERS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "SESSION",
    "COOKIE",
    "AUTH",
    "PRIVATE",
    "AWS_",
    "AZURE_",
    "GCP_",
    "GITHUB_",
    "OPENAI",
    "ANTHROPIC",
    "GEMINI",
    "DATABASE_URL",
)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    low = raw.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f"{name} must be a boolean-ish value, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _env_list(name: str) -> list[str]:
    raw = _env(name, "") or ""
    return [p.strip() for p in raw.split(",") if p.strip()]


def _env_int_list(name: str) -> list[int]:
    values: list[int] = []
    for item in _env_list(name):
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{name} must contain comma-separated integers") from exc
    return values


def _resolve(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


class Settings(BaseModel):
    """Resolved runtime settings (single source of truth for the harness)."""

    repo_root: Path = REPO_ROOT
    state_dir: Path = Field(default=REPO_ROOT / ".ultron")
    tools_dir: Path = Field(default=REPO_ROOT / "tools")
    tools_schema: Path = Field(default=REPO_ROOT / "tools" / "schema.json")

    # --- cost architecture -------------------------------------------------
    router_model: str = "gpt-4o-mini"
    planner_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    synth_model: str = "gpt-4o-mini"
    # auto = live if a provider key exists else offline; live = require a key;
    # offline = never call a model; stub = run every call site with priced,
    # deterministic canned answers (CI cost gate without keys or network).
    llm_mode: Literal["auto", "live", "offline", "stub"] = "auto"
    llm_api_base: str | None = None
    llm_timeout_s: float = 30.0
    cost_simulate: bool = False
    # Optional, explicitly enabled network tiers. Defaults remain hermetic.
    allow_zero_auth: bool = False
    scavenge_enabled: bool = False

    # --- Telegram cockpit --------------------------------------------------
    telegram_bot_token: str | None = None
    telegram_allowed_user_ids: list[int] = Field(default_factory=list)

    # --- budget kill-switch ------------------------------------------------
    budget_max_usd: float = 0.05
    budget_max_steps: int = 6
    budget_max_seconds: float = 120.0
    budget_max_llm_calls: int = 12

    # --- cache -------------------------------------------------------------
    cache_path: Path = Field(default=REPO_ROOT / ".ultron" / "cache.db")
    cache_ttl_llm: int = 86_400
    cache_ttl_tool: int = 3_600
    cache_ttl_web: int = 900
    cache_max_entries: int = 200_000

    # --- sandbox -----------------------------------------------------------
    sandbox_backend: Literal["docker", "local"] = "docker"
    sandbox_image: str = "ultron-tools:phase1"
    sandbox_build_image: bool = False
    sandbox_timeout_s: float = 60.0
    sandbox_memory: str = "512m"
    sandbox_cpus: str = "1"
    sandbox_pids: int = 128
    sandbox_tmpfs_mb: int = 64
    allow_local_sandbox: bool = False
    env_allowlist: list[str] = Field(default_factory=list)
    docker_bin: str = "docker"
    #: Offline fixture corpus for network tools (``ULTRON_WEB_MOCK``). Scoped to
    #: the Settings instance instead of the process environment: two sandboxes
    #: with different values must not contaminate each other, and a global
    #: ``os.environ`` write leaks into every other tool, test and subprocess in
    #: the process. ``None`` means "no fixtures" (tools may then go live).
    web_mock: str | None = None

    # --- live egress (opt-in, never used by CI) ----------------------------
    #: When True, the network tools may reach the real internet, with a
    #: URL-keyed SQLite cache and the TTLs below. Off by default; the eval
    #: harness never sets it, so CI stays hermetic.
    eval_live: bool = False
    #: URL-keyed page cache used by the tools in live mode. Separate file from
    #: the harness cache so a ``ultron cache clear`` can never drop it by
    #: accident, and so the docker backend can mount exactly this directory.
    web_cache_path: Path = Field(default=REPO_ROOT / ".ultron" / "webcache.db")

    # --- policy ------------------------------------------------------------
    policy_network: Literal["auto", "deny"] = "auto"
    #: When False (default), a LOW-risk tool that requests network access is
    #: escalated to MEDIUM and a human is asked. Phase 3 auto-discovers
    #: manifests, so "LOW + egress runs unattended" is not a safe default.
    policy_network_low_auto: bool = False
    policy_assume_yes: bool = False
    approvals_file: Path = Field(default=REPO_ROOT / ".ultron" / "approvals.json")
    audit_log: Path = Field(default=REPO_ROOT / ".ultron" / "audit.jsonl")

    # --- verification ------------------------------------------------------
    enable_llm_judge: bool = False
    enable_synthesis: bool = False

    # --- memory / eval -----------------------------------------------------
    memory_path: Path = Field(default=REPO_ROOT / ".ultron" / "memory.db")
    eval_tasks: Path = Field(default=REPO_ROOT / "eval" / "tasks.jsonl")
    eval_baseline: Path = Field(default=REPO_ROOT / "eval" / "baseline.json")
    eval_min_success_rate: float = 0.80
    eval_max_avg_cost_usd: float = 0.02
    eval_max_avg_latency_s: float = 20.0

    @property
    def cache_hits_cost_nothing(self) -> bool:
        return True

    @property
    def is_docker_backend(self) -> bool:
        return self.sandbox_backend == "docker"

    def ensure_dirs(self) -> None:
        for path in (
            self.state_dir,
            self.cache_path.parent,
            self.memory_path.parent,
            self.audit_log.parent,
            self.approvals_file.parent,
            self.web_cache_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_settings(**overrides: object) -> Settings:
    """Build :class:`Settings` from the environment, then apply overrides."""
    state_dir = _resolve(_env("ULTRON_STATE_DIR"), REPO_ROOT / ".ultron")
    data = {
        "repo_root": REPO_ROOT,
        "state_dir": state_dir,
        "tools_dir": _resolve(_env("ULTRON_TOOLS_DIR"), REPO_ROOT / "tools"),
        "tools_schema": _resolve(_env("ULTRON_TOOLS_SCHEMA"), REPO_ROOT / "tools" / "schema.json"),
        "router_model": _env("ULTRON_ROUTER_MODEL", "gpt-4o-mini"),
        "planner_model": _env("ULTRON_PLANNER_MODEL", "gpt-4o-mini"),
        "judge_model": _env("ULTRON_JUDGE_MODEL", "gpt-4o-mini"),
        "synth_model": _env("ULTRON_SYNTH_MODEL", "gpt-4o-mini"),
        "llm_mode": _env("ULTRON_LLM_MODE", "auto"),
        "llm_api_base": _env("ULTRON_LLM_API_BASE"),
        "llm_timeout_s": _env_float("ULTRON_LLM_TIMEOUT_S", 30.0),
        "cost_simulate": _env_bool("ULTRON_COST_SIMULATE", False),
        "allow_zero_auth": _env_bool("ULTRON_ALLOW_ZERO_AUTH", False),
        "scavenge_enabled": _env_bool("ULTRON_SCAVENGE", False),
        "telegram_bot_token": _env("TELEGRAM_BOT_TOKEN"),
        "telegram_allowed_user_ids": _env_int_list("TELEGRAM_ALLOWED_USERS"),
        "budget_max_usd": _env_float("ULTRON_BUDGET_MAX_USD", 0.05),
        "budget_max_steps": _env_int("ULTRON_BUDGET_MAX_STEPS", 6),
        "budget_max_seconds": _env_float("ULTRON_BUDGET_MAX_SECONDS", 120.0),
        "budget_max_llm_calls": _env_int("ULTRON_BUDGET_MAX_LLM_CALLS", 12),
        "cache_path": _resolve(_env("ULTRON_CACHE_PATH"), state_dir / "cache.db"),
        "cache_ttl_llm": _env_int("ULTRON_CACHE_TTL_LLM", 86_400),
        "cache_ttl_tool": _env_int("ULTRON_CACHE_TTL_TOOL", 3_600),
        "cache_ttl_web": _env_int("ULTRON_CACHE_TTL_WEB", 900),
        "cache_max_entries": _env_int("ULTRON_CACHE_MAX_ENTRIES", 200_000),
        "sandbox_backend": _env("ULTRON_SANDBOX", "docker"),
        "sandbox_image": _env("ULTRON_SANDBOX_IMAGE", "ultron-tools:phase1"),
        "sandbox_build_image": _env_bool("ULTRON_SANDBOX_BUILD_IMAGE", False),
        "sandbox_timeout_s": _env_float("ULTRON_SANDBOX_TIMEOUT_S", 60.0),
        "sandbox_memory": _env("ULTRON_SANDBOX_MEMORY", "512m"),
        "sandbox_cpus": _env("ULTRON_SANDBOX_CPUS", "1"),
        "sandbox_pids": _env_int("ULTRON_SANDBOX_PIDS", 128),
        "sandbox_tmpfs_mb": _env_int("ULTRON_SANDBOX_TMPFS_MB", 64),
        "allow_local_sandbox": _env_bool("ULTRON_ALLOW_LOCAL_SANDBOX", False),
        "env_allowlist": _env_list("ULTRON_ENV_ALLOWLIST"),
        "docker_bin": _env("ULTRON_DOCKER_BIN", "docker"),
        # Backward compatible: the env var was the only way to configure this
        # before, so it keeps working as the *default*; an explicit
        # ``web_mock=`` override (what the eval does) wins.
        "web_mock": _env("ULTRON_WEB_MOCK"),
        "eval_live": _env_bool("ULTRON_EVAL_LIVE", False),
        "web_cache_path": _resolve(_env("ULTRON_WEB_CACHE"), state_dir / "webcache.db"),
        "policy_network": _env("ULTRON_POLICY_NETWORK", "auto"),
        "policy_network_low_auto": _env_bool("ULTRON_POLICY_NETWORK_LOW_AUTO", False),
        "policy_assume_yes": _env_bool("ULTRON_POLICY_ASSUME_YES", False),
        "approvals_file": _resolve(_env("ULTRON_APPROVALS_FILE"), state_dir / "approvals.json"),
        "audit_log": _resolve(_env("ULTRON_AUDIT_LOG"), state_dir / "audit.jsonl"),
        "enable_llm_judge": _env_bool("ULTRON_ENABLE_LLM_JUDGE", False),
        "enable_synthesis": _env_bool("ULTRON_ENABLE_SYNTHESIS", False),
        "memory_path": _resolve(_env("ULTRON_MEMORY_PATH"), state_dir / "memory.db"),
        "eval_tasks": _resolve(_env("ULTRON_EVAL_TASKS"), REPO_ROOT / "eval" / "tasks.jsonl"),
        "eval_baseline": _resolve(
            _env("ULTRON_EVAL_BASELINE"), REPO_ROOT / "eval" / "baseline.json"
        ),
        "eval_min_success_rate": _env_float("ULTRON_EVAL_MIN_SUCCESS_RATE", 0.80),
        "eval_max_avg_cost_usd": _env_float("ULTRON_EVAL_MAX_AVG_COST_USD", 0.02),
        "eval_max_avg_latency_s": _env_float("ULTRON_EVAL_MAX_AVG_LATENCY_S", 20.0),
    }
    data.update({k: v for k, v in overrides.items() if v is not None})
    settings = Settings(**data)
    settings.ensure_dirs()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Tests call ``reset_settings()``."""
    return load_settings()


def reset_settings() -> Settings:
    """Drop the cache and re-read the environment (used by tests/CLI)."""
    get_settings.cache_clear()
    return get_settings()
