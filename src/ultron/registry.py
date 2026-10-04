"""Tool registry: loads, validates and indexes JSON tool manifests from ``tools/``.

A manifest is the *contract* between the agent, the policy gate, the sandbox and
the verifier:

* the sandbox reads ``entrypoint`` + ``permissions`` to build the container,
* the policy gate reads ``risk`` + ``permissions`` to decide allow/ask/deny,
* the verifier reads ``inputs``/``outputs`` to check what came back,
* the planner reads ``description``/``tags``/``price_estimate_usd`` to choose.

Versioning: several versions of one tool may coexist (``name@1.2.0``). The
registry resolves "latest" by semver and every lookup can pin an exact version,
so a cached plan can never silently execute mutated code. ``content_hash`` pins
the manifest *bytes*, and approvals are bound to it (see ``policy``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Settings, get_settings
from .errors import ManifestError, ToolNotFound

#: Tool names and (input/output) field names. Field names may be one character
#: (`q`, `n`, ...) because single-letter keys are common in tool payloads.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}$")
FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,48}$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")

#: Words that carry no routing signal. Without this, a description containing
#: "the" would match every goal containing "the" (a real bug we hit: it made the
#: research planner pick the HTTP fetcher over the research tool).
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "over",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "our",
        "we",
        "us",
        "can",
        "may",
        "not",
        "no",
        "all",
        "any",
        "also",
        "use",
        "used",
        "using",
        "run",
        "runs",
        "such",
        "via",
        "per",
    ]
)


def _tokens(text: str) -> set[str]:
    """Content tokens of a string (>2 chars, no stopwords)."""
    return {
        t
        for t in re.split(r"[^a-z0-9_]+", (text or "").lower())
        if len(t) > 2 and t not in STOPWORDS
    }


#: Input/output types allowed in manifests -> python types used by the verifier.
TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "list[string]": list,
    "list[float]": list,
    "list[int]": list,
    "dict": dict,
    "any": object,
}

#: Permission scopes we understand. Anything else is a hard manifest error
#: (fail-closed: an unknown scope might mean something the gate can't reason about).
PERMISSION_SCOPES = {"network", "fs", "proc", "env", "secrets", "clock", "gpu"}
#: Secrets never cross into a sandbox in Phase 1, regardless of what is declared.
PHASE1_DENIED_SCOPES = {"secrets"}


class RiskTier(StrEnum):
    """Risk classification that drives the policy gate."""

    LOW = "low"  # auto-approved, still sandboxed
    MEDIUM = "medium"  # human approval required
    HIGH = "high"  # denied unless a pinned, explicit approval exists


@dataclass(frozen=True, slots=True)
class Permission:
    """A parsed ``scope:detail`` permission, e.g. ``network:http``."""

    raw: str
    scope: str
    detail: str = ""

    @classmethod
    def parse(cls, raw: str) -> Permission:
        if not isinstance(raw, str) or (":" not in raw and raw not in PERMISSION_SCOPES):
            raise ManifestError(f"malformed permission {raw!r} (expected 'scope:detail')")
        scope, _, detail = raw.partition(":")
        if scope not in PERMISSION_SCOPES:
            raise ManifestError(f"unknown permission scope {scope!r} in {raw!r}")
        return cls(raw=raw, scope=scope, detail=detail)

    @property
    def is_network(self) -> bool:
        return self.scope == "network"

    @property
    def is_write(self) -> bool:
        return self.scope == "fs" and self.detail.startswith(("write", "rw"))


class ToolManifest(BaseModel):
    """Validated tool manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    name: str
    version: str
    entrypoint: str
    risk: RiskTier
    description: str = ""
    permissions: list[str] = Field(default_factory=list)
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    #: True when the tool cannot benefit from an LLM (pure function). The planner
    #: strongly prefers these: deterministic-first is the cheapest path.
    deterministic: bool = True
    #: Declared cap on network calls; informational for the cost model.
    timeout_s: float | None = None
    cache_ttl_s: int | None = None
    #: Rough expected spend per call (USD). 0.0 for deterministic tools.
    price_estimate_usd: float = 0.0
    author: str = "ultron"
    source_path: str = ""

    # --------------------------------------------------------------- validation
    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ManifestError(f"invalid tool name {v!r}: expected ^[a-z][a-z0-9_]{{1,48}}$")
        return v

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if not VERSION_RE.match(v):
            raise ManifestError(f"invalid version {v!r}: expected semver like 0.1.0")
        return v

    @field_validator("entrypoint")
    @classmethod
    def _check_entrypoint(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ManifestError("entrypoint must not be empty")
        # The sandbox executes this WITHOUT a shell and rejects metacharacters, so
        # a manifest cannot smuggle `; rm -rf /`.
        if re.search(r"[;&|`$><\n\\]", stripped):
            raise ManifestError(f"entrypoint {v!r} contains shell metacharacters; use argv form")
        if not re.match(r"^[A-Za-z0-9_./\- ]+$", stripped):
            raise ManifestError(f"entrypoint {v!r} contains disallowed characters")
        return stripped

    @field_validator("inputs", "outputs")
    @classmethod
    def _check_types(cls, v: dict[str, str]) -> dict[str, str]:
        for key, typ in v.items():
            if not FIELD_NAME_RE.match(key):
                raise ManifestError(f"invalid field name {key!r}")
            if typ not in TYPE_MAP:
                raise ManifestError(f"unsupported type {typ!r} for field {key!r}")
        return v

    @field_validator("permissions")
    @classmethod
    def _check_permissions(cls, v: list[str]) -> list[str]:
        for raw in v:
            Permission.parse(raw)
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> ToolManifest:
        if self.cache_ttl_s is not None and self.cache_ttl_s < 0:
            raise ManifestError("cache_ttl_s must be >= 0")
        if self.price_estimate_usd < 0:
            raise ManifestError("price_estimate_usd must be >= 0")
        return self

    # ------------------------------------------------------------------ derived
    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def version_tuple(self) -> tuple[int, int, int]:
        core = self.version.split("-")[0].split("+")[0]
        major, minor, patch = core.split(".")
        return int(major), int(minor), int(patch)

    @property
    def parsed_permissions(self) -> list[Permission]:
        return [Permission.parse(p) for p in self.permissions]

    @property
    def wants_network(self) -> bool:
        return any(p.is_network for p in self.parsed_permissions)

    @property
    def wants_write(self) -> bool:
        return any(p.is_write for p in self.parsed_permissions)

    @property
    def network_detail(self) -> str:
        for perm in self.parsed_permissions:
            if perm.is_network:
                return perm.detail or "any"
        return ""

    @property
    def required_inputs(self) -> list[str]:
        return list(self.inputs)

    @property
    def content_hash(self) -> str:
        """Stable hash of everything semantically meaningful in the manifest.

        Used to pin approvals and to key cached plans/routes, so it must cover the
        fields the planner searches on (``description``, ``tags``) as well as the
        ones the sandbox and verifier use — otherwise editing a description would
        silently keep serving decisions made under different metadata.
        """
        from .cache import make_key

        return make_key(
            self.name,
            self.version,
            self.entrypoint,
            self.risk.value,
            sorted(self.permissions),
            self.inputs,
            self.outputs,
            self.description,
            sorted(self.tags),
            self.deterministic,
            self.timeout_s,
            self.cache_ttl_s,
            self.price_estimate_usd,
        )[:32]

    def describe(self) -> str:
        return f"{self.key} risk={self.risk.value} perms={self.permissions or ['none']}"


@dataclass(slots=True)
class RegistryIndex:
    """Cheap, LLM-free relevance index used by the planner."""

    terms: dict[str, list[str]] = field(default_factory=dict)
    by_tag: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, manifests: list[ToolManifest]) -> RegistryIndex:
        terms: dict[str, list[str]] = {}
        by_tag: dict[str, list[str]] = {}
        for m in manifests:
            blob = f"{m.name} {m.description} {' '.join(m.tags)}"
            tokens = _tokens(blob)
            # `web_research` must also be findable as "research" / "web", so split
            # compound identifiers on underscores as well as on punctuation.
            for token in list(tokens):
                tokens |= {
                    part for part in token.split("_") if len(part) > 2 and part not in STOPWORDS
                }
            for token in tokens:
                terms.setdefault(token, []).append(m.key)
            for tag in m.tags:
                by_tag.setdefault(tag.lower(), []).append(m.key)
        return cls(terms=terms, by_tag=by_tag)


class Registry:
    """Loads manifests from a directory tree, resolving versions."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.tools_dir = Path(self.settings.tools_dir)
        #: name -> {version -> manifest}
        self._by_name: dict[str, dict[str, ToolManifest]] = {}
        self._index: RegistryIndex | None = None
        self.load_errors: list[str] = []

    # ------------------------------------------------------------------ loading
    def load(self, *, strict: bool = False) -> Registry:
        self._by_name.clear()
        self.load_errors.clear()
        for path in sorted(self.tools_dir.rglob("*.json")):
            if path.name in {"schema.json", "package.json"} or path.name.startswith("_"):
                continue
            try:
                manifest = self.load_manifest(path)
            except (ManifestError, json.JSONDecodeError, ValueError) as exc:
                message = f"{path.relative_to(self.tools_dir)}: {exc}"
                if strict:
                    raise ManifestError(message) from exc
                self.load_errors.append(message)
                continue
            bucket = self._by_name.setdefault(manifest.name, {})
            existing = bucket.get(manifest.version)
            if existing is not None:
                self.load_errors.append(
                    f"duplicate manifest {manifest.key} in {path.name} and {existing.source_path}"
                )
                continue
            bucket[manifest.version] = manifest
        self._index = None
        return self

    @staticmethod
    def load_manifest(path: Path) -> ToolManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ManifestError(f"{path.name}: manifest must be a JSON object")
        data = dict(data)
        data["source_path"] = str(path)
        return ToolManifest(**data)

    # ------------------------------------------------------------------ queries
    def get(self, name: str, version: str | None = None) -> ToolManifest:
        if "@" in name and version is None:
            name, _, version = name.partition("@")
        versions = self._by_name.get(name)
        if not versions:
            raise ToolNotFound(
                f"no manifest for tool {name!r} (have: {', '.join(sorted(self._by_name)) or 'none'})"
            )
        if version is None:
            return versions[max(versions, key=lambda v: versions[v].version_tuple)]
        if version not in versions:
            raise ToolNotFound(f"tool {name!r} has no version {version!r}")
        return versions[version]

    def versions_of(self, name: str) -> list[ToolManifest]:
        versions = self._by_name.get(name, {})
        return [
            versions[v]
            for v in sorted(versions, key=lambda v: versions[v].version_tuple, reverse=True)
        ]

    def latest(self) -> list[ToolManifest]:
        return [self.get(name) for name in sorted(self._by_name)]

    def all(self) -> list[ToolManifest]:
        return [
            m
            for name in sorted(self._by_name)
            for m in sorted(self._by_name[name].values(), key=lambda m: m.version)
        ]

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name.split("@")[0] in self._by_name

    def __iter__(self) -> Iterator[ToolManifest]:
        return iter(self.latest())

    def __len__(self) -> int:
        return len(self._by_name)

    @property
    def fingerprint(self) -> str:
        """Hash of the whole registry (used to key cached plans)."""
        from .cache import make_key

        return make_key(sorted(m.content_hash for m in self.all()))[:16]

    @property
    def index(self) -> RegistryIndex:
        if self._index is None:
            self._index = RegistryIndex.build(self.latest())
        return self._index

    def snapshot(self) -> dict[str, Any]:
        """Compact, LLM-friendly description of available tools."""
        return {
            "fingerprint": self.fingerprint,
            "tools": [
                {
                    "name": m.name,
                    "version": m.version,
                    "description": m.description,
                    "risk": m.risk.value,
                    "tags": m.tags,
                    "inputs": m.inputs,
                    "outputs": m.outputs,
                    "network": m.wants_network,
                    "deterministic": m.deterministic,
                }
                for m in self.latest()
            ],
        }

    def search(
        self, text: str, *, limit: int = 5, deterministic_only: bool = False
    ) -> list[ToolManifest]:
        """Deterministic keyword relevance search. Zero LLM cost.

        Used as the *first* planner step so the common case never needs a model.
        """
        tokens = _tokens(text)
        scores: dict[str, float] = {}
        # Iterate over the (few) query tokens, not over the whole index: scoring
        # must depend on what the goal actually says. Weight by inverse document
        # frequency so a keyword shared by many tools counts for less than a
        # distinctive one.
        for token in tokens:
            for key in self.index.terms.get(token, ()):
                scores[key] = scores.get(key, 0.0) + 1.0 / len(self.index.terms[token])
            for key in self.index.by_tag.get(token, ()):
                scores[key] = scores.get(key, 0.0) + 1.5 / len(self.index.by_tag[token])
        # Stem-ish partial credit for morphological variants: "calculate" ~
        # "calculator", "summarize" ~ "summary". Weak weight, never a match on
        # its own ahead of a real keyword hit.
        unmatched = [t for t in tokens if t not in self.index.terms and t not in self.index.by_tag]
        for token in unmatched:
            if len(token) < 4:
                continue
            for term, keys in self.index.terms.items():
                if len(term) < 4 or term == token:
                    continue
                if token in term or term in token:
                    for key in keys:
                        scores[key] = scores.get(key, 0.0) + 0.35
        ranked: list[tuple[float, ToolManifest]] = []
        for key, score in scores.items():
            manifest = self.get(key)
            if manifest.risk is RiskTier.HIGH:  # never auto-shortlist high risk
                score *= 0.15
            if manifest.deterministic:
                score += 0.25  # deterministic-first bias
            if deterministic_only and not manifest.deterministic:
                continue
            score -= manifest.price_estimate_usd * 10  # cheaper tools win ties
            ranked.append((score, manifest))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [m for _, m in ranked[:limit]]

    # --------------------------------------------------------------- json schema
    def validate_against_schema(self, manifest: ToolManifest) -> list[str]:
        """Validate a manifest against ``tools/schema.json`` when jsonschema exists."""
        schema_path = Path(self.settings.tools_schema)
        if not schema_path.exists():
            return []
        try:
            import jsonschema  # type: ignore[import-not-found]
        except ImportError:
            return []  # optional dependency; pydantic validation already ran
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        payload = manifest.model_dump(mode="json")
        # `source_path` is injected by the loader, never written by tool authors,
        # so it is dropped before validating against `additionalProperties: false`.
        payload.pop("source_path", None)
        validator = jsonschema.Draft7Validator(schema)
        return [
            f"{'/'.join(map(str, e.path))}: {e.message}" for e in validator.iter_errors(payload)
        ]
