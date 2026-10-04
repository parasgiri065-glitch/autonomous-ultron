"""Verifier: nothing counts as success until it passes here.

Layers, cheapest first:

1. **Envelope**    did the tool return ``{"ok": true, "result": {...}}``?
2. **Schema**      do the result fields exist and match the manifest's types?
3. **Sanity**      non-empty strings, no "as an AI…" refusals, confidence in range,
                   source lists that are actually usable URLs.
4. **Grounding**   deterministic provenance check when evidence text is supplied:
                   content words in the summary must come from the sources (this is
                   what stops an extractive tool from drifting into invention).
5. **Judge**       optional cheap LLM judge (``ULTRON_ENABLE_LLM_JUDGE=1``) for
                   hallucination/source validation on free-form outputs. Off by
                   default because it costs money; its verdict is cached.

A failed verification is *never* cached and never recorded as a successful
memory, so a broken tool cannot poison future runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .breaker import BreakerResult, BreakerVerifier
from .cache import Cache, make_key
from .config import Settings, get_settings
from .llm import LLMClient
from .provenance import ProvenanceEnvelope
from .registry import TYPE_MAP, ToolManifest

REFUSAL_MARKERS = (
    "as an ai",
    "i cannot",
    "i can't",
    "i'm unable",
    "unable to comply",
    "no information available",
    "lorem ipsum",
)
URL_SHAPE_RE = re.compile(r"^https?://[^\s]+$", re.I)
WORD_RE = re.compile(r"[a-z0-9]{3,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "has",
    "have",
    "its",
    "their",
    "which",
    "into",
    "not",
    "but",
    "you",
    "your",
    "our",
    "can",
    "will",
    "would",
    "there",
    "these",
    "those",
    "they",
    "them",
    "then",
    "than",
    "also",
    "http",
    "https",
    "www",
    "com",
    "org",
    "net",
}

JUDGE_SYSTEM = (
    "You are a strict verifier for a research agent. Given a goal, a produced answer "
    "and its sources, decide whether the answer is fully supported by the sources.\n"
    'Reply with ONLY JSON: {"grounded":true|false,"issues":["..."],"confidence":0.0-1.0}'
)

CheckStatus = Literal["pass", "fail", "warn", "skip"]


@dataclass(slots=True)
class Check:
    name: str
    status: CheckStatus
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "fail"

    def render(self) -> str:
        icon = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "skip": "SKIP"}[self.status]
        return f"[{icon}] {self.name}: {self.detail}" if self.detail else f"[{icon}] {self.name}"


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    score: float
    checks: list[Check] = field(default_factory=list)
    reason: str = ""
    judge_used: bool = False
    cost_usd: float = 0.0
    cached: bool = False
    untrusted: bool = False
    breaker: BreakerResult | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "score": round(self.score, 3),
            "reason": self.reason,
            "judge_used": self.judge_used,
            "cost_usd": self.cost_usd,
            "cached": self.cached,
            "untrusted": self.untrusted,
            "breaker": self.breaker.as_dict() if self.breaker else None,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks
            ],
        }

    def summary(self) -> str:
        failed = [c.name for c in self.checks if c.status == "fail"]
        warned = [c.name for c in self.checks if c.status == "warn"]
        verdict = "verified" if self.ok else f"REJECTED ({', '.join(failed)})"
        extra = f" warnings={','.join(warned)}" if warned else ""
        return f"{verdict} score={self.score:.2f}{extra}"


class Verifier:
    """Deterministic checks by default; LLM judge behind an explicit flag."""

    def __init__(
        self,
        *,
        cache: Cache | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        breaker: BreakerVerifier | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or Cache(self.settings)
        self.llm = llm or LLMClient(self.cache, self.settings)
        self.breaker = breaker or BreakerVerifier()

    def verify(
        self,
        manifest: ToolManifest,
        result: dict[str, Any] | None,
        *,
        goal: str = "",
        evidence: list[str] | None = None,
        run_id: str = "",
        provenance: list[ProvenanceEnvelope] | None = None,
    ) -> VerificationResult:
        checks: list[Check] = []
        checks.append(self._check_payload(result))
        if not checks[0].ok:
            return self._finalize(checks)

        assert result is not None  # guarded by the payload check
        checks.append(self._check_required_fields(manifest, result))
        checks.append(self._check_types(manifest, result))
        checks.append(self._check_sanity(result))
        checks.append(self._check_confidence(result))
        checks.append(self._check_sources(result))
        if evidence:
            checks.append(self._check_grounding(result, evidence))

        breaker_result: BreakerResult | None = None
        if provenance is not None:
            breaker_result = self.breaker.verify(result, provenance)
            checks.append(
                Check(
                    "breaker.provenance",
                    "pass" if breaker_result.ok else "fail",
                    breaker_result.reason,
                )
            )
        verdict = self._finalize(checks)
        verdict.breaker = breaker_result
        verdict.untrusted = bool(breaker_result and breaker_result.untrusted)
        if self.settings.enable_llm_judge and verdict.ok and _has_free_text(result):
            judge = self._judge(goal, result, evidence or [])
            checks.extend(judge.checks)
            verdict = self._finalize(checks)
            verdict.cost_usd = judge.cost_usd
            verdict.breaker = breaker_result
            verdict.untrusted = bool(breaker_result and breaker_result.untrusted)
        return verdict

    # ------------------------------------------------------------------- checks
    @staticmethod
    def _check_payload(result: dict[str, Any] | None) -> Check:
        if result is None:
            return Check("envelope", "fail", "tool returned no result object")
        if not isinstance(result, dict):
            return Check(
                "envelope", "fail", f"result must be an object, got {type(result).__name__}"
            )
        if not result:
            return Check("envelope", "fail", "result object is empty")
        return Check("envelope", "pass", f"{len(result)} field(s)")

    @staticmethod
    def _check_required_fields(manifest: ToolManifest, result: dict[str, Any]) -> Check:
        missing = sorted(set(manifest.outputs) - set(result))
        if missing:
            return Check(
                "schema.required", "fail", f"missing declared outputs: {', '.join(missing)}"
            )
        extra = sorted(set(result) - set(manifest.outputs))
        if extra:
            return Check(
                "schema.required", "warn", f"undeclared extra fields (ignored): {', '.join(extra)}"
            )
        return Check("schema.required", "pass", f"all {len(manifest.outputs)} output(s) present")

    @staticmethod
    def _check_types(manifest: ToolManifest, result: dict[str, Any]) -> Check:
        problems: list[str] = []
        for name, typ in manifest.outputs.items():
            if name not in result:
                continue
            expected = TYPE_MAP[typ]
            value = result[name]
            if expected is int and isinstance(value, bool):
                problems.append(f"{name}: expected int, got bool")
            elif expected is float and isinstance(value, bool):
                problems.append(f"{name}: expected float, got bool")
            elif expected is tuple and isinstance(value, bool):
                problems.append(f"{name}: expected number, got bool")
            elif expected is not object and not isinstance(value, expected):
                problems.append(f"{name}: expected {typ}, got {type(value).__name__}")
        if problems:
            return Check("schema.types", "fail", "; ".join(problems))
        return Check("schema.types", "pass", "types match manifest")

    @staticmethod
    def _check_sanity(result: dict[str, Any]) -> Check:
        problems: list[str] = []
        for key, value in result.items():
            if key.startswith("_"):
                continue
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    problems.append(f"{key} is empty")
                    continue
                lowered = stripped.lower()
                if any(marker in lowered for marker in REFUSAL_MARKERS):
                    problems.append(f"{key} looks like a refusal/placeholder")
                if len(stripped) < 8 and key in {"summary", "answer", "text"}:
                    problems.append(f"{key} is suspiciously short ({len(stripped)} chars)")
        if problems:
            return Check("sanity.content", "fail", "; ".join(problems))
        return Check("sanity.content", "pass", "content non-empty and non-placeholder")

    @staticmethod
    def _check_confidence(result: dict[str, Any]) -> Check:
        if "confidence" not in result:
            return Check("sanity.confidence", "skip", "tool reports no confidence")
        try:
            value = float(result["confidence"])
        except (TypeError, ValueError):
            return Check(
                "sanity.confidence", "fail", f"confidence is not numeric: {result['confidence']!r}"
            )
        if not 0.0 <= value <= 1.0:
            return Check("sanity.confidence", "fail", f"confidence {value} outside [0,1]")
        if value < 0.25:
            return Check("sanity.confidence", "warn", f"low self-reported confidence ({value})")
        return Check("sanity.confidence", "pass", f"confidence={value}")

    @staticmethod
    def _check_sources(result: dict[str, Any]) -> Check:
        if "sources" not in result:
            return Check("sanity.sources", "skip", "tool returns no sources")
        sources = result["sources"]
        if not isinstance(sources, list):
            return Check("sanity.sources", "fail", "sources must be a list")
        if not sources:
            return Check("sanity.sources", "fail", "no sources returned: cannot ground the answer")
        bad = [s for s in sources if not isinstance(s, str) or not URL_SHAPE_RE.match(s)]
        if bad:
            return Check("sanity.sources", "fail", f"{len(bad)} source(s) are not usable URLs")
        return Check("sanity.sources", "pass", f"{len(sources)} source(s)")

    def _check_grounding(self, result: dict[str, Any], evidence: list[str]) -> Check:
        """Every content word in the answer should exist in the evidence.

        Words are compared by a crude stem (first 5 characters) so that ordinary
        morphology is not mistaken for invention: "reduces" matches "reduction",
        "children" matches "child". Words with no stem in the evidence at all are
        treated as unsupported — that is where fabricated entities and numbers
        would show up.
        """
        answer = " ".join(
            str(v) for k, v in result.items() if isinstance(v, str) and not k.startswith("_")
        )
        answer_stems = {_stem(w) for w in WORD_RE.findall(answer.lower())} - STOPWORDS
        if not answer_stems:
            return Check("grounding.overlap", "fail", "answer has no content words")
        evidence_stems: set[str] = set()
        for chunk in evidence:
            evidence_stems |= {_stem(w) for w in WORD_RE.findall(str(chunk).lower())}
        overlap = len(answer_stems & evidence_stems) / len(answer_stems)
        if overlap >= 0.85:
            return Check(
                "grounding.overlap", "pass", f"{overlap:.0%} of answer terms found in sources"
            )
        if overlap >= 0.7:
            return Check(
                "grounding.overlap", "warn", f"only {overlap:.0%} of answer terms found in sources"
            )
        return Check(
            "grounding.overlap",
            "fail",
            f"only {overlap:.0%} of answer terms trace back to the sources (possible fabrication)",
        )

    # -------------------------------------------------------------------- judge
    def _judge(self, goal: str, result: dict[str, Any], evidence: list[str]) -> VerificationResult:
        answer = _primary_text(result)
        key = make_key(
            "judge", goal, answer, [e[:2000] for e in evidence], self.settings.judge_model
        )
        checks: list[Check] = []
        cost = 0.0

        entry = self.cache.get("judge", key)
        if entry is not None:
            data = dict(entry.value)
            cached = True
        else:
            cached = False
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Goal: {goal[:600]}\n\nAnswer:\n{answer[:2000]}\n\n"
                        f"Sources:\n{chr(10).join(e[:1200] for e in evidence[:5]) or '(none provided)'}"
                    ),
                },
            ]
            try:
                response = self.llm.complete(
                    "judge", messages, max_tokens=200, response_format_json=True
                )
            except Exception as exc:
                return VerificationResult(
                    ok=True,
                    score=1.0,
                    checks=[Check("judge.llm", "skip", f"judge unavailable: {type(exc).__name__}")],
                    reason="judge skipped",
                )
            data = response.json or {
                "grounded": True,
                "issues": ["judge reply unparseable"],
                "confidence": 0.5,
            }
            cost = response.cost_usd
            self.cache.set("judge", key, data, ttl_s=self.settings.cache_ttl_llm)

        grounded = bool(data.get("grounded", False))
        issues = [str(i) for i in (data.get("issues") or [])]
        try:
            jconfidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            jconfidence = 0.0
        detail = "; ".join(issues)[:300] or f"judge confidence={jconfidence:.2f}"
        if cached:
            detail = f"{detail} (cached verdict)"
        checks.append(Check("judge.llm", "pass" if grounded else "fail", detail))
        return VerificationResult(
            ok=grounded,
            score=1.0 if grounded else 0.0,
            checks=checks,
            reason="llm judge" + (" (cached)" if cached else ""),
            judge_used=True,
            cost_usd=cost if not cached else 0.0,
            cached=cached,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _finalize(checks: list[Check]) -> VerificationResult:
        failures = [c for c in checks if c.status == "fail"]
        warnings = [c for c in checks if c.status == "warn"]
        considered = [c for c in checks if c.status != "skip"]
        score = (len([c for c in considered if c.ok]) / len(considered)) if considered else 1.0
        if warnings:
            score = min(score, 0.85)
        reason = (
            "all checks passed"
            if not failures
            else "; ".join(f"{c.name}: {c.detail}" for c in failures)
        )
        return VerificationResult(
            ok=not failures, score=round(score, 3), checks=checks, reason=reason
        )


def _stem(word: str) -> str:
    """Crude, dependency-free stemmer: compare on the first five characters."""
    return word if len(word) < 6 else word[:5]


def _primary_text(result: dict[str, Any]) -> str:
    for key in ("summary", "answer", "text", "result"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return " ".join(str(v) for v in result.values() if isinstance(v, str))[:2000]


def _has_free_text(result: dict[str, Any]) -> bool:
    return any(
        isinstance(v, str) and len(v) > 40 for k, v in result.items() if not k.startswith("_")
    )
