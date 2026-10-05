"""Conservative final-result checks that sit below the normal verifier.

The Breaker is intentionally lax about prose and strict about claims that are
easy to prove wrong: numbers, named entities, and success claims. It never
creates evidence. A value is trusted only when a raw sandbox or web-fetch
provenance envelope contains it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .provenance import ProvenanceEnvelope

_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z])")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b")
_QUOTED_RE = re.compile(r"[\"']([^\"']{3,})[\"']")
_SUCCESS_RE = re.compile(
    r"\b(?:success(?:ful(?:ly)?)?|completed|retrieved|fetched|found|confirmed)\b",
    re.IGNORECASE,
)
_COMMON_CAPITALIZED = {"The", "This", "That", "There", "Result", "Sources", "Confidence"}


@dataclass(slots=True)
class BreakerResult:
    ok: bool
    untrusted: bool = False
    reasons: list[str] = field(default_factory=list)
    grounded_numbers: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    grounded_entities: list[str] = field(default_factory=list)
    unsupported_entities: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "all breaker checks passed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "untrusted": self.untrusted,
            "reason": self.reason,
            "reasons": self.reasons,
            "grounded_numbers": self.grounded_numbers,
            "unsupported_numbers": self.unsupported_numbers,
            "grounded_entities": self.grounded_entities,
            "unsupported_entities": self.unsupported_entities,
        }


class BreakerVerifier:
    """Check final claims against raw, non-LLM provenance evidence."""

    def verify(
        self,
        final_payload: Any,
        provenance: Iterable[ProvenanceEnvelope] | None = None,
        *,
        raw_tool_envelopes: Iterable[ProvenanceEnvelope] | None = None,
    ) -> BreakerResult:
        items = list(raw_tool_envelopes if raw_tool_envelopes is not None else (provenance or ()))
        support = [
            item for item in items if item.origin in {"sandbox_tool", "web_fetch", "local_file"}
        ]
        raw_text = "\n".join(_evidence_text(item) for item in support)
        final_text = _payload_text(final_payload)
        result = BreakerResult(ok=True)

        # Check 1: exact numeric tokens are cheap, high-signal grounding.
        final_numbers = sorted({_normal_number(value) for value in _NUMBER_RE.findall(final_text)})
        raw_numbers = {_normal_number(value) for value in _NUMBER_RE.findall(raw_text)}
        result.grounded_numbers = [value for value in final_numbers if value in raw_numbers]
        result.unsupported_numbers = [value for value in final_numbers if value not in raw_numbers]
        if result.unsupported_numbers:
            result.ok = False
            result.reasons.append(
                "groundedness: unsupported numeric value(s) "
                + ", ".join(result.unsupported_numbers)
            )

        # Named entities and quoted identifiers are deliberately a smaller set
        # than every content word: this is the "lax" part of the filter.
        final_entities = _entities(final_text)
        raw_lower = raw_text.casefold()
        result.grounded_entities = sorted(
            entity for entity in final_entities if entity.casefold() in raw_lower
        )
        result.unsupported_entities = sorted(
            entity for entity in final_entities if entity.casefold() not in raw_lower
        )
        if result.unsupported_entities:
            result.ok = False
            result.reasons.append(
                "groundedness: unsupported key entity/entities "
                + ", ".join(result.unsupported_entities)
            )

        # Check 2: an error/empty raw envelope cannot support a synthesized
        # success, regardless of how polished the prose is.
        failed_sources = [item for item in support if _is_failed_or_empty(item)]
        if failed_sources and _claims_success(final_payload, final_text):
            result.ok = False
            result.reasons.append(
                "contradiction: final output claims success despite an error/empty raw tool result"
            )

        # Check 3: an LLM may draft prose, but it cannot introduce data without
        # at least one raw tool/web source to anchor it. Mark it untrusted so the
        # caller can prevent both cache and long-term-memory writes.
        has_llm = any(item.origin == "llm_generated" for item in items)
        if has_llm and not support:
            result.ok = False
            result.untrusted = True
            result.reasons.append(
                "hallucination: LLM-generated data has no supporting tool provenance"
            )
        elif not support and final_text.strip():
            result.ok = False
            result.untrusted = True
            result.reasons.append("hallucination: final data has no supporting tool provenance")

        return result

    __call__ = verify


def _payload_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        values: list[str] = []
        for key, value in payload.items():
            if str(key).startswith("_"):
                continue
            values.append(_payload_text(value))
        return " ".join(values)
    if isinstance(payload, (list, tuple, set)):
        return " ".join(_payload_text(value) for value in payload)
    if payload is None or isinstance(payload, bool):
        return ""
    return str(payload)


def _evidence_text(item: ProvenanceEnvelope) -> str:
    chunks = [_payload_text(item.payload)]
    raw_stdout = item.metadata.get("raw_stdout")
    raw_stderr = item.metadata.get("raw_stderr")
    if raw_stdout:
        chunks.append(str(raw_stdout))
    if raw_stderr:
        chunks.append(str(raw_stderr))
    return "\n".join(chunks)


def _normal_number(value: str) -> str:
    value = value.rstrip("%").replace(",", "")
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def _entities(text: str) -> set[str]:
    entities = {value for value in _ENTITY_RE.findall(text) if value not in _COMMON_CAPITALIZED}
    entities.update(match.strip() for match in _QUOTED_RE.findall(text))
    return {value for value in entities if len(value) >= 3}


def _is_failed_or_empty(item: ProvenanceEnvelope) -> bool:
    metadata = item.metadata
    if metadata.get("ok") is False or metadata.get("error"):
        return True
    if item.payload is None or item.payload == {} or item.payload == "":
        return True
    return isinstance(item.payload, dict) and item.payload.get("ok") is False


def _claims_success(payload: Any, text: str) -> bool:
    if isinstance(payload, dict):
        for key in ("ok", "success", "successful", "completed"):
            if payload.get(key) is True:
                return True
        status = str(payload.get("status", "")).casefold()
        if status in {"ok", "success", "successful", "completed"}:
            return True
    return bool(_SUCCESS_RE.search(text))
