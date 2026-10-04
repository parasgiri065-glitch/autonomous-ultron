"""Zero-auth provider adapters.

The adapter is intentionally stdlib-only and never runs unless its caller has
explicitly enabled the zero-auth tier. Network failures become a normal
provider miss so the waterfall can continue to the next provider or deterministic
fallback.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..breaker import BreakerResult, BreakerVerifier
from ..provenance import ProvenanceEnvelope


class ZeroAuthUnavailable(RuntimeError):
    """A zero-auth endpoint was unavailable, rate-limited, or malformed."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status in {None, 408, 425, 429} or (
            self.status is not None and self.status >= 500
        )


@dataclass(slots=True)
class AdapterResponse:
    """Provider output plus the provenance and Breaker result for that output."""

    text: str
    provider: str
    model: str
    origin: str = "llm_generated"
    raw: dict[str, Any] = field(default_factory=dict)
    provenance: list[ProvenanceEnvelope] = field(default_factory=list)
    breaker: BreakerResult | None = None

    @property
    def json(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.text)
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def verify_with_breaker(self, payload: Any | None = None) -> BreakerResult:
        self.breaker = BreakerVerifier().verify(
            payload if payload is not None else self.text, self.provenance
        )
        return self.breaker


Transport = Callable[[urllib.request.Request, float], bytes | str | dict[str, Any]]


class PollinationsAdapter:
    """OpenAI-compatible Pollinations text endpoint; no signup or API key."""

    endpoint = "https://text.pollinations.ai/openai/chat/completions"
    provider = "pollinations"
    supported_models = ("llama", "qwen", "mistral", "gpt-4o-mini")

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        model: str = "llama",
        transport: Transport | None = None,
    ) -> None:
        self.timeout = timeout
        self.model = model
        self.transport = transport or self._transport

    def complete(
        self,
        messages: Iterable[dict[str, str]],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> AdapterResponse | None:
        payload = {
            "model": model or self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ultron-zero-auth/2.4",
            },
            method="POST",
        )
        try:
            raw = self.transport(request, timeout or self.timeout)
            data = _decode_json(raw)
            text = _completion_text(data)
            if not text:
                raise ZeroAuthUnavailable("Pollinations returned no completion")
        except ZeroAuthUnavailable:
            raise
        except urllib.error.HTTPError as exc:
            raise ZeroAuthUnavailable(f"Pollinations HTTP {exc.code}", status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise ZeroAuthUnavailable(f"Pollinations unavailable: {type(exc).__name__}") from exc

        response = AdapterResponse(
            text=text,
            provider=self.provider,
            model=model or self.model,
            raw=data,
        )
        response.provenance = [
            ProvenanceEnvelope.create(
                "llm_generated",
                f"llm:{self.provider}:{uuid.uuid4().hex[:16]}",
                text,
                verified=False,
                metadata={"provider": self.provider, "model": response.model},
            )
        ]
        # Always execute the safety layer. LLM-only text will normally remain
        # untrusted until a tool/web provenance envelope supports it.
        response.verify_with_breaker()
        return response

    @staticmethod
    def _transport(request: urllib.request.Request, timeout: float) -> bytes:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()


def _decode_json(raw: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    value = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    if not isinstance(value, dict):
        raise ZeroAuthUnavailable("provider response was not a JSON object")
    return value


def _completion_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message["content"]).strip()
            if first.get("text") is not None:
                return str(first["text"]).strip()
    if data.get("text") is not None:
        return str(data["text"]).strip()
    return ""
