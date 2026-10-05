"""Grounded, single-source data extraction with fail-closed live fetching."""

from __future__ import annotations

import csv
import html
import http.client
import io
import ipaddress
import json
import re
import socket
import ssl
import urllib.parse
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .breaker import BreakerVerifier
from .charter import Charter
from .config import Settings, load_settings
from .errors import HumanApprovalRequired, PolicyDenied, UltronError
from .policy import PolicyGate, PolicyRequest
from .provenance import ProvenanceEnvelope
from .registry import TYPE_MAP, RiskTier, ToolManifest

MAX_RESPONSE_BYTES = 5_000_000
MAX_REDIRECTS = 3
DEFAULT_TIMEOUT_S = 15.0
SUPPORTED_SUFFIXES = {".html", ".htm", ".csv", ".json", ".txt", ".text", ".pdf"}
REDIRECT_CODES = {301, 302, 303, 307, 308}


class ExtractionError(UltronError):
    """A malformed request, unsupported source, or unsafe fetch."""


class ExtractionPolicyError(ExtractionError):
    """The caller did not provide the policy/charter approval required for egress."""


@dataclass(slots=True)
class FetchResponse:
    """Small transport-neutral response used by tests and the safe HTTP client."""

    url: str
    content: bytes
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.headers = {str(key).casefold(): str(value) for key, value in self.headers.items()}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")


@dataclass(slots=True)
class FieldEvidence:
    value: Any
    evidence: str
    selector: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "evidence": self.evidence,
            "selector": self.selector,
            "source": self.source,
        }


@dataclass(slots=True)
class ExtractionResult:
    source: str
    content_type: str
    fields: dict[str, Any]
    evidence: dict[str, FieldEvidence]
    provenance: list[ProvenanceEnvelope]
    errors: list[str] = field(default_factory=list)
    verified: bool = False
    rejected: bool = False
    output_path: Path | None = None

    @property
    def data(self) -> dict[str, Any]:
        return self.fields

    @property
    def ok(self) -> bool:
        return not self.errors and not self.rejected

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "content_type": self.content_type,
            "fields": dict(self.fields),
            "data": dict(self.fields),
            "evidence": {key: value.as_dict() for key, value in self.evidence.items()},
            "provenance": [item.as_dict() for item in self.provenance],
            "errors": list(self.errors),
            "verified": self.verified,
            "rejected": self.rejected,
        }

    def render(self, output_format: str = "json") -> str:
        if output_format == "json":
            return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str) + "\n"
        if output_format == "csv":
            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow(["field", "value", "evidence", "selector", "source"])
            for key, value in self.fields.items():
                row = self.evidence.get(key)
                writer.writerow(
                    [
                        key,
                        _csv_value(value),
                        row.evidence if row else "",
                        row.selector if row else "",
                        row.source if row else self.source,
                    ]
                )
            return stream.getvalue()
        if output_format == "md":
            lines = [
                f"# Extracted data: {self.source}",
                "",
                "| Field | Value | Evidence |",
                "| --- | --- | --- |",
            ]
            for key, value in self.fields.items():
                row = self.evidence.get(key)
                lines.append(
                    f"| `{key}` | `{_csv_value(value)}` | {row.evidence if row else 'not found'} |"
                )
            if self.errors:
                lines.extend(["", "## Errors", "", *[f"- {error}" for error in self.errors]])
            lines.extend(
                [
                    "",
                    "## Provenance",
                    "",
                    *[f"- `{item.source_id}` at {item.timestamp}" for item in self.provenance],
                ]
            )
            return "\n".join(lines) + "\n"
        raise ExtractionError("format must be json, csv, or md")

    def write_artifact(self, path: str | Path, output_format: str = "json") -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(output_format), encoding="utf-8")
        self.output_path = target
        return target


@dataclass(slots=True)
class _HTMLNode:
    tag: str
    attrs: dict[str, str]
    text: str
    selector: str


class _HTMLExtractor:
    """Conservative page text and label/selector collector."""

    SKIP: ClassVar[set[str]] = {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
    }
    BLOCK: ClassVar[set[str]] = {
        "p",
        "div",
        "li",
        "dt",
        "dd",
        "tr",
        "br",
        "article",
        "section",
        "h1",
        "h2",
        "h3",
    }

    def __init__(self) -> None:
        from html.parser import HTMLParser

        class Parser(HTMLParser):
            def __init__(self, outer: _HTMLExtractor) -> None:
                super().__init__(convert_charrefs=True)
                self.outer = outer
                self.skip_depth = 0
                self.stack: list[dict[str, Any]] = []
                self.title_parts: list[str] = []
                self.text_parts: list[str] = []
                self.nodes: list[_HTMLNode] = []

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                tag = tag.casefold()
                attrs_map = {key.casefold(): value or "" for key, value in attrs}
                if self.skip_depth:
                    self.skip_depth += 1
                    return
                classes = f"{attrs_map.get('class', '')} {attrs_map.get('id', '')}".casefold()
                if tag in self.outer.SKIP or any(
                    word in classes for word in ("advert", "cookie", "sidebar", "navbar", "social")
                ):
                    self.skip_depth = 1
                    return
                state = {"tag": tag, "attrs": attrs_map, "parts": []}
                self.stack.append(state)
                if tag == "title":
                    state["is_title"] = True

            def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                self.handle_starttag(tag, attrs)
                self.handle_endtag(tag)

            def handle_endtag(self, tag: str) -> None:
                tag = tag.casefold()
                if self.skip_depth:
                    self.skip_depth -= 1
                    return
                if not self.stack:
                    return
                index = next(
                    (i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i]["tag"] == tag),
                    None,
                )
                if index is None:
                    return
                state = self.stack.pop(index)
                value = _clean_text(" ".join(state["parts"]))
                if state.get("is_title"):
                    self.title_parts.append(value)
                if value and tag not in {"html", "head", "body"}:
                    self.nodes.append(
                        _HTMLNode(
                            tag, state["attrs"], value, _selector(state["tag"], state["attrs"])
                        )
                    )
                if tag in self.outer.BLOCK and value:
                    # Data nodes are already appended in document order; only
                    # add a boundary here so label extraction cannot consume
                    # the following paragraph.
                    self.text_parts.append("\n")

            def handle_data(self, data: str) -> None:
                if self.skip_depth:
                    return
                value = _clean_text(data)
                if not value:
                    return
                self.text_parts.append(value)
                for state in self.stack:
                    state["parts"].append(value)

        self.parser = Parser(self)

    def parse(self, text: str) -> tuple[str, str, list[_HTMLNode]]:
        self.parser.feed(text)
        title = _clean_text(" ".join(self.parser.title_parts))
        body = re.sub(r"[ \t]+", " ", html.unescape("".join(self.parser.text_parts))).strip()
        return title, body, self.parser.nodes


class GroundedDataExtractor:
    """Extract explicitly requested fields from one URL or local file."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        policy_gate: PolicyGate | None = None,
        fetcher: Callable[[str], FetchResponse | str | bytes | dict[str, Any]] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_redirects: int = MAX_REDIRECTS,
        resolver: Callable[..., list[tuple[Any, ...]]] | None = None,
        interactive: bool = False,
    ) -> None:
        # Do not populate config.get_settings' process-wide cache from a
        # library object: callers and tests may intentionally change ULTRON_*
        # settings between isolated extraction runs.
        self.settings = settings or load_settings()
        self.policy_gate = policy_gate
        self.fetcher = fetcher
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = int(max_response_bytes)
        self.max_redirects = int(max_redirects)
        self.resolver = resolver or socket.getaddrinfo
        self.interactive = bool(interactive)

    def extract(
        self,
        source: str | Path,
        fields: dict[str, Any],
        *,
        output_format: str = "json",
        output: str | Path | None = None,
    ) -> ExtractionResult:
        schema = normalize_schema(fields)
        if output_format not in {"json", "csv", "md"}:
            raise ExtractionError("format must be json, csv, or md")
        if _is_url(str(source)):
            response = self._fetch_public(str(source))
            content = response.content
            source_name = response.url
            content_type = _content_type(response.headers.get("content-type", ""), response.url)
        else:
            path = Path(source).expanduser()
            content, source_name, content_type = self._read_file(path)
        result = self._extract_content(source_name, content_type, content, schema)
        if output is not None:
            result.write_artifact(output, output_format)
        return result

    extract_data = extract

    def _read_file(self, path: Path) -> tuple[bytes, str, str]:
        if not path.is_file():
            raise ExtractionError(f"source file does not exist: {path}")
        suffix = path.suffix.casefold()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ExtractionError(f"unsupported local file type: {suffix or '<none>'}")
        if path.stat().st_size > self.max_response_bytes:
            raise ExtractionError(f"source exceeds {self.max_response_bytes} byte limit")
        data = path.read_bytes()
        return data, str(path.resolve()), _content_type_from_suffix(suffix)

    def _fetch_public(self, url: str) -> FetchResponse:
        # Syntax, credentials, and literal-IP checks are local. Hostname DNS
        # must wait until live egress has passed opt-in and PolicyGate.
        safe = validate_public_url(url, resolver=self.resolver, resolve_dns=False)
        if self.fetcher is not None:
            current = safe
            for _ in range(self.max_redirects + 1):
                try:
                    response = _coerce_response(self.fetcher(current), current)
                except ExtractionError:
                    raise
                except Exception as exc:
                    raise ExtractionError(f"safe fetch failed: {exc}") from exc
                if len(response.content) > self.max_response_bytes:
                    raise ExtractionError(f"response exceeds {self.max_response_bytes} byte limit")
                if response.status_code in REDIRECT_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise ExtractionError("redirect response did not contain Location")
                    current = validate_public_url(
                        urllib.parse.urljoin(current, location),
                        resolver=self.resolver,
                        resolve_dns=False,
                    )
                    continue
                if response.status_code == 429:
                    retry = response.headers.get("retry-after", "later")
                    raise ExtractionError(f"source returned HTTP 429; retry after {retry}")
                if response.status_code >= 400:
                    raise ExtractionError(f"source returned HTTP {response.status_code}")
                return response
            raise ExtractionError(f"redirect limit exceeded ({self.max_redirects})")

        if not self.settings.eval_live:
            raise ExtractionPolicyError(
                "live network is disabled; set ULTRON_EVAL_LIVE=1 explicitly"
            )
        self._authorize_network(safe)
        return safe_http_get(
            safe,
            timeout_s=self.timeout_s,
            max_bytes=self.max_response_bytes,
            max_redirects=self.max_redirects,
            resolver=self.resolver,
        )

    def _authorize_network(self, url: str) -> None:
        manifest = ToolManifest(
            name="extract_public_page",
            version="0.1.0",
            entrypoint="python -c pass",
            risk=RiskTier.MEDIUM,
            description="Fetch one public page for explicit grounded extraction.",
            permissions=["network:http"],
            inputs={"url": "string"},
            outputs={"content": "string"},
        )
        gate = self.policy_gate or PolicyGate(
            self.settings, charter=Charter(self.settings.state_dir)
        )
        request = PolicyRequest(
            tool=manifest,
            inputs={"url": url},
            goal="single-page grounded extraction",
            action_type="tool_execution",
            allow_network=True,
        )
        try:
            gate.evaluate(request, interactive=self.interactive)
        except (HumanApprovalRequired, PolicyDenied) as exc:
            raise ExtractionPolicyError(
                f"network extraction requires explicit approval: {exc}"
            ) from exc

    def _extract_content(
        self, source: str, content_type: str, content: bytes, schema: dict[str, str]
    ) -> ExtractionResult:
        if len(content) > self.max_response_bytes:
            raise ExtractionError(f"source exceeds {self.max_response_bytes} byte limit")
        provenance_origin = "web_fetch" if _is_url(source) else "local_file"
        envelope = ProvenanceEnvelope.create(
            provenance_origin,
            source,
            content.decode("utf-8", "replace"),
            metadata={"source": source, "content_type": content_type},
        )
        try:
            if content_type == "application/pdf":
                text = _pdf_text(content)
                parsed: Any = text
                kind = "text"
            elif content_type == "application/json":
                parsed = json.loads(content.decode("utf-8", "replace"))
                kind = "json"
            elif content_type == "text/csv":
                parsed = list(csv.DictReader(io.StringIO(content.decode("utf-8", "replace"))))
                kind = "csv"
            elif content_type in {"text/html", "application/xhtml+xml"}:
                parsed = _HTMLExtractor().parse(content.decode("utf-8", "replace"))
                kind = "html"
            else:
                parsed = content.decode("utf-8", "replace")
                kind = "text"
        except (UnicodeDecodeError, json.JSONDecodeError, csv.Error, ExtractionError) as exc:
            raise ExtractionError(f"could not parse {source}: {exc}") from exc

        fields: dict[str, Any] = {}
        evidence: dict[str, FieldEvidence] = {}
        errors: list[str] = []
        for name, declared_type in schema.items():
            raw_value, excerpt, selector = _find_value(name, parsed, kind, source)
            if isinstance(raw_value, list) and not declared_type.startswith("list["):
                # A scalar schema over a multi-row CSV has one deterministic
                # interpretation: the first non-empty row. Callers that need
                # every row must declare a list type.
                raw_value = raw_value[0] if raw_value else None
            try:
                value = coerce_value(raw_value, declared_type)
            except ValueError as exc:
                value = None
                errors.append(f"{name}: {exc}")
            fields[name] = value
            # Missing values retain the same source and an explicit not-found
            # selector; provenance must not disappear merely because evidence
            # was absent.
            evidence[name] = FieldEvidence(value, _short_excerpt(excerpt), selector, source)
        breaker = BreakerVerifier()
        breaker_result = breaker.verify(fields, [envelope])
        if not breaker_result.ok:
            errors.append(
                "Breaker rejected unsupported extracted value(s): " + breaker_result.reason
            )
        return ExtractionResult(
            source,
            content_type,
            fields,
            evidence,
            [envelope],
            errors,
            verified=not errors and breaker_result.ok,
            rejected=not breaker_result.ok,
        )


DataExtractor = GroundedDataExtractor
Extractor = GroundedDataExtractor
SinglePageExtractor = GroundedDataExtractor


def normalize_schema(fields: dict[str, Any]) -> dict[str, str]:
    if not isinstance(fields, dict) or not fields:
        raise ExtractionError("--fields must be a non-empty JSON object")
    normalized: dict[str, str] = {}
    for name, value in fields.items():
        if not isinstance(name, str) or not name.strip():
            raise ExtractionError("field names must be non-empty strings")
        declared = value.get("type") if isinstance(value, dict) else value
        if not isinstance(declared, str) or declared not in TYPE_MAP:
            raise ExtractionError(f"unsupported type for field {name!r}: {declared!r}")
        normalized[name] = declared
    return normalized


def validate_public_url(
    url: str,
    *,
    resolver: Callable[..., list[tuple[Any, ...]]] = socket.getaddrinfo,
    resolve_dns: bool = True,
) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ExtractionError("only http and https URLs are supported")
    try:
        username, password, hostname, port = (
            parsed.username,
            parsed.password,
            parsed.hostname,
            parsed.port,
        )
    except ValueError as exc:
        raise ExtractionError(f"malformed URL: {exc}") from exc
    if username is not None or password is not None:
        raise ExtractionError("URL credentials are forbidden")
    if not hostname:
        raise ExtractionError("URL hostname is missing")
    hostname = hostname.rstrip(".").casefold()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
        raise ExtractionError("localhost destinations are forbidden")
    try:
        literal = ipaddress.ip_address(hostname)
        _assert_public_ip(literal)
    except ValueError as invalid_literal:
        if resolve_dns:
            try:
                answers = resolver(
                    hostname,
                    port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
                addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
                for answer in answers:
                    if len(answer) <= 4:
                        continue
                    try:
                        addresses.add(ipaddress.ip_address(answer[4][0]))
                    except (IndexError, TypeError, ValueError) as exc:
                        raise ExtractionError(
                            "DNS returned a malformed address; refusing fetch"
                        ) from exc
            except ExtractionError:
                raise
            except (OSError, socket.gaierror) as exc:
                raise ExtractionError(
                    f"DNS resolution failed for {hostname!r}; refusing fetch"
                ) from exc
            if not addresses:
                raise ExtractionError(
                    "DNS returned no usable addresses; refusing fetch"
                ) from invalid_literal
            for address in addresses:
                _assert_public_ip(address)
    return urllib.parse.urlunsplit(
        (parsed.scheme.casefold(), parsed.netloc, parsed.path or "/", parsed.query, "")
    )


def _assert_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        raise ExtractionError(
            f"private, reserved, or link-local destination is forbidden: {address}"
        )


def safe_http_get(
    url: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = MAX_RESPONSE_BYTES,
    max_redirects: int = MAX_REDIRECTS,
    resolver: Callable[..., list[tuple[Any, ...]]] = socket.getaddrinfo,
) -> FetchResponse:
    """HTTP client that connects to a validated resolved IP, not a re-resolved hostname."""
    current = validate_public_url(url, resolver=resolver)
    for _ in range(max_redirects + 1):
        response = _one_safe_http_request(current, timeout_s, max_bytes, resolver)
        if response.status_code in REDIRECT_CODES:
            location = response.headers.get("location")
            if not location:
                raise ExtractionError("redirect response did not contain Location")
            current = validate_public_url(
                urllib.parse.urljoin(current, location), resolver=resolver
            )
            continue
        if response.status_code == 429:
            retry = response.headers.get("retry-after", "later")
            raise ExtractionError(f"source returned HTTP 429; retry after {retry}")
        if response.status_code >= 400:
            raise ExtractionError(f"source returned HTTP {response.status_code}")
        return response
    raise ExtractionError(f"redirect limit exceeded ({max_redirects})")


def _one_safe_http_request(
    url: str, timeout_s: float, max_bytes: int, resolver: Callable[..., list[tuple[Any, ...]]]
) -> FetchResponse:
    parsed = urllib.parse.urlsplit(validate_public_url(url, resolver=resolver))
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        answers = resolver(parsed.hostname, port, type=socket.SOCK_STREAM)
        ips: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        for answer in answers:
            if len(answer) <= 4:
                continue
            try:
                ips.add(ipaddress.ip_address(answer[4][0]))
            except (IndexError, TypeError, ValueError) as exc:
                raise ExtractionError("DNS returned a malformed address; refusing fetch") from exc
    except ExtractionError:
        raise
    except (OSError, socket.gaierror) as exc:
        raise ExtractionError("DNS resolution failed; refusing fetch") from exc
    for address in ips:
        _assert_public_ip(address)
    if not ips:
        raise ExtractionError("DNS returned no usable addresses; refusing fetch")
    last_error: Exception | None = None
    for address in sorted(ips, key=str):
        sock: socket.socket | None = None
        try:
            sock = socket.create_connection((str(address), port), timeout=timeout_s)
            if parsed.scheme == "https":
                context = ssl.create_default_context()
                sock = context.wrap_socket(sock, server_hostname=parsed.hostname)
            host_header = parsed.hostname or ""
            if ":" in host_header and not host_header.startswith("["):
                host_header = f"[{host_header}]"
            if parsed.port and parsed.port not in {80, 443}:
                host_header = f"{host_header}:{parsed.port}"
            request = (
                f"GET {parsed.path or '/'}{('?' + parsed.query) if parsed.query else ''} HTTP/1.1\r\n"
                f"Host: {host_header}\r\nUser-Agent: ultron-extractor/4.2\r\nAccept: text/html,application/json,text/csv,text/plain,application/pdf\r\nConnection: close\r\n\r\n"
            ).encode("ascii", "strict")
            sock.sendall(request)
            response = http.client.HTTPResponse(sock)
            response.begin()
            headers = {key.casefold(): value for key, value in response.getheaders()}
            declared = int(headers.get("content-length", "0") or 0)
            if declared > max_bytes:
                raise ExtractionError(f"response exceeds {max_bytes} byte limit")
            content = response.read(max_bytes + 1)
            if len(content) > max_bytes:
                raise ExtractionError(f"response exceeds {max_bytes} byte limit")
            return FetchResponse(url, content, response.status, headers)
        except ExtractionError:
            raise
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            last_error = exc
        finally:
            if sock is not None:
                with suppress(OSError):
                    sock.close()
    raise ExtractionError(f"safe connection failed: {last_error}")


def _coerce_response(raw: FetchResponse | str | bytes | dict[str, Any], url: str) -> FetchResponse:
    if isinstance(raw, FetchResponse):
        return raw
    if isinstance(raw, dict):
        body = raw.get("content", raw.get("body", ""))
        if isinstance(body, str):
            body = body.encode()
        return FetchResponse(
            str(raw.get("url", url)),
            bytes(body),
            int(raw.get("status_code", 200)),
            {str(k).casefold(): str(v) for k, v in dict(raw.get("headers", {})).items()},
        )
    return FetchResponse(url, raw.encode() if isinstance(raw, str) else bytes(raw))


def _find_value(name: str, parsed: Any, kind: str, source: str) -> tuple[Any, str, str]:
    if kind == "json":
        value = parsed
        path = "$"
        for part in name.split("."):
            if not isinstance(value, dict) or part not in value:
                return None, "not found", path + "." + part
            value = value[part]
            path += "." + part
        return value, json.dumps(value, default=str), path
    if kind == "csv":
        rows = parsed if isinstance(parsed, list) else []
        values = [
            row.get(name)
            for row in rows
            if isinstance(row, dict) and row.get(name) not in (None, "")
        ]
        if name.startswith("rows."):
            name = name.split(".", 1)[1]
        if len(values) > 1:
            return values, "; ".join(str(value) for value in values), f"CSV column {name}"
        return (
            (values[0], str(values[0]), f"CSV row 1 column {name}")
            if values
            else (None, "not found", f"CSV column {name}")
        )
    if kind == "html":
        title, body, nodes = parsed
        if name.casefold() == "title":
            return (title or None, title or "not found", "html > title")
        key = _norm_name(name)
        for index, node in enumerate(nodes):
            attrs = {attr.casefold(): value for attr, value in node.attrs.items()}
            if key in {
                _norm_name(attrs.get("id", "")),
                _norm_name(attrs.get("name", "")),
                _norm_name(attrs.get("data-field", "")),
            }:
                return node.text, node.text, node.selector
            if key and key in {_norm_name(item) for item in attrs.get("class", "").split()}:
                return node.text, node.text, node.selector
            # Common definition-list/table/label layout: an explicit field
            # label immediately followed by its value is deterministic evidence.
            if _norm_name(node.text) == key and index + 1 < len(nodes):
                following = nodes[index + 1]
                if following.text and following.tag in {"dd", "td", "span", "div", "p"}:
                    return following.text, following.text, f"{node.selector} + {following.selector}"
        return _find_labeled(name, body, source)
    return _find_labeled(name, str(parsed), source)


def _find_labeled(name: str, text: str, source: str) -> tuple[Any, str, str]:
    label = r"[ _-]+".join(re.escape(part) for part in name.split("_"))
    pattern = re.compile(rf"(?im)(?:^|[\n|;]|\b)\s*{label}\s*(?:[:=\-])\s*([^\n|;]+)")
    match = pattern.search(text)
    if match:
        value = _clean_text(match.group(1))
        return value, value, f"label:{name}"
    if name.casefold() in {"text", "content", "body"}:
        value = _clean_text(text)
        return (value or None, value[:240], "document body")
    return None, "not found", f"missing:{name}"


def coerce_value(value: Any, declared: str) -> Any:
    if value is None or value == "":
        return None
    if declared == "any":
        return value
    if declared == "string":
        return str(value).strip()
    if declared == "int":
        if isinstance(value, bool):
            raise ValueError("expected int")
        match = re.search(r"[-+]?\d[\d,]*", str(value))
        if not match:
            raise ValueError("expected int")
        return int(match.group(0).replace(",", ""))
    if declared == "float":
        if isinstance(value, bool):
            raise ValueError("expected float")
        match = re.search(r"[-+]?\d[\d,.]*", str(value))
        if not match:
            raise ValueError("expected float")
        return float(match.group(0).replace(",", ""))
    if declared == "bool":
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().casefold()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
        raise ValueError("expected bool")
    if declared.startswith("list["):
        values = (
            value
            if isinstance(value, list)
            else [part.strip() for part in str(value).split(",") if part.strip()]
        )
        subtype = declared[5:-1]
        return [coerce_value(item, subtype) for item in values]
    if declared == "dict":
        if not isinstance(value, dict):
            raise ValueError("expected object")
        return value
    raise ValueError(f"unsupported type {declared!r}")


def _pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ExtractionError(
                "PDF extraction is unavailable: no existing PDF parser dependency"
            ) from exc
    try:
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise ExtractionError(f"malformed or unreadable PDF: {exc}") from exc


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _selector(tag: str, attrs: dict[str, str]) -> str:
    if attrs.get("id"):
        return f"#{attrs['id']}"
    if attrs.get("data-field"):
        return f"[data-field={attrs['data-field']!r}]"
    return tag + ("." + attrs["class"].split()[0] if attrs.get("class") else "")


def _short_excerpt(value: str, limit: int = 240) -> str:
    value = _clean_text(value)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _csv_value(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, (dict, list))
        else ""
        if value is None
        else str(value)
    )


def _is_url(value: str) -> bool:
    return urllib.parse.urlsplit(value).scheme.casefold() in {"http", "https"}


def _content_type(value: str, source: str) -> str:
    value = value.split(";", 1)[0].strip().casefold()
    return (
        value
        if value
        in {
            "text/html",
            "application/xhtml+xml",
            "application/json",
            "text/csv",
            "text/plain",
            "application/pdf",
        }
        else _content_type_from_suffix(Path(urllib.parse.urlsplit(source).path).suffix.casefold())
    )


def _content_type_from_suffix(suffix: str) -> str:
    return {
        ".html": "text/html",
        ".htm": "text/html",
        ".json": "application/json",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".text": "text/plain",
    }.get(suffix, "text/plain")


def _within_path(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


__all__ = [
    "MAX_REDIRECTS",
    "MAX_RESPONSE_BYTES",
    "DataExtractor",
    "ExtractionError",
    "ExtractionPolicyError",
    "ExtractionResult",
    "Extractor",
    "FetchResponse",
    "FieldEvidence",
    "GroundedDataExtractor",
    "SinglePageExtractor",
    "coerce_value",
    "normalize_schema",
    "safe_http_get",
    "validate_public_url",
]
