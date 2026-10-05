"""Hermetic-friendly public-source ingestion and corroborated intelligence briefs."""

from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..config import Settings, get_settings
from ..errors import UltronError
from ..provenance import ProvenanceEnvelope
from ..registry import Registry, RiskTier, ToolManifest

DEFAULT_TIMEOUT_S = 15.0
DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
CLICKBAIT_WORDS = frozenset(
    {
        "shocking",
        "unbelievable",
        "secret",
        "you won't believe",
        "you wont believe",
        "breaking",
        "miracle",
        "exposed",
        "this changes everything",
    }
)
SPECULATION_RE = re.compile(
    r"\b(?:may|might|could|possibly|potentially|likely|allegedly|reportedly|rumou?r|speculat(?:e|es|ed|ion))\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[,.]\d+)*(?:\.\d+)?%?(?![A-Za-z])")
DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)?\d{2}\b",
    re.IGNORECASE,
)
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&'_-]{2,}(?:\s+[A-Z][A-Za-z0-9&'_-]{2,}){0,3}\b")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|\d+(?:\.\d+)?%?")
EVENT_WORDS = frozenset(
    {
        "announced",
        "approved",
        "closed",
        "confirmed",
        "declined",
        "discovered",
        "expanded",
        "fell",
        "found",
        "grew",
        "increased",
        "launched",
        "opened",
        "reported",
        "reports",
        "signed",
        "surged",
        "won",
        "decreased",
        "acquired",
    }
)
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
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
        "with",
        "which",
        "who",
        "will",
        "would",
        "said",
        "says",
        "than",
        "also",
        "about",
        "after",
        "before",
        "during",
        "according",
        "than",
    ]
)


class IntelError(UltronError):
    """A public-source ingestion or report synthesis failure."""


@dataclass(slots=True)
class IntelItem:
    """One fetched public item and its immutable source evidence."""

    url: str
    title: str
    text: str
    source_domain: str
    published: str = ""
    provenance: ProvenanceEnvelope | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return self.source_domain

    @property
    def provenance_envelope(self) -> ProvenanceEnvelope | None:
        return self.provenance

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "source_domain": self.source_domain,
            "published": self.published,
            "provenance": self.provenance.as_dict() if self.provenance else None,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class _HTMLDocument:
    title: str = ""
    text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)


class _DocumentParser(HTMLParser):
    """Small dependency-free article/search HTML parser."""

    SKIP_TAGS = frozenset(
        {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form"}
    )
    NOISE_WORDS = frozenset(
        {
            "ad",
            "ads",
            "advert",
            "advertisement",
            "cookie",
            "banner",
            "sidebar",
            "navbar",
            "menu",
            "share",
            "social",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.doc = _HTMLDocument()
        self._skip = 0
        self._capture_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._link_href = ""
        self._link_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_map = {key.lower(): value or "" for key, value in attrs}
        classes = f"{attrs_map.get('class', '')} {attrs_map.get('id', '')}".casefold()
        if self._skip:
            self._skip += 1
            return
        if tag in self.SKIP_TAGS or any(word in classes for word in self.NOISE_WORDS):
            self._skip = 1
            return
        if tag == "title":
            self._capture_title = True
        if tag == "a":
            self._link_href = attrs_map.get("href", "")
            self._link_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip:
            self._skip -= 1
            return
        if tag == "title":
            self._capture_title = False
        if tag == "a" and self._link_href:
            self.doc.links.append((self._link_href, " ".join(self._link_parts).strip()))
            self._link_href = ""
            self._link_parts = []

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        clean = " ".join(data.split())
        if not clean:
            return
        if self._capture_title:
            self._title_parts.append(clean)
        self._text_parts.append(clean)
        if self._link_href:
            self._link_parts.append(clean)

    def finish(self) -> _HTMLDocument:
        self.doc.title = " ".join(self._title_parts).strip()
        self.doc.text = _normalize_text(" ".join(self._text_parts))
        return self.doc


class IntelIngester:
    """Fetch RSS, zero-auth search results, and article text with failover."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        fetcher: Callable[[str], str | bytes | dict[str, Any]] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        search_endpoint: str = DDG_HTML_ENDPOINT,
    ) -> None:
        self.settings = settings or get_settings()
        self.fetcher = fetcher or self._fetch_url
        self.timeout_s = float(timeout_s)
        self.search_endpoint = search_endpoint
        self.failures: list[dict[str, str]] = []

    def fetch_rss(self, url: str) -> list[IntelItem]:
        try:
            raw = self.fetcher(url)
            return self._parse_feed(url, _as_text(raw))
        except Exception as exc:
            self._failure(url, exc)
            return []

    ingest_rss = fetch_rss

    def search(
        self, query: str, *, endpoint: str | None = None, limit: int = 10
    ) -> list[IntelItem]:
        url = endpoint or self.search_endpoint
        if "{query}" in url:
            request_url = url.replace("{query}", urllib.parse.quote_plus(query))
        else:
            separator = "&" if "?" in url else "?"
            request_url = f"{url}{separator}{urllib.parse.urlencode({'q': query})}"
        try:
            raw = self.fetcher(request_url)
            if isinstance(raw, dict):
                return self._parse_search_json(request_url, raw, limit=limit)
            text = _as_text(raw)
            if "format=json" in request_url or text.lstrip().startswith("{"):
                try:
                    return self._parse_search_json(request_url, json.loads(text), limit=limit)
                except json.JSONDecodeError:
                    pass
            return self._parse_search_html(request_url, text, limit=limit)
        except Exception as exc:
            self._failure(request_url, exc)
            return []

    def fetch_article(self, url: str, *, html_text: str | None = None) -> IntelItem | None:
        try:
            raw = html_text if html_text is not None else self.fetcher(url)
            document = _parse_html(_as_text(raw))
            title = document.title or urlparse(url).netloc
            text = document.text
            if not text:
                raise IntelError("article contained no readable main text")
            return self._item(url, title, text)
        except Exception as exc:
            self._failure(url, exc)
            return None

    extract_article = fetch_article

    def ingest(
        self,
        topic: str,
        *,
        depth: str = "light",
        rss_urls: Iterable[str] = (),
        article_urls: Iterable[str] = (),
        search_endpoints: Iterable[str] = (),
    ) -> list[IntelItem]:
        """Collect search/RSS/article evidence; one failed source never aborts."""
        if depth not in {"light", "deep"}:
            raise IntelError("depth must be light or deep")
        items: list[IntelItem] = []
        for feed in rss_urls:
            items.extend(self.fetch_rss(feed))
        search_results = self.search(
            topic, endpoint=next(iter(search_endpoints), None), limit=8 if depth == "deep" else 4
        )
        items.extend(search_results)
        for url in article_urls:
            article = self.fetch_article(url)
            if article:
                items.append(article)
        if depth == "deep":
            seen = {item.url for item in items if item.text and len(item.text) > 250}
            for result in search_results:
                if result.url in seen:
                    continue
                article = self.fetch_article(result.url)
                if article:
                    items.append(article)
                    seen.add(result.url)
        return _dedupe_items(items)

    collect = ingest

    def _item(
        self,
        url: str,
        title: str,
        text: str,
        *,
        published: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> IntelItem:
        domain = _domain(url)
        envelope = ProvenanceEnvelope.create(
            "web_fetch",
            url,
            text,
            metadata={
                "url": url,
                "source_title": title,
                "domain": domain,
                "fetched_at": datetime.now(UTC).isoformat(),
            },
        )
        return IntelItem(url, title, text, domain, published, envelope, dict(metadata or {}))

    def _parse_feed(self, url: str, text: str) -> list[IntelItem]:
        root = ET.fromstring(text)
        nodes = list(root.findall(".//item"))
        if not nodes:
            nodes = [node for node in root.iter() if _local_name(node.tag) == "entry"]
        items: list[IntelItem] = []
        for node in nodes:
            fields = {_local_name(child.tag): (child.text or "").strip() for child in list(node)}
            link = fields.get("link", "")
            if not link:
                for child in list(node):
                    if _local_name(child.tag) == "link":
                        link = child.attrib.get("href", "")
                        break
            title = _clean_markup(fields.get("title", ""))
            description = _clean_markup(
                fields.get("description", fields.get("summary", fields.get("content", "")))
            )
            if not link and not description:
                continue
            item_url = urllib.parse.urljoin(url, html.unescape(link))
            text_value = description or title
            items.append(
                self._item(
                    item_url or url,
                    title or item_url,
                    text_value,
                    published=fields.get("pubDate", fields.get("published", "")),
                    metadata={"source_type": "rss", "feed_url": url},
                )
            )
        return items

    def _parse_search_html(self, url: str, text: str, *, limit: int) -> list[IntelItem]:
        document = _parse_html(text)
        results: list[IntelItem] = []
        for raw_href, anchor_title in document.links:
            href = _resolve_search_href(raw_href)
            if not href.startswith(("http://", "https://")):
                continue
            title = anchor_title or href
            results.append(
                self._item(href, title, title, metadata={"source_type": "search", "query_url": url})
            )
            if len(results) >= limit:
                break
        return results

    def _parse_search_json(self, url: str, data: dict[str, Any], *, limit: int) -> list[IntelItem]:
        results: list[IntelItem] = []
        for row in data.get("results", []) if isinstance(data.get("results", []), list) else []:
            if not isinstance(row, dict) or not row.get("url"):
                continue
            title = str(row.get("title") or row["url"])
            snippet = str(row.get("content") or row.get("snippet") or title)
            results.append(
                self._item(
                    str(row["url"]),
                    title,
                    snippet,
                    metadata={"source_type": "search", "query_url": url},
                )
            )
            if len(results) >= limit:
                break
        return results

    def _fetch_url(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "ultron-intel/4.1"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise IntelError(f"source request failed: {exc}") from exc

    def _failure(self, url: str, exc: Exception) -> None:
        self.failures.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})


class ClaimStatus(StrEnum):
    VERIFIED = "VERIFIED"
    UNCONFIRMED = "UNCONFIRMED"
    CONFLICTING = "CONFLICTING"


@dataclass(slots=True)
class Claim:
    text: str
    status: str
    kind: str
    sources: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    key: str = ""

    @property
    def source_urls(self) -> list[str]:
        return self.sources

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status,
            "kind": self.kind,
            "sources": self.sources,
            "domains": self.domains,
            "entities": self.entities,
            "metrics": self.metrics,
            "dates": self.dates,
            "flags": self.flags,
            "key": self.key,
        }


class ClaimTriangulator:
    """Group source claims by factual subject and detect corroboration/conflict."""

    def extract_claims(self, items: Iterable[IntelItem]) -> list[Claim]:
        claims: list[Claim] = []
        for item in items:
            clickbait = _is_clickbait(item.title)
            for sentence in _sentences(item.text):
                entities = sorted(set(ENTITY_RE.findall(sentence)))
                metrics = NUMBER_RE.findall(sentence)
                dates = DATE_RE.findall(sentence)
                tokens = {token.casefold() for token in TOKEN_RE.findall(sentence)}
                event = bool(tokens.intersection(EVENT_WORDS))
                if not (entities or metrics or dates or event):
                    continue
                flags: list[str] = []
                if clickbait:
                    flags.append("clickbait_title")
                speculative = bool(SPECULATION_RE.search(sentence))
                if speculative:
                    flags.append("speculation")
                if len(sentence.split()) < 5:
                    flags.append("fluff")
                base = _claim_key(sentence)
                kind = (
                    "metric" if metrics else ("date" if dates else ("event" if event else "entity"))
                )
                claims.append(
                    Claim(
                        sentence,
                        ClaimStatus.UNCONFIRMED,
                        kind,
                        [item.url],
                        [item.domain],
                        entities,
                        metrics,
                        dates,
                        flags,
                        base,
                    )
                )
        return claims

    def triangulate(self, items: Iterable[IntelItem]) -> list[Claim]:
        claims = self.extract_claims(items)
        groups: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            groups[claim.key].append(claim)
        for group in groups.values():
            domains = {domain for claim in group for domain in claim.domains if domain}
            values = {_value_signature(claim) for claim in group}
            conflicting = (
                len(domains) >= 2
                and len(values) > 1
                and any(claim.metrics or claim.dates for claim in group)
            )
            for claim in group:
                if conflicting:
                    claim.status = ClaimStatus.CONFLICTING
                elif len(domains) >= 2 and "speculation" not in claim.flags:
                    claim.status = ClaimStatus.VERIFIED
                else:
                    claim.status = ClaimStatus.UNCONFIRMED
                claim.sources = sorted({source for row in group for source in row.sources})
                claim.domains = sorted(domains)
        return _unique_claims(claims)

    corroborate = triangulate
    analyze = triangulate


@dataclass(slots=True)
class IntelBrief:
    topic: str
    executive_summary: list[str]
    verified_facts: list[Claim]
    timeline: list[Claim]
    sources: list[IntelItem]
    disclosures: dict[str, list[str]]
    markdown: str
    json_data: dict[str, Any]
    markdown_path: Path | None = None
    json_path: Path | None = None

    @property
    def report_markdown(self) -> str:
        return self.markdown

    def as_dict(self) -> dict[str, Any]:
        return self.json_data


class IntelSynthesizer:
    """Compile claims and provenance into Markdown and structured JSON."""

    def __init__(
        self, settings: Settings | None = None, *, registry: Registry | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or Registry(self.settings).load()

    def synthesize(
        self,
        topic: str,
        items: Iterable[IntelItem],
        claims: Iterable[Claim],
        *,
        output: str | Path | None = None,
    ) -> IntelBrief:
        source_items = _dedupe_items(list(items))
        all_claims = list(claims)
        verified = [
            claim
            for claim in all_claims
            if claim.status == ClaimStatus.VERIFIED and "speculation" not in claim.flags
        ]
        timeline = sorted(
            [claim for claim in all_claims if claim.dates],
            key=lambda claim: (claim.dates[0], claim.text),
        )
        domains = sorted({item.domain for item in source_items if item.domain})
        disclosures = {
            "checked": [
                "Public RSS/search/article sources available without authentication.",
                f"{len(source_items)} fetched item(s) across {len(domains)} independent domain(s).",
                "Claim overlap, numeric/date conflicts, source domains, and provenance envelopes.",
            ],
            "not_checked": [
                "Paywalled, authenticated, private, or unavailable sources.",
                "The truth of claims beyond the fetched text and source publication context.",
                "Expert review, primary documents, or real-time events not present in the fetched sources.",
            ],
        }
        summary = [
            f"{len(verified)} claim(s) were corroborated by at least two independent public domains.",
            f"The brief checked {len(source_items)} source item(s) spanning {len(domains)} domain(s).",
            f"{sum(claim.status == ClaimStatus.CONFLICTING for claim in all_claims)} conflicting and {sum(claim.status == ClaimStatus.UNCONFIRMED for claim in all_claims)} unconfirmed claim(s) remain flagged.",
        ]
        json_data = {
            "topic": topic,
            "executive_summary": summary,
            "verified_facts": [claim.as_dict() for claim in verified],
            "timeline": [claim.as_dict() for claim in timeline],
            "claims": [claim.as_dict() for claim in all_claims],
            "sources": [item.as_dict() for item in source_items],
            "disclosures": disclosures,
        }
        markdown = self._markdown(
            topic, summary, verified, timeline, all_claims, source_items, disclosures
        )
        brief = IntelBrief(
            topic, summary, verified, timeline, source_items, disclosures, markdown, json_data
        )
        self.register_capabilities(self.registry)
        if output is not None:
            brief.markdown_path, brief.json_path = self.write_report(brief, output)
        return brief

    build = synthesize

    def write_report(self, brief: IntelBrief, output: str | Path) -> tuple[Path, Path]:
        markdown_path = Path(output).expanduser()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(brief.markdown, encoding="utf-8")
        json_path = markdown_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(brief.json_data, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return markdown_path, json_path

    def capability_manifest(self) -> ToolManifest:
        return ToolManifest(
            name="intel_report",
            version="0.1.0",
            entrypoint="python -m ultron.domains.intel",
            risk=RiskTier.MEDIUM,
            description="Public-source intelligence ingestion, triangulation, and executive brief synthesis.",
            provides=["intel.report", "news.raw"],
            requires=[],
            permissions=[],
            inputs={"topic": "string"},
            outputs={"report": "string", "sources": "list[string]"},
            tags=["intel", "research", "provenance", "triangulation"],
            deterministic=False,
            author="ultron-intel",
        )

    def register_capabilities(self, registry: Registry | None = None) -> list[ToolManifest]:
        target = registry or self.registry
        manifest = self.capability_manifest()
        try:
            target.register(manifest)
        except Exception:
            # Re-registering the same content is harmless; preserve the existing
            # registry contract if another version is already present.
            existing = target.get(manifest.name)
            if existing.content_hash != manifest.content_hash:
                raise
        return [target.get(manifest.name)]

    register_tools = register_capabilities

    @staticmethod
    def _markdown(
        topic: str,
        summary: list[str],
        verified: list[Claim],
        timeline: list[Claim],
        claims: list[Claim],
        sources: list[IntelItem],
        disclosures: dict[str, list[str]],
    ) -> str:
        lines = [f"# Intelligence Brief: {topic}", "", "## Executive Summary", ""]
        lines.extend(f"- {item}" for item in summary)
        lines.extend(["", "## Corroborated Facts", ""])
        if verified:
            lines.extend(
                f"- **{claim.status}** {claim.text} ([{len(claim.sources)} sources]({claim.sources[0]}))"
                for claim in verified
            )
        else:
            lines.append("- No claims met the two-independent-domain corroboration threshold.")
        lines.extend(["", "## Corroborated Facts & Timeline", ""])
        if timeline:
            lines.extend(
                f"- **{claim.dates[0]}** — {claim.text} ({claim.status})" for claim in timeline
            )
        else:
            lines.append("- No dated claims were extracted.")
        lines.extend(["", "## Sources & Evidence Map", ""])
        for item in sources:
            provenance = item.provenance.source_id if item.provenance else item.url
            lines.append(
                f"- [{item.title or item.url}]({item.url}) — `{item.domain}`; provenance `{provenance}`"
            )
        lines.extend(["", "## Claim Status Notes", ""])
        for claim in claims:
            if claim.status != ClaimStatus.VERIFIED or claim.flags:
                flags = f"; flags: {', '.join(claim.flags)}" if claim.flags else ""
                lines.append(
                    f"- **{claim.status}** {claim.text} — sources: {', '.join(claim.sources)}{flags}"
                )
        lines.extend(["", "## Transparent Disclosures", "", "### What Ultron Checked", ""])
        lines.extend(f"- {item}" for item in disclosures["checked"])
        lines.extend(["", "### What Ultron Did NOT Check", ""])
        lines.extend(f"- {item}" for item in disclosures["not_checked"])
        lines.append("")
        return "\n".join(lines)


class IntelResearchEngine:
    """Orchestrate ingestion, triangulation, synthesis, and report output."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ingester: IntelIngester | None = None,
        triangulator: ClaimTriangulator | None = None,
        synthesizer: IntelSynthesizer | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ingester = ingester or IntelIngester(self.settings)
        self.triangulator = triangulator or ClaimTriangulator()
        self.synthesizer = synthesizer or IntelSynthesizer(self.settings)

    def research(
        self,
        topic: str,
        *,
        depth: str = "light",
        output: str | Path | None = None,
        rss_urls: Iterable[str] = (),
        article_urls: Iterable[str] = (),
    ) -> IntelBrief:
        items = self.ingester.ingest(
            topic, depth=depth, rss_urls=rss_urls, article_urls=article_urls
        )
        claims = self.triangulator.triangulate(items)
        if output is None:
            slug = re.sub(r"[^a-z0-9]+", "_", topic.casefold()).strip("_")[:80] or "topic"
            output = self.settings.state_dir / "intel" / f"{slug}.md"
        return self.synthesizer.synthesize(topic, items, claims, output=output)

    run = research


def _as_text(raw: str | bytes | dict[str, Any]) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    if isinstance(raw, dict):
        return json.dumps(raw)
    return str(raw)


def _parse_html(text: str) -> _HTMLDocument:
    parser = _DocumentParser()
    parser.feed(text)
    return parser.finish()


def _clean_markup(text: str) -> str:
    return _parse_html(text).text


def _normalize_text(text: str) -> str:
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    return text


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _domain(url: str) -> str:
    return urlparse(url).netloc.casefold().removeprefix("www.") or "unknown"


def _resolve_search_href(href: str) -> str:
    parsed = urlparse(html.unescape(href))
    if parsed.netloc.endswith("duckduckgo.com"):
        query = urllib.parse.parse_qs(parsed.query).get("uddg", [])
        if query:
            return query[0]
    return href


def _sentences(text: str) -> list[str]:
    return [
        part.strip().rstrip(".!?")
        for part in re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(part.strip()) >= 20
    ]


def _claim_key(text: str) -> str:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(text.casefold()):
        if token.isdigit() or NUMBER_RE.fullmatch(token) or DATE_RE.fullmatch(token):
            continue
        if token in STOPWORDS or token in {word.strip() for word in EVENT_WORDS}:
            continue
        tokens.append(token)
    return " ".join(sorted(set(tokens))) or text.casefold().strip()


def _value_signature(claim: Claim) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(sorted(claim.metrics)), tuple(sorted(claim.dates))


def _is_clickbait(title: str) -> bool:
    lowered = title.casefold()
    return any(word in lowered for word in CLICKBAIT_WORDS) or title.count("!") >= 2


def _unique_claims(claims: list[Claim]) -> list[Claim]:
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    result: list[Claim] = []
    for claim in claims:
        key = (claim.key, claim.status, tuple(claim.metrics + claim.dates))
        if key not in seen:
            seen.add(key)
            result.append(claim)
    return result


def _dedupe_items(items: Iterable[IntelItem]) -> list[IntelItem]:
    seen: set[str] = set()
    result: list[IntelItem] = []
    for item in items:
        if item.url in seen:
            continue
        seen.add(item.url)
        result.append(item)
    return result


if __name__ == "__main__":  # pragma: no cover - manifest entrypoint marker
    print(json.dumps({"ok": False, "error": "use IntelResearchEngine from Python"}))
