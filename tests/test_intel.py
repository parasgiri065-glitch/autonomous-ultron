from __future__ import annotations

import json
from pathlib import Path

from ultron.config import REPO_ROOT, load_settings
from ultron.domains.intel import (
    ClaimStatus,
    ClaimTriangulator,
    IntelBrief,
    IntelIngester,
    IntelItem,
    IntelSynthesizer,
)
from ultron.interfaces.telegram import TelegramBotClient, TelegramCockpit
from ultron.main import build_parser

RSS = """\
<rss version="2.0"><channel><title>Public feed</title>
<item><title>Acme confirms launch</title><link>https://alpha.example/launch</link>
<description><![CDATA[Acme launched Project Nova in 2024.]]></description>
<pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate></item>
</channel></rss>
"""
ARTICLE_ALPHA = """\
<html><head><title>Alpha article</title></head><body>
<nav>Advertisement navigation</nav><main><p>Acme launched Project Nova in 2024.</p>
<p>Acme reported 42% growth in 2024.</p></main><footer>Copyright and ads</footer>
</body></html>
"""
ARTICLE_BETA = """\
<html><head><title>Beta article</title></head><body><article>
<p>Acme launched Project Nova in 2024.</p><p>Acme reported 42% growth in 2024.</p>
</article></body></html>
"""


def _settings(tmp_path: Path):
    return load_settings(
        state_dir=tmp_path / "state",
        tools_dir=tmp_path / "tools",
        cache_path=tmp_path / "state" / "cache.db",
        memory_path=tmp_path / "state" / "memory.db",
        approvals_file=tmp_path / "state" / "approvals.json",
        audit_log=tmp_path / "state" / "audit.jsonl",
        llm_mode="offline",
        repo_root=REPO_ROOT,
    )


def test_ingestion_is_hermetic_and_attaches_provenance(tmp_path):
    responses = {
        "https://alpha.example/feed.xml": RSS,
        "https://alpha.example/launch": ARTICLE_ALPHA,
        "https://beta.example/launch": ARTICLE_BETA,
    }

    def fetch(url: str):
        if url.startswith("https://html.duckduckgo.com"):
            return '<a href="https://alpha.example/launch">Alpha</a><a href="https://beta.example/launch">Beta</a>'
        return responses[url]

    ingester = IntelIngester(_settings(tmp_path), fetcher=fetch)
    feed_items = ingester.fetch_rss("https://alpha.example/feed.xml")
    article = ingester.fetch_article("https://alpha.example/launch")
    items = ingester.ingest("Acme", depth="deep", rss_urls=["https://alpha.example/feed.xml"])
    assert feed_items[0].provenance is not None
    assert feed_items[0].provenance.origin == "web_fetch"
    assert feed_items[0].provenance.metadata["source_title"] == "Acme confirms launch"
    assert article is not None
    assert "Advertisement navigation" not in article.text
    assert "Copyright and ads" not in article.text
    assert len(items) >= 1
    assert ingester.failures == []

    failed = IntelIngester(
        _settings(tmp_path / "fail"),
        fetcher=lambda _url: (_ for _ in ()).throw(TimeoutError("slow")),
    )
    assert failed.fetch_rss("https://slow.example/feed") == []
    assert failed.failures[0]["url"] == "https://slow.example/feed"


def _item(url: str, text: str) -> IntelItem:
    domain = url.split("//", 1)[1].split("/", 1)[0]
    return IntelItem(url, "source", text, domain)


def test_triangulation_verifies_unconfirmed_and_conflicting_claims():
    items = [
        _item("https://alpha.example/a", "Acme launched Project Nova in 2024."),
        _item("https://beta.example/b", "Acme launched Project Nova in 2024."),
        _item("https://gamma.example/c", "Acme operates 7 facilities."),
        _item("https://delta.example/d", "Acme reported 42% growth in 2024."),
        _item("https://epsilon.example/e", "Acme reported 18% growth in 2024."),
    ]
    claims = ClaimTriangulator().triangulate(items)
    statuses = {claim.text: claim.status for claim in claims}
    assert statuses["Acme launched Project Nova in 2024"] == ClaimStatus.VERIFIED
    assert statuses["Acme operates 7 facilities"] == ClaimStatus.UNCONFIRMED
    assert statuses["Acme reported 42% growth in 2024"] == ClaimStatus.CONFLICTING
    assert statuses["Acme reported 18% growth in 2024"] == ClaimStatus.CONFLICTING

    flagged = ClaimTriangulator().triangulate(
        [_item("https://click.example/a", "Experts may reveal a shocking secret next year.")]
    )
    assert flagged
    assert "speculation" in flagged[0].flags
    assert "clickbait_title" not in flagged[0].flags  # the title is not clickbait here


def test_brief_markdown_json_and_capability_graph(tmp_path):
    items = [
        _item("https://alpha.example/a", "Acme launched Project Nova in 2024."),
        _item("https://beta.example/b", "Acme launched Project Nova in 2024."),
    ]
    claims = ClaimTriangulator().triangulate(items)
    synthesizer = IntelSynthesizer(_settings(tmp_path))
    output = tmp_path / "brief.md"
    brief = synthesizer.synthesize("Acme", items, claims, output=output)
    assert isinstance(brief, IntelBrief)
    assert "## Executive Summary" in brief.markdown
    assert "## Sources & Evidence Map" in brief.markdown
    assert "What Ultron Checked" in brief.markdown
    assert "What Ultron Did NOT Check" in brief.markdown
    assert brief.markdown_path == output
    assert brief.json_path == output.with_suffix(".json")
    assert json.loads(brief.json_path.read_text())["topic"] == "Acme"
    assert synthesizer.registry.get("intel_report").provides == ["intel.report", "news.raw"]
    assert synthesizer.registry.get("intel_report").requires == []


def test_cli_intel_dispatch_and_telegram_handler(tmp_path, monkeypatch, capsys):
    class FakeEngine:
        def __init__(self, _settings):
            pass

        def research(self, topic, *, depth, output):
            path = Path(output) if output else None
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Fake Intel", encoding="utf-8")
            return IntelBrief(
                topic,
                ["one", "two", "three"],
                [],
                [],
                [],
                {"checked": [], "not_checked": []},
                "# Fake Intel",
                {},
                path,
                path.with_suffix(".json") if path else None,
            )

    monkeypatch.setattr("ultron.main.IntelResearchEngine", FakeEngine)
    parser = build_parser()
    args = parser.parse_args(
        ["intel", "Acme", "--depth", "deep", "--output", str(tmp_path / "cli.md"), "--json"]
    )
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    sent: list[tuple[str, dict]] = []

    def transport(method, _url, kwargs):
        sent.append((method, kwargs))
        return {"ok": True, "result": {}}

    settings = _settings(tmp_path / "telegram")
    client = TelegramBotClient("token", [42], settings=settings, transport=transport)
    cockpit = TelegramCockpit(client, settings=settings, intel_factory=lambda: FakeEngine(settings))
    result = cockpit.handle_update(
        {"message": {"from": {"id": 42}, "chat": {"id": 99}, "text": "/intel Acme"}}
    )
    assert result["command"] == "/intel"
    assert result["topic"] == "Acme"
    assert any(method == "sendDocument" for method, _ in sent)
    assert any(
        "Researching" in kwargs["json"]["text"]
        for method, kwargs in sent
        if method == "sendMessage"
    )
