from __future__ import annotations

from pathlib import Path

from ultron.config import REPO_ROOT, load_settings
from ultron.forge import ForgeEngine
from ultron.ledger import FailureLedger
from ultron.registry import Registry
from ultron.scavenger import Scavenger

OAS3_URL = "https://raw.githubusercontent.com/example/apis/weather.yaml"
SWAGGER_URL = "https://api.apis.guru/example/swagger.json"

OAS3 = {
    "openapi": "3.0.0",
    "info": {"title": "Weather API", "description": "fixture weather"},
    "servers": [{"url": "http://fixture.test/v1"}],
    "paths": {"/weather": {"get": {"operationId": "get_weather"}}},
}
SWAGGER = {
    "swagger": "2.0",
    "info": {"title": "Books API"},
    "host": "fixture.test",
    "basePath": "/api",
    "schemes": ["http"],
    "paths": {"/books": {"get": {"operationId": "list_books"}}},
}


def _settings(tmp_path: Path):
    return load_settings(
        state_dir=tmp_path / "state",
        tools_dir=tmp_path / "tools",
        cache_path=tmp_path / "state" / "cache.db",
        memory_path=tmp_path / "state" / "memory.db",
        approvals_file=tmp_path / "state" / "approvals.json",
        audit_log=tmp_path / "state" / "audit.jsonl",
        sandbox_backend="local",
        allow_local_sandbox=True,
        llm_mode="offline",
        repo_root=REPO_ROOT,
    )


def test_discover_enforces_domain_allowlist_and_parses_both_spec_versions(tmp_path):
    documents = {OAS3_URL: OAS3, SWAGGER_URL: SWAGGER}

    def fetch(url):
        return documents[url]

    scavenger = Scavenger(_settings(tmp_path), fetcher=fetch, rate_limit_s=0)
    candidates = scavenger.discover(
        ["https://not-allowed.example/spec.json", OAS3_URL, SWAGGER_URL, OAS3_URL]
    )
    assert [candidate.api_version for candidate in candidates] == ["3.0.0", "2.0"]
    assert candidates[0].endpoint == "http://fixture.test/v1/weather"
    assert candidates[1].endpoint == "http://fixture.test/api/books"
    assert scavenger.rejections[0]["url"].startswith("https://not-allowed")
    assert len({candidate.spec_hash for candidate in candidates}) == 2


def test_wrap_is_stdlib_python_and_forges_with_mock_vector(tmp_path):
    settings = _settings(tmp_path)
    scavenger = Scavenger(settings, rate_limit_s=0)
    candidate = scavenger._candidate_from_document(SWAGGER, SWAGGER_URL)
    assert candidate is not None
    code, manifest = scavenger.wrap(candidate)
    compile(code, "scavenged.py", "exec")
    assert "urllib.request" in code
    assert manifest["risk"] == "medium"
    assert manifest["provides"] == ["api.books_api.result"]

    ledger = FailureLedger(settings.state_dir)
    registry = Registry(settings).load(strict=True)
    engine = ForgeEngine(settings, registry=registry, ledger=ledger, backend="local")
    spec = ledger.record_gap(
        "fixture books",
        expected_outputs={"result": "any"},
        suggested_provides=manifest["provides"],
    )
    temporary = engine.synthesize_tool(spec, code, manifest)
    assert engine.test_tool(
        temporary,
        {"params": {}, "body": {}, "mock_response": {"books": []}},
        {"result": "any"},
    )
    assert engine.registry.get("api_books_api").risk.value == "medium"
