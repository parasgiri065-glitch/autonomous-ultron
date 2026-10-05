from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultron.config import load_settings
from ultron.extractor import (
    ExtractionError,
    ExtractionPolicyError,
    FetchResponse,
    GroundedDataExtractor,
    normalize_schema,
    validate_public_url,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_html_extraction_is_typed_grounded_and_preserves_evidence():
    result = GroundedDataExtractor().extract(
        FIXTURES / "extract_product.html",
        {"name": "string", "price": "float", "category": "string", "in_stock": "bool", "missing": "string"},
    )
    assert result.ok
    assert result.fields == {
        "name": "Acme Widget",
        "price": 1299.5,
        "category": "Hardware",
        "in_stock": True,
        "missing": None,
    }
    assert result.evidence["price"].selector == "div.price"
    assert result.evidence["category"].source.endswith("extract_product.html")
    assert result.evidence["in_stock"].evidence == "yes"
    assert result.provenance[0].origin == "local_file"


def test_json_nested_and_csv_list_extraction():
    json_result = GroundedDataExtractor().extract(
        FIXTURES / "extract_profile.json",
        {"name": "string", "age": "int", "contact.email": "string", "missing": "string"},
    )
    assert json_result.ok
    assert json_result.fields["contact.email"] == "ada@example.test"
    assert json_result.fields["missing"] is None
    assert json_result.evidence["contact.email"].selector == "$.contact.email"

    csv_result = GroundedDataExtractor().extract(
        FIXTURES / "extract_people.csv",
        {"name": "list[string]", "score": "list[float]", "active": "list[bool]"},
    )
    assert csv_result.fields["name"] == ["Ada Lovelace", "Grace Hopper"]
    assert csv_result.fields["score"] == [98.5, 97.0]


def test_text_and_missing_values_are_not_invented():
    result = GroundedDataExtractor().extract(
        FIXTURES / "extract_notes.txt",
        {"project": "string", "owner": "string", "unknown": "string"},
    )
    assert result.ok
    assert result.fields == {"project": "Grounded Extractor", "owner": "Ada Lovelace", "unknown": None}
    assert result.evidence["unknown"].evidence == "not found"
    assert result.evidence["unknown"].source.endswith("extract_notes.txt")


def test_schema_and_input_fail_closed(tmp_path: Path):
    with pytest.raises(ExtractionError, match="non-empty JSON object"):
        normalize_schema({})
    with pytest.raises(ExtractionError, match="unsupported type"):
        normalize_schema({"field": "date"})
    with pytest.raises(ExtractionError, match="does not exist"):
        GroundedDataExtractor().extract(tmp_path / "nope.txt", {"x": "string"})
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"name":', encoding="utf-8")
    with pytest.raises(ExtractionError, match="could not parse"):
        GroundedDataExtractor().extract(malformed, {"name": "string"})
    oversized = tmp_path / "large.txt"
    oversized.write_text("x" * 20)
    with pytest.raises(ExtractionError, match="byte limit"):
        GroundedDataExtractor(max_response_bytes=10).extract(oversized, {"x": "string"})


def _public_resolver(host: str, port: int, **_: object):
    return [(2, 1, 6, "", ("93.184.216.34", port))]


def test_url_credentials_private_destinations_and_unsafe_redirects_are_rejected():
    with pytest.raises(ExtractionError, match="credentials"):
        validate_public_url("https://user:pass@example.com/page", resolver=_public_resolver)
    for url in ("http://127.0.0.1/", "http://10.0.0.1/", "http://localhost/"):
        with pytest.raises(ExtractionError):
            validate_public_url(url, resolver=_public_resolver)

    calls: list[str] = []

    def redirect_fetch(url: str):
        calls.append(url)
        return FetchResponse(url, b"", 302, {"location": "http://127.0.0.1/admin"})

    with pytest.raises(ExtractionError, match="forbidden"):
        GroundedDataExtractor(fetcher=redirect_fetch, resolver=_public_resolver).extract(
            "https://example.com", {"name": "string"}
        )
    assert calls == ["https://example.com/"]


def test_fetch_limits_timeout_429_and_redirect_limit_are_polite():
    def too_large(_url: str):
        return FetchResponse(_url, b"x" * 11)

    with pytest.raises(ExtractionError, match="byte limit"):
        GroundedDataExtractor(fetcher=too_large, resolver=_public_resolver, max_response_bytes=10).extract(
            "https://example.com", {"name": "string"}
        )

    def timed_out(_url: str):
        raise TimeoutError("deadline")

    with pytest.raises(ExtractionError, match="safe fetch failed"):
        GroundedDataExtractor(fetcher=timed_out, resolver=_public_resolver).extract(
            "https://example.com", {"name": "string"}
        )

    def rate_limited(_url: str):
        return FetchResponse(_url, b"", 429, {"retry-after": "60"})

    with pytest.raises(ExtractionError, match="retry after 60"):
        GroundedDataExtractor(fetcher=rate_limited, resolver=_public_resolver).extract(
            "https://example.com", {"name": "string"}
        )

    def redirect_loop(url: str):
        return FetchResponse(url, b"", 302, {"location": "https://example.com/next"})

    with pytest.raises(ExtractionError, match="redirect limit"):
        GroundedDataExtractor(fetcher=redirect_loop, resolver=_public_resolver, max_redirects=1).extract(
            "https://example.com", {"name": "string"}
        )


def test_live_fetch_requires_opt_in_and_policy_approval(tmp_path: Path):
    settings = load_settings(
        state_dir=tmp_path / "state",
        approvals_file=tmp_path / "state" / "approvals.json",
        audit_log=tmp_path / "state" / "audit.jsonl",
        eval_live=False,
        policy_network="deny",
    )
    with pytest.raises(ExtractionPolicyError, match="live network is disabled"):
        GroundedDataExtractor(settings, resolver=_public_resolver).extract(
            "https://example.com", {"name": "string"}
        )

    approved_settings = load_settings(
        state_dir=tmp_path / "approved-state",
        approvals_file=tmp_path / "approved-state" / "approvals.json",
        audit_log=tmp_path / "approved-state" / "audit.jsonl",
        eval_live=True,
        policy_network="deny",
    )
    with pytest.raises(ExtractionPolicyError, match="requires explicit approval"):
        GroundedDataExtractor(approved_settings, resolver=_public_resolver).extract(
            "https://example.com", {"name": "string"}
        )


def test_injected_fetcher_is_offline_and_redirects_revalidate():
    body = json.dumps({"name": "Offline page"}).encode()

    def fetch(url: str):
        return FetchResponse(url, body, 200, {"content-type": "application/json"})

    result = GroundedDataExtractor(fetcher=fetch, resolver=_public_resolver).extract(
        "https://example.com/data.json", {"name": "string"}
    )
    assert result.ok
    assert result.fields["name"] == "Offline page"
    assert json.loads(result.render("json"))["evidence"]["name"]["source"] == "https://example.com/data.json"
