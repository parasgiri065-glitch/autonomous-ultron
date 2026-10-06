from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from ultron.config import REPO_ROOT, load_settings
from ultron.engine.wheel_fetch import WheelFetcher, WheelFetchError
from ultron.policy import PolicyDecision
from ultron.registry import ToolManifest
from ultron.sandbox import Sandbox


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str, *, content_length: int | None = None):
        super().__init__(payload)
        self._url = url
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _settings(tmp_path: Path):
    return load_settings(
        state_dir=tmp_path / "state",
        tools_dir=REPO_ROOT / "tools",
        cache_path=tmp_path / "state" / "cache.db",
        memory_path=tmp_path / "state" / "memory.db",
        approvals_file=tmp_path / "state" / "approvals.json",
        audit_log=tmp_path / "state" / "audit.jsonl",
        sandbox_backend="docker",
        allow_local_sandbox=True,
        llm_mode="offline",
    )


def _opener(*, artifact_url: str, artifact: bytes, content_length: int | None = None):
    metadata = {
        "info": {"name": "fixturepkg", "version": "1.2.3"},
        "urls": [{"filename": "fixturepkg-1.2.3-py3-none-any.whl", "url": artifact_url}],
    }

    def open_url(request, timeout=0):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        if url.startswith("https://pypi.org/"):
            return _Response(json.dumps(metadata).encode(), url)
        return _Response(artifact, url, content_length=content_length)

    return open_url


def test_rejects_non_allowlisted_artifact_host(tmp_path: Path):
    settings = _settings(tmp_path)
    opener = _opener(
        artifact_url="https://evil.example/fixturepkg.whl",
        artifact=b"wheel",
    )
    with pytest.raises(WheelFetchError, match="non-allowlisted"):
        WheelFetcher(settings, opener=opener).fetch("fixturepkg")


def test_rejects_package_path_chars(tmp_path: Path):
    fetcher = WheelFetcher(_settings(tmp_path), opener=lambda *_args, **_kwargs: None)
    with pytest.raises(WheelFetchError, match="invalid PyPI package name"):
        fetcher.fetch("../fixturepkg")
    with pytest.raises(WheelFetchError, match="invalid PyPI package name"):
        fetcher.fetch("fixture/pkg")


def test_enforces_size_cap(tmp_path: Path):
    settings = _settings(tmp_path)
    opener = _opener(
        artifact_url="https://files.pythonhosted.org/packages/fixturepkg.whl",
        artifact=b"123456",
        content_length=6,
    )
    with pytest.raises(WheelFetchError, match="size cap"):
        WheelFetcher(settings, opener=opener, max_bytes=5).fetch("fixturepkg")


def test_records_and_verifies_sha256(tmp_path: Path):
    settings = _settings(tmp_path)
    artifact = b"offline wheel fixture"
    fetcher = WheelFetcher(
        settings,
        opener=_opener(
            artifact_url="https://files.pythonhosted.org/packages/fixturepkg-1.2.3-py3-none-any.whl",
            artifact=artifact,
        ),
    )
    path = fetcher.fetch("fixturepkg", version="1.2.3")
    record = fetcher.record(path)
    assert record["sha256"] == hashlib.sha256(artifact).hexdigest()
    assert record["path"] == str(path)
    assert fetcher.record_path(path).exists()
    path.write_bytes(b"tampered")
    with pytest.raises(WheelFetchError, match="sha256 mismatch"):
        fetcher.verify(path)


def test_wheel_is_mounted_read_only_and_network_stays_sealed(tmp_path: Path):
    settings = _settings(tmp_path)
    wheel = settings.state_dir / "wheels" / "fixturepkg-1.2.3-py3-none-any.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"fixture")
    record = {
        "package": "fixturepkg",
        "version": "1.2.3",
        "filename": wheel.name,
        "path": str(wheel),
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }
    manifest = ToolManifest(
        name="wheel_fixture",
        version="0.1.0",
        entrypoint="python tool.py",
        risk="low",
        meta={"dependencies": {"wheels": [record]}},
    )
    decision = PolicyDecision(
        action="allow",
        reason="offline fixture",
        risk="low",
        tool=manifest.name,
        version=manifest.version,
        granted=True,
        network="none",
    )
    argv = Sandbox(settings, backend="docker").docker_command_preview(manifest, decision)
    joined = " ".join(argv)
    assert "--network=none" in argv
    assert f"{wheel}:/wheels/{wheel.name}:ro" in argv
    assert "pip install" in joined
    assert "--no-index" in joined
    assert "--find-links /wheels" in joined
    assert "fixturepkg==1.2.3" in joined
    assert "--network=bridge" not in argv
