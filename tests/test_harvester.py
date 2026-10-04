from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ultron.config import REPO_ROOT, load_settings
from ultron.forge import ForgeEngine
from ultron.harvester import HarvestError, LicenseRejected, PyPIHarvester
from ultron.ledger import FailureLedger
from ultron.registry import Registry


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


def fixture(name: str, license_name: str = "MIT"):
    return {
        "info": {
            "name": name,
            "version": "1.2.3",
            "summary": "A deterministic fixture package",
            "license": license_name,
            "project_urls": {"Source": "https://example.invalid/source"},
            "top_level_modules": ["tools.calc"],
        }
    }


def test_inspect_extracts_metadata_and_enforces_license(tmp_path):
    responses = {"https://pypi.org/pypi/demo/json": fixture("demo")}
    harvester = PyPIHarvester(_settings(tmp_path), fetcher=responses.__getitem__)
    metadata = harvester.inspect_package("demo")
    assert metadata.name == "demo"
    assert metadata.version == "1.2.3"
    assert metadata.project_urls["Source"].startswith("https://")
    assert metadata.top_level_modules == ["tools.calc"]

    rejected = PyPIHarvester(
        _settings(tmp_path / "reject"), fetcher=lambda _url: fixture("bad", "GPL-3.0-only")
    )
    with pytest.raises(LicenseRejected):
        rejected.inspect_package("bad")

    apache = PyPIHarvester(
        _settings(tmp_path / "apache"),
        fetcher=lambda _url: fixture("apache", "Apache Software License"),
    )
    assert apache.inspect_package("apache").license_key == "apache-2.0"


def test_wrapper_is_valid_python_and_has_safe_error_envelope(tmp_path):
    harvester = PyPIHarvester(_settings(tmp_path), fetcher=lambda _url: fixture("demo"))
    metadata = harvester.inspect_package("demo")
    code, manifest = harvester.synthesize_wrapper(
        metadata, "run", {"expression": "string"}, "result"
    )
    compile(code, "pypi_wrapper.py", "exec")
    assert "except Exception" in code
    assert manifest["name"] == "pypi_demo"
    assert manifest["risk"] == "medium"
    assert manifest["permissions"] == []
    assert manifest["requires"] == []
    assert manifest["provides"] == ["pkg.demo.result"]

    script = tmp_path / "wrapper.py"
    script.write_text(code, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    failed = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps({"expression": "not valid"}),
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    assert failed.returncode != 0
    assert json.loads(failed.stdout)["ok"] is False

    with pytest.raises(HarvestError):
        harvester.synthesize_wrapper(metadata, "__import__", {}, "result")


def test_harvest_forges_and_registers_tools_for_graph_bfs(tmp_path):
    settings = _settings(tmp_path)
    ledger = FailureLedger(settings.state_dir)
    registry = Registry(settings).load(strict=True)
    forge = ForgeEngine(settings, registry=registry, ledger=ledger, backend="local")
    harvester = PyPIHarvester(
        settings,
        fetcher=lambda _url: fixture("tools"),
        forge=forge,
    )
    tool = harvester.harvest_and_forge(
        "tools",
        "run",
        {"expression": "2+2"},
        {"result": "dict"},
    )
    assert tool.name == "pypi_tools"
    assert tool.risk.value == "medium"
    assert tool.provides == ["pkg.tools.result"]
    assert tool.requires == []
    assert "pypi_tools" in forge.registry.graph()
    assert forge.registry.find_chain([], ["pkg.tools.result"]) == ["pypi_tools"]
