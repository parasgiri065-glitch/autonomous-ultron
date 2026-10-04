from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ultron.config import REPO_ROOT, load_settings
from ultron.forge import ForgeEngine
from ultron.ledger import FailureLedger
from ultron.refinery import ToolRefinery
from ultron.registry import Registry

RAW_SCRIPT = """
import argparse
import sys
import tkinter

def run(payload):
    print("debug output must not escape")
    return {"result": payload["value"] * 2}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print(parser.parse_args())
    sys.exit(1)
"""


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


def test_refinery_ast_strips_cli_noise_and_adds_error_envelope(tmp_path):
    refinery = ToolRefinery(_settings(tmp_path))
    code, manifest = refinery.refine_code(
        RAW_SCRIPT,
        "run",
        ["demo.result:int"],
        ["demo.input"],
    )
    compile(code, "refined_tool.py", "exec")
    assert "argparse" not in code
    assert "sys.exit" not in code
    assert "tkinter" not in code
    assert "debug output must not escape" not in code
    assert '"trace"' in code
    assert manifest["risk"] == "medium"
    assert manifest["provides"] == ["demo.result:int"]
    assert manifest["requires"] == ["demo.input"]
    assert manifest["entrypoint"] == "python refined_tool.py"

    script = tmp_path / "refined.py"
    script.write_text(code, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps({"value": 3}),
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.splitlines() == [json.dumps({"ok": True, "result": {"result": 6}})]

    failed = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps({}),
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    error = json.loads(failed.stdout)
    assert failed.returncode != 0
    assert error["ok"] is False
    assert error["error"]
    assert error["trace"]


def test_cook_sandbox_tests_and_registers_capability(tmp_path):
    settings = _settings(tmp_path)
    ledger = FailureLedger(settings.state_dir)
    registry = Registry(settings).load(strict=True)
    forge = ForgeEngine(settings, registry=registry, ledger=ledger, backend="local")
    refinery = ToolRefinery(settings, forge=forge)
    tool = refinery.cook_and_register(
        RAW_SCRIPT,
        "run",
        {"value": 5},
        {"result": "int"},
    )
    assert tool.name == "refined_run"
    assert tool.risk.value == "medium"
    assert tool.provides == ["refined.result"]
    assert tool.requires == []
    assert forge.registry.find_chain([], ["refined.result"]) == ["refined_run"]
