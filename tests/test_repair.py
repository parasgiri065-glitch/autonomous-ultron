from __future__ import annotations

import os
from pathlib import Path

import pytest

from ultron.config import REPO_ROOT, load_settings
from ultron.repair import RepairEngine

pytestmark = pytest.mark.skipif(
    os.environ.get("ULTRON_REPAIR_VALIDATION") == "1",
    reason="repair validator must not recursively validate itself",
)


PASS_PATCH = """\
diff --git a/tests/test_repair_candidate.py b/tests/test_repair_candidate.py
new file mode 100644
--- /dev/null
+++ b/tests/test_repair_candidate.py
@@ -0,0 +1,2 @@
+def test_candidate_patch_is_validated():
+    assert 2 + 2 == 4
"""

FAIL_PATCH = """\
diff --git a/tests/test_repair_candidate.py b/tests/test_repair_candidate.py
new file mode 100644
--- /dev/null
+++ b/tests/test_repair_candidate.py
@@ -0,0 +1,2 @@
+def test_candidate_patch_is_rejected():
+    assert False
"""


def _settings(tmp_path: Path):
    return load_settings(
        state_dir=tmp_path / "state",
        cache_path=tmp_path / "state" / "cache.db",
        memory_path=tmp_path / "state" / "memory.db",
        approvals_file=tmp_path / "state" / "approvals.json",
        audit_log=tmp_path / "state" / "audit.jsonl",
        llm_mode="offline",
        repo_root=REPO_ROOT,
    )


def test_repair_ticket_records_and_validates_candidate_patch(tmp_path):
    engine = RepairEngine(_settings(tmp_path))
    ticket = engine.record_fault(
        "synthetic.component",
        "synthetic failure",
        "Traceback (most recent call last): ...",
        {
            "test_path": "tests/test_repair_candidate.py",
            "test_name": "test_candidate_patch_is_validated",
        },
    )
    assert ticket.is_open
    assert engine.validate_patch(ticket, PASS_PATCH) is True
    assert ticket.status == "validated"
    assert ticket.validation["pytest"]["ok"] is True
    assert ticket.validation["eval"]["ok"] is True
    assert ticket.validation["target_fault"]["ok"] is True
    assert len(engine.open_tickets()) == 0
    assert engine.path.exists()


def test_repair_rejects_patch_when_candidate_tests_fail(tmp_path):
    engine = RepairEngine(_settings(tmp_path))
    ticket = engine.record_fault("synthetic.component", "broken", "trace", {})
    assert engine.validate_patch(ticket, FAIL_PATCH) is False
    assert ticket.status == "rejected"
    assert ticket.failure_reason == "pytest failed"
    assert engine.open_tickets() == []
