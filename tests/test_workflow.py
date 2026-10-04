from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "compounding.yml"


def test_compounding_workflow_is_pr_only_and_does_not_reference_secrets():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'cron: "0 2 * * *"' in text
    assert "workflow_dispatch" in text
    assert "gh pr create" in text
    assert 'git push --set-upstream origin "$branch"' in text
    assert "git merge" not in text
    assert "gh pr merge" not in text
    assert "secrets." not in text
    assert "OPENAI_API_KEY" not in text
    assert "GITHUB_TOKEN" not in text
    assert "github.token" in text
    assert "ALLOW_SCAVENGE" in text
    assert 'ULTRON_SCAVENGE: "0"' in text
