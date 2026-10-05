from __future__ import annotations

import json
from pathlib import Path

from ultron.config import load_settings
from ultron.soul.identity import SoulIdentity
from ultron.soul.memory import MemoryEngine
from ultron.soul.persona import Persona


def test_identity_persists_profile_and_grounding(tmp_path: Path):
    settings = load_settings(state_dir=tmp_path / "state")
    identity = SoulIdentity(settings)
    identity.update(
        {
            "identity": {"user": "Ada", "mission": "build verified tools"},
            "preferences": {"format": "concise"},
            "assets": ["workspace"],
            "lore": ["Phase five begins with evidence"],
        }
    )
    assert identity.path == tmp_path / "state" / "soul_profile.json"
    assert json.loads(identity.path.read_text(encoding="utf-8"))["identity"]["user"] == "Ada"

    reloaded = SoulIdentity(settings)
    assert reloaded.context()["identity"]["user"] == "Ada"
    assert reloaded.context()["preferences"]["format"] == "concise"
    assert reloaded.ground_goal("solve the task").startswith("For Ada:")


def test_memory_recall_and_semantic_reflection(tmp_path: Path):
    memory = MemoryEngine(tmp_path / "soul.db")
    episode = memory.record_episode(
        "compile a verified report",
        ["collect", "summarize"],
        {"report": "verified report"},
        verified=True,
        answer="verified report",
    )
    assert episode.startswith("episode-")
    recalled = memory.recall("compile a verified report")
    assert recalled is not None
    assert recalled["answer"] == "verified report"
    assert recalled["trajectory"] == ["collect", "summarize"]
    reflection = memory.reflect("verified report")
    assert reflection["nodes"]
    assert any(node["kind"] == "fact" for node in reflection["nodes"])
    assert memory.stats()["verified_episodes"] == 1

    memory.record_episode("unverified claim", ["bad"], {"answer": "no"}, verified=False)
    assert memory.recall("unverified claim") is None


def test_persona_is_direct_and_grounded():
    persona = Persona()
    assert "zero-sycophancy" in persona.traits
    assert "never fabricate" in persona.system_prompt.lower()
    grounded = persona.ground("do the work", {"user": "Ada"})
    assert grounded["identity"]["user"] == "Ada"
    assert "uncertainty" in persona.response_guidance(success=False, verified=False)
