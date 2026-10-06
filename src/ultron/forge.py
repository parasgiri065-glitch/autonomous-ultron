"""Phase 2.3 Forge scaffolding: validate, sandbox-test, and register tools."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .breaker import BreakerVerifier
from .charter import Charter
from .config import Settings, get_settings
from .errors import ManifestError
from .ledger import FailureLedger, MissingCapabilitySpec
from .policy import AutoApprovePrompter, PolicyGate, PolicyRequest
from .registry import Registry, ToolManifest
from .sandbox import Sandbox
from .verifier import Verifier


@dataclass(slots=True)
class ForgeArtifact:
    spec: MissingCapabilitySpec
    temp_dir: Path
    script_path: Path
    manifest_path: Path
    test_inputs: dict[str, Any]
    expected_schema: dict[str, str]


class ForgeEngine:
    """Safe v0 forge path; code is explicit and never generated here."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        registry: Registry | None = None,
        ledger: FailureLedger | None = None,
        backend: Literal["docker", "local"] | None = None,
        charter: Charter | None = None,
        templates: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.registry = registry or Registry(self.settings).load()
        self.ledger = ledger or FailureLedger(self.settings.state_dir)
        self.backend = backend or self.settings.sandbox_backend
        self.charter = charter or Charter(self.settings.state_dir)
        self.templates = dict(templates or {})
        self._artifacts: dict[str, ForgeArtifact] = {}
        self.last_report: dict[str, list[dict[str, str]]] = {
            "forged": [],
            "failed": [],
            "skipped": [],
        }
        self.last_test_error = ""

    def add_template(
        self,
        goal: str,
        code_str: str,
        manifest_dict: dict[str, Any],
        *,
        test_inputs: dict[str, Any] | None = None,
        expected_schema: dict[str, str] | None = None,
    ) -> None:
        """Provide explicit v0 code for a ledger goal."""
        self.templates[goal] = (
            code_str,
            {
                "manifest": dict(manifest_dict),
                "test_inputs": dict(test_inputs or {}),
                "expected_schema": dict(expected_schema or {}),
            },
        )

    def synthesize_tool(
        self,
        spec: MissingCapabilitySpec,
        code_str: str,
        manifest_dict: dict[str, Any],
    ) -> ToolManifest:
        """Write a validated temporary implementation and manifest."""
        if not code_str.strip():
            raise ManifestError("forged implementation is empty")
        data = dict(manifest_dict)
        data.setdefault("name", _name_for_spec(spec))
        data.setdefault("version", "0.1.0")
        data.setdefault("description", f"Forged capability for: {spec.goal[:300]}")
        data.setdefault("inputs", dict(spec.required_inputs))
        data.setdefault("outputs", dict(spec.expected_outputs))
        data["provides"] = list(spec.suggested_provides)
        data["requires"] = list(spec.suggested_requires)
        # v0 has no autonomous trust escalation: every forged tool is medium.
        data["risk"] = "medium"

        forge_root = Path(self.settings.state_dir) / "forge"
        forge_root.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", str(data["name"]))[:80] or "tool"
        temp_dir = forge_root / f"{safe_stem}-{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir()
        script_path = temp_dir / f"{safe_stem}.py"
        manifest_path = temp_dir / f"{safe_stem}.json"
        script_path.write_text(code_str, encoding="utf-8")
        try:
            relative_script = script_path.resolve().relative_to(self.settings.repo_root.resolve())
            entrypoint = f"python {relative_script.as_posix()}"
        except ValueError:
            entrypoint = f"python {script_path}"
        data["entrypoint"] = entrypoint
        data["source_path"] = str(manifest_path)
        try:
            manifest = ToolManifest(**data)
            schema_errors = Registry(self.settings).validate_against_schema(manifest)
            if schema_errors:
                raise ManifestError(
                    "forged manifest failed schema validation: " + "; ".join(schema_errors)
                )
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        manifest_path.write_text(
            json.dumps(_manifest_payload(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._artifacts[manifest.key] = ForgeArtifact(
            spec=spec,
            temp_dir=temp_dir,
            script_path=script_path,
            manifest_path=manifest_path,
            test_inputs={},
            expected_schema=dict(manifest.outputs),
        )
        return manifest

    def test_tool(
        self,
        manifest: ToolManifest,
        test_inputs: dict[str, Any],
        expected_schema: dict[str, str],
    ) -> bool:
        """Run, verify, and register a forged tool; discard failures."""
        self.last_test_error = ""
        artifact = self._artifacts.get(manifest.key)
        if artifact is None:
            raise ManifestError(f"no temporary forge artifact for {manifest.key}")
        artifact.test_inputs = dict(test_inputs)
        artifact.expected_schema = dict(expected_schema)
        settings = self.settings.model_copy(update={"policy_network_low_auto": False})
        sandbox = Sandbox(settings, backend=self.backend, run_id=f"forge-{uuid.uuid4().hex[:8]}")
        gate = PolicyGate(
            settings,
            prompter=AutoApprovePrompter(),
            charter=self.charter,
            run_id=sandbox.run_id,
        )
        request = PolicyRequest(
            tool=manifest,
            inputs=test_inputs,
            run_id=sandbox.run_id,
            goal=artifact.spec.goal,
            action_type="forge_test",
            # v0 forge tests are offline by construction. A future explicit
            # operator approval path may change this request, but a template
            # cannot silently request egress.
            allow_network=False,
        )
        try:
            decision = gate.evaluate(request, interactive=True)
            outcome = sandbox.run(
                manifest, test_inputs, decision, use_cache=False, defer_cache_write=True
            )
            verify_manifest = manifest.model_copy(update={"outputs": dict(expected_schema)})
            verifier = Verifier(settings=settings, breaker=BreakerVerifier())
            verification = verifier.verify(
                verify_manifest,
                outcome.result,
                goal=artifact.spec.goal,
                provenance=outcome.provenance,
            )
            breaker = BreakerVerifier().verify(outcome.result, outcome.provenance)
            passed = bool(outcome.ok and verification.ok and breaker.ok)
            # Prefer the sandbox error: a failed run (e.g. pip resolver failure
            # inside the container) is the real cause, while the verifier's
            # "no result object" is only its downstream symptom.
            reason = outcome.error or (
                verification.reason if not verification.ok else breaker.reason
            )
            if passed:
                for item in outcome.provenance:
                    if item.origin in {"sandbox_tool", "web_fetch"}:
                        item.verified = True
                sandbox.commit_cache(outcome)
                self.ledger.record_attempt(artifact.spec, failure_reason="forge verified")
                self._register_artifact(manifest, artifact)
                return True
            self.last_test_error = reason or outcome.error or "forge verification failed"
            self._discard_artifact(artifact, self.last_test_error)
            return False
        except Exception as exc:
            self.last_test_error = f"forge test error: {type(exc).__name__}: {exc}"
            self._discard_artifact(artifact, self.last_test_error)
            return False

    def auto_forge(self, **kwargs: Any) -> list[ToolManifest]:
        """Compatibility-facing name for the bounded ledger forge pass."""
        return self.auto_forge_from_ledger(**kwargs)

    def auto_forge_from_ledger(
        self,
        *,
        top_n: int = 3,
        code_str: str | None = None,
        manifest_dict: dict[str, Any] | None = None,
        test_inputs: dict[str, Any] | None = None,
        expected_schema: dict[str, str] | None = None,
    ) -> list[ToolManifest]:
        forged: list[ToolManifest] = []
        self.last_report = {"forged": [], "failed": [], "skipped": []}
        for spec in self.ledger.top(top_n):
            if spec.attempt_count >= 3:
                self.last_report["skipped"].append(
                    {"goal": spec.goal, "reason": "attempt_count >= 3"}
                )
                continue
            template = self.templates.get(spec.goal)
            supplied_code = code_str
            supplied_manifest = manifest_dict
            supplied_inputs = test_inputs
            supplied_schema = expected_schema
            if template:
                if isinstance(template, tuple):
                    supplied_code, details = template
                else:
                    supplied_code = template.get("code") or template.get("code_str")
                    details = template
                supplied_manifest = details.get("manifest")
                supplied_inputs = details.get("test_inputs")
                supplied_schema = details.get("expected_schema")
            if not supplied_code:
                self.last_report["skipped"].append(
                    {"goal": spec.goal, "reason": "no explicit code (v0)"}
                )
                continue  # v0 never asks an LLM to invent implementation code
            try:
                manifest = self.synthesize_tool(spec, supplied_code, supplied_manifest or {})
                inputs = supplied_inputs or _sample_inputs(manifest.inputs)
                schema = supplied_schema or manifest.outputs
                if self.test_tool(manifest, inputs, schema):
                    final = self.registry.get(manifest.name)
                    forged.append(final)
                    self.last_report["forged"].append({"goal": spec.goal, "tool": final.key})
                else:
                    self.last_report["failed"].append(
                        {"goal": spec.goal, "reason": "sandbox or breaker rejection"}
                    )
            except (ManifestError, ValueError) as exc:
                self.ledger.record_attempt(spec, failure_reason=f"synthesis failed: {exc}")
                self.last_report["failed"].append({"goal": spec.goal, "reason": str(exc)})
        return forged

    def _register_artifact(self, manifest: ToolManifest, artifact: ForgeArtifact) -> ToolManifest:
        dynamic = Path(self.settings.tools_dir) / "dynamic"
        dynamic.mkdir(parents=True, exist_ok=True)
        final_script = dynamic / artifact.script_path.name
        final_manifest_path = dynamic / artifact.manifest_path.name
        shutil.move(str(artifact.script_path), str(final_script))
        try:
            relative_script = final_script.resolve().relative_to(self.settings.repo_root.resolve())
            script_arg = relative_script.as_posix()
        except ValueError:
            script_arg = str(final_script.resolve())
        final_entrypoint = f"python {script_arg}"
        final = manifest.model_copy(
            update={"entrypoint": final_entrypoint, "source_path": str(final_manifest_path)}
        )
        final_manifest_path.write_text(
            json.dumps(_manifest_payload(final), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.registry.register(final)
        self.charter.log("tool_registration", detail=final.key)
        shutil.rmtree(artifact.temp_dir, ignore_errors=True)
        self._artifacts.pop(manifest.key, None)
        return final

    def _discard_artifact(self, artifact: ForgeArtifact, reason: str) -> None:
        self.ledger.record_attempt(artifact.spec, failure_reason=reason)
        shutil.rmtree(artifact.temp_dir, ignore_errors=True)
        for key, item in list(self._artifacts.items()):
            if item is artifact:
                self._artifacts.pop(key, None)


def _manifest_payload(manifest: ToolManifest) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    payload.pop("source_path", None)
    return payload


def _name_for_spec(spec: MissingCapabilitySpec) -> str:
    candidates = spec.suggested_provides or [spec.goal]
    slug = re.sub(r"[^a-z0-9]+", "_", candidates[0].lower()).strip("_")
    return ("forge_" + (slug or "tool"))[:48].rstrip("_")


def _sample_inputs(schema: dict[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, typ in schema.items():
        values[name] = {
            "string": "forge-test",
            "int": 1,
            "float": 1.0,
            "bool": True,
            "list[string]": ["forge-test"],
            "list[float]": [1.0],
            "list[int]": [1],
            "dict": {},
            "any": "forge-test",
        }.get(typ, "forge-test")
    return values
