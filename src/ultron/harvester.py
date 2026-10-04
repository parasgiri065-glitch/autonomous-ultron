"""Safe PyPI metadata harvesting into the existing Forge pipeline."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings
from .errors import UltronError
from .forge import ForgeEngine
from .registry import ToolManifest

PYPI_JSON = "https://pypi.org/pypi/{package}/json"
ALLOWED_LICENSES = frozenset({"mit", "apache-2.0", "bsd", "isc", "cc0"})
_FUNCTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


class HarvestError(UltronError):
    """Metadata, license, or wrapper generation failure."""


class LicenseRejected(HarvestError):
    """The package license is not explicitly in the permissive allowlist."""


@dataclass(slots=True)
class PackageMetadata:
    name: str
    version: str
    summary: str
    license: str
    project_urls: dict[str, str] = field(default_factory=dict)
    top_level_modules: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def normalized_name(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_") or "package"

    @property
    def license_key(self) -> str | None:
        return _license_key(self.license)


Fetcher = Callable[[str], dict[str, Any]]


class PyPIHarvester:
    """Inspect permissively licensed packages and hand explicit wrappers to Forge."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        fetcher: Fetcher | None = None,
        forge: ForgeEngine | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.fetcher = fetcher or self._fetch_json
        self.forge = forge or ForgeEngine(self.settings)

    def inspect_package(self, package_name: str) -> PackageMetadata:
        package = _validate_package_name(package_name)
        try:
            document = self.fetcher(PYPI_JSON.format(package=package))
        except Exception as exc:
            raise HarvestError(f"could not inspect PyPI package {package!r}: {exc}") from exc
        if not isinstance(document, dict):
            raise HarvestError("PyPI response was not a JSON object")
        info = document.get("info")
        if not isinstance(info, dict):
            raise HarvestError("PyPI response did not contain an info object")
        metadata = PackageMetadata(
            name=str(info.get("name") or package),
            version=str(info.get("version") or "0.0.0"),
            summary=str(info.get("summary") or ""),
            license=str(info.get("license") or ""),
            project_urls=_project_urls(info.get("project_urls")),
            top_level_modules=_top_level_modules(info, document, package),
            raw=document,
        )
        if metadata.license_key is None:
            raise LicenseRejected(
                f"package {metadata.name!r} has a non-allowlisted or unknown license: "
                f"{metadata.license or 'unspecified'}"
            )
        return metadata

    def synthesize_wrapper(
        self,
        meta: PackageMetadata,
        target_function: str,
        inputs_schema: dict[str, str],
        output_key: str,
    ) -> tuple[str, dict[str, Any]]:
        """Create a small JSON-envelope wrapper without shell or dynamic eval."""
        if not _FUNCTION_RE.fullmatch(target_function) or "__" in target_function:
            raise HarvestError("target_function must be a dotted Python identifier")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,48}", output_key):
            raise HarvestError("output_key must be a lowercase manifest field name")
        if not inputs_schema:
            inputs_schema = {}
        package_module = (
            meta.top_level_modules[0] if meta.top_level_modules else meta.normalized_name
        )
        module_name, function_name = _target_parts(package_module, target_function)
        code = _wrapper_code(module_name, function_name, output_key)
        safe_name = f"pypi_{meta.normalized_name}"[:49].rstrip("_")
        manifest: dict[str, Any] = {
            "name": safe_name,
            "version": "0.1.0",
            "description": f"Safe wrapper for {meta.name}: {meta.summary}"[:500],
            "risk": "medium",
            "provides": [f"pkg.{meta.name}.result"],
            "requires": [],
            "permissions": [],
            "inputs": dict(inputs_schema),
            "outputs": {output_key: "any"},
            "tags": ["pypi", "package", meta.normalized_name],
            "deterministic": True,
            "author": "ultron-harvester",
        }
        return code, manifest

    def harvest_and_forge(
        self,
        package_name: str,
        target_function: str,
        test_inputs: dict[str, Any],
        expected_schema: dict[str, str],
    ) -> ToolManifest:
        """Inspect, wrap, sandbox-test, Breaker-verify, and dynamically register."""
        try:
            metadata = self.inspect_package(package_name)
        except HarvestError as exc:
            self._record_failure(package_name, str(exc))
            raise
        output_key = next(iter(expected_schema), "result")
        try:
            code, manifest = self.synthesize_wrapper(
                metadata,
                target_function,
                {key: _schema_type(value) for key, value in test_inputs.items()},
                output_key,
            )
            manifest["inputs"] = _infer_inputs_from_metadata(test_inputs, manifest["inputs"])
            manifest["outputs"] = dict(expected_schema)
            spec = self.forge.ledger.record_gap(
                f"pypi:{metadata.name}:{target_function}",
                required_inputs=dict(manifest["inputs"]),
                expected_outputs=dict(expected_schema),
                suggested_provides=list(manifest["provides"]),
                failure_reason="PyPI wrapper harvest",
            )
            temporary = self.forge.synthesize_tool(spec, code, manifest)
            if not self.forge.test_tool(temporary, test_inputs, expected_schema):
                raise HarvestError(f"sandbox or Breaker rejected wrapper for {metadata.name}")
            return self.forge.registry.get(temporary.name)
        except Exception as exc:
            if isinstance(exc, HarvestError):
                error = str(exc)
            else:
                error = f"harvest failed: {type(exc).__name__}: {exc}"
            self._record_failure(package_name, error)
            raise exc if isinstance(exc, HarvestError) else HarvestError(error) from exc

    def _record_failure(self, package_name: str, reason: str) -> None:
        self.forge.ledger.record_gap(
            f"pypi:{package_name}",
            expected_outputs={"result": "any"},
            failure_reason=reason,
        )

    @staticmethod
    def _fetch_json(url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": "ultron-harvester/3.1"})
        try:
            with urllib.request.urlopen(request, timeout=15.0) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise HarvestError(f"PyPI request failed: {exc}") from exc
        if not isinstance(value, dict):
            raise HarvestError("PyPI response was not an object")
        return value


def harvest_and_forge(
    package_name: str,
    target_function: str,
    test_inputs: dict[str, Any],
    expected_schema: dict[str, str],
) -> ToolManifest:
    """Convenience entry point using the process settings and default Forge."""
    return PyPIHarvester().harvest_and_forge(
        package_name, target_function, test_inputs, expected_schema
    )


def _validate_package_name(value: str) -> str:
    package = (value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,126}", package):
        raise HarvestError("invalid PyPI package name")
    return package


def _project_urls(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if isinstance(item, str)}


def _top_level_modules(info: dict[str, Any], document: dict[str, Any], package: str) -> list[str]:
    for key in ("top_level_modules", "top_level", "modules"):
        value = info.get(key, document.get(key))
        if isinstance(value, list):
            modules = [
                str(item)
                for item in value
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", str(item))
            ]
            if modules:
                return modules
    return [re.sub(r"[-.]", "_", str(info.get("name") or package)).lower()]


def _license_key(value: str) -> str | None:
    text = re.sub(r"[®™]", "", (value or "").casefold()).strip()
    if not text or any(marker in text for marker in ("gpl", "agpl", "lgpl", "proprietary")):
        return None
    candidate: str | None = None
    if "apache" in text and ("2" in text or "software" in text):
        candidate = "apache-2.0"
    elif text == "mit" or "mit license" in text:
        candidate = "mit"
    elif "isc" in text:
        candidate = "isc"
    elif "cc0" in text or "creative commons zero" in text:
        candidate = "cc0"
    elif "bsd" in text:
        candidate = "bsd"
    return candidate if candidate in ALLOWED_LICENSES else None


def _target_parts(package_module: str, target_function: str) -> tuple[str, str]:
    pieces = target_function.split(".")
    if len(pieces) == 1:
        return package_module, pieces[0]
    return ".".join([package_module, *pieces[:-1]]), pieces[-1]


def _wrapper_code(module_name: str, function_name: str, output_key: str) -> str:
    return f"""import importlib
import json
import sys

MODULE = {module_name!r}
FUNCTION = {function_name!r}
OUTPUT_KEY = {output_key!r}

def run(payload):
    if not isinstance(payload, dict):
        raise TypeError("inputs must be a JSON object")
    module = importlib.import_module(MODULE)
    function = getattr(module, FUNCTION)
    try:
        value = function(**payload)
    except TypeError as keyword_error:
        # Existing Ultron tools use the standard run(payload) contract;
        # ordinary package functions commonly use keyword arguments. Support
        # both without eval, exec, or shell interpolation.
        try:
            value = function(payload)
        except TypeError:
            raise keyword_error
    return {{OUTPUT_KEY: value}}

def main():
    try:
        payload = json.loads(sys.stdin.read() or "{{}}")
        print(json.dumps({{"ok": True, "result": run(payload)}}, default=str))
    except Exception as exc:
        print(json.dumps({{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
"""


def _schema_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list[string]"
    if isinstance(value, dict):
        return "dict"
    return "string"


def _infer_inputs_from_metadata(
    test_inputs: dict[str, Any], schema: dict[str, str]
) -> dict[str, str]:
    return {key: schema.get(key, _schema_type(value)) for key, value in test_inputs.items()}
