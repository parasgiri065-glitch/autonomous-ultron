"""Host-side, sealed wheel acquisition for JIT tools.

The sandbox never performs package discovery or network access.  This module is
called by the host before a forge test, validates PyPI metadata and artifact
hosts, records a digest, and hands the sandbox only a read-only wheel mount.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from ..config import Settings, get_settings
from ..errors import UltronError

ALLOWED_HOSTS = frozenset({"pypi.org", "files.pythonhosted.org"})
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_PACKAGES = 24
_PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,126}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.!+~-]{0,126}$")
_ARTIFACT_SUFFIXES = (".whl", ".tar.gz", ".zip")
# Base-install marker environment: everything PEP 508 defines on the host, with
# ``extra`` pinned to "" so extra-conditional requirements evaluate false
# exactly as pip treats them for a plain (no-extras) install.
_MARKER_ENV = {**default_environment(), "extra": ""}


class WheelFetchError(UltronError):
    """A wheel failed host-side validation or acquisition."""


Opener = Callable[..., Any]


class WheelFetcher:
    """Fetch and attest PyPI wheels on the host, never from a sandbox."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        opener: Opener | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.settings = settings or get_settings()
        self.opener = opener or urllib.request.urlopen
        self.max_bytes = int(max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.wheels_dir = Path(self.settings.state_dir) / "wheels"
        self.wheels_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, package: str, version: str | None = None) -> Path:
        """Download an allowlisted PyPI artifact and write its SHA-256 attestation."""
        path, _record, _info = self._fetch_one(package, version)
        return path

    def fetch_closure(
        self,
        package: str,
        version: str | None = None,
        *,
        max_packages: int = DEFAULT_MAX_PACKAGES,
    ) -> list[dict[str, Any]]:
        """Seal an artifact for ``package`` and its whole dependency closure.

        Walks ``requires_dist`` breadth-first from PyPI JSON metadata,
        evaluating PEP 508 markers for a plain base install (no extras), and
        returns one verified SHA-256 record per package.  The sandbox mounts
        every record read-only and pip installs them with ``--no-index
        --find-links``, so the closure is what makes offline installs of
        JIT-forged tools succeed.
        """
        if max_packages <= 0:
            raise ValueError("max_packages must be positive")
        root_version = self.validate_version(version) if version is not None else None
        queue: deque[tuple[str, str | None]] = deque(
            [(self.validate_package(package), root_version)]
        )
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        while queue:
            name, pin = queue.popleft()
            key = canonicalize_name(name)
            if key in seen:
                continue
            if len(seen) >= max_packages:
                raise WheelFetchError(
                    f"dependency closure for {package} exceeds {max_packages} packages"
                )
            seen.add(key)
            _path, record, info = self._fetch_one(name, pin)
            records.append(record)
            for dep_name, dep_specifier in self._base_requirements(info):
                queue.append((dep_name, self._resolve_version(dep_name, dep_specifier)))
        return records

    def _fetch_one(
        self, package: str, version: str | None = None
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        """Fetch one artifact; return its path, attestation record, and PyPI info."""
        package = self.validate_package(package)
        if version is not None:
            version = self.validate_version(version)
        metadata_url = (
            f"https://pypi.org/pypi/{package}/{version}/json"
            if version
            else f"https://pypi.org/pypi/{package}/json"
        )
        metadata = self._json(metadata_url)
        info = metadata.get("info")
        if not isinstance(info, dict):
            raise WheelFetchError("PyPI metadata did not contain an info object")
        resolved_version = str(info.get("version") or version or "")
        if not resolved_version:
            raise WheelFetchError("PyPI metadata did not contain a version")
        files = metadata.get("urls")
        if not isinstance(files, list):
            raise WheelFetchError("PyPI metadata did not contain release files")
        artifact = self._choose_artifact(files)
        url = str(artifact.get("url") or "")
        self._check_url(url)
        filename = self._filename(url, package, resolved_version)
        destination = self.wheels_dir / filename
        digest, size = self._download(url, destination)
        record = {
            "package": package,
            "version": resolved_version,
            "filename": filename,
            "url": url,
            "sha256": digest,
            "size": size,
            "kind": "wheel" if filename.endswith(".whl") else "sdist",
            "path": str(destination),
        }
        self._write_record(destination, record)
        return destination, record, info

    @staticmethod
    def _base_requirements(info: dict[str, Any]) -> list[tuple[str, str]]:
        """Parse ``requires_dist`` entries needed for a base install.

        Returns ``(name, specifier)`` pairs for requirements whose markers
        hold without any extra selected; unparseable or unresolvable entries
        are skipped because they are never required by a plain install.
        """
        raw = info.get("requires_dist")
        if not isinstance(raw, list):
            return []
        requirements: list[tuple[str, str]] = []
        for entry in raw:
            if not isinstance(entry, str) or not entry.strip():
                continue
            try:
                requirement = Requirement(entry)
                if requirement.marker is not None and not requirement.marker.evaluate(_MARKER_ENV):
                    continue
            except Exception:
                # Malformed or unresolvable metadata never applies to a base
                # install, so it must not abort the closure walk.
                continue
            requirements.append((requirement.name, str(requirement.specifier)))
        return requirements

    def _resolve_version(self, package: str, specifier_text: str) -> str | None:
        """Pick an exact release satisfying ``specifier_text`` (None = latest).

        Latest is used whenever it satisfies the specifier; otherwise the
        highest matching non-empty release is pinned so the sandbox's
        ``pkg==version`` install is resolvable offline.
        """
        package = self.validate_package(package)
        if not specifier_text:
            return None
        try:
            specifier = SpecifierSet(specifier_text)
        except InvalidSpecifier as exc:
            raise WheelFetchError(
                f"invalid dependency specifier for {package}: {specifier_text}"
            ) from exc
        metadata = self._json(f"https://pypi.org/pypi/{package}/json")
        info = metadata.get("info")
        latest = str(info.get("version") or "") if isinstance(info, dict) else ""
        if latest:
            try:
                if specifier.contains(latest):
                    return None
            except InvalidVersion:
                pass
        releases = metadata.get("releases")
        candidates: list[Version] = []
        if isinstance(releases, dict):
            for raw_version, files in releases.items():
                if not isinstance(files, list) or not files:
                    continue  # yanked or withdrawn releases ship no files
                try:
                    parsed = Version(str(raw_version))
                except InvalidVersion:
                    continue
                if specifier.contains(str(parsed)):
                    candidates.append(parsed)
        if not candidates:
            raise WheelFetchError(f"no release of {package} satisfies {specifier_text}")
        return str(max(candidates))

    def verify(self, path: Path | str) -> dict[str, Any]:
        """Recompute and verify a previously recorded artifact digest."""
        artifact = Path(path)
        record_path = self.record_path(artifact)
        if not artifact.is_file():
            raise WheelFetchError(f"wheel artifact does not exist: {artifact}")
        if not record_path.is_file():
            raise WheelFetchError(f"wheel sha256 record does not exist: {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected = str(record.get("sha256") or "")
        actual = _sha256_file(artifact)
        if not expected or actual != expected:
            raise WheelFetchError(
                f"wheel sha256 mismatch for {artifact.name}: expected {expected}, got {actual}"
            )
        return record

    def record(self, path: Path | str) -> dict[str, Any]:
        """Return a verified wheel record for manifest embedding."""
        return self.verify(path)

    @staticmethod
    def validate_package(package: str) -> str:
        value = (package or "").strip()
        if not _PACKAGE_RE.fullmatch(value) or any(char in value for char in ("/", "\\")):
            raise WheelFetchError(f"invalid PyPI package name: {package!r}")
        return value

    @staticmethod
    def validate_version(version: str) -> str:
        value = (version or "").strip()
        if not _VERSION_RE.fullmatch(value) or any(char in value for char in ("/", "\\")):
            raise WheelFetchError(f"invalid PyPI package version: {version!r}")
        return value

    def _json(self, url: str) -> dict[str, Any]:
        self._check_url(url)
        try:
            with self.opener(
                urllib.request.Request(url, headers={"User-Agent": "ultron-wheel-fetch/5.0"}),
                timeout=30.0,
            ) as response:
                self._check_response_url(response, url)
                value = json.loads(response.read(4 * 1024 * 1024).decode("utf-8"))
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise WheelFetchError(f"PyPI metadata fetch failed: {exc}") from exc
        if not isinstance(value, dict):
            raise WheelFetchError("PyPI metadata response was not a JSON object")
        return value

    def _download(self, url: str, destination: Path) -> tuple[str, int]:
        temporary = destination.with_name(destination.name + ".part")
        digest = hashlib.sha256()
        size = 0
        try:
            with self.opener(
                urllib.request.Request(url, headers={"User-Agent": "ultron-wheel-fetch/5.0"}),
                timeout=60.0,
            ) as response:
                self._check_response_url(response, url)
                content_length = _content_length(response)
                if content_length is not None and content_length > self.max_bytes:
                    raise WheelFetchError(
                        f"artifact exceeds size cap: {content_length} > {self.max_bytes} bytes"
                    )
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(min(1024 * 1024, self.max_bytes + 1))
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise WheelFetchError(
                                f"artifact exceeds size cap: {size} > {self.max_bytes} bytes"
                            )
                        digest.update(chunk)
                        handle.write(chunk)
            temporary.replace(destination)
        except WheelFetchError:
            temporary.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            temporary.unlink(missing_ok=True)
            raise WheelFetchError(f"wheel download failed: {exc}") from exc
        return digest.hexdigest(), size

    def _write_record(self, artifact: Path, record: dict[str, Any]) -> None:
        self.record_path(artifact).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def record_path(self, artifact: Path | str) -> Path:
        path = Path(artifact)
        return path.with_name(path.name + ".sha256.json")

    @staticmethod
    def _choose_artifact(files: list[Any]) -> dict[str, Any]:
        candidates = [item for item in files if isinstance(item, dict) and item.get("url")]
        candidates = [
            item
            for item in candidates
            if str(item.get("filename") or urllib.parse.urlparse(str(item["url"])).path)
            .lower()
            .endswith(_ARTIFACT_SUFFIXES)
        ]
        if not candidates:
            raise WheelFetchError("PyPI release has no wheel or source distribution")
        wheels = [item for item in candidates if str(item.get("filename", "")).endswith(".whl")]
        return sorted(
            wheels or candidates, key=lambda item: str(item.get("filename") or item["url"])
        )[0]

    @staticmethod
    def _filename(url: str, package: str, version: str) -> str:
        raw = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
        if not raw or raw in {".", ".."} or not raw.lower().endswith(_ARTIFACT_SUFFIXES):
            raise WheelFetchError("PyPI artifact URL did not contain a supported filename")
        # Keep the final path component only; no URL path can escape wheels_dir.
        return re.sub(r"[^A-Za-z0-9_.+~-]", "_", raw) or f"{package}-{version}.whl"

    @staticmethod
    def _check_url(url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise WheelFetchError(
                f"refusing non-allowlisted wheel host: {url} "
                f"(allowed: {', '.join(sorted(ALLOWED_HOSTS))})"
            )

    def _check_response_url(self, response: Any, requested_url: str) -> None:
        final_url = response.geturl() if hasattr(response, "geturl") else requested_url
        self._check_url(str(final_url))


def _content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ALLOWED_HOSTS",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_PACKAGES",
    "WheelFetchError",
    "WheelFetcher",
]
