"""AST-based hardening of raw Python scripts before they enter Forge."""

from __future__ import annotations

import ast
import re
from typing import Any

from .config import Settings, get_settings
from .errors import UltronError
from .forge import ForgeEngine
from .registry import TYPE_MAP, ToolManifest

_UNSAFE_IMPORT_ROOTS = frozenset(
    {
        "argparse",
        "tkinter",
        "matplotlib",
        "pandasgui",
        "pyqt5",
        "wx",
        "streamlit",
        "gradio",
        "telemetry",
        "opentelemetry",
        "prometheus_client",
        "sentry_sdk",
        "logging",
        "loguru",
        "logger",
        "metrics",
        "tracer",
    }
)
_UNSAFE_CALL_ROOTS = _UNSAFE_IMPORT_ROOTS | {"print", "exit", "quit", "track", "record_metric"}


class RefineryError(UltronError):
    """The source could not be safely converted into a tool."""


class _Sanitizer(ast.NodeTransformer):
    """Remove CLI/observability side effects while preserving library code."""

    @staticmethod
    def _root_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                current = current.value
            return current.id if isinstance(current, ast.Name) else None
        return None

    @classmethod
    def _is_unsafe_call(cls, node: ast.Call) -> bool:
        root = cls._root_name(node.func)
        if root in _UNSAFE_CALL_ROOTS:
            return True
        if isinstance(node.func, ast.Attribute):
            dotted = ast.unparse(node.func).lower()
            return any(dotted.startswith(f"{name}.") for name in _UNSAFE_CALL_ROOTS)
        return False

    @staticmethod
    def _is_main_guard(node: ast.If) -> bool:
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            return False
        if not isinstance(test.ops[0], ast.Eq):
            return False
        left, right = test.left, test.comparators[0]
        return (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and isinstance(right, ast.Constant)
            and right.value == "__main__"
        ) or (
            isinstance(right, ast.Name)
            and right.id == "__name__"
            and isinstance(left, ast.Constant)
            and left.value == "__main__"
        )

    def visit_If(self, node: ast.If) -> ast.If | None:
        if self._is_main_guard(node):
            return None
        node = self.generic_visit(node)
        if not node.body:
            node.body = [ast.Pass()]
        if not node.orelse:
            node.orelse = []
        return node

    def visit_Import(self, node: ast.Import) -> ast.Import | None:
        kept = [
            alias for alias in node.names if alias.name.split(".", 1)[0] not in _UNSAFE_IMPORT_ROOTS
        ]
        return ast.Import(names=kept) if kept else None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        root = (node.module or "").split(".", 1)[0]
        if root in _UNSAFE_IMPORT_ROOTS or root == "sys":
            return None
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.stmt | None:
        if isinstance(node.value, ast.Call) and self._is_unsafe_call(node.value):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST | None:
        if self._is_unsafe_call(node):
            return ast.Constant(value=None)
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | None:
        if isinstance(node.value, ast.Call) and self._is_unsafe_call(node.value):
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt | None:
        if isinstance(node.value, ast.Call) and self._is_unsafe_call(node.value):
            return None
        return self.generic_visit(node)


def _safe_identifier(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value or "") or "__" in value:
        raise RefineryError(f"{label} must be a simple Python function identifier")
    return value


def _semantic_output(tag: str) -> tuple[str, str]:
    """Map a semantic provide tag to an output key and optional type."""
    raw = str(tag).strip()
    name, separator, declared_type = raw.rpartition(":")
    if separator and declared_type in TYPE_MAP:
        raw = name
        return raw.rsplit(".", 1)[-1], declared_type
    return raw.rsplit(".", 1)[-1], "any"


def _schema_from_provides(provides: list[str]) -> dict[str, str]:
    schema: dict[str, str] = {}
    for tag in provides:
        key, typ = _semantic_output(tag)
        if re.fullmatch(r"[a-z][a-z0-9_]{0,48}", key):
            schema[key] = typ
    return schema or {"result": "any"}


def _schema_literal(schema: dict[str, str]) -> str:
    # Values were validated before this is emitted; repr keeps the generated
    # wrapper free from interpolation or executable user expressions.
    return repr(dict(schema))


def _envelope_code(target_func: str, schema: dict[str, str]) -> str:
    return f"""\n\n# Ultron refinery execution envelope.\nimport json as _ultron_json\nimport traceback as _ultron_traceback\nimport sys as _ultron_sys\n\n_ULTRON_TARGET = {target_func!r}\n_ULTRON_SCHEMA = {_schema_literal(schema)}\n\ndef _ultron_type_matches(value, declared):\n    if declared == "any":\n        return True\n    if declared == "int":\n        return isinstance(value, int) and not isinstance(value, bool)\n    if declared == "float":\n        return isinstance(value, (int, float)) and not isinstance(value, bool)\n    if declared == "bool":\n        return isinstance(value, bool)\n    if declared == "string":\n        return isinstance(value, str)\n    if declared == "list[string]":\n        return isinstance(value, list) and all(isinstance(item, str) for item in value)\n    if declared == "list[int]":\n        return isinstance(value, list) and all(\n            isinstance(item, int) and not isinstance(item, bool) for item in value\n        )\n    if declared == "list[float]":\n        return isinstance(value, list) and all(\n            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value\n        )\n    if declared == "dict":\n        return isinstance(value, dict)\n    return False\n\ndef _ultron_execute(payload):\n    if not isinstance(payload, dict):\n        raise TypeError("tool input must be a JSON object")\n    target = globals().get(_ULTRON_TARGET)\n    if not callable(target):\n        raise NameError(f"target function {{_ULTRON_TARGET!r}} was not found")\n    try:\n        value = target(**payload)\n    except TypeError as keyword_error:\n        try:\n            value = target(payload)\n        except TypeError:\n            raise keyword_error\n    output = value if isinstance(value, dict) else {{next(iter(_ULTRON_SCHEMA), "result"): value}}\n    if not isinstance(output, dict):\n        raise TypeError("target function must produce a JSON object or declared scalar")\n    for _key, _declared in _ULTRON_SCHEMA.items():\n        if _key not in output:\n            raise ValueError(f"missing declared output {{_key!r}}")\n        if not _ultron_type_matches(output[_key], _declared):\n            raise TypeError(f"output {{_key!r}} does not match declared type {{_declared}}")\n    _ultron_json.dumps(output)\n    return output\n\ndef _ultron_main():\n    try:\n        _payload = _ultron_json.loads(_ultron_sys.stdin.read() or "{{}}")\n        _ultron_sys.stdout.write(_ultron_json.dumps({{"ok": True, "result": _ultron_execute(_payload)}}) + "\\n")\n    except Exception as _exc:\n        _ultron_sys.stdout.write(_ultron_json.dumps({{"ok": False, "error": str(_exc), "trace": _ultron_traceback.format_exc()}}) + "\\n")\n        _ultron_sys.stdout.flush()\n        raise SystemExit(1)\n\nif __name__ == "__main__":\n    _ultron_main()\n"""


class ToolRefinery:
    """Turn noisy scripts into deterministic Forge-compatible tools."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        forge: ForgeEngine | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.forge = forge or ForgeEngine(self.settings)

    def refine_code(
        self,
        raw_code: str,
        target_func_name: str,
        provides: list[str],
        requires: list[str],
    ) -> tuple[str, dict[str, Any]]:
        """Sanitize source with AST and append the standard JSON envelope."""
        return self._refine_code(raw_code, target_func_name, provides, requires, None)

    def _refine_code(
        self,
        raw_code: str,
        target_func_name: str,
        provides: list[str],
        requires: list[str],
        output_schema: dict[str, str] | None,
    ) -> tuple[str, dict[str, Any]]:
        target = _safe_identifier(target_func_name, "target_func_name")
        if not isinstance(raw_code, str) or not raw_code.strip():
            raise RefineryError("raw_code must not be empty")
        try:
            tree = ast.parse(raw_code, filename="raw_tool.py", mode="exec")
        except SyntaxError as exc:
            raise RefineryError(f"raw_code is not valid Python: {exc}") from exc
        tree = _Sanitizer().visit(tree)
        tree = ast.fix_missing_locations(tree)
        source = ast.unparse(tree).strip()
        schema = dict(output_schema or _schema_from_provides(provides))
        for key, typ in schema.items():
            if (
                key not in {k for k, _ in (_semantic_output(item) for item in provides)}
                and output_schema is None
            ):
                continue
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,48}", key) or typ not in TYPE_MAP:
                raise RefineryError(f"invalid output schema entry {key!r}: {typ!r}")
        hardened = source + _envelope_code(target, schema)
        compile(hardened, "refined_tool.py", "exec")
        tool_name = "refined_" + re.sub(r"[^a-z0-9_]", "_", target.lower())
        tool_name = tool_name[:49].rstrip("_")
        manifest: dict[str, Any] = {
            "name": tool_name,
            "version": "0.1.0",
            "description": f"AST-refined wrapper for {target}",
            "entrypoint": "python refined_tool.py",
            "risk": "medium",
            "provides": list(provides),
            "requires": list(requires),
            "permissions": [],
            "inputs": {},
            "outputs": schema,
            "tags": ["refined", "ast-hardened"],
            "deterministic": True,
            "author": "ultron-refinery",
        }
        return hardened + "\n", manifest

    def cook_and_register(
        self,
        raw_code: str,
        target_func: str,
        test_input: dict[str, Any],
        expected_schema: dict[str, str],
    ) -> ToolManifest:
        """Refine, Forge-test, Breaker-verify, and register a cooked tool."""
        provides = [f"refined.{key}" for key in expected_schema] or ["refined.result"]
        requires: list[str] = []
        try:
            code, manifest = self._refine_code(
                raw_code, target_func, provides, requires, expected_schema
            )
            manifest["inputs"] = {key: _infer_type(value) for key, value in test_input.items()}
            spec = self.forge.ledger.record_gap(
                f"refine:{target_func}",
                required_inputs=dict(manifest["inputs"]),
                expected_outputs=dict(expected_schema),
                suggested_provides=list(provides),
                suggested_requires=requires,
                failure_reason="refinery cook",
            )
            temporary = self.forge.synthesize_tool(spec, code, manifest)
            if not self.forge.test_tool(temporary, test_input, expected_schema):
                raise RefineryError(f"sandbox or Breaker rejected refined tool {target_func!r}")
            return self.forge.registry.get(temporary.name)
        except RefineryError:
            raise
        except Exception as exc:
            raise RefineryError(f"cook failed: {type(exc).__name__}: {exc}") from exc


def _infer_type(value: Any) -> str:
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
