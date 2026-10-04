"""Deterministic arithmetic tool: ``python -m tools.calc``.

Demonstrates the cheapest possible tool: no network, no LLM, pure function, so
its output is cached forever and a re-run costs nothing. Evaluation is done on a
whitelisted AST instead of ``eval`` -- a manifest cannot smuggle code execution
through an input string.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from tools._io import main_guard, require

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPONENT = 12  # refuse 2**999999 style resource bombs


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return float(op(left, right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary operator {type(node.op).__name__} is not allowed")
        return float(op(_eval(node.operand)))
    raise ValueError(f"expression element {type(node).__name__} is not allowed")


def run(payload: dict[str, Any]) -> dict[str, Any]:
    expression = require(payload, "expression", str)
    if len(expression) > 500:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")
    result = _eval(tree)
    return {
        "result": result,
        "expression": expression,
        "normalized": ast.unparse(tree),
    }


if __name__ == "__main__":
    main_guard(run)
