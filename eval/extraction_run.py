"""Offline fixture-accuracy evaluation for the Phase 4.2 extractor.

This is intentionally separate from eval/run.py and never reads or writes
``eval/baseline.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ultron.extractor import GroundedDataExtractor  # noqa: E402


@dataclass(slots=True)
class ExtractionEvalReport:
    tasks: int
    exact_tasks: int
    fields: int
    matched_fields: int
    cases: list[dict[str, Any]]

    @property
    def field_accuracy(self) -> float:
        return self.matched_fields / self.fields if self.fields else 0.0

    @property
    def task_accuracy(self) -> float:
        return self.exact_tasks / self.tasks if self.tasks else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": "phase-4.2-extraction-fixtures",
            "tasks": self.tasks,
            "exact_tasks": self.exact_tasks,
            "fields": self.fields,
            "matched_fields": self.matched_fields,
            "field_accuracy": self.field_accuracy,
            "task_accuracy": self.task_accuracy,
            "cases": self.cases,
        }


def run_extraction_eval(
    tasks_path: Path | None = None, *, quiet: bool = False
) -> ExtractionEvalReport:
    path = tasks_path or ROOT / "eval" / "extraction_tasks.jsonl"
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    extractor = GroundedDataExtractor()
    cases: list[dict[str, Any]] = []
    matched_fields = 0
    field_count = 0
    exact_tasks = 0
    for row in rows:
        source = (
            ROOT / row["source"] if not Path(row["source"]).is_absolute() else Path(row["source"])
        )
        result = extractor.extract(source, row["fields"])
        actual = result.fields
        expected = row["expected"]
        matches = {name: actual.get(name) == value for name, value in expected.items()}
        matched_fields += sum(matches.values())
        field_count += len(matches)
        exact = bool(matches) and all(matches.values()) and result.ok
        exact_tasks += int(exact)
        cases.append({"id": row["id"], "exact": exact, "matches": matches, "errors": result.errors})
    report = ExtractionEvalReport(len(rows), exact_tasks, field_count, matched_fields, cases)
    if not quiet:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument(
        "--json", action="store_true", help="same JSON report; retained for CLI symmetry"
    )
    args = parser.parse_args(argv)
    report = run_extraction_eval(args.tasks, quiet=False)
    return 0 if report.exact_tasks == report.tasks else 1


if __name__ == "__main__":
    raise SystemExit(main())
