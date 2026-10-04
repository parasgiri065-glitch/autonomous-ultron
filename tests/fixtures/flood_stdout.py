"""Test fixture: a misbehaving tool that floods stdout.

Used by ``tests/test_smoke.py`` and by the Docker CI job to prove that the
sandbox bounds what it reads from a tool and kills the producer when the cap is
hit (issue #2). It deliberately never prints an envelope: a real flooder would
not get the chance to finish.

Not a tool: it has no manifest and is never registered.
"""

from __future__ import annotations

import sys

CHUNK = b"x" * (1024 * 1024)
DEFAULT_MB = 10


def main(argv: list[str]) -> int:
    megabytes = int(argv[1]) if len(argv) > 1 else DEFAULT_MB
    out = sys.stdout.buffer
    for _ in range(megabytes):
        out.write(CHUNK)
    out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
