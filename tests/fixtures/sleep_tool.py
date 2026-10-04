"""Test fixture: a tool that hangs, so the executor's timeout path is exercised.

Prints nothing until it is done, so it also proves the timeout does not depend on
output: the sandbox must kill it and report ``timed_out``.

Not a tool: it has no manifest and is never registered.
"""

from __future__ import annotations

import sys
import time

if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    time.sleep(seconds)
    sys.stdout.write(f'{{"ok": true, "result": {{"slept": {seconds}}}}}\n')
