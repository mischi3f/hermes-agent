#!/usr/bin/env python3
"""Opt-in real CLI regression for Honcho shutdown-time SIGABRT.

Runs against the current operator profile only when explicitly enabled. Raw
stdout/stderr live in an auto-deleted temporary directory; generated session
IDs are counted but never printed or persisted in the repository.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path


_RUN_ENV = "HERMES_RUN_HONCHO_CLI_SHUTDOWN_TEST"
_SESSION_ID_RE = re.compile(
    r"(?im)^session[_ ]?id\s*:\s*[A-Za-z0-9_.:-]+\s*$"
)


def main() -> int:
    if os.environ.get(_RUN_ENV) != "1":
        print(f"Refusing real provider test; set {_RUN_ENV}=1 to opt in")
        return 2

    launcher = Path(
        os.environ.get(
            "HERMES_TEST_LAUNCHER", str(Path.home() / ".local" / "bin" / "hermes")
        )
    )
    if not launcher.is_file():
        print("Hermes launcher was not found")
        return 2

    try:
        run_count = int(os.environ.get("HERMES_HONCHO_SHUTDOWN_RUNS", "10"))
    except ValueError:
        print("HERMES_HONCHO_SHUTDOWN_RUNS must be an integer")
        return 2
    if not 10 <= run_count <= 20:
        print("HERMES_HONCHO_SHUTDOWN_RUNS must be between 10 and 20")
        return 2

    failures = 0
    with tempfile.TemporaryDirectory(prefix="honcho-cli-shutdown-") as temp_dir:
        output_dir = Path(temp_dir)
        for run_number in range(1, run_count + 1):
            try:
                completed = subprocess.run(
                    [
                        str(launcher),
                        "chat",
                        "-q",
                        "Reply with exactly: OK",
                        "-t",
                        "web",
                        "-Q",
                        "--max-turns",
                        "4",
                        "--source",
                        "groupme",
                        "--no-restore-cwd",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=120,
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONFAULTHANDLER": "1"},
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                (output_dir / f"run-{run_number}.out").write_text(stdout)
                (output_dir / f"run-{run_number}.err").write_text(stderr)
                failures += 1
                print(f"run={run_number} timeout=1 pass=0")
                continue

            (output_dir / f"run-{run_number}.out").write_text(completed.stdout)
            (output_dir / f"run-{run_number}.err").write_text(completed.stderr)
            answer_count = len(re.findall(r"(?m)^OK$", completed.stdout))
            session_id_count = len(_SESSION_ID_RE.findall(completed.stderr))
            fatal = "Fatal Python error" in completed.stderr
            aborted = "Aborted" in completed.stderr
            passed = (
                completed.returncode == 0
                and answer_count == 1
                and session_id_count == 1
                and not fatal
                and not aborted
            )
            failures += int(not passed)
            print(
                f"run={run_number} exit={completed.returncode} "
                f"answer_count={answer_count} session_id_count={session_id_count} "
                f"fatal={int(fatal)} aborted={int(aborted)} pass={int(passed)}"
            )

    print(f"summary runs={run_count} passed={run_count - failures} failed={failures}")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
