"""Run every analysis script and save a combined, UTF-8 execution log."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "out"
LOG_FILE = OUT / "run_log.txt"
SCRIPTS = (
    "p1_tennis.py",
    "p2_pareto.py",
    "p3_fires.py",
    "p4_kl.py",
    "build_report.py",
)


def write_log(lines: list[str]) -> None:
    """Persist the current log even if a later script fails."""

    LOG_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = [
        f"run started {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
        f"python {sys.version.split()[0]}",
    ]

    for name in SCRIPTS:
        banner = f"\n{'=' * 72}\n$ {sys.executable} src/{name}\n{'=' * 72}"
        print(banner)
        log.append(banner)

        completed = subprocess.run(
            [sys.executable, name],
            cwd=SRC,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        print(completed.stdout, end="")
        log.append(completed.stdout.rstrip())

        if completed.returncode:
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
                log.append(completed.stderr.rstrip())
            write_log(log)
            print(f"{name} failed with exit code {completed.returncode}", file=sys.stderr)
            return completed.returncode

    write_log(log)
    print(f"\nall scripts finished; log written to {LOG_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
