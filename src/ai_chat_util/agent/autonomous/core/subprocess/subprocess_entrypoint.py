"""Subprocess backend entrypoint.

This module is executed as a separate Python process.
It runs the actual agent command, streams stdout/stderr to files, and writes the
exit code to a file. This enables detached execution while still allowing
TaskManager.get_status() to determine final outcome.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from ..process_utils import popen_new_process_group_kwargs


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous agent subprocess entrypoint")
    parser.add_argument("--workspace", required=True, help="Workspace directory")
    parser.add_argument("--exit-code-file", required=True, help="Path to write exit code")
    parser.add_argument("--stdout-file", required=True, help="Path to write stdout")
    parser.add_argument("--stderr-file", required=True, help="Path to write stderr")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to execute (prefix with --)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(list(sys.argv[1:] if argv is None else argv))

    workspace = Path(ns.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    exit_code_file = Path(ns.exit_code_file).expanduser().resolve()
    stdout_file = Path(ns.stdout_file).expanduser().resolve()
    stderr_file = Path(ns.stderr_file).expanduser().resolve()
    exit_code_file.parent.mkdir(parents=True, exist_ok=True)
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = list(ns.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command provided. Use: -- <command...>")

    env = os.environ.copy()
    env.setdefault("WORKSPACE", workspace.as_posix())

    # Use line-buffered text IO to keep logs readable.
    with stdout_file.open("w", encoding="utf-8", buffering=1) as out, stderr_file.open(
        "w", encoding="utf-8", buffering=1
    ) as err:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=workspace.as_posix(),
                env=env,
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                **popen_new_process_group_kwargs(),
                close_fds=True,
                text=True,
            )
            rc = proc.wait()
        except BaseException as e:
            # Ensure we always write an exit code file.
            err.write(f"subprocess_entrypoint error: {e}\n")
            rc = 1

    exit_code_file.write_text(str(int(rc)), encoding="utf-8")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
