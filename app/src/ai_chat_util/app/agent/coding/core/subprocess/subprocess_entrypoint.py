"""Subprocess backend entrypoint.

This module is executed as a separate Python process.
It runs the actual agent command, streams stdout/stderr to files, and writes the
exit code to a file. This enables detached execution while still allowing
TaskManager.get_status() to determine final outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _redact_cmd(argv: list[str]) -> list[str]:
    """Best-effort redaction for secrets in command arguments.

    This is a safety net: users should avoid putting secrets directly into
    command-line args. We still mask common flags and obvious API key patterns.
    """

    if not argv:
        return argv

    redact_next_for = {
        "--password",
        "-p",
        "--api-key",
        "--apikey",
        "--token",
        "--auth",
        "--authorization",
        "--bearer",
        "--secret",
        "--key",
    }
    redacted: list[str] = []
    mask_next = False
    for a in argv:
        if mask_next:
            redacted.append("***")
            mask_next = False
            continue
        low = (a or "").lower()
        if low in redact_next_for:
            redacted.append(a)
            mask_next = True
            continue

        # Flag=value style
        for flag in redact_next_for:
            if low.startswith(flag + "="):
                redacted.append(a.split("=", 1)[0] + "=***")
                break
        else:
            # OpenAI-style keys (best-effort)
            if isinstance(a, str) and a.startswith("sk-") and len(a) >= 12:
                redacted.append("sk-***")
            else:
                redacted.append(a)
    return redacted


def _format_cmd_for_text(argv: list[str]) -> str:
    if os.name == "nt":
        # Windows-friendly cmdline rendering.
        return subprocess.list2cmdline(argv)
    # POSIX-ish rendering.
    return shlex.join(argv)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coding agent subprocess entrypoint")
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

    # Persist executed command for debugging.
    try:
        redacted = _redact_cmd(cmd)
        (workspace / "command.resolved.txt").write_text(
            _format_cmd_for_text(redacted) + "\n",
            encoding="utf-8",
        )
        (workspace / "command.resolved.json").write_text(
            json.dumps(
                {
                    "cmd": redacted,
                    "cwd": workspace.as_posix(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "platform": sys.platform,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        # Do not fail task execution due to debug logging.
        pass

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
