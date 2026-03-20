from __future__ import annotations

import os
import pathlib

from .subprocess_coding_agent_runner import SubprocessCodingAgentRunner


def _windows_wrap_shell_shims(argv: list[str]) -> list[str]:
    """Make Windows command execution robust for shell shims.

    On Windows, commands installed by various package managers (npm, pipx, etc.)
    are often exposed as `*.cmd` / `*.bat` / `*.ps1` shims. When running a
    subprocess with `shell=False`, CreateProcess cannot directly execute these
    shims (especially `.ps1`), which can surface as WinError 2.

    This function resolves the first argv entry against PATH and wraps the
    invocation with the appropriate interpreter:
    - `.exe` / `.com`: run directly
    - `.cmd` / `.bat`: run via `cmd.exe /c`
    - `.ps1`: run via `powershell.exe -File`
    """

    if os.name != "nt" or not argv:
        return argv

    head = (argv[0] or "").strip().strip('"')
    if not head:
        return argv

    head_path = pathlib.Path(head)
    # If an explicit path is provided, only wrap by extension.
    if head_path.is_absolute() or any(sep in head for sep in ("\\", "/")):
        ext = head_path.suffix.lower()
        if ext == ".ps1":
            return [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(head_path),
                *argv[1:],
            ]
        if ext in {".cmd", ".bat"}:
            import subprocess as _subprocess

            cmdline = _subprocess.list2cmdline([str(head_path), *argv[1:]])
            return ["cmd.exe", "/d", "/s", "/c", cmdline]
        return argv

    # Resolve via PATH by probing common Windows shim extensions.
    path_entries = [p for p in (os.environ.get("PATH") or "").split(os.pathsep) if p]
    exts = [".exe", ".com", ".cmd", ".bat", ".ps1"]
    found: pathlib.Path | None = None

    for ext in exts:
        for p in path_entries:
            cand = pathlib.Path(p) / f"{head}{ext}"
            if cand.exists():
                found = cand
                break
        if found is not None:
            break

    if found is None:
        # Could be a built-in / alias / already resolvable by CreateProcess.
        return argv

    ext = found.suffix.lower()
    if ext in {".exe", ".com"}:
        return [str(found), *argv[1:]]
    if ext in {".cmd", ".bat"}:
        import subprocess as _subprocess

        cmdline = _subprocess.list2cmdline([str(found), *argv[1:]])
        return ["cmd.exe", "/d", "/s", "/c", cmdline]
    if ext == ".ps1":
        return [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(found),
            *argv[1:],
        ]
    return argv


class WindowsProcessCodingAgentRunner(SubprocessCodingAgentRunner):
    """Windows-specific local process runner.

    Adds shim wrapping so common Windows launchers (`*.cmd`/`*.ps1`) work with
    detached `shell=False` execution.
    """

    def _wrap_command_for_platform(self, argv: list[str]) -> list[str]:
        return _windows_wrap_shell_shims(argv)
