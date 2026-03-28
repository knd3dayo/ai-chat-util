from __future__ import annotations

import asyncio
import os

import pytest

from coding_agent_util.core.subprocess.windows_process_coding_agent_runner import (
    WindowsProcessCodingAgentRunner,
)


def test_windows_cmd_shim_wrap_includes_prompt_inside_cmdline(tmp_path) -> None:
    if os.name != "nt":
        pytest.skip("Windows-only test")

    shim = tmp_path / "dummy_tool.cmd"
    shim.write_text("@echo off\r\necho dummy\r\n", encoding="utf-8")

    prompt = "hello world"
    runner = asyncio.run(
        WindowsProcessCodingAgentRunner.create_runner(
            prompt=prompt,
            workspace_path=tmp_path / "ws",
            command_base=str(shim),
        )
    )

    # The Windows shim wrapper should convert `.cmd` into `cmd.exe /c "<cmdline>"`.
    assert runner.command[:4] == ["cmd.exe", "/d", "/s", "/c"]
    assert len(runner.command) == 5

    cmdline = runner.command[4]
    # Ensure the prompt is part of the actual command string passed to cmd.exe,
    # not a separate argv item.
    assert "hello" in cmdline
    assert '"hello world"' in cmdline
