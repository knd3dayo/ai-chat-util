from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import AsyncGenerator, Optional

from ...util.logging import get_application_logger

from ai_chat_util.common.model.agent_util_models import TaskStatus
from ..abstract_agent_runner import AbstractAgentRunner
from ..abstract_task_service import AbstractTaskService
from ..process_utils import kill_process_tree, popen_new_process_group_kwargs
from .subprocess_coding_agent_runner import SubprocessCodingAgentRunner
from .windows_process_coding_agent_runner import WindowsProcessCodingAgentRunner
from .linux_process_coding_agent_runner import LinuxProcessCodingAgentRunner
from ai_chat_util.common.config.runtime import (
    get_coding_runtime_config,
    get_coding_runtime_config_path,
)

logger = get_application_logger()


class SubprocessTaskService(AbstractTaskService):
    """Task service implementation for local subprocess backend."""

    def __init__(self) -> None:
        self.runner: Optional[SubprocessCodingAgentRunner] = None

    async def prepare(
        self,
        prompt: str,
        sources: Optional[list[Path]],
        task_id: Optional[str],
        workspace_path: Optional[Path] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> None:
        cfg = get_coding_runtime_config()
        backend = (cfg.backend.task_backend or "process").strip().lower()
        # `process` auto-selects based on current platform.
        if backend == "process":
            runner_cls: type[SubprocessCodingAgentRunner]
            runner_cls = WindowsProcessCodingAgentRunner if os.name == "nt" else LinuxProcessCodingAgentRunner
        elif backend == "windows_process":
            runner_cls = WindowsProcessCodingAgentRunner
        elif backend == "linux_process":
            runner_cls = LinuxProcessCodingAgentRunner
        else:
            # Should not happen since config validation normalizes/validates.
            runner_cls = SubprocessCodingAgentRunner

        params: dict[str, object] = {"prompt": prompt, "task_id": task_id}
        if sources:
            params["source_paths"] = sources
        if workspace_path is not None:
            params["workspace_path"] = workspace_path
        if extra_env:
            params["extra_env"] = extra_env

        self.runner = await runner_cls.create_runner(**params)  # type: ignore[arg-type]

    def get_agent_runner(self) -> AbstractAgentRunner:
        if self.runner is None:
            raise RuntimeError("Runner not initialized")
        return self.runner

    def spawn_detached_monitor(self, task_id: str, timeout: int) -> None:
        cfg = get_coding_runtime_config()
        if cfg.monitor.disable_detach_monitor:
            return

        max_seconds = max(int(timeout) + 60, 120)
        interval = float(cfg.monitor.detach_monitor_interval)

        env = os.environ.copy()
        cfg_path = get_coding_runtime_config_path()
        if cfg_path:
            env.setdefault("AI_CHAT_UTIL_CONFIG", str(cfg_path))

        cmd = [
            SubprocessCodingAgentRunner.resolve_python_executable(),
            "-m",
            "ai_chat_util.agent.coding._cli_.docker_main",
            "monitor",
            task_id,
            "--interval",
            str(interval),
            "--max-seconds",
            str(max_seconds),
            "--quiet",
        ]

        subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            **popen_new_process_group_kwargs(),
            close_fds=True,
        )

    def cancel_task(self, task: TaskStatus) -> None:
        md = task.metadata if isinstance(task.metadata, dict) else {}
        pid = md.get("pid")
        if isinstance(pid, int) and pid > 1:
            kill_process_tree(pid)

    def start(self, *, wait: bool, timeout: int) -> TaskStatus:
        if self.runner is None:
            raise RuntimeError("Runner not initialized")

        run_result = self.runner.start()
        task_status = self.runner.get_task_status()
        task_status.metadata["pid"] = getattr(run_result, "pid", None)

        if wait:
            task_status.starting_foregrond()
        else:
            task_status.starting_background()

        return task_status

    async def monitor(self, timeout: int) -> AsyncGenerator[TaskStatus, None]:
        if self.runner is None:
            return

        loop = asyncio.get_running_loop()
        start = loop.time()

        exit_path = self.runner.exit_code_file
        while True:
            if exit_path.exists():
                try:
                    rc = int(exit_path.read_text(encoding="utf-8").strip())
                except Exception:
                    rc = 1

                task_status = self.runner.get_task_status()
                if rc == 0:
                    task_status.completed()
                else:
                    task_status.failed()

                try:
                    task_status.stdout = self.runner.stdout_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    task_status.stdout = task_status.stdout or ""
                try:
                    task_status.stderr = self.runner.stderr_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    task_status.stderr = task_status.stderr or ""

                try:
                    base = self.runner.get_workspace_path()
                    task_status.artifacts = [
                        str(p.relative_to(base).as_posix())
                        for p in base.rglob("*")
                        if p.is_file()
                    ]
                except Exception:
                    pass

                yield task_status
                return

            if loop.time() - start > timeout:
                task_status = self.runner.get_task_status()
                self.cancel_task(task_status)
                task_status.timeouted(timeout)
                yield task_status
                return

            await asyncio.sleep(1.0)
