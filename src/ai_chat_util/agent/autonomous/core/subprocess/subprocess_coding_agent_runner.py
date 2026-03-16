from __future__ import annotations

import os
import pathlib
import shlex
import sys
import uuid
from dataclasses import dataclass
from typing import Optional, Union

from ..abstract_agent_runner import AbstractAgentRunner

from ai_chat_util_base.model.autonomous_agent_util_models import TaskStatus, CodingAgentConfig
from ..utils import ExecutorUtil
from ..process_utils import popen_new_process_group_kwargs
from ai_chat_util_base.config.autonomous_agent_util_runtime import get_runtime_config


@dataclass
class SubprocessRunResult:
    pid: int


class SubprocessCodingAgentRunner(AbstractAgentRunner):
    """Runner that executes the agent locally via Python subprocess.

    It prepares a workspace (same semantics as Docker runner), then starts a
    detached entrypoint process that runs the actual agent command and writes
    stdout/stderr/exit code to files in the workspace.
    """

    def __init__(
        self,
        task_id: Optional[str] = None,
        workspace_path: Optional[Union[str, pathlib.Path]] = None,
        command_base: Optional[str] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> None:
        self.task_id = task_id or str(uuid.uuid4())
        self._runtime_cfg = get_runtime_config()
        runtime_cfg = self._runtime_cfg
        cfg = CodingAgentConfig(
            llm_provider=runtime_cfg.llm.provider,
            llm_model=runtime_cfg.llm.model,
            llm_base_url=runtime_cfg.llm.base_url,
            workspace_root=runtime_cfg.paths.workspace_root,
        )

        if workspace_path is not None:
            self.workspace = pathlib.Path(workspace_path)
        else:
            self.workspace = pathlib.Path(cfg.workspace_root) / self.task_id
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.command_base = command_base or (runtime_cfg.subprocess.command or runtime_cfg.compose.command)
        self.command: list[str] = shlex.split(self.command_base)

        # Per-task env (e.g., Authorization) to be inherited by the entrypoint process.
        self.extra_env: dict[str, str] = {
            str(k): str(v) for k, v in (extra_env or {}).items() if v is not None
        }

        workspace_path = self.workspace.resolve().as_posix()
        self.task_status = TaskStatus.create(task_id=self.task_id, workspace_path=workspace_path)
        self.task_status.metadata["backend"] = "subprocess"
        self.task_status.metadata["workspace_path"] = workspace_path

        # Log/exit-code files
        self.stdout_file = self.workspace / "stdout.log"
        self.stderr_file = self.workspace / "stderr.log"
        self.exit_code_file = self.workspace / ".exit_code"

        self.task_status.metadata["stdout_path"] = self.stdout_file.as_posix()
        self.task_status.metadata["stderr_path"] = self.stderr_file.as_posix()
        self.task_status.metadata["exit_code_path"] = self.exit_code_file.as_posix()

    def prepare_workspace(
        self,
        initial_files: Optional[dict[str, str]] = None,
        source_paths: Optional[list[pathlib.Path]] = None,
    ) -> None:
        if initial_files:
            ExecutorUtil.add_data(initial_files, self.workspace)
        if source_paths:
            ExecutorUtil.add_files(source_paths, self.workspace)

    def get_task_status(self) -> TaskStatus:
        return self.task_status

    def get_workspace_path(self) -> pathlib.Path:
        return self.workspace.resolve()

    @classmethod
    async def create_runner(
        cls,
        prompt: str,
        task_id: Optional[str] = None,
        source_paths: Optional[list[pathlib.Path]] = None,
        workspace_path: Optional[Union[str, pathlib.Path]] = None,
        detach: bool = True,
        command_base: Optional[str] = None,
        extra_env: Optional[dict[str, str]] = None,
        **_kwargs,
    ) -> "SubprocessCodingAgentRunner":
        runner = cls(
            task_id=task_id,
            workspace_path=workspace_path,
            command_base=command_base,
            extra_env=extra_env,
        )
        if prompt:
            runner.command = [*runner.command, prompt]
        runner.prepare_workspace(source_paths=source_paths)
        return runner

    def start(self) -> SubprocessRunResult:
        # Spawn entrypoint wrapper which will run the actual agent command.
        entrypoint_cmd = [
            sys.executable,
            "-m",
            "autonomous_agent_util.core.subprocess.subprocess_entrypoint",
            "--workspace",
            self.workspace.as_posix(),
            "--exit-code-file",
            self.exit_code_file.as_posix(),
            "--stdout-file",
            self.stdout_file.as_posix(),
            "--stderr-file",
            self.stderr_file.as_posix(),
            "--",
            *self.command,
        ]

        import subprocess  # local import to keep module lightweight

        runtime_cfg = self._runtime_cfg
        env = os.environ.copy()
        # Ensure non-secret runtime settings are passed even if this process
        # does not have them in os.environ.
        env["LLM_PROVIDER"] = str(runtime_cfg.llm.provider)
        env["LLM_MODEL"] = str(runtime_cfg.llm.model)
        if runtime_cfg.llm.base_url:
            env["LLM_BASE_URL"] = str(runtime_cfg.llm.base_url)
        api_key = runtime_cfg.llm.api_key or env.get("LLM_API_KEY")
        providers_requiring_key = {"openai", "azure", "azure_openai", "anthropic"}
        provider = (runtime_cfg.llm.provider or "").lower()
        if provider in providers_requiring_key and not api_key:
            raise RuntimeError(
                "LLM API key が未設定です。.env/環境変数で LLM_API_KEY を設定するか、"
                "autonomous-agent-util-config.yml で 'llm.api_key: os.environ/LLM_API_KEY' のように参照設定してください。"
            )
        if api_key:
            env["LLM_API_KEY"] = str(api_key)
        # Do not persist secrets to disk; pass per-task env via subprocess env.
        for k, v in self.extra_env.items():
            if v is None:
                continue
            env[str(k)] = str(v)

        proc = subprocess.Popen(
            entrypoint_cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            **popen_new_process_group_kwargs(),
            close_fds=True,
        )
        self.task_status.metadata["pid"] = int(proc.pid)
        return SubprocessRunResult(pid=int(proc.pid))
