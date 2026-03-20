from __future__ import annotations

import os
import pathlib
import shlex
import sys
import uuid
import json
from dataclasses import dataclass
from typing import Optional, Union
from datetime import datetime, timezone

from ..abstract_agent_runner import AbstractAgentRunner

from ai_chat_util_base.model.agent_util_models import TaskStatus, CodingAgentConfig
from ..utils import ExecutorUtil
from ..process_utils import popen_new_process_group_kwargs
from ai_chat_util_base.config.runtime import get_autonomous_runtime_config


def _find_project_root(start: pathlib.Path) -> pathlib.Path | None:
    p = start.resolve()
    for parent in (p, *p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return None


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
        self._runtime_cfg = get_autonomous_runtime_config()
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

        # `process` is the preferred config section name.
        # Keep reading `subprocess` for backward compatibility.
        process_cmd = getattr(getattr(runtime_cfg, "process", None), "command", None)
        subprocess_cmd = getattr(getattr(runtime_cfg, "subprocess", None), "command", None)
        self.command_base = command_base or (process_cmd or subprocess_cmd)
        if not (self.command_base or "").strip():
            raise ValueError(
                "process.command が未設定です。ai-chat-util-config.yml の ai_chat_util_config.autonomous_agent_util.process.command を設定してください。"
            )

        self.command_requested: list[str] = shlex.split(self.command_base)
        self.command: list[str] = self._wrap_command_for_platform(list(self.command_requested))

        # Per-task env (e.g., Authorization) to be inherited by the entrypoint process.
        self.extra_env: dict[str, str] = {
            str(k): str(v) for k, v in (extra_env or {}).items() if v is not None
        }

        workspace_path = self.workspace.resolve().as_posix()
        self.task_status = TaskStatus.create(task_id=self.task_id, workspace_path=workspace_path)
        backend_name = (getattr(getattr(runtime_cfg, "backend", None), "task_backend", None) or "process")
        self.task_status.metadata["backend"] = str(backend_name)
        self.task_status.metadata["workspace_path"] = workspace_path

        # Log/exit-code files
        self.stdout_file = self.workspace / "stdout.log"
        self.stderr_file = self.workspace / "stderr.log"
        self.exit_code_file = self.workspace / ".exit_code"

        # Debug artifacts
        self.command_txt_file = self.workspace / "command.txt"
        self.command_json_file = self.workspace / "command.json"
        self.command_resolved_txt_file = self.workspace / "command.resolved.txt"
        self.command_resolved_json_file = self.workspace / "command.resolved.json"

        self.task_status.metadata["stdout_path"] = self.stdout_file.as_posix()
        self.task_status.metadata["stderr_path"] = self.stderr_file.as_posix()
        self.task_status.metadata["exit_code_path"] = self.exit_code_file.as_posix()
        self.task_status.metadata["command_path"] = self.command_txt_file.as_posix()
        self.task_status.metadata["command_json_path"] = self.command_json_file.as_posix()
        self.task_status.metadata["command_resolved_path"] = self.command_resolved_txt_file.as_posix()
        self.task_status.metadata["command_resolved_json_path"] = self.command_resolved_json_file.as_posix()

    def _wrap_command_for_platform(self, argv: list[str]) -> list[str]:
        """Hook to adjust argv per-platform.

        Windows implementations may need to wrap shell shims (`*.cmd`/`*.ps1`).
        """
        return argv

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
            runner.command_requested = [*runner.command_requested, prompt]
            runner.command = [*runner.command, prompt]
        runner.prepare_workspace(source_paths=source_paths)
        return runner

    @staticmethod
    def _redact_cmd(argv: list[str]) -> list[str]:
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
            for flag in redact_next_for:
                if low.startswith(flag + "="):
                    redacted.append(a.split("=", 1)[0] + "=***")
                    break
            else:
                if isinstance(a, str) and a.startswith("sk-") and len(a) >= 12:
                    redacted.append("sk-***")
                else:
                    redacted.append(a)
        return redacted

    def _persist_requested_command_debug_artifacts(self) -> None:
        try:
            redacted = self._redact_cmd(list(self.command_requested))
            # Human friendly
            try:
                import subprocess as _subprocess

                txt = _subprocess.list2cmdline(redacted) if os.name == "nt" else shlex.join(redacted)
            except Exception:
                txt = " ".join(redacted)

            self.command_txt_file.write_text(txt + "\n", encoding="utf-8")
            self.command_json_file.write_text(
                json.dumps(
                    {
                        "cmd": redacted,
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
            pass

    def start(self) -> SubprocessRunResult:
        python_exe = self.resolve_python_executable()
        # Persist user-requested command (before platform shim resolution) for debugging.
        self._persist_requested_command_debug_artifacts()
        # Spawn entrypoint wrapper which will run the actual agent command.
        entrypoint_cmd = [
            python_exe,
            "-m",
            "ai_chat_util.agent.autonomous.core.subprocess.subprocess_entrypoint",
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

        # Ensure tools installed into the project's virtualenv
        # are available to the detached entrypoint and its child process.
        try:
            root = _find_project_root(pathlib.Path(__file__))
            if root is not None:
                if os.name == "nt":
                    venv_bin = root / ".venv" / "Scripts"
                else:
                    venv_bin = root / ".venv" / "bin"
                if venv_bin.exists():
                    old_path = env.get("PATH") or ""
                    venv_bin_str = str(venv_bin)
                    if venv_bin_str not in old_path.split(os.pathsep):
                        env["PATH"] = venv_bin_str + os.pathsep + old_path
        except Exception:
            pass
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
                "ai-chat-util-config.yml で 'ai_chat_util_config.llm.api_key: os.environ/LLM_API_KEY' または "
                "'ai_chat_util_config.autonomous_agent_util.llm.api_key: os.environ/LLM_API_KEY' のように参照設定してください。"
            )
        if api_key:
            env["LLM_API_KEY"] = str(api_key)

        # Some external runners read OpenAI-style environment variables, not this
        # project's LLM_* variables. Map them as a best-effort compatibility layer.
        if api_key and not env.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = str(api_key)

        base_url = runtime_cfg.llm.base_url or env.get("LLM_BASE_URL")
        if base_url:
            openai_base_url = str(base_url).rstrip("/")
            if not openai_base_url.endswith("/v1"):
                openai_base_url = openai_base_url + "/v1"
            env.setdefault("OPENAI_BASE_URL", openai_base_url)
            env.setdefault("OPENAI_API_BASE", openai_base_url)
            env.setdefault("OPENAI_API_BASE_URL", openai_base_url)
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
        # Useful for debugging "no logs created" cases.
        self.task_status.metadata["python_executable"] = str(python_exe)
        self.task_status.metadata["pid"] = int(proc.pid)
        return SubprocessRunResult(pid=int(proc.pid))

    @classmethod
    def resolve_python_executable(cls) -> str:
        """Select a Python executable capable of importing this project.

        Some Windows setups start the MCP server with the system Python. In that
        case, `sys.executable` cannot import `ai_chat_util` when spawning a
        child process, causing detached tasks to fail with no logs.

        Prefer the project's `.venv` interpreter when it exists.
        """

        override = (
            os.environ.get("AI_CHAT_UTIL_PYTHON_EXECUTABLE")
            or os.environ.get("AI_CHAT_UTIL_PYTHON")
            or os.environ.get("AUTONOMOUS_AGENT_PYTHON_EXECUTABLE")
        )
        if override:
            try:
                cand = pathlib.Path(override).expanduser()
                if cand.exists():
                    return str(cand)
            except Exception:
                # Fall back to auto-detection.
                pass

        root = _find_project_root(pathlib.Path(__file__))
        if root is not None:
            if os.name == "nt":
                cand = root / ".venv" / "Scripts" / "python.exe"
            else:
                cand = root / ".venv" / "bin" / "python"
            if cand.exists():
                return str(cand)

        return sys.executable
