from __future__ import annotations

import os
import pathlib
import shlex
import sys
import uuid
from dataclasses import dataclass
from typing import Optional, Union

from ..abstract_agent_runner import AbstractAgentRunner

from ai_chat_util_base.model.agent_util_models import TaskStatus, CodingAgentConfig
from ..utils import ExecutorUtil
from ..process_utils import popen_new_process_group_kwargs
from ai_chat_util_base.config.ai_chat_util_runtime import get_autonomous_runtime_config


def _find_project_root(start: pathlib.Path) -> pathlib.Path | None:
    p = start.resolve()
    for parent in (p, *p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return None


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

        self.command_base = command_base or (runtime_cfg.subprocess.command or runtime_cfg.compose.command)
        self.command: list[str] = shlex.split(self.command_base)
        # Windows: wrap common shell shims (`*.cmd` / `*.bat` / `*.ps1`) so they
        # work under `shell=False` detached execution.
        self.command = _windows_wrap_shell_shims(self.command)

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
        python_exe = self.resolve_python_executable()
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

        # Ensure tools installed into the project's virtualenv (e.g., opencode)
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
                "autonomous-agent-util-config.yml で 'autonomous_agent_util_config.llm.api_key: os.environ/LLM_API_KEY' のように参照設定してください。"
            )
        if api_key:
            env["LLM_API_KEY"] = str(api_key)

        # opencode (and other external runners) typically read OpenAI-style
        # environment variables, not this project's LLM_* variables.
        # Map them so `opencode run -m openai/...` can use LiteLLM seamlessly.
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
