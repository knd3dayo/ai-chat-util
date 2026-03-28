import asyncio
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, AsyncGenerator, Generator
import signal
from python_on_whales import docker as whales, Container

# 内部パッケージのインポート
from ..abstract_agent_runner import AbstractAgentRunner
from ..abstract_task_service import AbstractTaskService
from .docker_coding_agent_runner import CodingAgentRunner
from ai_chat_util_base.model.agent_util_models import TaskStatus

from ...util.logging import get_application_logger
from ..process_utils import popen_new_process_group_kwargs
from ai_chat_util_base.config.runtime import (
    get_coding_runtime_config,
    get_coding_runtime_config_path,
)

logger = get_application_logger()


# --- Logic Layer: Typerに依存しないサービス ---
class DockerTaskService(AbstractTaskService):

    def __init__(self):
        self.runner: Optional[CodingAgentRunner] = None
        self.container: Optional[Container] = None

    def spawn_detached_monitor(self, task_id: str, timeout: int) -> None:
        """wait=False の場合でも tasks_db.json が自動更新されるよう、別プロセスで monitor を起動する。"""
        cfg = get_coding_runtime_config()
        if cfg.monitor.disable_detach_monitor:
            return

        # timeout に少し余裕を持たせる（ログ保存・docker の状態反映の遅延吸収）
        max_seconds = max(int(timeout) + 60, 120)
        interval = float(cfg.monitor.detach_monitor_interval)

        env = os.environ.copy()
        cfg_path = get_coding_runtime_config_path()
        if cfg_path:
            env.setdefault("AI_CHAT_UTIL_CONFIG", str(cfg_path))

        cmd = [
            sys.executable,
            "-m",
            "coding_agent_util._cli_.docker_main",
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
        if task.status != "running" or not task.container_id:
            return
        whales.container.kill(task.container_id)

    async def prepare(
        self,
        prompt: str,
        sources: Optional[list[Path]],
        task_id: Optional[str],
        workspace_path: Optional[Path] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> None:
        params = {
            "prompt": prompt,
            "task_id": task_id,
        }
        if sources:
            params["source_paths"] = sources

        if workspace_path is not None:
            params["workspace_path"] = workspace_path

        if extra_env:
            params["extra_env"] = extra_env

        self.runner = await CodingAgentRunner.create_runner(**params)

    def start(self, *, wait: bool, timeout: int) -> TaskStatus:
        if self.runner is None:
            raise RuntimeError("Runner not initialized")

        self.container = self.runner.start()

        task_status = self.runner.get_task_status()
        # container_id が無いと logs/状態取得ができず、running のまま固まるため必ず保存する
        task_status.container_id = getattr(self.container, "id", None)

        if wait:
            task_status.starting_foregrond()
        else:
            task_status.starting_background()

        return task_status
        
    def get_agent_runner(self) -> AbstractAgentRunner:
        """コーディングエージェントのランナーを返す。"""
        if not self.runner:
            raise RuntimeError("Runner not initialized")
        return self.runner

    
    async def monitor(self, timeout: int) -> AsyncGenerator[TaskStatus, None]:
        if not self.container or not self.runner:
            return
        async for status in DockerTaskService.monitor_container(self.container, self.runner, timeout):
            yield status

    @classmethod
    def print_confg(cls, container: Container) -> None:
        """Container の設定をデバッグ出力する（ENVは秘匿して出す）。"""
        try:
            cfg = getattr(container, "config", None)
            args = getattr(container, "args", None)

            safe_env = None
            env_list = getattr(cfg, "env", None) if cfg is not None else None
            if isinstance(env_list, list):
                safe_env = []
                for item in env_list:
                    if not isinstance(item, str) or "=" not in item:
                        safe_env.append(item)
                        continue
                    key, _value = item.split("=", 1)
                    upper = key.upper()
                    if any(k in upper for k in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                        safe_env.append(f"{key}=***")
                    else:
                        safe_env.append(item)

            logger.debug(
                "container.debug id=%s image=%s cmd=%s entrypoint=%s env=%s",
                getattr(container, "id", None),
                getattr(cfg, "image", None) if cfg is not None else None,
                getattr(cfg, "cmd", None) if cfg is not None else None,
                getattr(cfg, "entrypoint", None) if cfg is not None else None,
                safe_env,
            )
            logger.debug("container.args=%s", args)
        except Exception:
            # デバッグログの失敗で本処理を落とさない
            pass

    @classmethod
    async def monitor_container(cls, container: Container, runner: AbstractAgentRunner, timeout: int) -> AsyncGenerator[TaskStatus, None]:
        # debug用: container設定/引数は秘匿情報(ENV)を含みうるため、デフォルトでは出さない。
        if get_coding_runtime_config().monitor.debug_container:
            cls.print_confg(container)
        try:
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            while True:
                # reload() で最新の状態（state）を同期
                container.reload()
                # debug用
                # print(f"Monitoring container {container.id[:12]}: status={container.state.status}, elapsed={int(asyncio.get_event_loop().time() - start_time)}s")
                # whales のコンテナ状態チェック
                if container.state.status == 'exited':
                    break
                
                if (loop.time() - start_time) > timeout:
                    container.kill()
                    task  = runner.get_task_status()
                    if task:
                        task.timeouted(timeout)
                    if task:
                        yield task
                    return
                await asyncio.sleep(1)

            # 終了コードの取得
            exit_code = container.state.exit_code
            
            # ログ取得 (whales では直接 logs() が呼べ、文字列で返ります)
            out_logs = container.logs()
            
            artifacts = [str(f.relative_to(runner.get_workspace_path())) for f in runner.get_workspace_path().glob("**/*") if f.is_file()]
            task  = runner.get_task_status()
            if task:
                if exit_code == 0:
                    task.completed()
                else:
                    task.failed()
                task.stdout = out_logs
                task.artifacts = artifacts
                yield task
 
        except Exception as e:
            task  = runner.get_task_status()
            if task:
                task.failed()
                task.stderr = f"Monitor Error: {str(e)}"
                yield task
        finally:
            # 既に remove=True で起動しているが、念のため
            try: container.remove(force=True)
            except: pass

    @classmethod
    def prune_containers(cls, compose_service_name: str) -> Generator[str, None, None]:
        """掃除ロジック"""
        containers = whales.container.list(filters={"label": f"managed_by={compose_service_name}"})
        for c in containers:
            c.remove(force=True)
            yield f"Removed container {c.id[:12]}"



