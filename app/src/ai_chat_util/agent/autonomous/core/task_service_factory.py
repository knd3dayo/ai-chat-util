from __future__ import annotations

from .abstract_task_service import AbstractTaskService
from .docker.docker_task_service import DockerTaskService
from .subprocess.subprocess_task_service import SubprocessTaskService
from ai_chat_util_base.config.runtime import get_autonomous_runtime_config


def select_task_service(backend: str | None = None) -> AbstractTaskService:
    cfg = get_autonomous_runtime_config()
    b = (backend or cfg.backend.task_backend or "process").strip().lower()
    if b in ("docker", "compose"):
        return DockerTaskService()
    if b in ("subprocess", "process", "windows_process", "linux_process"):
        return SubprocessTaskService()
    raise ValueError(f"Unknown AI_PLATFORM_TASK_BACKEND: {b}")
