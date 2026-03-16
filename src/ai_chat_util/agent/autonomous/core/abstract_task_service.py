from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, Optional

from ai_chat_util_base.model.autonomous_agent_util_models import TaskStatus
from .abstract_agent_runner import AbstractAgentRunner


class AbstractTaskService(ABC):
    """Backend-agnostic task service.

    - `prepare()` creates/configures the runner.
    - `start()` starts the backend synchronously and returns the initial TaskStatus.
    - `monitor()` converges the task to a final exited status.
    """

    @abstractmethod
    async def prepare(
        self,
        prompt: str,
        sources: Optional[list[Path]],
        task_id: Optional[str],
        workspace_path: Optional[Path] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def start(self, *, wait: bool, timeout: int) -> TaskStatus:
        raise NotImplementedError

    @abstractmethod
    def get_agent_runner(self) -> AbstractAgentRunner:
        raise NotImplementedError

    @abstractmethod
    def spawn_detached_monitor(self, task_id: str, timeout: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def cancel_task(self, task: TaskStatus) -> None:
        raise NotImplementedError

    @abstractmethod
    def monitor(self, timeout: int) -> AsyncGenerator[TaskStatus, None]:
        raise NotImplementedError


