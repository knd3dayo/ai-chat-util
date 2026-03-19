from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ai_chat_util_base.model.agent_util_models import TaskStatus


class AbstractAgentRunner(ABC):
    """Execution backend runner.

    Runner is responsible for starting the execution backend (Docker container or
    local subprocess) and exposing the associated TaskStatus/workspace.
    """

    @abstractmethod
    def start(self) -> Any:
        """Start execution and return a backend-specific handle (e.g. Container, pid)."""
        raise NotImplementedError

    @abstractmethod
    def get_task_status(self) -> TaskStatus:
        """Return current TaskStatus object."""
        raise NotImplementedError

    @abstractmethod
    def get_workspace_path(self) -> Path:
        """Return workspace path."""
        raise NotImplementedError