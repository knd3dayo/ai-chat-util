from typing import Generator, Any
from abc import ABC, abstractmethod

# 内部パッケージのインポート
from ai_chat_util.app.agent.agent_util_models import TaskStatus
class AbstractActions(ABC):
    @abstractmethod
    def after_start_task_action(self, tid: str) -> None:
        pass

    @abstractmethod
    def after_start_detach_task_action(self, tid: str) -> None:
        pass

    @abstractmethod
    async def progress_action(self, tid: str) -> TaskStatus:
        pass

    @abstractmethod
    def after_complete_action(
            self, runner: Any) -> None:
        pass
    @abstractmethod
    def after_task_not_found_action(self) -> None:
        pass
    @abstractmethod
    def after_list_action(self, table: list) -> None:
        pass
    @abstractmethod
    def after_cancel_action(self, task_id: str, result: dict[str, Any] | None = None) -> None:
        pass
    @abstractmethod
    def after_get_status_action(self, task_id: str, status_data: TaskStatus) -> None:
        pass

    @abstractmethod
    def prune_progress_action(self, generator: Generator[str, None, None]):
        pass