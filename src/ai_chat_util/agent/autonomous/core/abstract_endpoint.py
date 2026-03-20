import os
import pathlib
from pathlib import Path
from typing import Any, Dict, Optional, Annotated
from abc import ABC, abstractmethod
from fastapi import Body, Path, Query

from .....ai_chat_util_base.model.agent_util_models import (
    ExecuteRequest,
    ExecuteResponse,
    HealthzResponse,
    TaskStatus,
    CancelResponse,
)
class AutonomousEndPointBase(ABC):

    @abstractmethod
    def rewrite_workspace_path(self, workspace_path: str) -> str:
        """Rewrite inbound workspace_path based on config.

        This is useful when the supervisor and executor run in different
        environments (host vs container) and see the same workspace under
        different absolute paths.
        """
        pass

    @abstractmethod
    def validate_workspace_path(self, workspace_path: str) -> pathlib.Path:
        """Validate the workspace_path and return the resolved Path object.

        This is a critical security measure to prevent directory traversal
        attacks or unauthorized access. The implementation should ensure that
        the provided workspace_path is within an allowed base directory and does
        not contain any malicious path components.
        """
        pass


    @abstractmethod
    async def healthz(self) -> HealthzResponse:
        """
        ヘルスチェック用エンドポイント。SVはこれを叩いてエージェントが生きているか確認する。
        """
        pass


    @abstractmethod
    async def _execute_main_(
        self,
        wait_for_completion: bool,
        req: Annotated[ExecuteRequest, Body(description="タスク実行リクエスト")],
    ) -> ExecuteResponse:
        """
        タスク実行用エンドポイント（同期版/非同期版）の共通処理。SVはこれを叩いてエージェントにタスク実行を指示する。
        引数のwait_for_completionで、タスク完了までHTTPレスポンスを返すかどうかを制御する。
        """
        pass


    @abstractmethod
    async def execute_sync(
        self,
        req: Annotated[ExecuteRequest, Body(description="タスク実行リクエスト")],
    ) -> ExecuteResponse:
        """
        タスク実行用エンドポイント（同期版）。ユーザーはこれを叩いてエージェントにタスク実行を指示する。
        executeとの違いは、タスク完了までHTTPレスポンを返さない点。小規模タスクやテスト用途向け。
        """
        pass


    @abstractmethod
    async def execute_async(
        self,
        req: Annotated[ExecuteRequest, Body(description="タスク実行リクエスト")],
    ) -> ExecuteResponse:
        """
        タスク実行用エンドポイント（非同期版）。ユーザーはこれを叩いてエージェントにタスク実行を指示する。
        処理が完了する前にHTTPレスポンスを返す。
        ユーザーは/statusエンドポイントを叩いてタスクの進捗や結果を取得する。   
        キャンセルを行う場合は、/cancelエンドポイントを叩く。
        """
        pass

    @abstractmethod
    # タスクが使用しているワークスペースのパスを返すエンドポイント
    async def workspace_path(
        self,
        task_id: Annotated[str, Path(description="task id")],
    ) -> Dict[str, str]:
        """
        タスクのワークスペースパス取得用エンドポイント。SVはこれを叩いてタスクのワークスペースパスを取得する。
        """
        pass

    @abstractmethod
    async def status(
        self,
        task_id: Annotated[str, Path(description="task id")],
        tail: Annotated[
            Optional[int],
            Query(description="ログの末尾 n 行（省略時は 200、null で全量）", ge=0),
        ] = 200,
    ) -> TaskStatus:
        """
         タスクステータス取得用エンドポイント。SVはこれを叩いてタスクの進捗や結果を取得する。
         tailはログの末尾n行を取得するためのパラメータ。
         既存のTaskStatus(JSONをそのまま返す。SV側は status/sub_status を見て完了判定する。
        """
        pass

    @abstractmethod
    async def get_result(
        self,
        task_id: Annotated[str, Path(description="task id")],
        tail: Annotated[
            Optional[int],
            Query(description="ログの末尾 n 行（省略時は 200、null で全量）", ge=0),
        ] = 200,
    ) -> Dict[str, Any]:
        """
         タスク結果取得用エンドポイント。SVはこれを叩いてタスクの結果を取得する。
         tailはログの末尾n行を取得するためのパラメータ。
        """
        pass

    @abstractmethod
    async def cancel(self, task_id: Annotated[str, Path(description="task id")]) -> CancelResponse:
        """
         タスクキャンセル用エンドポイント。SVはこれを叩いてタスクのキャンセルを指示する。
        """
        pass

