from collections import deque
from typing import Any, Dict, Optional, List, ClassVar, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_serializer
import os
import uuid
from ..common.config.runtime import get_coding_runtime_config

from typing import Optional
from pydantic import BaseModel, Field

class CodingAgentConfig(BaseModel):

    env_file: ClassVar[str] = ".env"  # デフォルトの環境変数ファイルパス

    workspace_root: str = Field(default="/tmp/coding_agent_tasks", description="Root directory for task workspaces")
    
    @classmethod
    def set_env_file(cls, env_file: str):
        cls.env_file = env_file
 
class ComposeConfig(BaseModel):

    env_file: ClassVar[str] = ".env"  # デフォルトの環境変数ファイルパス
    
    compose_directory: str = Field(..., description="Path to the directory containing docker-compose.yml")
    compose_file: str = Field(..., description="Name of the docker-compose file")
    compose_service_name: str = Field(..., description="Name of the service in docker-compose to run")
    compose_command : str = Field(..., description="Command to execute in the container (overrides default)")

    @classmethod
    def set_env_file(cls, env_file: str):
        cls.env_file = env_file

    @classmethod
    def from_env(cls):
        cfg = get_coding_runtime_config()
        params = {
            "compose_directory": cfg.compose.directory,
            "compose_file": cfg.compose.file,
            "compose_service_name": cfg.compose.service_name,
            "compose_command": cfg.compose.command,
        }

        return cls(**params)

    def get_compose_path(self) -> str:
        return os.path.join(self.compose_directory, self.compose_file)

    def get_compose_paths(self) -> List[str]:
        """Return compose file paths, supporting docker-compose style COMPOSE_FILE lists.

        Docker Compose supports specifying multiple compose files via COMPOSE_FILE using
        the OS path separator (Linux/macOS: ':', Windows: ';').
        """
        raw = (self.compose_file or "").strip()
        if not raw:
            return [os.path.join(self.compose_directory, "docker-compose.yml")]

        # First split using OS path separator (Compose convention)
        parts = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
        # Allow comma-separated lists as a convenience
        if len(parts) == 1 and "," in parts[0]:
            parts = [p.strip() for p in parts[0].split(",") if p.strip()]

        paths: List[str] = []
        for part in parts:
            if os.path.isabs(part):
                paths.append(part)
            else:
                paths.append(os.path.join(self.compose_directory, part))
        return paths

class CodingAgentRequest(BaseModel):
    prompt: str = Field(..., examples=["hello.py を修正して"])
    initial_files: Optional[Dict[str, str]] = None # 事前に配置したいファイル
    timeout: int = Field(default=300, ge=1, le=1800)

class TaskStatus(BaseModel):
    task_id: str
    workspace_path: str
    # SV実行全体の相関ID（task_idとは別）。
    # executorのtask_idはワークスペース/compose名衝突回避のためサブタスク単位で一意化されうる。
    trace_id: Optional[str] = None
    status: Optional[Literal[
        "pending", "running", "exited"
        ]] = None
    sub_status: Optional[Literal[
        "not-started", "running-foreground", "running-background", "starting", "failed", "timeout", "cancelled", "completed"
        ]] = None  # より詳細な状態（例: "starting", "running", "exited"など）
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    artifacts: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    container_id: Optional[str] = None

    # 逐次通知/統合向けの拡張メタ情報。
    # SV層では server_logs(リングバッファ) 等を入れることがある。
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_serializer("metadata")
    def _serialize_metadata(self, metadata: Dict[str, Any]):
        if not isinstance(metadata, dict):
            return metadata

        server_logs = metadata.get("server_logs")
        if isinstance(server_logs, deque):
            # shallow copy して server_logs だけ list 化
            return {**metadata, "server_logs": list(server_logs)}

        return metadata

    @classmethod
    def create(cls, task_id: str, workspace_path: str) -> "TaskStatus":
        """新規タスク作成用のファクトリーメソッド"""
        return TaskStatus(
            task_id=task_id,
            status="pending",
            sub_status="not-started",
            created_at=datetime.now(timezone.utc),
            # workspace_pathがNoneの場合はワークスペースルート + task_id のパスを想定。実際のワークスペースパスはランナー側で設定される。
            workspace_path=workspace_path,
        )

    def pendding(self):
        self.status = "pending"
        self.sub_status = "not-started"

    def starting_foregrond(self):
        self.status = "running"
        self.sub_status = "running-foreground"
    
    def starting_background(self):
        self.status = "running"
        self.sub_status = "running-background"
    
    def timeouted(self, timeout: int):
        self.status = "exited"
        self.sub_status = "timeout"
        self.stderr=f"Task timed out after {timeout} seconds"
    
    def completed(self):
        self.status = "exited"
        self.sub_status = "completed"
    
    def failed(self):
        self.status = "exited"
        self.sub_status = "failed"

    def cancelled(self):
        self.status = "exited"
        self.sub_status = "cancelled"
        
    def is_exited(self) -> bool:
        return self.status == "exited"

class ExecuteRequest(BaseModel):
    prompt: str = Field(..., description="指示内容")
    workspace_path: str = Field(..., description="ホスト側の共有workspace（絶対パス）")
    timeout: int = Field(default=300, ge=1, le=1800)
    task_id: Optional[str] = Field(default=None, description="任意のtask_id（未指定なら自動採番）")
    trace_id: Optional[str] = Field(default=None, description="SV実行全体の相関ID")

class ExecuteResponse(BaseModel):
    task_id: str = Field(..., description="実行されたタスクのID")


class HealthzResponse(BaseModel):
    status: str = Field(default="ok", description="health status")

class CancelResponse(BaseModel):
    task_id: Optional[str] = Field(default=None, description="task id")
    status: Optional[str] = Field(default=None, description="task status")
    sub_status: Optional[str] = Field(default=None, description="task sub status")
    message: str = Field(default="cancel requested", description="result message")
