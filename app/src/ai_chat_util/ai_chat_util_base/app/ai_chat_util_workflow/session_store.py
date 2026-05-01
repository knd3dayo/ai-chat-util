from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ai_chat_util.ai_chat_util_base.core.common.config.runtime import AiChatUtilConfig, get_runtime_config, get_runtime_config_path


class WorkflowSessionRecord(BaseModel):
    trace_id: str = Field(...)
    phase: Literal["plan", "graph"] = Field(default="plan")
    workflow_file_path: str = Field(default="")
    original_markdown: str = Field(default="")
    prepared_markdown: str = Field(default="")
    message: str = Field(default="")
    max_node_visits: int = Field(default=8, ge=1)


class WorkflowSessionStore:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_runtime_config(cls, runtime_config: AiChatUtilConfig | None = None) -> "WorkflowSessionStore":
        config = runtime_config or get_runtime_config()
        base_dir = config.mcp.working_directory or str(get_runtime_config_path().parent)
        root_dir = Path(base_dir).expanduser().resolve() / ".ai_chat_util" / "workflow_sessions"
        return cls(root_dir)

    def load(self, trace_id: str) -> WorkflowSessionRecord | None:
        path = self._path_for(trace_id)
        if not path.exists():
            return None
        return WorkflowSessionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, record: WorkflowSessionRecord) -> None:
        path = self._path_for(record.trace_id)
        path.write_text(json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")

    def delete(self, trace_id: str) -> None:
        path = self._path_for(trace_id)
        if path.exists():
            path.unlink()

    def _path_for(self, trace_id: str) -> Path:
        normalized = "".join(ch for ch in str(trace_id).strip().lower() if ch.isalnum() or ch in {"-", "_"})
        return self.root_dir / f"{normalized or 'workflow'}.json"