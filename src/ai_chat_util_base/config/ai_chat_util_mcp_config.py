import json
from typing import Dict, Optional, List, Literal

from pydantic import BaseModel, Field

try:
    # Pydantic v2
    from pydantic import AliasChoices  # type: ignore
except Exception:  # pragma: no cover
    AliasChoices = None  # type: ignore

from langchain_mcp_adapters.sessions import Connection

# サーバー設定のモデル定義
class MCPServerConfigEntry(BaseModel):
    # langchain-mcp-adapters は `transport` を要求する。
    # 既存設定との互換のため、入力では `type` も受け付ける。
    if AliasChoices is not None:
        transport: Literal["stdio", "sse", "websocket", "http"] = Field(
            default="stdio",
            validation_alias=AliasChoices("transport", "type"),
            serialization_alias="transport",
        )
    else:  # pragma: no cover
        transport: Literal["stdio", "sse", "websocket", "http"] = Field(default="stdio", alias="transport")

    name: str = Field(..., description="サーバーの識別名。")
    command: Optional[str] = None
    args: Optional[List[str]] = Field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    # 許可するツール名のリスト (Noneなら全許可)
    allowed_tools: Optional[List[str]] = Field(default=None, alias="allowedTools")

    class Config:
        populate_by_name = True

class MCPServerConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            data = json.load(f)

        raw_servers = data.get("mcpServers", {})
        if not isinstance(raw_servers, dict):
            raise ValueError("mcpServers must be an object")

        normalized: Dict[str, dict] = {}
        for name, cfg in raw_servers.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"mcpServers.{name} must be an object")

            cfg2 = dict(cfg)
            # Backward/compat input: allow `type` but normalize it to `transport`.
            if "transport" not in cfg2 and "type" in cfg2:
                cfg2["transport"] = cfg2.get("type")
            normalized[name] = cfg2

        self.servers = {name: MCPServerConfigEntry(**cfg) for name, cfg in normalized.items()}


    def to_langchain_config(self) -> dict[str, Connection]:
        """
        langchain_mcp_adapters.client.MultiServerMCPClient の引数として
        そのまま渡せる辞書を生成します。
        """
        lc_config = {}
        for name, cfg in self.servers.items():
            transport = cfg.transport
            if transport == "stdio":
                # stdio形式のパラメータ
                lc_config[name] = {
                    "transport": "stdio",
                    "command": cfg.command,
                    "args": cfg.args,
                    "env": cfg.env,
                }
            elif transport in {"sse", "http", "websocket"}:
                # SSE/HTTP/WebSocket形式のパラメータ
                lc_config[name] = {
                    "transport": transport,
                    "url": cfg.url,
                    "headers": cfg.headers,
                }
        return lc_config

    def get_allowed_tools_map(self) -> Dict[str, Optional[List[str]]]:
        """
        ツール制限用のマッピングを返します（後続のフィルタリング用）。
        """
        return {name: cfg.allowed_tools for name, cfg in self.servers.items()}