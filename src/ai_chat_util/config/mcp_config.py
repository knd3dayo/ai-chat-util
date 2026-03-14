import json
import asyncio
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field

from langchain_mcp_adapters.sessions import Connection

# サーバー設定のモデル定義
class MCPServerConfig(BaseModel):
    type: Literal["stdio", "sse", "http"] = "stdio" # httpもsseとして扱うのが一般的
    command: Optional[str] = None
    args: Optional[List[str]] = Field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    # 許可するツール名のリスト (Noneなら全許可)
    allowed_tools: Optional[List[str]] = Field(default=None, alias="allowedTools")

    class Config:
        populate_by_name = True

class MCPConfigParser:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            data = json.load(f)
        self.servers: Dict[str, MCPServerConfig] = {
            name: MCPServerConfig(**cfg) 
            for name, cfg in data.get("mcpServers", {}).items()
        }


    def to_langchain_config(self) -> dict[str, Connection]:
        """
        langchain_mcp_adapters.client.MultiServerMCPClient の引数として
        そのまま渡せる辞書を生成します。
        """
        lc_config = {}
        for name, cfg in self.servers.items():
            if cfg.type == "stdio":
                # stdio形式のパラメータ
                lc_config[name] = {
                    "command": cfg.command,
                    "args": cfg.args,
                    "env": cfg.env
                }
            elif cfg.type in ["sse", "http"]:
                # SSE/HTTP形式のパラメータ
                lc_config[name] = {
                    "url": cfg.url,
                    "headers": cfg.headers
                }
        return lc_config

    def get_allowed_tools_map(self) -> Dict[str, Optional[List[str]]]:
        """
        ツール制限用のマッピングを返します（後続のフィルタリング用）。
        """
        return {name: cfg.allowed_tools for name, cfg in self.servers.items()}