import json
import os
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
    def __init__(self):
        self.servers = {}

    @staticmethod
    def _resolve_env_value(value: str, *, config_path: str, server_name: str, env_key: str) -> str:
        """Resolve env-ref syntax used across this project.

        mcp.json supports literal env values as-is, but also allows the
        `os.environ/VAR_NAME` form to reference a process env var.
        """

        prefix = "os.environ/"
        if not value.startswith(prefix):
            return value

        env_name = value[len(prefix) :].strip()
        if not env_name:
            raise ValueError(
                "Invalid env reference in mcp.json (empty env var name): "
                f"config={config_path!r} server={server_name!r} env.{env_key}={value!r}"
            )

        resolved = os.getenv(env_name)
        if resolved is None or resolved == "":
            raise ValueError(
                "Environment variable is not set for mcp.json env reference: "
                f"{env_name} (config={config_path!r} server={server_name!r} env.{env_key})"
            )
        return resolved

    def load_server_config(self, config_path: str):
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

            # Resolve env refs (os.environ/VAR_NAME) for stdio transports.
            env = cfg2.get("env")
            if env is not None:
                if not isinstance(env, dict):
                    raise ValueError(f"mcpServers.{name}.env must be an object")
                resolved_env: Dict[str, str] = {}
                for env_key, env_val in env.items():
                    if not isinstance(env_key, str) or not env_key.strip():
                        raise ValueError(
                            f"mcpServers.{name}.env has an invalid key (must be non-empty string): {env_key!r}"
                        )
                    if not isinstance(env_val, str):
                        raise ValueError(
                            f"mcpServers.{name}.env.{env_key} must be a string: {env_val!r}"
                        )
                    resolved_env[env_key] = self._resolve_env_value(
                        env_val,
                        config_path=config_path,
                        server_name=name,
                        env_key=env_key,
                    )
                cfg2["env"] = resolved_env

            # Many configs use the mcpServers.<key> as the server name.
            # MCPServerConfigEntry requires `name`, so inject it when omitted.
            if "name" not in cfg2:
                cfg2["name"] = name
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


    def get_allowed_tools_config(self) -> "MCPServerConfig":
        """
        ツール制限用のマッピングを返します（後続のフィルタリング用）。
        """
        allowed_servers =  {name: cfg.allowed_tools for name, cfg in self.servers.items()}
        allowed_mcp_config = MCPServerConfig()
        allowed_mcp_config.servers = allowed_servers

        return allowed_mcp_config
    
    def filter(self, include_name: str |None = None, exclude_name: str| None = None) -> "MCPServerConfig":
        """
        指定されたサーバー名やツール名に基づいて、MCPServerConfigをフィルタリングします。
        - include_name: このサーバー名を含むサーバーのみを許可
        - exclude_name: このサーバー名を除外するサーバーのみを許可

        """
        filtered_config = MCPServerConfig()
        for name, cfg in self.servers.items():
            if include_name is not None and name != include_name:
                continue
            if exclude_name is not None and name == exclude_name:
                continue
            filtered_config.servers[name] = cfg

        return filtered_config