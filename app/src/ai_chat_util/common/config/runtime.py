from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator, ConfigDict
from .ai_chat_util_mcp_config import MCPServerConfigEntry, MCPServerConfig
from ai_chat_util.ai_chat_util_base.analyze_file_util.util.file_path_resolver import resolve_existing_file_path

from .config_util import (
    CONFIG_ENV_VAR,
    CODING_AGENT_UTIL_CONFIG_ROOT_KEY,
    CODING_DEFAULT_CONFIG_FILENAME,
    ConfigError, load_resolved_yaml, load_yaml_config, resolve_config_path, resolve_coding_config_path,
    resolve_env_ref,extract_required_root_section, extract_optional_ai_section_dict,
    apply_secret_overrides_from_yaml,
    AI_CHAT_UTIL_CONFIG_ROOT_KEY,
    AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME,
    _ENV_REF_PREFIX,
    resolve_path_placeholders,
    
)
import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

# =====================
# Coding Agent Util Runtime (embedded)
# =====================

_runtime_state: _RuntimeState | None = None
_coding_runtime_state: _CodingRuntimeState | None = None


def init_runtime(config_path: str | None = None) -> AiChatUtilConfig:
    global _runtime_state

    resolved, raw_root = load_resolved_yaml(config_path, resolver=resolve_config_path)
    config = _build_ai_chat_util_config(raw_root=raw_root, resolved=resolved)

    _runtime_state = _RuntimeState(config_path=resolved, config=config)
    _apply_non_secret_runtime_side_effects(config)

    return config


def init_coding_runtime(config_path: str | None = None) -> CodingAgentUtilConfig:
    global _coding_runtime_state

    resolved, raw_root = load_resolved_yaml(config_path, resolver=resolve_coding_config_path)
    config = _build_coding_agent_util_config(raw_root=raw_root, resolved=resolved)

    # Use model_construct() to avoid re-validating `config`.
    # At this point, env-ref secrets (e.g., llm.api_key) may already be resolved
    # into literal values, and re-validation would incorrectly re-trigger the
    # secret-policy validator.
    _coding_runtime_state = _CodingRuntimeState.model_construct(config_path=resolved, config=config)
    _configure_python_logging(config)
    return config



def apply_logging_overrides(level: str | None = None, file: str | None = None) -> None:
    """Apply process-local logging overrides.

    This is intended for CLI flags like --loglevel/--logfile.
    It does not write to environment variables.
    """
    cfg = get_runtime_config()
    effective = cfg.model_copy(deep=True)

    if level:
        effective.logging.level = level
    if file:
        effective.logging.file = file

    _configure_python_logging(effective)


def get_runtime_config() -> AiChatUtilConfig:
    if _runtime_state is None:
        return init_runtime(None)
    return _runtime_state.config

def get_coding_runtime_config() -> CodingAgentUtilConfig:
    if _coding_runtime_state is None:
        return init_coding_runtime(None)
    return _coding_runtime_state.config


def get_coding_runtime_config_path() -> Path:
    if _coding_runtime_state is None:
        init_coding_runtime(None)
    assert _coding_runtime_state is not None
    return _coding_runtime_state.config_path


def get_runtime_config_path() -> Path:
    if _runtime_state is None:
        init_runtime(None)
    assert _runtime_state is not None
    return _runtime_state.config_path


def get_runtime_config_info() -> dict[str, Any]:
    """Return the resolved config file path and the raw file content.

    The returned `config` value is loaded directly from the YAML file so that
    env references remain unresolved and secrets are not materialized.
    """
    config_path = get_runtime_config_path()
    raw_root = load_yaml_config(config_path)
    return {
        "path": str(config_path),
        "config": raw_root,
    }


def _apply_non_secret_runtime_side_effects(config: AiChatUtilConfig) -> None:
    _configure_python_logging(config)
    _configure_litellm(config)


def _build_redacting_formatter(fmt: str):
    import logging
    import re

    class _RedactingFormatter(logging.Formatter):
        _replacements: list[tuple[re.Pattern[str], str]] = [
            # common api_key patterns
            (re.compile(r"(api_key\s*=\s*)(['\"])[^'\"]+\2", re.IGNORECASE), r"\1\2***\2"),
            (re.compile(r"(api_key\s*:\s*)(['\"])[^'\"]+\2", re.IGNORECASE), r"\1\2***\2"),
            # Bearer tokens
            (re.compile(r"(Authorization\s*:\s*Bearer\s+)[^\s\"]+", re.IGNORECASE), r"\1***"),
            # OpenAI-style keys (best-effort)
            (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), "sk-***"),
        ]

        def format(self, record: logging.LogRecord) -> str:
            s = super().format(record)
            for pattern, repl in self._replacements:
                s = pattern.sub(repl, s)
            return s

    return _RedactingFormatter(fmt)


def _configure_python_logging(config: AiChatUtilConfig | CodingAgentUtilConfig) -> None:
    # Keep this lightweight but deterministic: CLI/test runs often reconfigure logging.
    # We reset handlers to avoid duplicated output and to enforce UTF-8 file encoding.
    import logging

    level_name = (config.logging.level or "INFO").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    formatter: logging.Formatter = _build_redacting_formatter(
        "%(asctime)s - %(levelname)s - %(pathname)s -  %(lineno)d - %(funcName)s - %(message)s"
    )

    # Reset handlers to avoid duplicates across repeated init/apply overrides.
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    if config.logging.file:
        file_handler = logging.FileHandler(config.logging.file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    # Prevent accidental secret leakage / huge logs from 3rd-party debug.
    noisy_loggers = {
        # LiteLLM uses several logger names depending on version
        "litellm": logging.WARNING,
        "LiteLLM": logging.WARNING,
        # OpenAI SDK can log request details at DEBUG
        "openai": logging.WARNING,
        # LangGraph checkpointing uses aiosqlite and can emit verbose SQL logs at DEBUG
        "aiosqlite": logging.WARNING,
        # HTTP clients
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "aiohttp": logging.WARNING,
        "urllib3": logging.WARNING,
        # pysmb can log authentication details such as username at INFO
        "SMB": logging.WARNING,
        "SMB.SMB": logging.WARNING,
        "SMB.SMBConnection": logging.WARNING,
        "SMB.SMBProtocol": logging.WARNING,
        "SMB.SMBFactory": logging.WARNING,
        "SMB.SMBMessage": logging.WARNING,
        "SMB.SMB2Message": logging.WARNING,
        # event loop chatter
        "asyncio": logging.WARNING,
        # sse_starlette.sse
        "sse_starlette": logging.WARNING,
    }
    for name, lvl in noisy_loggers.items():
        logging.getLogger(name).setLevel(lvl)


def _configure_litellm(config: AiChatUtilConfig) -> None:
    # Non-secret: base_url can be configured here.
    try:
        import litellm
    except Exception:
        return

    # Windows で aiohttp transport 経由の実行後にプロセスが異常終了することがあるため、
    # 既定では httpx transport を使う（必要なら env: DISABLE_AIOHTTP_TRANSPORT=False で上書き可能）。
    try:
        if os.name == "nt" and getattr(litellm, "disable_aiohttp_transport", False) is False:
            litellm.disable_aiohttp_transport = True
    except Exception:
        pass

    if config.llm.base_url:
        litellm.api_base = config.llm.base_url

    if config.llm.api_version:
        # LiteLLM uses api_version for some providers (e.g. Azure).
        litellm.api_version = config.llm.api_version


def _expand_mapping_path_field(
    container: dict[str, Any],
    section_key: str,
    field_key: str,
    *,
    config_path: Path,
    field_prefix: str,
) -> None:
    section = container.get(section_key)
    if not isinstance(section, dict):
        return
    value = section.get(field_key)
    if isinstance(value, str) and value.strip():
        section[field_key] = resolve_path_placeholders(
            value,
            config_path=config_path,
            field_path=f"{field_prefix}{section_key}.{field_key}",
        )


def _expand_allowlisted_ai_paths(raw: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    copied = dict(raw)

    for section_key, field_key in (
        ("mcp", "mcp_config_path"),
        ("mcp", "custom_instructions_file_path"),
        ("mcp", "working_directory"),
        ("logging", "file"),
        ("features", "audit_log_path"),
        ("network", "ca_bundle"),
        ("office2pdf", "libreoffice_path"),
    ):
        _expand_mapping_path_field(
            copied,
            section_key,
            field_key,
            config_path=config_path,
            field_prefix=f"{AI_CHAT_UTIL_CONFIG_ROOT_KEY}.",
        )

    file_server = copied.get("file_server")
    if isinstance(file_server, dict):
        allowed_roots = file_server.get("allowed_roots")
        if isinstance(allowed_roots, list):
            for idx, root in enumerate(allowed_roots):
                if not isinstance(root, dict):
                    continue
                value = root.get("path")
                if isinstance(value, str) and value.strip():
                    root["path"] = resolve_path_placeholders(
                        value,
                        config_path=config_path,
                        field_path=(
                            f"{AI_CHAT_UTIL_CONFIG_ROOT_KEY}.file_server.allowed_roots[{idx}].path"
                        ),
                    )

    return copied


def _expand_allowlisted_coding_paths(raw: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    copied = dict(raw)

    for section_key, field_key in (
        ("paths", "workspace_root"),
        ("paths", "host_projects_root"),
        ("paths", "executor_allowed_workspace_root"),
        ("compose", "directory"),
        ("compose", "file"),
        ("logging", "file"),
    ):
        _expand_mapping_path_field(
            copied,
            section_key,
            field_key,
            config_path=config_path,
            field_prefix="coding_agent_util.",
        )

    paths = copied.get("paths")
    if isinstance(paths, dict):
        rewrites = paths.get("workspace_path_rewrites")
        if isinstance(rewrites, list):
            for idx, rule in enumerate(rewrites):
                if not isinstance(rule, dict):
                    continue
                for key in ("from", "to"):
                    value = rule.get(key)
                    if isinstance(value, str) and value.strip():
                        rule[key] = resolve_path_placeholders(
                            value,
                            config_path=config_path,
                            field_path=f"coding_agent_util.paths.workspace_path_rewrites[{idx}].{key}",
                        )

    return copied


def _build_ai_chat_util_config(*, raw_root: dict[str, Any], resolved: Path) -> AiChatUtilConfig:
    raw_section = extract_required_root_section(raw_root=raw_root, resolved=resolved)

    # This project intentionally does NOT support the deprecated `ai_chat_util_config.paths`.
    # The new config surface is `ai_chat_util_config.mcp`.
    if "paths" in raw_section:
        raise ConfigError(
            "設定ファイルの形式が古いか、キーが誤っています: "
            f"{resolved}\n"
            "'ai_chat_util_config.paths' はサポートされません。'ai_chat_util_config.mcp' に移行してください。"
        )

    # Secrets may be declared in YAML only via env reference (os.environ/VAR).
    raw = apply_secret_overrides_from_yaml(
        dict(raw_section),
        config_path=resolved,
        field_prefix=f"{AI_CHAT_UTIL_CONFIG_ROOT_KEY}.",
    )
    raw = _expand_allowlisted_ai_paths(raw, config_path=resolved)

    try:
        return AiChatUtilConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"設定ファイルの検証に失敗しました: {resolved}\n{e}") from e


def _build_coding_agent_util_config(
    *, raw_root: dict[str, Any], resolved: Path
) -> CodingAgentUtilConfig:
    raw: dict[str, Any]

    ai_section_dict = extract_optional_ai_section_dict(raw_root=raw_root, resolved=resolved)

    # Integrated format:
    # - ai-chat-util-config.yml root contains ONLY:
    #     - ai_chat_util_config: ...
    #     - coding_agent_util: ...
    nested_embedded_key = next(
        (
            key for key in ("coding_agent_util",)
            if ai_section_dict is not None and ai_section_dict.get(key, "__missing__") != "__missing__"
        ),
        None,
    )
    if nested_embedded_key is not None:
        raise ConfigError(
            "設定ファイルの形式が不正です: "
            f"{resolved}\n"
            "統合設定の coding-agent-util は 'ai_chat_util_config.coding_agent_util' ではなく、"
            "ルート直下の 'coding_agent_util:' に記述してください。"
        )

    root_embedded = "__missing__"
    if isinstance(raw_root, dict):
        root_embedded = raw_root.get("coding_agent_util", "__missing__")
    if root_embedded != "__missing__":
        if root_embedded is None:
            root_embedded = {}
        if not isinstance(root_embedded, dict):
            raise ConfigError(
                f"coding_agent_util は mapping(dict) である必要があります: {resolved} (coding_agent_util)"
            )

        inherited = dict(root_embedded)
        # Inherit selected LLM settings from ai_chat_util_config.llm when not specified.
        root_llm = ai_section_dict.get("llm") if isinstance(ai_section_dict, dict) else None
        if isinstance(root_llm, dict):
            inherited_llm = dict(inherited.get("llm") or {}) if isinstance(inherited.get("llm"), dict) else {}
            for k in ("provider", "base_url", "api_key", "base_model"):
                if k not in inherited_llm and k in root_llm:
                    inherited_llm[k] = root_llm.get(k)
            if inherited_llm:
                inherited["llm"] = inherited_llm
        raw = inherited
    else:
        # Standalone coding-agent-util-config.yml uses the new root key.
        if not isinstance(raw_root, dict):
            raise ConfigError(f"設定ファイルのルートは mapping(dict) である必要があります: {resolved}")

        standalone_section = raw_root.get(CODING_AGENT_UTIL_CONFIG_ROOT_KEY, "__missing__")
        if standalone_section == "__missing__":
            # If it looks like an ai-chat-util config (or integrated config), give a clear hint.
            llm = raw_root.get("llm")
            llm_looks_ai = isinstance(llm, dict) and any(k in llm for k in ("completion_model", "embedding_model"))
            root_looks_ai = any(k in raw_root for k in ("features", "office2pdf", "network"))
            ai_section_present = ai_section_dict is not None
            path_looks_ai = resolved.name == AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME
            if llm_looks_ai or root_looks_ai or path_looks_ai or ai_section_present:
                raise ConfigError(
                    "ai-chat-util-config.yml 形式の設定が指定されましたが、coding-agent-util 用の設定が見つかりません。\n"
                    f"対処: {resolved} の ルート直下に 'coding_agent_util:' セクションを追加するか、{CODING_DEFAULT_CONFIG_FILENAME} を指定してください。"
                )

            # Otherwise, treat as old autonomous format and fail fast with migration instructions.
            raise ConfigError(
                f"設定ファイルの形式が不正です: {resolved}\n"
                f"ルートに '{CODING_AGENT_UTIL_CONFIG_ROOT_KEY}:' が必要です。\n\n"
                "旧フォーマット（ルート直下に llm/compose/backend...）はサポートされません。\n"
                "対処: 既存の llm/compose/backend... を coding_agent_util_config: 配下へ 1段インデントして移動してください。"
            )

        if standalone_section is None:
            standalone_section = {}
        if not isinstance(standalone_section, dict):
            raise ConfigError(
                f"{CODING_AGENT_UTIL_CONFIG_ROOT_KEY} は mapping(dict) である必要があります: {resolved} ({CODING_AGENT_UTIL_CONFIG_ROOT_KEY})"
            )

        raw = dict(standalone_section)

    for section_key in (
        "endpoint",
        "llm",
        "compose",
        "backend",
        "monitor",
        "paths",
        "host",
        "process",
        "subprocess",
        "logging",
    ):
        if raw.get(section_key, "__missing__") is None:
            raw[section_key] = {}

    raw = _expand_allowlisted_coding_paths(raw, config_path=resolved)

    try:
        config = CodingAgentUtilConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"設定ファイルの検証に失敗しました: {resolved}\n{e}") from e

    _resolve_coding_llm_api_key_in_place(config, config_path=resolved)
    return config


def _resolve_coding_llm_api_key_in_place(
    config: CodingAgentUtilConfig, *, config_path: Path
) -> None:
    api_key_value = config.llm.api_key
    if not api_key_value:
        return
    if not isinstance(api_key_value, str):
        raise ConfigError(f"llm.api_key は文字列である必要があります: {config_path} (llm.api_key)")
    if not api_key_value.startswith(_ENV_REF_PREFIX):
        # Should be guarded by model validation, but keep a clear error.
        raise ConfigError(
            "llm.api_key は秘密情報のため設定ファイルに直書きできません。"
            "'llm.api_key: os.environ/ENV_VAR_NAME' の形式にしてください。"
        )

    resolved = resolve_env_ref(api_key_value, config_path=config_path, field_path="llm.api_key")
    config.llm.api_key = resolved


class LLMSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="openai")
    completion_model: str = Field(default="gpt-5")
    embedding_model: str = Field(default="text-embedding-3-small")

    # non-secret default timeout (seconds) for chat completion calls
    timeout_seconds: float = Field(default=60.0, ge=0.0)

    # secret API key (must be provided via env reference; e.g. os.environ/ENV_VAR_NAME)
    api_key: str | None = Field(default=None)

    # non-secret endpoint/base url
    base_url: str | None = Field(default=None)

    # non-secret api version (mainly for Azure OpenAI via LiteLLM)
    api_version: str | None = Field(default=None)

    # secret/non-secret mixed: extra headers for outbound LLM requests.
    # For safety, values must be provided via env reference in ai-chat-util-config.yml and are resolved at load time.
    extra_headers: dict[str, str] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_api_key_required(self) -> "LLMSection":
        provider = (self.provider or "").lower()
        providers_requiring_key = {"openai", "azure", "azure_openai", "anthropic"}
        if provider in providers_requiring_key and not self.api_key:
            raise ValueError(
                "llm.api_key が未設定です。ai-chat-util-config.yml で 'ai_chat_util_config.llm.api_key: os.environ/ENV_VAR_NAME' を設定し、"
                "参照先の環境変数を設定してください。"
            )
        return self

    def create_litellm_model_list(self) -> list[dict[str, Any]]:
        """Create a list of model names to try with LiteLLM, based on the config."""
        litellm_extra_headers: dict[str, str] | None = None
        if self.extra_headers:
            # Reserve x-mcp-* for MCP forwarding (do NOT send to LiteLLM).
            filtered = {
                k: v for k, v in self.extra_headers.items() if not (k or "").lower().startswith("x-mcp-")
            }
            if filtered:
                litellm_extra_headers = filtered

        completion_litellm_dict = {}
        completion_litellm_dict["model"] = f"{self.provider}/{self.completion_model}"
        completion_litellm_dict["api_key"] = self.api_key
        if self.base_url:
            completion_litellm_dict["api_base"] = self.base_url
        if self.api_version:
            completion_litellm_dict["api_version"] = self.api_version
        if litellm_extra_headers:
            completion_litellm_dict["extra_headers"] = litellm_extra_headers

        completion_model_dict = {
            "model_name": self.completion_model,
            "litellm_params": completion_litellm_dict
        }

        embedding_litellm_dict = {}
        embedding_litellm_dict["model"] = f"{self.provider}/{self.embedding_model}"
        embedding_litellm_dict["api_key"] = self.api_key
        if self.base_url:
            embedding_litellm_dict["api_base"] = self.base_url
        if self.api_version:
            embedding_litellm_dict["api_version"] = self.api_version
        if litellm_extra_headers:
            embedding_litellm_dict["extra_headers"] = litellm_extra_headers
        embedding_model_dict = {
            "model_name": self.embedding_model,
            "litellm_params": embedding_litellm_dict
        }

        models: list[dict[str, Any]] = []
        if self.completion_model:
            models.append(completion_model_dict)
        if self.embedding_model:
            models.append(embedding_model_dict)
        return models

class PathsSection(BaseModel):
    # Deprecated: kept only for type references in older versions.
    # This project now uses MCPSection (ai_chat_util_config.mcp).
    mcp_config_path: str | None = Field(default=None)
    custom_instructions_file_path: str | None = Field(default=None)
    working_directory: str | None = Field(default=None)


class CodingAgentEndpointSection(BaseModel):
    """Selector for the MCP server key assigned to the coding-agent-family route."""

    model_config = ConfigDict(extra="forbid")

    # mcpServers.<name> in mcp.json
    mcp_server_name: str = Field(default="coding-agent")

    @model_validator(mode="after")
    def _validate_mcp_server_name(self) -> "CodingAgentEndpointSection":
        name = self.mcp_server_name
        if not (isinstance(name, str) and name.strip()):
            raise ValueError("mcp.coding_agent_endpoint.mcp_server_name must be a non-empty string")
        self.mcp_server_name = name.strip()
        return self


class MCPSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Optional: MCP client settings JSON path
    mcp_config_path: str | None = Field(default=None)
    custom_instructions_file_path: str | None = Field(default=None)
    working_directory: str | None = Field(default=None)

    coding_agent_endpoint: CodingAgentEndpointSection = Field(
        default_factory=CodingAgentEndpointSection,
        description="coding-agent 系 route に割り当てる MCP サーバー名（mcp.json の mcpServers.<name>）",
    )

    # Optional: extra headers/env forwarding for MCP transports ONLY.
    # Keys must start with:
    # - x-mcp-<Header-Name>
    # - x-mcp-env-<ENV_NAME>
    extra_headers: dict[str, str] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_mcp_extra_headers_keys(self) -> "MCPSection":
        extra = self.extra_headers
        if extra is None:
            return self
        if not isinstance(extra, dict):
            raise ValueError("mcp.extra_headers は mapping(dict) である必要があります")

        for k, v in extra.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("mcp.extra_headers のキーは空でない文字列である必要があります")
            if not isinstance(v, str):
                raise ValueError(f"mcp.extra_headers.{k} は文字列である必要があります")
            lower = k.strip().lower()
            if not (lower.startswith("x-mcp-") or lower.startswith("x-mcp-env-")):
                raise ValueError(
                    "mcp.extra_headers は MCP 転送専用です。キーは 'x-mcp-' または 'x-mcp-env-' で始まる必要があります: "
                    + repr(k)
                )

        return self


class FileServerSMBSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False)
    server: str | None = Field(default=None)
    share: str | None = Field(default=None)
    port: int = Field(default=445)
    domain: str | None = Field(default=None)
    username: str | None = Field(default=None)
    password: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_port(self) -> "FileServerSMBSection":
        if self.port <= 0:
            raise ValueError("file_server.smb.port は 1 以上である必要があります")
        return self

    @model_validator(mode="after")
    def _validate_credentials(self) -> "FileServerSMBSection":
        if self.enabled and self.username and not self.password:
            raise ValueError("file_server.smb.password を設定してください")
        if self.enabled and self.password and not self.username:
            raise ValueError("file_server.smb.username を設定してください")
        return self


class FileServerAllowedRoot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: Literal["local", "smb"] = Field(default="local")
    path: str = Field(default=".")
    description: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_name(self) -> "FileServerAllowedRoot":
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("file_server.allowed_roots[].name は空文字にできません")
        return self


class FileServerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False)
    default_provider: Literal["local", "smb"] = Field(default="local")
    default_root: str | None = Field(default=None)
    allowed_roots: list[FileServerAllowedRoot] = Field(default_factory=list)
    max_depth: int = Field(default=3)
    max_entries: int = Field(default=1000)
    include_hidden_default: bool = Field(default=False)
    follow_symlinks: bool = Field(default=False)
    include_mime_default: bool = Field(default=False)
    smb: FileServerSMBSection = Field(default_factory=FileServerSMBSection)

    @model_validator(mode="after")
    def _validate_limits(self) -> "FileServerSection":
        if self.max_depth < 0:
            raise ValueError("file_server.max_depth は 0 以上である必要があります")
        if self.max_entries <= 0:
            raise ValueError("file_server.max_entries は 1 以上である必要があります")
        return self

    @model_validator(mode="after")
    def _validate_roots(self) -> "FileServerSection":
        names: set[str] = set()
        for root in self.allowed_roots:
            if root.name in names:
                raise ValueError(f"file_server.allowed_roots の name が重複しています: {root.name}")
            names.add(root.name)

        if self.default_root and self.default_root not in names:
            raise ValueError("file_server.default_root は allowed_roots の name と一致する必要があります")
        return self

class FeaturesSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_outside_modifications: bool = Field(default=False)
    use_custom_pdf_analyzer: bool = Field(default=False)
    enable_deep_agent: bool = Field(
        default=False,
        description="deep_agent route を有効化します。未導入時は deepagents パッケージのインストールが必要です。",
    )
    type_selection_mode: Literal["disabled", "deterministic"] = Field(
        default="disabled",
        description="WF型 / SV型 / 自律型の上位型選択モード。deterministic で Coordinator を有効化します。",
    )
    type_selection_default_route: Literal["supervisor", "autonomous"] = Field(
        default="supervisor",
        description="cross-type 判定で明確なシグナルがない場合の既定ルート。",
    )
    type_selection_workflow_requires_definition: bool = Field(
        default=True,
        description="WF型を選ぶには workflow_file_path の明示を必須にするかどうか。",
    )
    type_selection_prefer_workflow_when_definition_available: bool = Field(
        default=True,
        description="workflow 定義がある場合に WF 型を優先するかどうか。",
    )
    type_selection_workflow_on_high_predictability: bool = Field(
        default=True,
        description="predictability=high のとき WF 型を優先するかどうか。",
    )
    type_selection_workflow_on_high_approval_frequency: bool = Field(
        default=True,
        description="approval_frequency=high のとき WF 型を優先するかどうか。",
    )
    type_selection_workflow_on_side_effects: bool = Field(
        default=True,
        description="has_side_effects=true のとき WF 型を優先するかどうか。",
    )
    type_selection_autonomous_on_explicit_coding_request: bool = Field(
        default=True,
        description="ユーザーが coding_agent を明示した場合に自律型を優先するかどうか。",
    )
    type_selection_autonomous_on_explicit_deep_request: bool = Field(
        default=True,
        description="ユーザーが deep_agent を明示した場合に自律型を優先するかどうか。",
    )
    type_selection_autonomous_on_high_exploration: bool = Field(
        default=True,
        description="exploration_level=high のとき自律型を優先するかどうか。",
    )
    type_selection_require_clarification_on_missing_workflow_definition: bool = Field(
        default=True,
        description="WF 型が適しそうだが workflow_file_path が未指定のとき clarification を要求するかどうか。",
    )
    type_selection_ambiguity_gap: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="cross-type 判定の上位候補スコア差がこの値未満なら clarification 候補にします。",
    )
    preferred_coding_route: Literal["coding_agent", "deep_agent"] = Field(
        default="coding_agent",
        description="複雑な調査系要求で優先する route。deep_agent を選ぶには enable_deep_agent も有効化してください。",
    )
    routing_mode: Literal["legacy", "structured", "hybrid"] = Field(
        default="legacy",
        description="Supervisor の routing 判定モード。legacy は従来挙動、structured/hybrid は RoutingDecision を併用します。",
    )
    routing_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="structured/hybrid routing で clarification へ切り替える信頼度しきい値。",
    )
    sufficiency_check_enabled: bool = Field(
        default=False,
        description="MCP 実行結果に対する SufficiencyDecision の構造化判定を有効化します。",
    )
    audit_log_enabled: bool = Field(
        default=False,
        description="Supervisor audit event の構造化ログを有効化します。",
    )
    audit_log_path: str | None = Field(
        default=None,
        description="Supervisor audit event を JSONL 形式で保存するパス。未指定時は work/supervisor_audit.jsonl を使います。",
    )
    mcp_recursion_limit: int = Field(
        default=50,
        ge=1,
        description=(
            "LangGraph supervisor の再帰(ステップ)上限。\n"
            "ツール/エージェントの無限ループを防ぐ安全弁です。"
        ),
    )
    mcp_tool_call_limit: int = Field(
        default=4,
        ge=1,
        description=(
            "1ユーザー入力(=1 trace_id)あたりのツール呼び出し回数上限。\n"
            "同一入力でツールが繰り返し実行されるのを防ぎます。\n"
            "既定値は、設定確認と文書調査を含む代表的な複合シナリオが最低限完走できる水準に調整されています。"
        ),
    )
    mcp_followup_tool_call_limit: int = Field(
        default=8,
        ge=1,
        description=(
            "status/get_result/workspace_path/cancel のような追跡系ツール専用の呼び出し回数上限。\n"
            "execute 後のポーリングや結果取得で通常ツールの予算を消費し切らないようにするための別枠です。"
        ),
    )
    mcp_followup_poll_interval_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="status の再ポーリング時に推奨する待機秒数。例: 10.0 を設定すると 10 秒間隔を指示できます。",
    )
    mcp_status_tail_lines: int = Field(
        default=20,
        ge=0,
        description="status ツールで取得するログ tail 行数の既定値。0 ならログ本文を返しません。",
    )
    mcp_get_result_tail_lines: int = Field(
        default=80,
        ge=0,
        description="get_result ツールで取得するログ tail 行数の既定値。0 ならログ本文を返しません。",
    )
    mcp_tool_timeout_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "MCPツール呼び出しのアプリ側ハードタイムアウト(秒)。\n"
            "未設定(None)の場合は llm.timeout_seconds を使用します。"
        ),
    )
    mcp_tool_timeout_retries: int = Field(
        default=1,
        ge=0,
        description=(
            "MCPツールがタイムアウトした場合の再試行回数。\n"
            "1なら『最大1回だけ再試行』です。"
        ),
    )
    hitl_approval_tools: list[str] = Field(
        default_factory=list,
        description=(
            "HITL（Human-in-the-loop）承認が必要なツール名のリスト。"
            "このリストに含まれるツールを実行する前に、エージェントは人間へ承認を求めて pause します。"
        ),
    )

    def get_hitl_approval_tools_text(self) -> str:
        if not self.hitl_approval_tools:
            return "(なし)"
        return ", ".join(self.hitl_approval_tools)


class LoggingSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = Field(default="INFO")
    file: str | None = Field(default=None)


class NetworkSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests_verify: bool = Field(default=True)
    ca_bundle: str | None = Field(default=None)


class Office2PDFSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    libreoffice_path: str | None = Field(default=None)


class _RuntimeState(BaseModel):
    config_path: Path
    config: AiChatUtilConfig



class CodingAgentUtilLoggingSection(BaseModel):
    level: str = Field(default="INFO")
    file: str | None = Field(default=None)


class CodingAgentUtilLLMSection(BaseModel):
    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o")

    # non-secret base url/endpoint
    base_url: str | None = Field(default=None)

    # non-secret: override LLM_BASE_URL passed into container
    base_url_in_container: str | None = Field(default=None)

    # secret API key (must be provided as env reference: os.environ/ENV_VAR)
    api_key: str | None = Field(default=None)

    # optional override for cost tracking etc.
    base_model: str | None = Field(default=None)


class CodingComposeSection(BaseModel):
    directory: str = Field(default=".")
    file: str = Field(default="docker-compose.yml")
    service_name: str = Field(default="executor-service")
    command: str = Field(default="")


class CodingBackendSection(BaseModel):
    # NOTE:
    # - `process` is the only local execution backend name.
    # - `subprocess` is deprecated but accepted for backward compatibility and
    #   normalized to `process`.
    task_backend: str = Field(default="process")

    @model_validator(mode="after")
    def _normalize_task_backend(self) -> "CodingBackendSection":
        b = (self.task_backend or "process").strip().lower()
        if b == "subprocess":
            b = "process"
        if b not in {"docker", "compose", "process", "windows_process", "linux_process"}:
            raise ValueError(
                "backend.task_backend は 'docker' | 'compose' | 'process' | 'windows_process' | 'linux_process' のいずれかである必要があります"
            )
        if b == "windows_process" and os.name != "nt":
            raise ValueError("backend.task_backend=windows_process は Windows 上でのみ使用できます")
        if b == "linux_process" and os.name == "nt":
            raise ValueError("backend.task_backend=linux_process は Linux/Unix 系 OS 上でのみ使用できます")
        self.task_backend = b
        return self


class CodingMonitorSection(BaseModel):
    disable_detach_monitor: bool = Field(default=False)
    detach_monitor_interval: float = Field(default=2.0, ge=0.1)
    debug_container: bool = Field(default=False)


class WorkspacePathRewriteRule(BaseModel):
    """Rewrite inbound workspace_path before validation."""

    model_config = ConfigDict(populate_by_name=True)

    from_prefix: str = Field(..., alias="from")
    to_prefix: str = Field(..., alias="to")

    @model_validator(mode="after")
    def _validate_prefixes(self) -> "WorkspacePathRewriteRule":
        if not (isinstance(self.from_prefix, str) and self.from_prefix.strip()):
            raise ValueError("paths.workspace_path_rewrites[].from must be a non-empty string")
        if not (isinstance(self.to_prefix, str) and self.to_prefix.strip()):
            raise ValueError("paths.workspace_path_rewrites[].to must be a non-empty string")
        if not os.path.isabs(self.from_prefix.strip()):
            raise ValueError("paths.workspace_path_rewrites[].from must be an absolute path prefix")
        if not os.path.isabs(self.to_prefix.strip()):
            raise ValueError("paths.workspace_path_rewrites[].to must be an absolute path prefix")
        return self


class CodingPathsSection(BaseModel):
    workspace_root: str = Field(default="/tmp/coding_agent_tasks")
    host_projects_root: str = Field(default_factory=lambda: str(Path.home() / "ai-platform" / "data" / "projects"))
    executor_allowed_workspace_root: str | None = Field(default=None)
    workspace_path_rewrites: list[WorkspacePathRewriteRule] = Field(default_factory=list)


class CodingHostSection(BaseModel):
    uid: int | None = Field(default=None)
    gid: int | None = Field(default=None)


class CodingEndpointSection(BaseModel):
    """Endpoint-related settings for future MCP-aware routing.

    NOTE: Currently not used by the executor runtime. This section exists to
    reserve a stable config surface so future versions can apply custom
    behavior when `mcp.json` contains a specific server definition.
    """

    mcp_server_name: str = Field(default="coding-agent")
    followup_poll_interval_seconds: float = Field(default=2.0, ge=0.0)
    status_default_tail_lines: int = Field(default=20, ge=0)
    get_result_default_tail_lines: int = Field(default=80, ge=0)
    max_tool_result_chars: int = Field(default=4000, ge=256)

    @model_validator(mode="after")
    def _validate_mcp_server_name(self) -> "CodingEndpointSection":
        name = self.mcp_server_name
        if not (isinstance(name, str) and name.strip()):
            raise ValueError("endpoint.mcp_server_name must be a non-empty string")
        self.mcp_server_name = name.strip()
        return self


class CodingSubprocessSection(BaseModel):
    # Deprecated alias section; keep for backward compatibility.
    command: str = Field(default="")


class CodingProcessSection(BaseModel):
    command: str = Field(default="")


class AiChatUtilConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMSection = Field(default_factory=LLMSection)
    mcp: MCPSection = Field(default_factory=MCPSection)
    file_server: FileServerSection = Field(default_factory=FileServerSection)
    features: FeaturesSection = Field(default_factory=FeaturesSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)
    network: NetworkSection = Field(default_factory=NetworkSection)
    office2pdf: Office2PDFSection = Field(default_factory=Office2PDFSection)

    def _apply_mcp_runtime_overrides(self, config: MCPServerConfig) -> None:
        """Apply runtime overrides from ai-chat-util-config.yml onto MCP server config.

        - mcp.extra_headers with reserved prefixes are forwarded into transports:
          - x-mcp-<Header-Name> => http/sse/websocket headers[Header-Name]
          - x-mcp-env-<ENV_NAME> => stdio env[ENV_NAME]
        - Ensure stdio servers can resolve the same ai-chat-util-config.yml by
          injecting an absolute CONFIG_ENV_VAR.
        """

        extra = getattr(self.mcp, "extra_headers", None)
        mcp_headers: dict[str, str] = {}
        mcp_env: dict[str, str] = {}

        if isinstance(extra, dict) and extra:
            import re

            for raw_key, raw_val in extra.items():
                if not isinstance(raw_key, str) or not isinstance(raw_val, str):
                    continue
                key = raw_key.strip()
                if not key:
                    continue

                lower = key.lower()
                if lower.startswith("x-mcp-env-"):
                    env_name = key[len("x-mcp-env-") :].strip()
                    if not env_name:
                        raise ConfigError(
                            "mcp.extra_headers の x-mcp-env- プレフィックス指定が不正です（ENV名が空）"
                        )
                    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", env_name):
                        raise ConfigError(
                            "mcp.extra_headers の環境変数名が不正です: "
                            f"{env_name} (key={raw_key!r})\n"
                            "対処: x-mcp-env-ENV_NAME の ENV_NAME は [A-Za-z_][A-Za-z0-9_]* を満たしてください。"
                        )
                    mcp_env[env_name] = raw_val
                    continue

                if lower.startswith("x-mcp-"):
                    header_name = key[len("x-mcp-") :].strip()
                    if not header_name:
                        raise ConfigError(
                            "mcp.extra_headers の x-mcp- プレフィックス指定が不正です（ヘッダー名が空）"
                        )
                    mcp_headers[header_name] = raw_val

        # Forward headers/env into each server entry.
        if (mcp_headers or mcp_env) and getattr(config, "servers", None):
            for _, entry in config.servers.items():
                transport = (getattr(entry, "transport", None) or "").lower()

                if transport == "stdio" and mcp_env:
                    env = getattr(entry, "env", None)
                    if env is None:
                        env = {}
                    if isinstance(env, dict):
                        env2 = dict(env)
                        # ai-chat-util-config.yml (runtime) takes precedence
                        env2.update(mcp_env)
                        entry.env = env2

                if transport in {"http", "sse", "websocket"} and mcp_headers:
                    headers = getattr(entry, "headers", None)
                    if headers is None:
                        headers = {}
                    if isinstance(headers, dict):
                        headers2 = dict(headers)
                        # ai-chat-util-config.yml (runtime) takes precedence
                        headers2.update(mcp_headers)
                        entry.headers = headers2

        # Always inject the absolute config path for stdio servers.
        runtime_config_path = str(get_runtime_config_path())
        if getattr(config, "servers", None):
            for _, entry in config.servers.items():
                transport = (getattr(entry, "transport", None) or "").lower()
                if transport != "stdio":
                    continue
                env = getattr(entry, "env", None)
                if env is None:
                    env = {}
                if isinstance(env, dict):
                    env2 = dict(env)
                    env2[CONFIG_ENV_VAR] = runtime_config_path
                    entry.env = env2

    def get_mcp_server_config(self) -> MCPServerConfig:
        if not self.mcp.mcp_config_path:
            logger.warning(
                "MCP 設定ファイルパスが未設定です。ai-chat-util-config.yml の mcp.mcp_config_path を設定してください。"
            )
            return MCPServerConfig()

        # ai-chat-util-config.yml からの相対パスも解決できるよう、設定ファイルのディレクトリも探索対象に入れる
        config_dir = str(get_runtime_config_path().parent)
        resolved = resolve_existing_file_path(
            self.mcp.mcp_config_path,
            working_directory=self.mcp.working_directory,
            extra_search_dirs=[config_dir],
        ).resolved_path

        config = MCPServerConfig()
        config.load_server_config(resolved)
 
        # Apply runtime overrides (extra_headers forwarding, config path injection).
        self._apply_mcp_runtime_overrides(config)
        return config


class CodingAgentUtilConfig(BaseModel):
    endpoint: CodingEndpointSection = Field(default_factory=CodingEndpointSection)
    llm: CodingAgentUtilLLMSection = Field(default_factory=CodingAgentUtilLLMSection)
    compose: CodingComposeSection = Field(default_factory=CodingComposeSection)
    backend: CodingBackendSection = Field(default_factory=CodingBackendSection)
    monitor: CodingMonitorSection = Field(default_factory=CodingMonitorSection)
    paths: CodingPathsSection = Field(default_factory=CodingPathsSection)
    host: CodingHostSection = Field(default_factory=CodingHostSection)
    # Preferred config section name.
    process: CodingProcessSection = Field(default_factory=CodingProcessSection)
    # Backward compatible alias (deprecated): `subprocess.command`.
    subprocess: CodingSubprocessSection = Field(default_factory=CodingSubprocessSection)
    logging: CodingAgentUtilLoggingSection = Field(default_factory=CodingAgentUtilLoggingSection)

    @model_validator(mode="after")
    def _coalesce_process_subprocess(self) -> "CodingAgentUtilConfig":
        fields = set(getattr(self, "model_fields_set", set()) or set())
        has_process = "process" in fields
        has_subprocess = "subprocess" in fields

        p_cmd = (self.process.command or "").strip() if self.process else ""
        s_cmd = (self.subprocess.command or "").strip() if self.subprocess else ""

        if has_process and has_subprocess:
            # If both sections are explicitly configured, they must agree.
            if p_cmd and s_cmd and p_cmd != s_cmd:
                raise ValueError(
                    "process.command と subprocess.command の両方が設定されていますが一致しません。process.command に統一してください。"
                )
            # Prefer process if present; keep alias in sync.
            effective = p_cmd or s_cmd
            if effective:
                self.process.command = effective
                self.subprocess.command = effective
            return self

        if has_subprocess and not has_process:
            # Old config: migrate in-memory.
            if s_cmd:
                self.process.command = s_cmd
            return self

        if has_process and not has_subprocess:
            # Keep deprecated alias in sync for internal callers.
            if p_cmd:
                self.subprocess.command = p_cmd
            return self

        # Neither explicitly set; keep defaults.
        return self

    @model_validator(mode="after")
    def _validate_required_commands(self) -> "CodingAgentUtilConfig":
        # NOTE:
        # This project may load coding_agent_util config for features that do not
        # execute tasks (e.g., workspace path rewrites). Requiring commands at load
        # time would break those scenarios. We therefore validate required commands
        # only when the user explicitly configures the relevant sections.
        fields = set(getattr(self, "model_fields_set", set()) or set())
        backend_explicit = "backend" in fields
        process_explicit = "process" in fields or "subprocess" in fields
        compose_explicit = "compose" in fields

        backend = (self.backend.task_backend or "process").strip().lower()
        proc_cmd = (self.process.command or "").strip() if self.process else ""
        compose_cmd = (self.compose.command or "").strip() if self.compose else ""

        if backend_explicit or process_explicit or compose_explicit:
            if backend in {"process", "windows_process", "linux_process"}:
                if not proc_cmd:
                    raise ValueError("process 系バックエンドでは process.command を設定してください")
            if backend in {"docker", "compose"}:
                if not compose_cmd:
                    raise ValueError("docker/compose バックエンドでは compose.command を設定してください")
        return self

    @model_validator(mode="after")
    def _validate_llm_secret_policy(self) -> "CodingAgentUtilConfig":
        # Allow only env references at validation time; init_coding_runtime resolves it later.
        if self.llm.api_key and self.llm.api_key.startswith(_ENV_REF_PREFIX):
            return self
        if self.llm.api_key:
            raise ValueError(
                "llm.api_key は秘密情報のため設定ファイルに直書きできません。"
                "'llm.api_key: os.environ/ENV_VAR_NAME' の形式にしてください。"
            )
        return self

    def get_compose_paths(self) -> list[str]:
        raw = (self.compose.file or "").strip()
        if not raw:
            return [str(Path(self.compose.directory) / "docker-compose.yml")]

        parts = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
        if len(parts) == 1 and "," in parts[0]:
            parts = [p.strip() for p in parts[0].split(",") if p.strip()]

        paths: list[str] = []
        for part in parts:
            if os.path.isabs(part):
                paths.append(part)
            else:
                paths.append(str(Path(self.compose.directory) / part))
        return paths

class AppConfigSection(BaseModel):
    ai_chat_util_config: AiChatUtilConfig = Field(default_factory=AiChatUtilConfig)
    agent_util_config: CodingAgentUtilConfig = Field(default_factory=CodingAgentUtilConfig)

class _CodingRuntimeState(BaseModel):
    # NOTE:
    # init_coding_runtime() resolves env-ref secrets (e.g., llm.api_key) into
    # literal values after validation. When storing the already-validated config
    # inside this state model, Pydantic may revalidate nested instances, which
    # would re-trigger the secret-policy validator and fail.
    model_config = ConfigDict(revalidate_instances="never")

    config_path: Path
    config: CodingAgentUtilConfig



