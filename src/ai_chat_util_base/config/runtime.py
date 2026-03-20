from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Literal, Callable

from pydantic import BaseModel, Field, model_validator, ConfigDict

from .config_util import (
    CONFIG_ENV_VAR,
    ConfigError, load_resolved_yaml, resolve_config_path, resolve_autonomous_config_path, 
    resolve_env_ref,extract_required_root_section, extract_optional_ai_section_dict, 
    apply_secret_overrides_from_yaml,
    AI_CHAT_UTIL_CONFIG_ROOT_KEY,
    AUTONOMOUS_AGENT_UTIL_CONFIG_ROOT_KEY,
    AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME,
    AUTONOMOUS_DEFAULT_CONFIG_FILENAME,
    _ENV_REF_PREFIX
    
)



# =====================
# Autonomous Agent Util Runtime (embedded)
# =====================

_runtime_state: _RuntimeState | None = None
_autonomous_runtime_state: _AutonomousRuntimeState | None = None


def init_runtime(config_path: str | None = None) -> AiChatUtilConfig:
    global _runtime_state

    resolved, raw_root = load_resolved_yaml(config_path, resolver=resolve_config_path)
    config = _build_ai_chat_util_config(raw_root=raw_root, resolved=resolved)

    _runtime_state = _RuntimeState(config_path=resolved, config=config)
    _apply_non_secret_runtime_side_effects(config)

    return config


def init_autonomous_runtime(config_path: str | None = None) -> AutonomousAgentUtilConfig:
    global _autonomous_runtime_state

    resolved, raw_root = load_resolved_yaml(config_path, resolver=resolve_autonomous_config_path)
    config = _build_autonomous_agent_util_config(raw_root=raw_root, resolved=resolved)

    # Use model_construct() to avoid re-validating `config`.
    # At this point, env-ref secrets (e.g., llm.api_key) may already be resolved
    # into literal values, and re-validation would incorrectly re-trigger the
    # secret-policy validator.
    _autonomous_runtime_state = _AutonomousRuntimeState.model_construct(config_path=resolved, config=config)
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

def get_autonomous_runtime_config() -> AutonomousAgentUtilConfig:
    if _autonomous_runtime_state is None:
        return init_autonomous_runtime(None)
    return _autonomous_runtime_state.config

def get_autonomous_runtime_config_path() -> Path:
    if _autonomous_runtime_state is None:
        init_autonomous_runtime(None)
    assert _autonomous_runtime_state is not None
    return _autonomous_runtime_state.config_path


def get_runtime_config_path() -> Path:
    if _runtime_state is None:
        init_runtime(None)
    assert _runtime_state is not None
    return _runtime_state.config_path


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


def _configure_python_logging(config: AiChatUtilConfig | AutonomousAgentUtilConfig) -> None:
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


def _build_ai_chat_util_config(*, raw_root: dict[str, Any], resolved: Path) -> AiChatUtilConfig:
    raw_section = extract_required_root_section(raw_root=raw_root, resolved=resolved)

    # Secrets may be declared in YAML only via env reference (os.environ/VAR).
    raw = apply_secret_overrides_from_yaml(
        dict(raw_section),
        config_path=resolved,
        field_prefix=f"{AI_CHAT_UTIL_CONFIG_ROOT_KEY}.",
    )

    try:
        return AiChatUtilConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"設定ファイルの検証に失敗しました: {resolved}\n{e}") from e


def _build_autonomous_agent_util_config(
    *, raw_root: dict[str, Any], resolved: Path
) -> AutonomousAgentUtilConfig:
    raw: dict[str, Any]

    ai_section_dict = extract_optional_ai_section_dict(raw_root=raw_root, resolved=resolved)

    # New integrated format:
    # - ai-chat-util-config.yml root contains ONLY:
    #     - ai_chat_util_config: ...
    #     - autonomous_agent_util: ...
    # - Backward compatibility for nested ai_chat_util_config.autonomous_agent_util is intentionally NOT supported.
    nested_embedded = (
        ai_section_dict.get("autonomous_agent_util", "__missing__")
        if ai_section_dict is not None
        else "__missing__"
    )
    if nested_embedded != "__missing__":
        raise ConfigError(
            "設定ファイルの形式が不正です: "
            f"{resolved}\n"
            "統合設定の autonomous-agent-util は 'ai_chat_util_config.autonomous_agent_util' ではなく、"
            "ルート直下の 'autonomous_agent_util:' に記述してください。"
        )

    root_embedded = raw_root.get("autonomous_agent_util", "__missing__") if isinstance(raw_root, dict) else "__missing__"
    if root_embedded != "__missing__":
        if root_embedded is None:
            root_embedded = {}
        if not isinstance(root_embedded, dict):
            raise ConfigError(
                f"autonomous_agent_util は mapping(dict) である必要があります: {resolved} (autonomous_agent_util)"
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
        # Standalone autonomous-agent-util-config.yml must use the new root key.
        if not isinstance(raw_root, dict):
            raise ConfigError(f"設定ファイルのルートは mapping(dict) である必要があります: {resolved}")

        standalone_section = raw_root.get(AUTONOMOUS_AGENT_UTIL_CONFIG_ROOT_KEY, "__missing__")
        if standalone_section == "__missing__":
            # If it looks like an ai-chat-util config (or integrated config), give a clear hint.
            llm = raw_root.get("llm")
            llm_looks_ai = isinstance(llm, dict) and any(k in llm for k in ("completion_model", "embedding_model"))
            root_looks_ai = any(k in raw_root for k in ("features", "office2pdf", "network"))
            ai_section_present = ai_section_dict is not None
            path_looks_ai = resolved.name == AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME
            if llm_looks_ai or root_looks_ai or path_looks_ai or ai_section_present:
                raise ConfigError(
                    "ai-chat-util-config.yml 形式の設定が指定されましたが、autonomous-agent-util 用の設定が見つかりません。\n"
                    f"対処: {resolved} の ルート直下に 'autonomous_agent_util:' セクションを追加するか、{AUTONOMOUS_DEFAULT_CONFIG_FILENAME} を指定してください。"
                )

            # Otherwise, treat as old autonomous format and fail fast with migration instructions.
            raise ConfigError(
                f"設定ファイルの形式が不正です: {resolved}\n"
                f"ルートに '{AUTONOMOUS_AGENT_UTIL_CONFIG_ROOT_KEY}:' が必要です。\n\n"
                "旧フォーマット（ルート直下に llm/compose/backend...）はサポートされません。\n"
                "対処: 既存の llm/compose/backend... を autonomous_agent_util_config: 配下へ 1段インデントして移動してください。"
            )

        if standalone_section is None:
            standalone_section = {}
        if not isinstance(standalone_section, dict):
            raise ConfigError(
                f"{AUTONOMOUS_AGENT_UTIL_CONFIG_ROOT_KEY} は mapping(dict) である必要があります: {resolved} ({AUTONOMOUS_AGENT_UTIL_CONFIG_ROOT_KEY})"
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

    try:
        config = AutonomousAgentUtilConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"設定ファイルの検証に失敗しました: {resolved}\n{e}") from e

    _resolve_autonomous_llm_api_key_in_place(config, config_path=resolved)
    return config

def _resolve_autonomous_llm_api_key_in_place(
    config: AutonomousAgentUtilConfig, *, config_path: Path
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
    # Preferred key name for MCP settings JSON path.
    # Kept compatible with older key: mcp_server_config_file_path.
    mcp_config_path: str | None = Field(default=None)
    mcp_server_config_file_path: str | None = Field(default=None)
    custom_instructions_file_path: str | None = Field(default=None)
    working_directory: str | None = Field(default=None)

    @model_validator(mode="after")
    def _coalesce_mcp_config_path(self) -> "PathsSection":
        a = self.mcp_config_path
        b = self.mcp_server_config_file_path
        if a and b and a != b:
            raise ValueError(
                "paths.mcp_config_path と paths.mcp_server_config_file_path の両方が設定されていますが一致しません。"
                "どちらか一方に統一してください。"
            )
        if a and not b:
            self.mcp_server_config_file_path = a
        if b and not a:
            self.mcp_config_path = b
        return self


class FeaturesSection(BaseModel):
    allow_outside_modifications: bool = Field(default=False)
    use_custom_pdf_analyzer: bool = Field(default=False)
    mcp_recursion_limit: int = Field(
        default=50,
        ge=1,
        description=(
            "LangGraph supervisor の再帰(ステップ)上限。\n"
            "ツール/エージェントの無限ループを防ぐ安全弁です。"
        ),
    )
    mcp_tool_call_limit: int = Field(
        default=2,
        ge=1,
        description=(
            "1ユーザー入力(=1 trace_id)あたりのツール呼び出し回数上限。\n"
            "同一入力でツールが繰り返し実行されるのを防ぎます。"
        ),
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


class LoggingSection(BaseModel):
    level: str = Field(default="INFO")
    file: str | None = Field(default=None)


class NetworkSection(BaseModel):
    requests_verify: bool = Field(default=True)
    ca_bundle: str | None = Field(default=None)


class Office2PDFSection(BaseModel):
    libreoffice_path: str | None = Field(default=None)


class _RuntimeState(BaseModel):
    config_path: Path
    config: AiChatUtilConfig



class AutonomousAgentUtilLoggingSection(BaseModel):
    level: str = Field(default="INFO")
    file: str | None = Field(default=None)


class AutonomousAgentUtilLLMSection(BaseModel):
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


class AutonomousComposeSection(BaseModel):
    directory: str = Field(default=".")
    file: str = Field(default="docker-compose.yml")
    service_name: str = Field(default="executor-service")
    command: str = Field(default="")


class AutonomousBackendSection(BaseModel):
    # NOTE:
    # - `process` is the only local execution backend name.
    # - `subprocess` is deprecated but accepted for backward compatibility and
    #   normalized to `process`.
    task_backend: str = Field(default="process")

    @model_validator(mode="after")
    def _normalize_task_backend(self) -> "AutonomousBackendSection":
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


class AutonomousMonitorSection(BaseModel):
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


class AutonomousPathsSection(BaseModel):
    workspace_root: str = Field(default="/tmp/autonomous_agent_tasks")
    host_projects_root: str = Field(default="/home/user/ai-platform/data/projects")
    executor_allowed_workspace_root: str | None = Field(default=None)
    workspace_path_rewrites: list[WorkspacePathRewriteRule] = Field(default_factory=list)


class AutonomousHostSection(BaseModel):
    uid: int | None = Field(default=None)
    gid: int | None = Field(default=None)


class AutonomousEndpointSection(BaseModel):
    """Endpoint-related settings for future MCP-aware routing.

    NOTE: Currently not used by the executor runtime. This section exists to
    reserve a stable config surface so future versions can apply custom
    behavior when `mcp.json` contains a specific server definition.
    """

    mcp_server_name: str = Field(default="coding-agent")

    @model_validator(mode="after")
    def _validate_mcp_server_name(self) -> "AutonomousEndpointSection":
        name = self.mcp_server_name
        if not (isinstance(name, str) and name.strip()):
            raise ValueError("endpoint.mcp_server_name must be a non-empty string")
        self.mcp_server_name = name.strip()
        return self


class AutonomousSubprocessSection(BaseModel):
    # Deprecated alias section; keep for backward compatibility.
    command: str = Field(default="")


class AutonomousProcessSection(BaseModel):
    command: str = Field(default="")


class AiChatUtilConfig(BaseModel):
    llm: LLMSection = Field(default_factory=LLMSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    features: FeaturesSection = Field(default_factory=FeaturesSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)
    network: NetworkSection = Field(default_factory=NetworkSection)
    office2pdf: Office2PDFSection = Field(default_factory=Office2PDFSection)


class AutonomousAgentUtilConfig(BaseModel):
    endpoint: AutonomousEndpointSection = Field(default_factory=AutonomousEndpointSection)
    llm: AutonomousAgentUtilLLMSection = Field(default_factory=AutonomousAgentUtilLLMSection)
    compose: AutonomousComposeSection = Field(default_factory=AutonomousComposeSection)
    backend: AutonomousBackendSection = Field(default_factory=AutonomousBackendSection)
    monitor: AutonomousMonitorSection = Field(default_factory=AutonomousMonitorSection)
    paths: AutonomousPathsSection = Field(default_factory=AutonomousPathsSection)
    host: AutonomousHostSection = Field(default_factory=AutonomousHostSection)
    # Preferred config section name.
    process: AutonomousProcessSection = Field(default_factory=AutonomousProcessSection)
    # Backward compatible alias (deprecated): `subprocess.command`.
    subprocess: AutonomousSubprocessSection = Field(default_factory=AutonomousSubprocessSection)
    logging: AutonomousAgentUtilLoggingSection = Field(default_factory=AutonomousAgentUtilLoggingSection)

    @model_validator(mode="after")
    def _coalesce_process_subprocess(self) -> "AutonomousAgentUtilConfig":
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
    def _validate_required_commands(self) -> "AutonomousAgentUtilConfig":
        # NOTE:
        # This project may load autonomous_agent_util config for features that do not
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
    def _validate_llm_secret_policy(self) -> "AutonomousAgentUtilConfig":
        # Allow only env references at validation time; init_autonomous_runtime resolves it later.
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
    agent_util_config: AutonomousAgentUtilConfig = Field(default_factory=AutonomousAgentUtilConfig)

class _AutonomousRuntimeState(BaseModel):
    # NOTE:
    # init_autonomous_runtime() resolves env-ref secrets (e.g., llm.api_key) into
    # literal values after validation. When storing the already-validated config
    # inside this state model, Pydantic may revalidate nested instances, which
    # would re-trigger the secret-policy validator and fail.
    model_config = ConfigDict(revalidate_instances="never")

    config_path: Path
    config: AutonomousAgentUtilConfig

