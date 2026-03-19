from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator, ConfigDict


CONFIG_ENV_VAR = "AI_CHAT_UTIL_CONFIG"
DEFAULT_CONFIG_FILENAME = "ai-chat-util-config.yml"

# New config format root key (required).
AI_CHAT_UTIL_CONFIG_ROOT_KEY = "ai_chat_util_config"


class ConfigError(RuntimeError):
    pass


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


class AiChatUtilConfig(BaseModel):
    llm: LLMSection = Field(default_factory=LLMSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    features: FeaturesSection = Field(default_factory=FeaturesSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)
    network: NetworkSection = Field(default_factory=NetworkSection)
    office2pdf: Office2PDFSection = Field(default_factory=Office2PDFSection)


class _RuntimeState(BaseModel):
    config_path: Path
    config: AiChatUtilConfig


_runtime_state: _RuntimeState | None = None


def _find_project_root(start: Path) -> Path | None:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def _default_project_root() -> Path | None:
    # Prefer cwd walk-up.
    cwd_root = _find_project_root(Path.cwd())
    if cwd_root is not None:
        return cwd_root

    # Fallback to package location walk-up.
    here = Path(__file__).resolve()
    pkg_root = _find_project_root(here)
    return pkg_root


def _abspath(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def resolve_config_path(cli_config_path: str | None) -> Path:
    tried: list[Path] = []

    if cli_config_path:
        candidate = _abspath(cli_config_path)
        tried.append(candidate)
        if not candidate.is_file():
            raise ConfigError(
                f"--config で指定された設定ファイルが見つかりません: {candidate}"
            )
        # propagate to env for downstream subprocess/tool execution
        os.environ[CONFIG_ENV_VAR] = str(candidate)
        return candidate

    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        candidate = _abspath(env_path)
        tried.append(candidate)
        if not candidate.is_file():
            raise ConfigError(
                f"環境変数 {CONFIG_ENV_VAR} で指定された設定ファイルが見つかりません: {candidate}"
            )
        return candidate

    cwd_candidate = (Path.cwd() / DEFAULT_CONFIG_FILENAME).resolve()
    tried.append(cwd_candidate)
    if cwd_candidate.is_file():
        return cwd_candidate

    project_root = _default_project_root()
    if project_root is not None:
        root_candidate = (project_root / DEFAULT_CONFIG_FILENAME).resolve()
        tried.append(root_candidate)
        if root_candidate.is_file():
            return root_candidate

    tried_str = "\n".join(f"- {p}" for p in tried)
    raise ConfigError(
        "設定ファイル ai-chat-util-config.yml が見つかりません。以下を探索しました:\n" + tried_str +
        f"\n\n対処: {CONFIG_ENV_VAR} にパスを設定するか、--config を指定するか、カレント/プロジェクトルートに {DEFAULT_CONFIG_FILENAME} を配置してください。"
    )


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ConfigError(
            "PyYAML がインストールされていません。requirements に PyYAML を追加してください。"
        ) from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  # type: ignore[attr-defined]

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"ai-chat-util-config.yml のルートは mapping(dict) である必要があります: {path}")
    return data


_ENV_REF_PREFIX = "os.environ/"


def _resolve_env_ref(value: str, *, config_path: Path, field_path: str) -> str:
    if not value.startswith(_ENV_REF_PREFIX):
        raise ConfigError(
            f"秘密情報は ai-chat-util-config.yml に直書きできません: {config_path} ({field_path})\n"
            f"対処: '{field_path}: os.environ/ENV_VAR_NAME' の形式で環境変数参照にしてください。"
        )

    env_name = value[len(_ENV_REF_PREFIX) :].strip()
    if not env_name:
        raise ConfigError(
            f"環境変数参照が不正です（変数名が空）: {config_path} ({field_path})"
        )

    resolved = os.getenv(env_name)
    if resolved is None or resolved == "":
        raise ConfigError(
            f"環境変数 {env_name} が設定されていません: {config_path} ({field_path})\n"
            "対処: .env もしくは環境変数で API キー等を設定してください。"
        )
    return resolved


def _apply_secret_overrides_from_yaml(
    raw: dict[str, Any], *, config_path: Path, field_prefix: str = ""
) -> dict[str, Any]:
    """Apply secret settings from YAML in a safe way.

    - Supports llm.api_key
    - Supports llm.extra_headers (values only)
    - Secrets must be provided as env reference: os.environ/VAR
    """

    llm = raw.get("llm")
    if not isinstance(llm, dict):
        return raw

    copied_llm = dict(llm)
    changed = False

    if "api_key" in llm:
        api_key_value = llm.get("api_key")
        if api_key_value is not None:
            if not isinstance(api_key_value, str):
                raise ConfigError(
                    f"llm.api_key は文字列である必要があります: {config_path} ({field_prefix}llm.api_key)"
                )
            copied_llm["api_key"] = _resolve_env_ref(
                api_key_value,
                config_path=config_path,
                field_path=f"{field_prefix}llm.api_key",
            )
            changed = True

    if "extra_headers" in llm:
        extra_headers_value = llm.get("extra_headers")
        if extra_headers_value is not None:
            if not isinstance(extra_headers_value, dict):
                raise ConfigError(
                    f"llm.extra_headers は mapping(dict) である必要があります: {config_path} ({field_prefix}llm.extra_headers)"
                )
            resolved_headers: dict[str, str] = {}
            for k, v in extra_headers_value.items():
                if not isinstance(k, str) or not k.strip():
                    raise ConfigError(
                        f"llm.extra_headers のキーは空でない文字列である必要があります: {config_path} (llm.extra_headers)"
                    )
                if not isinstance(v, str):
                    raise ConfigError(
                        f"llm.extra_headers.{k} は文字列である必要があります: {config_path} ({field_prefix}llm.extra_headers.{k})"
                    )
                resolved_headers[k] = _resolve_env_ref(
                    v,
                    config_path=config_path,
                    field_path=f"{field_prefix}llm.extra_headers.{k}",
                )

            copied_llm["extra_headers"] = resolved_headers
            changed = True

    if not changed:
        return raw

    copied = dict(raw)
    copied["llm"] = copied_llm
    return copied


def init_runtime(config_path: str | None = None) -> AiChatUtilConfig:
    global _runtime_state

    # Load secrets from .env / env. Non-secrets are not read from env.
    load_dotenv()

    resolved = resolve_config_path(config_path)
    raw_root = _load_yaml_config(resolved)

    # New format: settings must live under ai_chat_util_config.
    raw_section = raw_root.get(AI_CHAT_UTIL_CONFIG_ROOT_KEY, "__missing__")
    if raw_section == "__missing__":
        raise ConfigError(
            f"設定ファイルの形式が不正です: {resolved}\n"
            f"ルートに '{AI_CHAT_UTIL_CONFIG_ROOT_KEY}:' が必要です。\n\n"
            "旧フォーマット（ルート直下に llm/paths/features...）はサポートされません。\n"
            "対処: 既存の llm/paths/features... を ai_chat_util_config: 配下へ 1段インデントして移動してください。"
        )
    if raw_section is None:
        raw_section = {}
    if not isinstance(raw_section, dict):
        raise ConfigError(
            f"{AI_CHAT_UTIL_CONFIG_ROOT_KEY} は mapping(dict) である必要があります: {resolved} ({AI_CHAT_UTIL_CONFIG_ROOT_KEY})"
        )

    # Secrets may be declared in YAML only via env reference (os.environ/VAR).
    raw = _apply_secret_overrides_from_yaml(
        dict(raw_section),
        config_path=resolved,
        field_prefix=f"{AI_CHAT_UTIL_CONFIG_ROOT_KEY}.",
    )

    try:
        config = AiChatUtilConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"設定ファイルの検証に失敗しました: {resolved}\n{e}") from e

    _runtime_state = _RuntimeState(config_path=resolved, config=config)
    _apply_non_secret_runtime_side_effects(config)

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


def get_runtime_config_path() -> Path:
    if _runtime_state is None:
        init_runtime(None)
    assert _runtime_state is not None
    return _runtime_state.config_path


def _apply_non_secret_runtime_side_effects(config: AiChatUtilConfig) -> None:
    _configure_python_logging(config)
    _configure_litellm(config)


def _configure_python_logging(config: AiChatUtilConfig) -> None:
    # Keep this lightweight but deterministic: CLI/test runs often reconfigure logging.
    # We reset handlers to avoid duplicated output and to enforce UTF-8 file encoding.
    import logging
    import re

    class _RedactingFormatter(logging.Formatter):
        _replacements: list[tuple[re.Pattern[str], str]] = [
            # common api_key patterns
            (re.compile(r"(api_key\s*=\s*)(['\"])\s*[^'\"]+\2", re.IGNORECASE), r"\\1\\2***\\2"),
            (re.compile(r"(api_key\s*:\s*)(['\"])\s*[^'\"]+\2", re.IGNORECASE), r"\\1\\2***\\2"),
            # Bearer tokens
            (re.compile(r"(Authorization\s*:\s*Bearer\s+)[^\s\"]+", re.IGNORECASE), r"\\1***"),
            # OpenAI-style keys (best-effort)
            (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), "sk-***"),
        ]

        def format(self, record: logging.LogRecord) -> str:
            s = super().format(record)
            for pattern, repl in self._replacements:
                s = pattern.sub(repl, s)
            return s

    level_name = (config.logging.level or "INFO").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    formatter: logging.Formatter = _RedactingFormatter(
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


# =====================
# Autonomous Agent Util Runtime (embedded)
# =====================

AUTONOMOUS_CONFIG_ENV_VAR = "AUTONOMOUS_AGENT_UTIL_CONFIG"
AUTONOMOUS_DEFAULT_CONFIG_FILENAME = "autonomous-agent-util-config.yml"
AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME = DEFAULT_CONFIG_FILENAME

# New autonomous-agent-util standalone config format root key (required).
AUTONOMOUS_AGENT_UTIL_CONFIG_ROOT_KEY = "autonomous_agent_util_config"


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
    command: str = Field(default="opencode run")


class AutonomousBackendSection(BaseModel):
    task_backend: Literal["docker", "compose", "subprocess", "process"] = Field(default="process")


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


class AutonomousSubprocessSection(BaseModel):
    command: str = Field(default="opencode run")


class AutonomousAgentUtilConfig(BaseModel):
    llm: AutonomousAgentUtilLLMSection = Field(default_factory=AutonomousAgentUtilLLMSection)
    compose: AutonomousComposeSection = Field(default_factory=AutonomousComposeSection)
    backend: AutonomousBackendSection = Field(default_factory=AutonomousBackendSection)
    monitor: AutonomousMonitorSection = Field(default_factory=AutonomousMonitorSection)
    paths: AutonomousPathsSection = Field(default_factory=AutonomousPathsSection)
    host: AutonomousHostSection = Field(default_factory=AutonomousHostSection)
    subprocess: AutonomousSubprocessSection = Field(default_factory=AutonomousSubprocessSection)
    logging: AutonomousAgentUtilLoggingSection = Field(default_factory=AutonomousAgentUtilLoggingSection)

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


class _AutonomousRuntimeState(BaseModel):
    config_path: Path
    config: AutonomousAgentUtilConfig


_autonomous_runtime_state: _AutonomousRuntimeState | None = None


def resolve_autonomous_config_path(cli_config_path: str | None) -> Path:
    tried: list[Path] = []

    if cli_config_path:
        candidate = _abspath(cli_config_path)
        tried.append(candidate)
        if not candidate.is_file():
            raise ConfigError(f"--config で指定された設定ファイルが見つかりません: {candidate}")
        return candidate

    # Integration mode: prefer resolving from ai-chat-util-config.yml via AI_CHAT_UTIL_CONFIG.
    ai_cfg_env = os.getenv(CONFIG_ENV_VAR)
    if ai_cfg_env:
        candidate = _abspath(ai_cfg_env)
        tried.append(candidate)
        if not candidate.is_file():
            raise ConfigError(f"環境変数 {CONFIG_ENV_VAR} で指定された設定ファイルが見つかりません: {candidate}")
        return candidate

    # Backward-compat: standalone autonomous-agent-util-config.yml via env.
    env_path = os.getenv(AUTONOMOUS_CONFIG_ENV_VAR)
    if env_path:
        candidate = _abspath(env_path)
        tried.append(candidate)
        if not candidate.is_file():
            raise ConfigError(
                f"環境変数 {AUTONOMOUS_CONFIG_ENV_VAR} で指定された設定ファイルが見つかりません: {candidate}"
            )
        return candidate

    # Integration mode: allow ai-chat-util-config.yml in CWD.
    cwd_ai_candidate = (Path.cwd() / AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME).resolve()
    tried.append(cwd_ai_candidate)
    if cwd_ai_candidate.is_file():
        return cwd_ai_candidate

    # Backward-compat: standalone autonomous-agent-util-config.yml in CWD.
    cwd_candidate = (Path.cwd() / AUTONOMOUS_DEFAULT_CONFIG_FILENAME).resolve()
    tried.append(cwd_candidate)
    if cwd_candidate.is_file():
        return cwd_candidate

    project_root = _default_project_root()
    if project_root is not None:
        root_ai_candidate = (project_root / AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME).resolve()
        tried.append(root_ai_candidate)
        if root_ai_candidate.is_file():
            return root_ai_candidate

        root_candidate = (project_root / AUTONOMOUS_DEFAULT_CONFIG_FILENAME).resolve()
        tried.append(root_candidate)
        if root_candidate.is_file():
            return root_candidate

    tried_str = "\n".join(f"- {p}" for p in tried)
    raise ConfigError(
        "設定ファイルが見つかりません。以下を探索しました:\n"
        + tried_str
        + (
            f"\n\n対処: {CONFIG_ENV_VAR} にパスを設定するか、--config を指定するか、"
            f"カレント/プロジェクトルートに {AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME} を配置してください。"
        )
    )


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

    resolved = _resolve_env_ref(api_key_value, config_path=config_path, field_path="llm.api_key")
    config.llm.api_key = resolved


def _configure_autonomous_python_logging(config: AutonomousAgentUtilConfig) -> None:
    import logging
    import re

    level_name = (config.logging.level or "INFO").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    # Reset handlers to avoid duplicates across repeated init.
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    class _RedactingFormatter(logging.Formatter):
        _replacements: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"(api_key\s*=\s*)(['\"])\s*[^'\"]+\2", re.IGNORECASE), r"\1\2***\2"),
            (re.compile(r"(api_key\s*:\s*)(['\"])\s*[^'\"]+\2", re.IGNORECASE), r"\1\2***\2"),
            (re.compile(r"(Authorization\s*:\s*Bearer\s+)[^\s\"]+", re.IGNORECASE), r"\1***"),
            (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), "sk-***"),
        ]

        def format(self, record: logging.LogRecord) -> str:
            s = super().format(record)
            for pattern, repl in self._replacements:
                s = pattern.sub(repl, s)
            return s

    formatter: logging.Formatter = _RedactingFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if config.logging.file:
        fh = logging.FileHandler(config.logging.file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    noisy_loggers = {
        "litellm": logging.WARNING,
        "LiteLLM": logging.WARNING,
        "openai": logging.WARNING,
        "aiosqlite": logging.WARNING,
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "aiohttp": logging.WARNING,
        "urllib3": logging.WARNING,
        "asyncio": logging.WARNING,
        "sse_starlette": logging.WARNING,
    }
    for name, lvl in noisy_loggers.items():
        logging.getLogger(name).setLevel(lvl)


def init_autonomous_runtime(config_path: str | None = None) -> AutonomousAgentUtilConfig:
    global _autonomous_runtime_state

    load_dotenv()

    resolved = resolve_autonomous_config_path(config_path)
    raw_root = _load_yaml_config(resolved)

    raw: dict[str, Any]
    ai_section = raw_root.get(AI_CHAT_UTIL_CONFIG_ROOT_KEY) if isinstance(raw_root, dict) else None
    if ai_section is None:
        ai_section_dict: dict[str, Any] | None = None
    elif not isinstance(ai_section, dict):
        raise ConfigError(
            f"{AI_CHAT_UTIL_CONFIG_ROOT_KEY} は mapping(dict) である必要があります: {resolved} ({AI_CHAT_UTIL_CONFIG_ROOT_KEY})"
        )
    else:
        ai_section_dict = ai_section

    embedded = (
        ai_section_dict.get("autonomous_agent_util", "__missing__")
        if ai_section_dict is not None
        else "__missing__"
    )

    if embedded != "__missing__":
        if embedded is None:
            embedded = {}
        if not isinstance(embedded, dict):
            raise ConfigError(
                f"ai_chat_util_config.autonomous_agent_util は mapping(dict) である必要があります: {resolved} (ai_chat_util_config.autonomous_agent_util)"
            )

        inherited = dict(embedded)
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
                    f"対処: {resolved} の '{AI_CHAT_UTIL_CONFIG_ROOT_KEY}.autonomous_agent_util' セクションを追加するか、{AUTONOMOUS_DEFAULT_CONFIG_FILENAME} を指定してください。"
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
        "llm",
        "compose",
        "backend",
        "monitor",
        "paths",
        "host",
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

    _autonomous_runtime_state = _AutonomousRuntimeState(config_path=resolved, config=config)
    _configure_autonomous_python_logging(config)
    return config


def get_autonomous_runtime_config() -> AutonomousAgentUtilConfig:
    if _autonomous_runtime_state is None:
        return init_autonomous_runtime(None)
    return _autonomous_runtime_state.config


def get_autonomous_runtime_config_path() -> Path:
    if _autonomous_runtime_state is None:
        init_autonomous_runtime(None)
    assert _autonomous_runtime_state is not None
    return _autonomous_runtime_state.config_path


def apply_autonomous_logging_overrides(level: str | None = None, file: str | None = None) -> None:
    cfg = get_autonomous_runtime_config()
    effective = cfg.model_copy(deep=True)
    if level:
        effective.logging.level = level
    if file:
        effective.logging.file = file
    _configure_autonomous_python_logging(effective)
