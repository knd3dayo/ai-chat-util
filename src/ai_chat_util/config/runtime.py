from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


CONFIG_ENV_VAR = "AI_CHAT_UTIL_CONFIG"
DEFAULT_CONFIG_FILENAME = "config.yml"


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
    # For safety, values must be provided via env reference in config.yml and are resolved at load time.
    extra_headers: dict[str, str] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_api_key_required(self) -> "LLMSection":
        provider = (self.provider or "").lower()
        providers_requiring_key = {"openai", "azure", "azure_openai", "anthropic"}
        if provider in providers_requiring_key and not self.api_key:
            raise ValueError(
                "llm.api_key が未設定です。config.yml で 'llm.api_key: os.environ/ENV_VAR_NAME' を設定し、"
                "参照先の環境変数を設定してください。"
            )
        return self

    def create_litellm_model_list(self) -> list[dict[str, Any]]:
        """Create a list of model names to try with LiteLLM, based on the config."""
        completion_litellm_dict = {}
        completion_litellm_dict["model"] = f"{self.provider}/{self.completion_model}"
        completion_litellm_dict["api_key"] = self.api_key
        if self.base_url:
            completion_litellm_dict["api_base"] = self.base_url
        if self.api_version:
            completion_litellm_dict["api_version"] = self.api_version
        if self.extra_headers:
            completion_litellm_dict["extra_headers"] = self.extra_headers

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
        if self.extra_headers:
            embedding_litellm_dict["extra_headers"] = self.extra_headers
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
        default=15,
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
        "設定ファイル config.yml が見つかりません。以下を探索しました:\n" + tried_str +
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
        raise ConfigError(f"config.yml のルートは mapping(dict) である必要があります: {path}")
    return data


_ENV_REF_PREFIX = "os.environ/"


def _resolve_env_ref(value: str, *, config_path: Path, field_path: str) -> str:
    if not value.startswith(_ENV_REF_PREFIX):
        raise ConfigError(
            f"秘密情報は config.yml に直書きできません: {config_path} ({field_path})\n"
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


def _apply_secret_overrides_from_yaml(raw: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
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
                    f"llm.api_key は文字列である必要があります: {config_path} (llm.api_key)"
                )
            copied_llm["api_key"] = _resolve_env_ref(
                api_key_value, config_path=config_path, field_path="llm.api_key"
            )
            changed = True

    if "extra_headers" in llm:
        extra_headers_value = llm.get("extra_headers")
        if extra_headers_value is not None:
            if not isinstance(extra_headers_value, dict):
                raise ConfigError(
                    f"llm.extra_headers は mapping(dict) である必要があります: {config_path} (llm.extra_headers)"
                )
            resolved_headers: dict[str, str] = {}
            for k, v in extra_headers_value.items():
                if not isinstance(k, str) or not k.strip():
                    raise ConfigError(
                        f"llm.extra_headers のキーは空でない文字列である必要があります: {config_path} (llm.extra_headers)"
                    )
                if not isinstance(v, str):
                    raise ConfigError(
                        f"llm.extra_headers.{k} は文字列である必要があります: {config_path} (llm.extra_headers.{k})"
                    )
                resolved_headers[k] = _resolve_env_ref(
                    v,
                    config_path=config_path,
                    field_path=f"llm.extra_headers.{k}",
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
    raw = _load_yaml_config(resolved)

    # Secrets may be declared in YAML only via env reference (os.environ/VAR).
    raw = _apply_secret_overrides_from_yaml(raw, config_path=resolved)

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
