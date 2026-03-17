from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator, ConfigDict


CONFIG_ENV_VAR = "AUTONOMOUS_AGENT_UTIL_CONFIG"
DEFAULT_CONFIG_FILENAME = "autonomous-agent-util-config.yml"


class ConfigError(RuntimeError):
    pass


_ENV_REF_PREFIX = "os.environ/"


def _abspath(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _find_project_root(start: Path) -> Path | None:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def _default_project_root() -> Path | None:
    cwd_root = _find_project_root(Path.cwd())
    if cwd_root is not None:
        return cwd_root

    here = Path(__file__).resolve()
    pkg_root = _find_project_root(here)
    return pkg_root


def resolve_config_path(cli_config_path: str | None) -> Path:
    tried: list[Path] = []

    if cli_config_path:
        candidate = _abspath(cli_config_path)
        tried.append(candidate)
        if not candidate.is_file():
            raise ConfigError(f"--config で指定された設定ファイルが見つかりません: {candidate}")
        os.environ[CONFIG_ENV_VAR] = str(candidate)
        return candidate

    env_path = os.getenv(CONFIG_ENV_VAR)
    if env_path:
        candidate = _abspath(env_path)
        tried.append(candidate)
        if not candidate.is_file():
            raise ConfigError(f"環境変数 {CONFIG_ENV_VAR} で指定された設定ファイルが見つかりません: {candidate}")
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
        "設定ファイル autonomous-agent-util-config.yml が見つかりません。以下を探索しました:\n"
        + tried_str
        + f"\n\n対処: {CONFIG_ENV_VAR} にパスを設定するか、--config を指定するか、カレント/プロジェクトルートに {DEFAULT_CONFIG_FILENAME} を配置してください。"
    )


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ConfigError("PyYAML がインストールされていません。dependencies に pyyaml を追加してください。") from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  # type: ignore[attr-defined]

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"autonomous-agent-util-config.yml のルートは mapping(dict) である必要があります: {path}")
    return data


def _resolve_env_ref(value: str, *, config_path: Path, field_path: str) -> str:
    if not value.startswith(_ENV_REF_PREFIX):
        raise ConfigError(
            f"秘密情報は autonomous-agent-util-config.yml に直書きできません: {config_path} ({field_path})\n"
            f"対処: '{field_path}: os.environ/ENV_VAR_NAME' の形式で環境変数参照にしてください。"
        )

    env_name = value[len(_ENV_REF_PREFIX) :].strip()
    if not env_name:
        raise ConfigError(f"環境変数参照が不正です（変数名が空）: {config_path} ({field_path})")

    resolved = os.getenv(env_name)
    if resolved is None or resolved == "":
        raise ConfigError(
            f"環境変数 {env_name} が設定されていません: {config_path} ({field_path})\n"
            "対処: .env もしくは環境変数で API キー等を設定してください。"
        )
    return resolved


class LoggingSection(BaseModel):
    level: str = Field(default="INFO")
    file: str | None = Field(default=None)


class LLMSection(BaseModel):
    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o")

    # non-secret base url/endpoint
    base_url: str | None = Field(default=None)

    # non-secret: override LLM_BASE_URL passed into container
    base_url_in_container: str | None = Field(default=None)

    # secret API key (must be provided as env reference or env outside YAML)
    api_key: str | None = Field(default=None)


class ComposeSection(BaseModel):
    directory: str = Field(default=".")
    file: str = Field(default="docker-compose.yml")
    service_name: str = Field(default="executor-service")
    command: str = Field(default="opencode run")


class BackendSection(BaseModel):
    task_backend: Literal["docker", "compose", "subprocess", "process"] = Field(default="process")


class MonitorSection(BaseModel):
    disable_detach_monitor: bool = Field(default=False)
    detach_monitor_interval: float = Field(default=2.0, ge=0.1)
    debug_container: bool = Field(default=False)


class WorkspacePathRewriteRule(BaseModel):
    """Rewrite inbound workspace_path before validation.

    Useful when supervisor and executor see different absolute paths.
    Example YAML:

    paths:
      workspace_path_rewrites:
        - from: /srv/ai_platform/workspaces
          to: /workspaces
    """

    model_config = ConfigDict(populate_by_name=True)

    from_prefix: str = Field(..., alias="from")
    to_prefix: str = Field(..., alias="to")

    @model_validator(mode="after")
    def _validate_prefixes(self) -> "WorkspacePathRewriteRule":
        if not (isinstance(self.from_prefix, str) and self.from_prefix.strip()):
            raise ValueError("paths.workspace_path_rewrites[].from must be a non-empty string")
        if not (isinstance(self.to_prefix, str) and self.to_prefix.strip()):
            raise ValueError("paths.workspace_path_rewrites[].to must be a non-empty string")
        if not self.from_prefix.strip().startswith("/"):
            raise ValueError("paths.workspace_path_rewrites[].from must be an absolute path prefix")
        if not self.to_prefix.strip().startswith("/"):
            raise ValueError("paths.workspace_path_rewrites[].to must be an absolute path prefix")
        return self


class PathsSection(BaseModel):
    workspace_root: str = Field(default="/tmp/autonomous_agent_tasks")

    # Task DB root on executor host
    host_projects_root: str = Field(default="/home/user/ai-platform/data/projects")

    # Optional security guard for inbound workspace_path
    executor_allowed_workspace_root: str | None = Field(default=None)

    # Optional mapping to rewrite inbound workspace_path (supervisor->executor).
    workspace_path_rewrites: list[WorkspacePathRewriteRule] = Field(default_factory=list)



class HostSection(BaseModel):
    uid: int | None = Field(default=None)
    gid: int | None = Field(default=None)


class SubprocessSection(BaseModel):
    command: str = Field(default="opencode run")


class AutonomousAgentUtilConfig(BaseModel):
    llm: LLMSection = Field(default_factory=LLMSection)
    compose: ComposeSection = Field(default_factory=ComposeSection)
    backend: BackendSection = Field(default_factory=BackendSection)
    monitor: MonitorSection = Field(default_factory=MonitorSection)
    paths: PathsSection = Field(default_factory=PathsSection)
    host: HostSection = Field(default_factory=HostSection)
    subprocess: SubprocessSection = Field(default_factory=SubprocessSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)

    @model_validator(mode="after")
    def _validate_llm_secret_policy(self) -> "AutonomousAgentUtilConfig":
        if self.llm.api_key and self.llm.api_key.startswith(_ENV_REF_PREFIX):
            # Allowed: env reference; will be resolved in init_runtime.
            return self
        if self.llm.api_key:
            # Disallow literal secrets in YAML.
            raise ValueError(
                "llm.api_key は秘密情報のため autonomous-agent-util-config.yml に直書きできません。"
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


class _RuntimeState(BaseModel):
    config_path: Path
    config: AutonomousAgentUtilConfig


_runtime_state: _RuntimeState | None = None


def _apply_secret_overrides_from_yaml(raw: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    llm = raw.get("llm")
    if not isinstance(llm, dict):
        return raw

    if "api_key" not in llm:
        return raw

    api_key_value = llm.get("api_key")
    if api_key_value is None:
        return raw

    if not isinstance(api_key_value, str):
        raise ConfigError(f"llm.api_key は文字列である必要があります: {config_path} (llm.api_key)")

    resolved = _resolve_env_ref(api_key_value, config_path=config_path, field_path="llm.api_key")

    copied = dict(raw)
    copied_llm = dict(llm)
    copied_llm["api_key"] = resolved
    copied["llm"] = copied_llm
    return copied


def _configure_python_logging(config: AutonomousAgentUtilConfig) -> None:
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
            # common api_key patterns
            (re.compile(r"(api_key\s*=\s*)(['\"])\s*[^'\"]+\2", re.IGNORECASE), r"\1\2***\2"),
            (re.compile(r"(api_key\s*:\s*)(['\"])\s*[^'\"]+\2", re.IGNORECASE), r"\1\2***\2"),
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

    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    formatter: logging.Formatter = _RedactingFormatter(fmt)

    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if config.logging.file:
        fh = logging.FileHandler(config.logging.file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Avoid noisy 3rd-party debug logs that may include request details.
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
    }
    for name, lvl in noisy_loggers.items():
        logging.getLogger(name).setLevel(lvl)


def init_runtime(config_path: str | None = None) -> AutonomousAgentUtilConfig:
    global _runtime_state

    # Load secrets from .env / env. Non-secrets are not read from env.
    load_dotenv()

    resolved = resolve_config_path(config_path)
    raw = _load_yaml_config(resolved)

    # YAML allows keys with only comments to become `null`.
    # Coalesce known section keys to empty dicts for ergonomic configs.
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

    # If secrets are declared in YAML, they must be env references.
    raw = _apply_secret_overrides_from_yaml(raw, config_path=resolved)

    try:
        config = AutonomousAgentUtilConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"設定ファイルの検証に失敗しました: {resolved}\n{e}") from e

    _runtime_state = _RuntimeState(config_path=resolved, config=config)
    _configure_python_logging(config)
    return config


def get_runtime_config() -> AutonomousAgentUtilConfig:
    if _runtime_state is None:
        return init_runtime(None)
    return _runtime_state.config


def get_runtime_config_path() -> Path:
    if _runtime_state is None:
        init_runtime(None)
    assert _runtime_state is not None
    return _runtime_state.config_path


def apply_logging_overrides(level: str | None = None, file: str | None = None) -> None:
    cfg = get_runtime_config()
    effective = cfg.model_copy(deep=True)
    if level:
        effective.logging.level = level
    if file:
        effective.logging.file = file
    _configure_python_logging(effective)
