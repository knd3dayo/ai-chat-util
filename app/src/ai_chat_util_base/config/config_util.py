from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Literal, Callable

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator, ConfigDict

CONFIG_ENV_VAR = "AI_CHAT_UTIL_CONFIG"
DEFAULT_CONFIG_FILENAME = "ai-chat-util-config.yml"
CODING_DEFAULT_CONFIG_FILENAME = "coding-agent-util-config.yml"
AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME = DEFAULT_CONFIG_FILENAME

# New config format root key (required).
AI_CHAT_UTIL_CONFIG_ROOT_KEY = "ai_chat_util_config"
# New coding-agent-util standalone config format root key (required).
CODING_AGENT_UTIL_CONFIG_ROOT_KEY = "coding_agent_util_config"

CODING_CONFIG_ENV_VAR = "CODING_AGENT_UTIL_CONFIG"

_ENV_REF_PREFIX = "os.environ/"


class ConfigError(RuntimeError):
    pass

def load_resolved_yaml(
    config_path: str | None,
    *,
    resolver: Callable[[str | None], Path],
) -> tuple[Path, dict[str, Any]]:
    # Load secrets from .env / env. Non-secrets are not read from env.
    load_dotenv()
    resolved = resolver(config_path)
    raw_root = load_yaml_config(resolved)
    return resolved, raw_root


def load_yaml_config(path: Path) -> dict[str, Any]:
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


def resolve_config_path(cli_config_path: str | None) -> Path:
    tried: list[Path] = []

    resolved = _resolve_cli_config_path(
        cli_config_path,
        tried=tried,
        # propagate to env for downstream subprocess/tool execution
        propagate_env_var=CONFIG_ENV_VAR,
    )
    if resolved is not None:
        return resolved

    resolved = _resolve_env_config_path(CONFIG_ENV_VAR, tried=tried)
    if resolved is not None:
        return resolved

    resolved = _resolve_in_cwd([DEFAULT_CONFIG_FILENAME], tried=tried)
    if resolved is not None:
        return resolved

    project_root = _default_project_root()
    resolved = _resolve_in_project_root([DEFAULT_CONFIG_FILENAME], tried=tried, project_root=project_root)
    if resolved is not None:
        return resolved

    tried_str = "\n".join(f"- {p}" for p in tried)
    raise ConfigError(
        "設定ファイル ai-chat-util-config.yml が見つかりません。以下を探索しました:\n" + tried_str +
        f"\n\n対処: {CONFIG_ENV_VAR} にパスを設定するか、--config を指定するか、カレント/プロジェクトルートに {DEFAULT_CONFIG_FILENAME} を配置してください。"
    )

def _resolve_cli_config_path(
    cli_config_path: str | None,
    *,
    tried: list[Path],
    propagate_env_var: str | None = None,
) -> Path | None:
    if not cli_config_path:
        return None

    candidate = _abspath(cli_config_path)
    tried.append(candidate)
    if not candidate.is_file():
        raise ConfigError(f"--config で指定された設定ファイルが見つかりません: {candidate}")

    if propagate_env_var:
        os.environ[propagate_env_var] = str(candidate)

    return candidate


def _resolve_env_config_path(env_var: str, *, tried: list[Path]) -> Path | None:
    env_path = os.getenv(env_var)
    if not env_path:
        return None

    candidate = _abspath(env_path)
    tried.append(candidate)
    if not candidate.is_file():
        raise ConfigError(f"環境変数 {env_var} で指定された設定ファイルが見つかりません: {candidate}")

    return candidate


def resolve_coding_config_path(cli_config_path: str | None) -> Path:
    tried: list[Path] = []

    resolved = _resolve_cli_config_path(
        cli_config_path,
        tried=tried,
        # propagate to env for downstream subprocess/tool execution
        # NOTE: coding MCP server passes ai-chat-util-config.yml here (integration mode).
        propagate_env_var=CONFIG_ENV_VAR,
    )
    if resolved is not None:
        return resolved

    # Integration mode: prefer resolving from ai-chat-util-config.yml via AI_CHAT_UTIL_CONFIG.
    resolved = _resolve_env_config_path(CONFIG_ENV_VAR, tried=tried)
    if resolved is not None:
        return resolved

    # Canonical standalone coding-agent-util-config.yml via env.
    resolved = _resolve_env_config_path(CODING_CONFIG_ENV_VAR, tried=tried)
    if resolved is not None:
        return resolved

    # Integration mode: allow ai-chat-util-config.yml in CWD.
    resolved = _resolve_in_cwd(
        [
            AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME,
            CODING_DEFAULT_CONFIG_FILENAME,
        ],
        tried=tried,
    )
    if resolved is not None:
        return resolved

    project_root = _default_project_root()
    resolved = _resolve_in_project_root(
        [
            AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME,
            CODING_DEFAULT_CONFIG_FILENAME,
        ],
        tried=tried,
        project_root=project_root,
    )
    if resolved is not None:
        return resolved

    tried_str = "\n".join(f"- {p}" for p in tried)
    raise ConfigError(
        "設定ファイルが見つかりません。以下を探索しました:\n"
        + tried_str
        + (
            f"\n\n対処: {CONFIG_ENV_VAR} にパスを設定するか、--config を指定するか、"
            f"カレント/プロジェクトルートに {AI_CHAT_UTIL_DEFAULT_CONFIG_FILENAME} または {CODING_DEFAULT_CONFIG_FILENAME} を配置してください。"
        )
    )

def _resolve_in_cwd(filenames: list[str], *, tried: list[Path]) -> Path | None:
    for filename in filenames:
        candidate = (Path.cwd() / filename).resolve()
        tried.append(candidate)
        if candidate.is_file():
            return candidate
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


def _resolve_in_project_root(
    filenames: list[str], *, tried: list[Path], project_root: Path | None
) -> Path | None:
    if project_root is None:
        return None
    for filename in filenames:
        candidate = (project_root / filename).resolve()
        tried.append(candidate)
        if candidate.is_file():
            return candidate
    return None


def _find_project_root(start: Path) -> Path | None:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None

def extract_required_root_section(*, raw_root: dict[str, Any], resolved: Path) -> dict[str, Any]:
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
    return dict(raw_section)

def extract_optional_ai_section_dict(
    *, raw_root: dict[str, Any], resolved: Path
) -> dict[str, Any] | None:
    ai_section = raw_root.get(AI_CHAT_UTIL_CONFIG_ROOT_KEY) if isinstance(raw_root, dict) else None
    if ai_section is None:
        return None
    if not isinstance(ai_section, dict):
        raise ConfigError(
            f"{AI_CHAT_UTIL_CONFIG_ROOT_KEY} は mapping(dict) である必要があります: {resolved} ({AI_CHAT_UTIL_CONFIG_ROOT_KEY})"
        )
    return ai_section

def apply_secret_overrides_from_yaml(
    raw: dict[str, Any], *, config_path: Path, field_prefix: str = ""
) -> dict[str, Any]:
    """Apply secret settings from YAML in a safe way.

    - Supports llm.api_key
    - Supports llm.extra_headers (values only)
    - Supports mcp.extra_headers (values only)
    - Secrets must be provided as env reference: os.environ/VAR
    """

    llm = raw.get("llm")
    mcp = raw.get("mcp")
    file_server = raw.get("file_server")

    copied_llm = dict(llm) if isinstance(llm, dict) else None
    copied_mcp = dict(mcp) if isinstance(mcp, dict) else None
    copied_file_server = dict(file_server) if isinstance(file_server, dict) else None
    changed = False

    if copied_llm is None and copied_mcp is None and copied_file_server is None:
        return raw

    if copied_llm is not None and "api_key" in copied_llm:
        api_key_value = copied_llm.get("api_key")
        if api_key_value is not None:
            if not isinstance(api_key_value, str):
                raise ConfigError(
                    f"llm.api_key は文字列である必要があります: {config_path} ({field_prefix}llm.api_key)"
                )
            copied_llm["api_key"] = resolve_env_ref(
                api_key_value,
                config_path=config_path,
                field_path=f"{field_prefix}llm.api_key",
            )
            changed = True

    if copied_llm is not None and "extra_headers" in copied_llm:
        extra_headers_value = copied_llm.get("extra_headers")
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
                resolved_headers[k] = resolve_env_ref(
                    v,
                    config_path=config_path,
                    field_path=f"{field_prefix}llm.extra_headers.{k}",
                )

            copied_llm["extra_headers"] = resolved_headers
            changed = True

    if copied_mcp is not None and "extra_headers" in copied_mcp:
        extra_headers_value = copied_mcp.get("extra_headers")
        if extra_headers_value is not None:
            if not isinstance(extra_headers_value, dict):
                raise ConfigError(
                    f"mcp.extra_headers は mapping(dict) である必要があります: {config_path} ({field_prefix}mcp.extra_headers)"
                )
            resolved_headers: dict[str, str] = {}
            for k, v in extra_headers_value.items():
                if not isinstance(k, str) or not k.strip():
                    raise ConfigError(
                        f"mcp.extra_headers のキーは空でない文字列である必要があります: {config_path} (mcp.extra_headers)"
                    )
                if not isinstance(v, str):
                    raise ConfigError(
                        f"mcp.extra_headers.{k} は文字列である必要があります: {config_path} ({field_prefix}mcp.extra_headers.{k})"
                    )
                resolved_headers[k] = resolve_env_ref(
                    v,
                    config_path=config_path,
                    field_path=f"{field_prefix}mcp.extra_headers.{k}",
                )

            copied_mcp["extra_headers"] = resolved_headers
            changed = True

    if copied_file_server is not None and "smb" in copied_file_server:
        smb_value = copied_file_server.get("smb")
        if smb_value is not None:
            if not isinstance(smb_value, dict):
                raise ConfigError(
                    f"file_server.smb は mapping(dict) である必要があります: {config_path} ({field_prefix}file_server.smb)"
                )
            copied_smb = dict(smb_value)
            smb_changed = False
            if bool(copied_smb.get("enabled", False)):
                for secret_key in ("username", "password"):
                    if secret_key not in copied_smb:
                        continue
                    secret_value = copied_smb.get(secret_key)
                    if secret_value is None:
                        continue
                    if not isinstance(secret_value, str):
                        raise ConfigError(
                            f"file_server.smb.{secret_key} は文字列である必要があります: {config_path} ({field_prefix}file_server.smb.{secret_key})"
                        )
                    copied_smb[secret_key] = resolve_env_ref(
                        secret_value,
                        config_path=config_path,
                        field_path=f"{field_prefix}file_server.smb.{secret_key}",
                    )
                    smb_changed = True

            if smb_changed:
                copied_file_server["smb"] = copied_smb
                changed = True

    if not changed:
        return raw

    copied = dict(raw)
    if copied_llm is not None:
        copied["llm"] = copied_llm
    if copied_mcp is not None:
        copied["mcp"] = copied_mcp
    if copied_file_server is not None:
        copied["file_server"] = copied_file_server
    return copied

def resolve_env_ref(value: str, *, config_path: Path, field_path: str) -> str:
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
