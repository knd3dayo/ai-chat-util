import json
from pathlib import Path
import pytest

from ai_chat_util.core.common.config.ai_chat_util_mcp_config import MCPServerConfig
from ai_chat_util.core.common.config import runtime as runtime_mod
from ai_chat_util.core.common.config.config_util import ConfigError, resolve_config_path


def _write_mcp_json(tmp_path: Path, env: dict | None, *, command: str = "python") -> str:
    data = {
        "mcpServers": {
            "TestServer": {
                "type": "stdio",
                "command": command,
                "args": ["-c", "print('ok')"],
            }
        }
    }
    if env is not None:
        data["mcpServers"]["TestServer"]["env"] = env

    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_env_ref_is_resolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")

    mcp_path = _write_mcp_json(tmp_path, {"LLM_API_KEY": "os.environ/LLM_API_KEY"})

    cfg = MCPServerConfig()
    cfg.load_server_config(mcp_path)

    assert cfg.servers["TestServer"].env is not None
    assert cfg.servers["TestServer"].env["LLM_API_KEY"] == "dummy-key"


def test_literal_env_value_is_kept(tmp_path: Path) -> None:
    mcp_path = _write_mcp_json(tmp_path, {"PYTHONUTF8": "1"})

    cfg = MCPServerConfig()
    cfg.load_server_config(mcp_path)

    assert cfg.servers["TestServer"].env is not None
    assert cfg.servers["TestServer"].env["PYTHONUTF8"] == "1"


def test_env_ref_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    mcp_path = _write_mcp_json(tmp_path, {"LLM_API_KEY": "os.environ/LLM_API_KEY"})

    cfg = MCPServerConfig()
    with pytest.raises(ValueError) as e:
        cfg.load_server_config(mcp_path)

    msg = str(e.value)
    assert "LLM_API_KEY" in msg
    assert "TestServer" in msg


def test_command_path_expands_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    mcp_path = _write_mcp_json(tmp_path, None, command="${HOME}/bin/python")

    cfg = MCPServerConfig()
    cfg.load_server_config(mcp_path)

    assert cfg.servers["TestServer"].command == f"{tmp_path.as_posix()}/bin/python"


def test_command_path_with_unresolved_env_raises(tmp_path: Path) -> None:
    mcp_path = _write_mcp_json(tmp_path, None, command="${UNSET_HOME}/bin/python")

    cfg = MCPServerConfig()
    with pytest.raises(ValueError) as e:
        cfg.load_server_config(mcp_path)

    assert "環境変数を解決できません" in str(e.value)


def test_resolve_config_path_expands_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    cfg_path.write_text("ai_chat_util_config: {}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = resolve_config_path("${HOME}/ai-chat-util-config.yml")

    assert resolved == cfg_path.resolve()


def test_init_runtime_expands_allowlisted_ai_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "logs").mkdir()
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    data = {
        "ai_chat_util_config": {
            "llm": {"api_key": "os.environ/LLM_API_KEY"},
            "mcp": {
                "mcp_config_path": "${HOME}/cfg/mcp.json",
                "working_directory": "${HOME}/workspace",
            },
            "logging": {"file": "${HOME}/logs/app.log"},
            "features": {"audit_log_path": "${HOME}/logs/audit.jsonl"},
            "network": {"ca_bundle": "${HOME}/certs/ca.pem"},
            "office2pdf": {"libreoffice_path": "${HOME}/bin/soffice"},
            "file_server": {
                "allowed_roots": [
                    {"name": "root", "path": "${HOME}/data"}
                ]
            },
        }
    }
    cfg_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    runtime_mod._runtime_state = None  # type: ignore[attr-defined]

    cfg = runtime_mod.init_runtime(str(cfg_path))

    assert cfg.mcp.mcp_config_path == f"{tmp_path.as_posix()}/cfg/mcp.json"
    assert cfg.mcp.working_directory == f"{tmp_path.as_posix()}/workspace"
    assert cfg.logging.file == f"{tmp_path.as_posix()}/logs/app.log"
    assert cfg.features.audit_log_path == f"{tmp_path.as_posix()}/logs/audit.jsonl"
    assert cfg.network.ca_bundle == f"{tmp_path.as_posix()}/certs/ca.pem"
    assert cfg.office2pdf.libreoffice_path == f"{tmp_path.as_posix()}/bin/soffice"
    assert cfg.file_server.allowed_roots[0].path == f"{tmp_path.as_posix()}/data"


def test_init_runtime_supports_method_based_office2pdf_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    data = {
        "ai_chat_util_config": {
            "llm": {"api_key": "os.environ/LLM_API_KEY"},
            "office2pdf": {
                "method": "pywin32",
                "pywin32": {"office_path": "${HOME}/Microsoft Office/root/Office16/WINWORD.EXE"},
                "libreoffice_exec": {"libreoffice_path": "${HOME}/bin/soffice"},
                "libreoffice_uno": {"host": "uno-host", "port": 8100},
            },
        }
    }
    cfg_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    runtime_mod._runtime_state = None  # type: ignore[attr-defined]

    cfg = runtime_mod.init_runtime(str(cfg_path))

    assert cfg.office2pdf.method == "pywin32"
    assert (
        cfg.office2pdf.pywin32.office_path
        == f"{tmp_path.as_posix()}/Microsoft Office/root/Office16/WINWORD.EXE"
    )
    assert cfg.office2pdf.libreoffice_exec.libreoffice_path == f"{tmp_path.as_posix()}/bin/soffice"
    assert cfg.office2pdf.libreoffice_uno.host == "uno-host"
    assert cfg.office2pdf.libreoffice_uno.port == 8100


def test_init_runtime_normalizes_legacy_office2pdf_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    data = {
        "ai_chat_util_config": {
            "llm": {"api_key": "os.environ/LLM_API_KEY"},
            "office2pdf": {"libreoffice_path": "soffice"},
        }
    }
    cfg_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    runtime_mod._runtime_state = None  # type: ignore[attr-defined]

    cfg = runtime_mod.init_runtime(str(cfg_path))

    assert cfg.office2pdf.method == "libreoffice_exec"
    assert cfg.office2pdf.libreoffice_exec.libreoffice_path == "soffice"
    assert cfg.office2pdf.libreoffice_path == "soffice"


def test_init_runtime_rejects_unresolved_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    cfg_path.write_text(
        json.dumps(
            {
                "ai_chat_util_config": {
                    "llm": {"api_key": "os.environ/LLM_API_KEY"},
                    "logging": {"file": "${UNSET_HOME}/logs/app.log"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    runtime_mod._runtime_state = None  # type: ignore[attr-defined]

    with pytest.raises(ConfigError):
        runtime_mod.init_runtime(str(cfg_path))
