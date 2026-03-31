import json
from pathlib import Path
import pytest

from ai_chat_util.common.config.ai_chat_util_mcp_config import MCPServerConfig


def _write_mcp_json(tmp_path: Path, env: dict | None) -> str:
    data = {
        "mcpServers": {
            "TestServer": {
                "type": "stdio",
                "command": "python",
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
