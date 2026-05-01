from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ai_chat_util.ai_chat_util_base.core.common.config.runtime import get_runtime_config, init_runtime
from ai_chat_util.ai_chat_util_base.app.analyze_file_util.api.api_server import app


def _write_local_file_server_config(config_path: Path, *, root_path: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                "ai_chat_util_config:",
                "  llm:",
                "    provider: openai",
                "    api_key: os.environ/LLM_API_KEY",
                "    completion_model: gpt-5",
                "  file_server:",
                "    enabled: true",
                "    default_root: workspace",
                "    max_depth: 3",
                "    max_entries: 100",
                "    allowed_roots:",
                "      - name: workspace",
                "        provider: local",
                f"        path: {root_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_list_file_server_entries_returns_recursive_tree(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "docs"
    nested.mkdir(parents=True)
    (workspace / "readme.txt").write_text("hello", encoding="utf-8")
    (workspace / ".hidden.txt").write_text("secret", encoding="utf-8")
    (nested / "guide.md").write_text("guide", encoding="utf-8")

    cfg_path = tmp_path / "ai-chat-util-config.yml"
    _write_local_file_server_config(cfg_path, root_path=workspace)
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    monkeypatch.setenv("AI_CHAT_UTIL_CONFIG", str(cfg_path))

    with TestClient(app) as client:
        response = client.get(
            "/api/file_util/list_file_server_entries",
            params={"recursive": True, "max_depth": 1},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "local"
    assert payload["root_name"] == "workspace"
    assert payload["path"] == "."
    assert payload["total_entries"] == 3

    docs_entry = payload["entries"][0]
    file_entry = payload["entries"][1]
    assert docs_entry["name"] == "docs"
    assert docs_entry["entry_type"] == "directory"
    assert docs_entry["children"][0]["name"] == "guide.md"
    assert file_entry["name"] == "readme.txt"
    assert file_entry["entry_type"] == "file"


def test_list_file_server_roots_returns_configured_roots(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cfg_path = tmp_path / "ai-chat-util-config.yml"
    _write_local_file_server_config(cfg_path, root_path=workspace)
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    monkeypatch.setenv("AI_CHAT_UTIL_CONFIG", str(cfg_path))

    with TestClient(app) as client:
        response = client.get("/api/file_util/list_file_server_roots")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["default_root"] == "workspace"
    assert payload["roots"] == [
        {
            "name": "workspace",
            "provider": "local",
            "path": str(workspace),
            "description": None,
            "is_default": True,
        }
    ]


def test_list_file_server_entries_rejects_path_traversal(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cfg_path = tmp_path / "ai-chat-util-config.yml"
    _write_local_file_server_config(cfg_path, root_path=workspace)
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    monkeypatch.setenv("AI_CHAT_UTIL_CONFIG", str(cfg_path))

    with TestClient(app) as client:
        response = client.get(
            "/api/file_util/list_file_server_entries",
            params={"path": "../outside"},
        )

    assert response.status_code == 400
    assert ".." in response.json()["detail"]


def test_init_runtime_resolves_file_server_smb_env_refs(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "ai-chat-util-config.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "ai_chat_util_config:",
                "  llm:",
                "    provider: openai",
                "    api_key: os.environ/LLM_API_KEY",
                "    completion_model: gpt-5",
                "  file_server:",
                "    enabled: true",
                "    allowed_roots:",
                "      - name: remote",
                "        provider: smb",
                "        path: documents",
                "    smb:",
                "      enabled: true",
                "      server: smb.example.com",
                "      share: shared",
                "      username: os.environ/FILE_SERVER_SMB_USERNAME",
                "      password: os.environ/FILE_SERVER_SMB_PASSWORD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FILE_SERVER_SMB_USERNAME", "user01")
    monkeypatch.setenv("FILE_SERVER_SMB_PASSWORD", "pass01")
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")

    init_runtime(str(cfg_path))
    config = get_runtime_config()

    assert config.file_server.smb.username == "user01"
    assert config.file_server.smb.password == "pass01"


def test_list_file_server_roots_keeps_empty_smb_root_path(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cfg_path = tmp_path / "ai-chat-util-config.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "ai_chat_util_config:",
                "  llm:",
                "    provider: openai",
                "    api_key: os.environ/LLM_API_KEY",
                "    completion_model: gpt-5",
                "  file_server:",
                "    enabled: true",
                "    default_root: workspace",
                "    allowed_roots:",
                "      - name: workspace",
                "        provider: local",
                f"        path: {workspace}",
                "      - name: smb-workspace",
                "        provider: smb",
                "        path: ''",
                "    smb:",
                "      enabled: true",
                "      server: 192.168.35.89",
                "      share: workspace",
                "      username: os.environ/FILE_SERVER_SMB_USERNAME",
                "      password: os.environ/FILE_SERVER_SMB_PASSWORD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    monkeypatch.setenv("FILE_SERVER_SMB_USERNAME", "user1")
    monkeypatch.setenv("FILE_SERVER_SMB_PASSWORD", "pass01")
    monkeypatch.setenv("AI_CHAT_UTIL_CONFIG", str(cfg_path))

    with TestClient(app) as client:
        response = client.get("/api/file_util/list_file_server_roots")

    assert response.status_code == 200
    payload = response.json()
    assert payload["roots"] == [
        {
            "name": "workspace",
            "provider": "local",
            "path": str(workspace),
            "description": None,
            "is_default": True,
        },
        {
            "name": "smb-workspace",
            "provider": "smb",
            "path": "",
            "description": None,
            "is_default": False,
        },
    ]