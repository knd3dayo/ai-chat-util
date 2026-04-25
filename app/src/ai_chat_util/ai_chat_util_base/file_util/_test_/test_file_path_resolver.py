from __future__ import annotations

from ai_chat_util.ai_chat_util_base.file_util.util.file_path_resolver import resolve_existing_path


def test_resolve_existing_path_accepts_directory_and_prefers_working_directory(tmp_path, monkeypatch) -> None:
    cwd_root = tmp_path / "cwd"
    workspace_root = tmp_path / "workspace"
    cwd_docs = cwd_root / "docs"
    workspace_docs = workspace_root / "docs"

    cwd_docs.mkdir(parents=True)
    workspace_docs.mkdir(parents=True)
    monkeypatch.chdir(cwd_root)

    result = resolve_existing_path(
        "docs",
        working_directory=str(workspace_root),
        allow_directory=True,
    )

    assert result.path_kind == "directory"
    assert result.resolved_path == str(workspace_docs.resolve())