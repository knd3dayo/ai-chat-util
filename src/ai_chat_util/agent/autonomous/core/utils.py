import os
import io
import pathlib
import io
import zipfile
from typing import Union
from fastapi import UploadFile
import zipfile
import tempfile
from pathlib import Path
import atexit
import shutil
import json

class ExecutorUtil:
    """タスクの実行と管理に関するユーティリティ関数をまとめたクラスです。"""
    @staticmethod
    def create_temporary_zip(source_dir: Path) -> Path:
        """ディレクトリを一時的なZIPファイルに固める"""
        tmp_zip = Path(tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name)
        atexit.register(lambda: tmp_zip.unlink(missing_ok=True))  # 終了時に自動削除
        with zipfile.ZipFile(tmp_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in source_dir.rglob('*'):
                if file.is_file():
                    # source_dir からの相対パスで格納
                    zf.write(file, file.relative_to(source_dir))
        return tmp_zip

    @staticmethod
    def make_zip_from_dir(src_dir: pathlib.Path, zip_path: pathlib.Path) -> None:
        """ディレクトリ全体をzip化します（zip_path は上書き）。"""
        if not src_dir.exists() or not src_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {src_dir}")

        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in src_dir.rglob("*"):
                if not p.is_file():
                    continue
                # zip 内のパスは src_dir からの相対
                zf.write(p, arcname=str(p.relative_to(src_dir)))

    @staticmethod
    def cleanup_file(path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    @staticmethod
    def get_container_logs(container, tail: int = 200) -> tuple[str, str]:
        """docker コンテナの stdout/stderr を取得して (stdout, stderr) を返します。"""
        # docker SDK の tail は str/int を受け付ける
        out = container.logs(stdout=True, stderr=False, tail=tail)
        err = container.logs(stdout=False, stderr=True, tail=tail)
        return out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")

    @staticmethod
    def extract_zip_to_dir(zip_file: Union[UploadFile, pathlib.Path], dest_dir: pathlib.Path) -> None:
        """UploadFile(API) または Path(CLI) から ZIP を展開します。"""
        
        # 1. バイナリデータの取得
        if isinstance(zip_file, pathlib.Path):
            # CLIの場合: Pathオブジェクトから直接読み込む
            with open(zip_file, "rb") as f:
                contents = f.read()
        else:
            # APIの場合: UploadFileから読み込む
            # ※同期的なread()を想定（非同期の場合は await が必要だが、通常 utils は同期的に書くことが多い）
            contents = zip_file.file.read()

        # 2. ZIPの展開
        with zipfile.ZipFile(io.BytesIO(contents)) as zip_ref:
            # Zip Slip 対策: 展開先が dest_dir の外に出ないかチェック
            for member in zip_ref.namelist():
                member_path = dest_dir / member
                if not str(member_path.resolve()).startswith(str(dest_dir.resolve())):
                    raise Exception(f"Unsafe zip member detected: {member}")
            
            zip_ref.extractall(dest_dir)

    @staticmethod
    def add_data(initial_files: dict[str, str] | None, workspace: pathlib.Path):
        """初期ファイルを task_dir に配置します。"""
        if not initial_files:
            return
        for name, content in initial_files.items():
            (workspace / name).write_text(content, encoding='utf-8')

    @staticmethod
    def add_zip_file( zip_file: UploadFile, workspace: pathlib.Path):
        """アップロードされた ZIP ファイルを task_dir に展開します。"""
        ExecutorUtil.extract_zip_to_dir(zip_file, workspace)

    @staticmethod
    def add_files(src_paths: list[pathlib.Path], workspace: pathlib.Path):
        for src_path in src_paths:
            """src_path のファイルを task_dir にコピーします。"""
            if src_path.is_file():
                shutil.copy(src_path, workspace / src_path.name)
            elif src_path.is_dir():
                for item in src_path.rglob('*'):
                    if item.is_file():
                        dest = workspace / item.relative_to(src_path)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(item, dest)

    @staticmethod
    def ensure_opencode_task_config_for_docker(workspace: pathlib.Path) -> pathlib.Path:
        """Create a per-task OpenCode config file inside workspace.

        This file contains MCP server definitions and uses `{env:VAR}` placeholders
        so that secrets (e.g., Authorization tokens) are not persisted to disk.

        The caller should set OPENCODE_CONFIG to `/workspace/.opencode/opencode.task.json`
        inside the container.
        """

        opencode_dir = workspace / ".opencode"
        opencode_dir.mkdir(parents=True, exist_ok=True)
        config_path = opencode_dir / "opencode.task.json"

        # Keep the config minimal and portable. It is merged on top of the global config.
        # NOTE: Use container paths for MCP projects because OpenCode runs inside the container.
        config: dict[str, object] = {
            "$schema": "https://opencode.ai/config.json",
            "mcp": {
                "ai-chat-util": {
                    "enabled": True,
                    "timeout": 60000,
                    "type": "local",
                    "command": [
                        "uv",
                        "--directory",
                        "/home/codeuser/app/mcp/ai-chat-util",
                        "run",
                        "-m",
                        "ai_chat_util.mcp.mcp_server",
                    ],
                    "environment": {
                        # LLM settings (best-effort)
                        "LLM_PROVIDER": "{env:LLM_PROVIDER}",
                        "COMPLETION_MODEL": "{env:LLM_MODEL}",
                        "OPENAI_API_KEY": "{env:LLM_API_KEY}",
                        "OPENAI_BASE_URL": "{env:LLM_BASE_URL}",
                        # Request-scoped context
                        "AI_PLATFORM_AUTHORIZATION": "{env:AI_PLATFORM_AUTHORIZATION}",
                        "AUTHORIZATION": "{env:AUTHORIZATION}",
                        "AI_PLATFORM_TRACE_ID": "{env:AI_PLATFORM_TRACE_ID}",
                        "TRACE_ID": "{env:TRACE_ID}",
                    },
                },
                "denodo-log-util": {
                    "enabled": True,
                    "timeout": 60000,
                    "type": "local",
                    "command": [
                        "uv",
                        "--directory",
                        "/home/codeuser/app/mcp/deonodo-log-util",
                        "run",
                        "-m",
                        "denodo_log_util.denodo_support_util_mcp",
                        "--mode",
                        "stdio",
                    ],
                    "environment": {
                        # Required by denodo_support_util_mcp.py
                        "APP_DATA_PATH": "{env:APP_DATA_PATH}",
                        # Request-scoped context
                        "AI_PLATFORM_AUTHORIZATION": "{env:AI_PLATFORM_AUTHORIZATION}",
                        "AUTHORIZATION": "{env:AUTHORIZATION}",
                        "AI_PLATFORM_TRACE_ID": "{env:AI_PLATFORM_TRACE_ID}",
                        "TRACE_ID": "{env:TRACE_ID}",
                    },
                },
            },
        }

        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config_path
