import typer
import asyncio
from pathlib import Path
from typing import Optional
import time

from ai_chat_util_base.config.runtime import init_coding_runtime

# 内部パッケージのインポート
from ..core.task_manager import TaskManager
from ..core.docker.docker_task_service import DockerTaskService
from .typer_actions import TyperActions

# --- CLI Layer: コマンドの定義 ---
actions = TyperActions()
docker_task_service = DockerTaskService()

app = typer.Typer(help="Coding Agent Executor CLI Tool")


@app.callback()
def _global_options(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help=(
            "Path to config YAML (ai-chat-util-config.yml). If omitted, resolved by env AI_CHAT_UTIL_CONFIG "
            "(with root-level coding_agent_util section), or searched from CWD/project root."
        ),
        envvar="AI_CHAT_UTIL_CONFIG",
    ),
):
    init_coding_runtime(str(config) if config else None)

@app.command()
def run(
    prompt: str = typer.Argument(..., help="指示内容"),
    sources: Optional[list[Path]] = typer.Option(
        None,
        "--src",
        "-s",
        help="コンテナへ渡すソース（複数指定可）",
    ),
    task_id: Optional[str] = typer.Option(None, "--id"),
    timeout: int = 300,
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        "--sync-mode/--async-mode",
        help=(
            "同期/非同期の実行モード。"
            " --wait/--sync-mode は完了まで待機し、--no-wait/--async-mode は task_id を返して終了します。"
        ),
    ),
):
    """新しいタスクを実行します。"""
    if sources:
        for src in sources:
            if not src.exists():
                raise typer.BadParameter(f"存在しないパスです: {src}", param_hint="--src/-s")
            if not (src.is_file() or src.is_dir()):
                raise typer.BadParameter(f"ファイル/ディレクトリではありません: {src}", param_hint="--src/-s")

    async def main():
        await TaskManager.run_task(
            task_service=docker_task_service,
            actions=actions,
            prompt=prompt,
            sources=sources,
            task_id=task_id,
            timeout=timeout,
            wait=wait,
        )
        
    asyncio.run(main())


@app.command(hidden=True)
def monitor(
    task_id: str,
    interval: float = typer.Option(2.0, help="ポーリング間隔（秒）"),
    max_seconds: int = typer.Option(3600, help="最大監視時間（秒）"),
    tail: int = typer.Option(200, help="ログ取得行数"),
    quiet: bool = typer.Option(True, help="出力せず DB 更新のみ行う"),
):
    """デタッチ実行用の内部モニタ（status の自動更新）"""

    async def main():
        start = time.monotonic()
        while True:
            status_data = await TaskManager.show_status(task_id, tail=tail)
            if not quiet:
                actions.after_get_status_action(task_id, status_data)

            if status_data.status not in ("running", "pending"):
                return

            if time.monotonic() - start > max_seconds:
                return

            await asyncio.sleep(interval)

    asyncio.run(main())

@app.command(name="list")
def list_tasks():
    """一覧表示"""
    actions.after_list_action(TaskManager.list_tasks())

@app.command()
def status(task_id: str, tail: int = 20):
    """状態確認"""
    async def main():
        status_data = await TaskManager.show_status(task_id, tail)
        actions.after_get_status_action(task_id, status_data)

    asyncio.run(main())

@app.command()
def cancel(task_id: str):
    """強制終了"""
    async def main():
        await TaskManager.cancel_task(task_id)
        actions.after_cancel_action(task_id)

    asyncio.run(main())

@app.command()
def prune(compose_service_name: str):
    """掃除実行"""
    actions.prune_progress_action(docker_task_service.prune_containers(compose_service_name))

if __name__ == "__main__":
    app()