from typing import Callable, Generator
import typer
import asyncio
from pathlib import Path
from tabulate import tabulate  # pip install tabulate (一覧表示用)
import signal

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from python_on_whales import docker as whales


# 内部パッケージのインポート
from ..core.abstract_actions import AbstractActions
from typing import Any
from ai_chat_util.app.agent.agent_util_models import TaskStatus

from ..core.task_manager import TaskManager

class TyperActions(AbstractActions) :

    console = Console()

    def after_start_task_action(self, tid: str) -> None:
        """タスク開始後のアクション"""
        self.console.print(f"[bold green]🚀 タスクを開始しました: {tid}[/bold green]")
        
    def after_start_detach_task_action(self, tid: str) -> None:
        self.console.print(f"🔗 [yellow]バックグラウンドで実行中です。'status {tid}' で確認してください。[/yellow]")

    # 進捗表示処理
    async def progress_action(self, tid: str) -> TaskStatus:
        # ここでは例として、タスクの状態を定期的にチェックして表示するロジックを記述します。
        # 実際には ComposeRunner.get_status() を呼び出して、タスクの状態やログを取得することになります。
        last_line_count = 0
        last_stderr_count = 0
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def handle_interrupt():
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, handle_interrupt)
            except NotImplementedError:
                # Windows等では未対応のことがある
                pass

        with Live(Spinner("dots", text="実行中...", style="cyan"), refresh_per_second=4, transient=True) as live:
            try:
                while not stop_event.is_set():
                    status_data = await TaskManager.get_status(tid, tail=1000)
                    
                    # 1. 標準出力の表示
                    if status_data.stdout:
                        lines = status_data.stdout.splitlines()
                        if len(lines) > last_line_count:
                            for line in lines[last_line_count:]:
                                live.console.print(f"  [white]{line}[/white]")
                            last_line_count = len(lines)

                    # 1. エラー(STDERR)の色分け表示
                    if status_data.stderr:
                        err_lines = status_data.stderr.splitlines()
                        if len(err_lines) > last_stderr_count:
                            for line in err_lines[last_stderr_count:]:
                                live.console.print(f"  [bold red]ERR> {line}[/bold red]")
                            last_stderr_count = len(err_lines)

                    # 終了判定
                    if status_data.status not in ["running", "pending"]:
                        break
                    
                    await asyncio.sleep(1.5)

            except Exception as e:
                self.console.print(f"[bold red]❌ エラーが発生しました: {e}[/bold red]")
            finally:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    try:
                        loop.remove_signal_handler(sig)
                    except Exception:
                        pass

        # ループを抜けた後の処理
        status_data = await TaskManager.get_status(tid)

        self.console.print(f"\n🏁 終了ステータス: [bold cyan]{status_data.status}[/bold cyan]")

        # 中断時の処理
        if stop_event.is_set():
            if typer.confirm("⚠️ 中断しました。コンテナを停止して削除しますか？", default=True):
                await TaskManager.cancel_task(status_data.task_id)
                self.console.print(f"🛑 タスク {status_data.task_id} をキャンセルしました。")
                status_data.cancelled()  # キャンセルフラグを立てる（必要に応じて）
        return status_data

    def after_complete_action(
            self, runner: Any) -> None:
        self.console.print(f"[bold green]✅ タスクが完了しました！[/bold green]")

    def after_task_not_found_action(self) -> None:
        typer.echo("タスクは見つかりませんでした。")

    def after_list_action(self, table: list) -> None:
        typer.echo(tabulate(table, headers=["Task ID", "Status", "Created At"]))

    def after_cancel_action(self, task_id: str, result: dict[str, Any] | None = None) -> None:
        if result and result.get("message"):
            self.console.print(
                f"[bold yellow]🛑 タスク {task_id}: {result.get('message')} (status={result.get('status')}, sub={result.get('sub_status')})[/bold yellow]"
            )
            return
        self.console.print(f"[bold yellow]🛑 タスク {task_id} をキャンセルしました。[/bold yellow]")

    def after_get_status_action(self, task_id: str, status_data: TaskStatus) -> None:
        self.console.print(f"=== Task: {task_id} (status={status_data.status}, sub={status_data.sub_status}) ===")
        if status_data.stdout:
            self.console.print(f"\n[STDOUT]\n{status_data.stdout}")
        if status_data.stderr:
            self.console.print(f"\n[STDERR]\n{status_data.stderr}", style="red")
        if status_data.artifacts:
            self.console.print(f"\n[Artifacts]\n{', '.join(status_data.artifacts)}")

    def prune_progress_action(self, generator: Generator[str, None, None]):
        """管理対象の孤立したコンテナを強制掃除します (APIサーバーの終了処理と同等)"""
        
        typer.echo("🧹 掃除を開始します...")
        try:
            for msg in generator:
                typer.echo(f"✅ {msg}")
                # コンテナを削除するロジックは generator 内で実行される想定
        except Exception as e:
            typer.secho(f"❌ 掃除中にエラー: {e}", fg=typer.colors.RED)

