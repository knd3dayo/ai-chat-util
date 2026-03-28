from coding_agent_util._cli_.typer_actions import TyperActions

__all__ = ["TyperActions"]

    def prune_progress_action(self, generator: Generator[str, None, None]):
        """管理対象の孤立したコンテナを強制掃除します (APIサーバーの終了処理と同等)"""
        
        typer.echo("🧹 掃除を開始します...")
        try:
            for msg in generator:
                typer.echo(f"✅ {msg}")
                # コンテナを削除するロジックは generator 内で実行される想定
        except Exception as e:
            typer.secho(f"❌ 掃除中にエラー: {e}", fg=typer.colors.RED)

