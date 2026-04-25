import logging


def get_application_logger() -> logging.Logger:
    """アプリケーション全体で使用するロガーを取得する関数。

    Note:
    - ログの秘匿マスク/出力先は `config.runtime._configure_python_logging()` が root logger に設定する。
    - ここで独自 handler を追加すると秘匿マスクを迂回したり、ハンドラ重複でログが多重出力される。
    """

    return logging.getLogger("ai_chat_util.ai_chat_util_agent.coding")
