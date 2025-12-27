import logging
import os
default_log_level = logging.INFO
log_format = "%(asctime)s - %(levelname)s - %(pathname)s -  %(lineno)d - %(funcName)s - %(message)s"

def getLogger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    :param name: Name of the logger.
    :return: Logger object.
    """
    logger = logging.getLogger(name)
    # LOGLEVEL環境変数の値を適切に変換
    log_level_env = os.environ.get("LOGLEVEL", None)
    if log_level_env is not None:
        if isinstance(log_level_env, str):
            log_level = logging.getLevelName(log_level_env.upper())
            if not isinstance(log_level, int):
                log_level = default_log_level
        else:
            log_level = log_level_env
    else:
        log_level = default_log_level

    logger.setLevel(log_level)
    formatter = logging.Formatter(log_format)

    # ハンドラの重複追加を防ぐ
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

        log_file = os.environ.get("LOGFILE", None)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    return logger

