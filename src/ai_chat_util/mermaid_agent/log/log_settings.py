import logging
default_log_level = logging.WARN
log_format = "%(asctime)s - %(levelname)s - %(filename)s -  %(lineno)d - %(funcName)s - %(message)s"

def getLogger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    :param name: Name of the logger.
    :return: Logger object.
    """
    # Logging configuration is performed in ai_chat_util.config.runtime.init_runtime().
    # This function intentionally avoids reading non-secret environment variables.
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=default_log_level, format=log_format)
    return logging.getLogger(name)

