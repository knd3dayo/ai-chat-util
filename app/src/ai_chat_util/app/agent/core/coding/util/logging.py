from __future__ import annotations

import logging


def get_application_logger(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(name)
    return logging.getLogger("ai_chat_util.app.agent.core.coding")