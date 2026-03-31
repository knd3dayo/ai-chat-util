from __future__ import annotations

import re
from typing import Any, Sequence
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from ai_chat_util.base.llm.prompts import PromptsBase
from .supervisor_support import AuditContext
from ai_chat_util.common.config.ai_chat_util_mcp_config import MCPServerConfig
from ai_chat_util.common.config.runtime import AiChatUtilConfig
from .tool_limits import ToolLimits
import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

# ... AgentBuilder クラス本体は agent.py から移植 ...
# ここにAgentBuilderの全実装を貼り付け
