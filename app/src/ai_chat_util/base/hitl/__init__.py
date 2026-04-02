from ai_chat_util.common.config.runtime import AiChatUtilConfig, get_runtime_config

from ..chat import AbstractChatClient
from .hitl_client import HITLClientBase, IOManagerBase, StdIOHITLClient, StdIOManager


def create_stdio_hitl_client(
    llm_client: AbstractChatClient,
    runtime_config: AiChatUtilConfig | None = None,
    trace_id: str | None = None,
) -> HITLClientBase:
    if runtime_config is None:
        runtime_config = get_runtime_config()
    return StdIOHITLClient(llm_client, runtime_config, trace_id=trace_id)

__all__ = [
    "HITLClientBase",
    "IOManagerBase",
    "StdIOHITLClient",
    "StdIOManager",
    "create_stdio_hitl_client",
]