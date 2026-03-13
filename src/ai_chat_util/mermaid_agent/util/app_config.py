import os
from pydantic import BaseModel, Field
from ai_chat_util.config.runtime import get_runtime_config

class AppConfig(BaseModel):
    # Non-secret settings are loaded from config.yml via runtime config.
    # Secret settings (API keys) remain in env/.env.
    model_id: str = Field(default="")
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    azure_openai: bool = Field(default=False)
    base_url: str = Field(default="")
    azure_api_version: str = Field(default="2024-12-01")
    azure_openai_endpoint: str = Field(default="")
    mcp_server_config_file_path: str = Field(default="")
    custom_instructions_file_path: str = Field(default="")

    def __init__(self, **data):
        if not data:
            cfg = get_runtime_config()
            data = {
                "model_id": cfg.llm.completion_model,
                "azure_openai": cfg.llm.provider == "azure_openai",
                "base_url": cfg.llm.base_url or "",
                "mcp_server_config_file_path": cfg.paths.mcp_server_config_file_path or "",
                "custom_instructions_file_path": cfg.paths.custom_instructions_file_path or "",
            }
        super().__init__(**data)
