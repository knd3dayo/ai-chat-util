import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

class AppConfig(BaseModel):
    # default_factoryでオブジェクト生成時に環境変数から読み込む
    model_id: str = Field(default_factory=lambda: os.getenv("OPENAI_COMPLETION_MODEL",""))
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY",""))
    azure_openai: bool = Field(default_factory=lambda: os.getenv("AZURE_OPENAI") == "true")
    base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL",""))
    azure_api_version: str = Field(default_factory=lambda: os.getenv("AZURE_API_VERSION", "2024-12-01"))
    azure_openai_endpoint: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    mcp_server_config_file_path: str = Field(default_factory=lambda: os.getenv("MCP_SERVER_CONFIG_FILE_PATH",""))
    custom_instructions_file_path: str = Field(default_factory=lambda: os.getenv("CUSTOM_INSTRUCTIONS_FILE_PATH",""))

    def __init__(self, **data):
        load_dotenv()
        super().__init__(**data)
