from typing import Optional
import os
from dotenv import load_dotenv

class LLMConfig:

    def __init__(self):
        load_dotenv()

        # mcp_server_config_file_path
        self.mcp_server_config_file_path: Optional[str] = os.getenv("MCP_SERVER_CONFIG_FILE_PATH", None)

        # custom_instructions_file_path
        self.custom_instructions_file_path: Optional[str] = os.getenv("CUSTOM_INSTRUCTIONS_FILE_PATH", None)

        # working_directory
        self.working_directory: Optional[str] = os.getenv("WORKING_DIRECTORY", None)

        # allow_outside_modifications
        self.allow_outside_modifications: bool = os.getenv("ALLOW_OUTSIDE_MODIFICATIONS","false").lower() == "true"

        # use_custom_pdf_analyzer
        self.use_custom_pdf_analyzer: bool = os.getenv("USE_CUSTOM_PDF_ANALYZER","false").lower() == "true"

        self.llm_provider: str = os.getenv("LLM_PROVIDER","openai")
        self.api_key: str = ""
        self.completion_model: str = ""
        self.embedding_model: str = ""
        self.api_version: Optional[str] = None
        self.endpoint: Optional[str] = None

        self.base_url: Optional[str] = None
        if self.llm_provider == "openai" or self.llm_provider == "azure_openai":
            self.api_key = os.getenv("OPENAI_API_KEY","")
            self.base_url = os.getenv("OPENAI_BASE_URL","") or None
            self.completion_model: str = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o")
            self.embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        if self.llm_provider == "azure_openai":
            self.api_version: Optional[str] = os.getenv("AZURE_OPENAI_API_VERSION","")
            self.endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT","")


