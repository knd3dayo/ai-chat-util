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
        self.completion_model: str =  os.getenv("COMPLETION_MODEL","gpt-5")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL","text-embedding-3-small")

    def get_model_path(self) -> str:
        return f"{self.llm_provider}/{self.completion_model}"
