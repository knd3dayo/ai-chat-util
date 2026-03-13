from typing import Optional

from ai_chat_util.config.runtime import get_runtime_config

class LLMConfig:

    def __init__(self):
        cfg = get_runtime_config()

        self.mcp_server_config_file_path = cfg.paths.mcp_server_config_file_path
        self.custom_instructions_file_path = cfg.paths.custom_instructions_file_path
        self.working_directory = cfg.paths.working_directory

        self.allow_outside_modifications = cfg.features.allow_outside_modifications
        self.use_custom_pdf_analyzer = cfg.features.use_custom_pdf_analyzer

        self.llm_provider = cfg.llm.provider
        self.completion_model = cfg.llm.completion_model
        self.embedding_model = cfg.llm.embedding_model

    def get_model_path(self) -> str:
        return f"{self.llm_provider}/{self.completion_model}"
