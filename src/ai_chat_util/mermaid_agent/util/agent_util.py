import os, json, argparse
from typing import Any, Union, ClassVar, Optional, Any, List

from pydantic import BaseModel, Field

from agent_framework import MCPStdioTool, HostedMCPTool
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.openai import OpenAIChatClient
from agent_framework import ChatAgent

from ai_chat_util.mermaid_agent.util.app_config import AppConfig

import ai_chat_util.mermaid_agent.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class MSAIAgentMcpSetting(BaseModel):
    # mcp_settings.jsonのルートとなる要素名
    servers_label: ClassVar[str] = "mcpServers"

    # typeはstdio or sse. defaultはstdio
    type: str = Field(default="stdio", description="Type of the MCP tool (e.g., 'stdio', 'sse').")
    name: str = Field(description="Name of the MCP tool.")
    autoApprove: Optional[list[str]] = Field(default=[], description="List of tool names allowed to auto-approve the tool execution.")
    disabled: Optional[bool] = Field(default=False, description="Whether the tool is disabled.")
    description: Optional[str] = Field(default=None, description="Description of the MCP tool.")
    timeout: Optional[int] = Field(default=60, description="Timeout in seconds for the MCP tool execution.")

    # for stdio
    command: Optional[str] = Field(default=None, description="Command to execute for the MCP tool.")
    args: Optional[List[str]] = Field(default=None, description="Arguments for the MCP tool command.")
    env: Optional[dict] = Field(default=None, description="Environment variables for the MCP tool.")
    # for sse
    url: Optional[str] = Field(default=None, description="URL for the MCP tool if type is 'sse'.")


    @staticmethod
    def create_from_file(file_path: str) -> dict[str, 'MSAIAgentMcpSetting']:
        try:
            settings_dict: dict[str, MSAIAgentMcpSetting] = {}
            with open(file_path, 'r') as file:
                data = json.load(file)
                tools_data = data.get(MSAIAgentMcpSetting.servers_label, {})
                for tool_name, tool in tools_data.items():
                    tool["name"] = tool_name
                    validated_tool = MSAIAgentMcpSetting.model_validate(tool)
                    settings_dict[validated_tool.name] = validated_tool
            return settings_dict
        except Exception as e:
            logger.error(f"Error loading MCP settings from {file_path}: {e}")
            return {}

    @staticmethod
    def create_mcp_tools_from_settings(mcp_settings_json_path: str) -> list[Any]:
        if not mcp_settings_json_path or os.path.isfile(mcp_settings_json_path) is False:
            logger.info("MCP settings JSON path is not provided or invalid.")
            return []
        mcp_settings = MSAIAgentMcpSetting.create_from_file(mcp_settings_json_path)
        tools = []
        for name, setting in mcp_settings.items():
            if setting.disabled:
                logger.info(f"MCP tool '{name}' is disabled. Skipping.")
                continue
            if setting.type == "stdio":
                if not setting.command:
                    logger.error(f"MCP tool '{name}' of type 'stdio' requires a command. Skipping.")
                    continue
                tool = MCPStdioTool(
                    name=setting.name,
                    command=setting.command,
                    args=setting.args,
                    env=setting.env,
                    auto_approve=setting.autoApprove,
                    description=setting.description,
                    timeout=setting.timeout
                )
                tools.append(tool)
                logger.debug(f"Created MCPStdioTool: {setting.name}")
                
            elif setting.type == "sse":
                if not setting.url:
                    logger.error(f"MCP tool '{setting.name}' of type 'sse' requires a URL. Skipping.")
                    continue
                tool = HostedMCPTool(
                    name=setting.name,
                    url=setting.url,
                    auto_approve=setting.autoApprove,
                    description=setting.description,
                    timeout=setting.timeout
                )
                tools.append(tool)
                logger.debug(f"Created MCPSseTool: {setting.name}")
            else:
                logger.error(f"Unknown MCP tool type '{setting.type}' for tool '{setting.name}'. Skipping.")

        return tools

class MSAIAgentParams(BaseModel):
    name: str = Field(default="Helpful Assistant", description="Name of the agent.")
    instructions: str = Field(default="", description="Instructions for the agent.")
    tools: list[Any] = Field(default_factory=list, description="List of tools available to the agent.")

class MSAIAgentUtil(BaseModel):

    app_config: AppConfig = Field(default_factory=lambda: AppConfig())

    def __create_openai_dict(self) -> dict:
        completion_dict = {}
        completion_dict["api_key"] = self.app_config.api_key
        completion_dict["model_id"] = self.app_config.model_id
        if self.app_config.base_url:
            completion_dict["base_url"] = self.app_config.base_url
        return completion_dict
    
    def __create_azure_openai_dict(self) -> dict:
        completion_dict = {}
        completion_dict["api_key"] = self.app_config.api_key
        if self.app_config.base_url:
            completion_dict["base_url"] = self.app_config.base_url
        else:
            completion_dict["endpoint"] = self.app_config.azure_openai_endpoint
            completion_dict["deployment_name"] = self.app_config.model_id
            completion_dict["api_version"] = self.app_config.azure_api_version
        return completion_dict

    def __create_tools(self) -> list[Any]:
        # MCP Stdio Toolのデフォルト設定を作成
        tools = [self.__create_default_mcp_server()]
        if self.app_config.mcp_server_config_file_path:
            mcp_tools = MSAIAgentMcpSetting.create_mcp_tools_from_settings(self.app_config.mcp_server_config_file_path)
            tools.extend(mcp_tools)
        return tools
    
    def create_default_agent_params(
        self, name: str = "Helpful Assistant", instructions: str = "", tools: list[Any] = []) -> MSAIAgentParams:
        params = MSAIAgentParams()
        params.name = name
        if instructions:
            params.instructions = instructions
        else:
            params.instructions = self.__create_default_instructions(self.app_config.custom_instructions_file_path) if self.app_config.custom_instructions_file_path else ""

        if tools:
            params.tools = tools
        else:
            params.tools = self.__create_tools()
        
        return params
 
    def create_agent(self, params: Optional[MSAIAgentParams] = None) -> ChatAgent:
        if params is None:
            params = self.create_default_agent_params()

        params_dict = params.model_dump()

        if (self.app_config.azure_openai):
            client = AzureOpenAIChatClient(
                **self.__create_azure_openai_dict()
            )

            return client.create_agent(**params_dict)
        else:
            client = OpenAIChatClient(
                **self.__create_openai_dict()
            )
            return client.create_agent(**params_dict)

    def __create_default_mcp_server(self) -> MCPStdioTool:
        tool = MCPStdioTool(
            name="mcp_server",
            command="uv",
            args=["run", "-m", "mermaid_workflow.tool.mcp_server"],
            env=os.environ.copy(),
            description="MCP server tools for file operations",
        )
        return tool

        
    def __create_default_instructions(self, custom_instructions_path: Optional[str] = None) -> str:
        logger.debug(f"Creating instructions with custom_instructions_path: {custom_instructions_path}")
        instructions = "You are a helpful assistant."
        custom_instructions = ""
        if custom_instructions_path and os.path.isfile(custom_instructions_path):
            with open(custom_instructions_path, 'r', encoding='utf-8') as file:
                custom_instructions = file.read()
        if custom_instructions.strip():
            # カスタムインストラクションの内容に従うことを明示的に指示
            instructions = f"""
                ユーザーからの指示を実行する前に必ず以下のカスタムインストラクションを確認し、遵守してください。
                カスタムインストラクションの内容とユーザーの指示を踏まえて、最適な計画を考えてください。
                回答する際には、回答がどのような計画に従って行われたかを説明してください。
                カスタムインストラクションの内容は以下の通りです。
                -----
                {custom_instructions}
                -----
                上記のカスタムインストラクションに従ってください。
                """
            logger.info(f"Loaded custom instructions from {custom_instructions_path}")

        return instructions

async def async_main():
    # 引数解析 -f mcp_settings_json_path
    parser = argparse.ArgumentParser(description="MS AI Agent Sample")
    parser.add_argument("-f", "--mcp_settings_json_path", type=str, help="Path to the MCP settings JSON file")
    # -d 作業ディレクトリ defaultはカレントディレクトリ
    parser.add_argument("-d", "--working_directory", type=str, default=".", help="Path to the working directory")
    # 作業フォルダ以外のファイル更新を許可するかどうかのフラグ
    parser.add_argument("--allow_outside_modifications", action="store_true", help="Allow modifications to files outside the working directory")
    # カスタムインストラクションのパス
    parser.add_argument("-c", "--custom_instructions_path", type=str, help="Path to the custom instructions file")

    args = parser.parse_args()
    mcp_settings_json_path = args.mcp_settings_json_path
    working_directory = args.working_directory
    allow_outside_modifications = args.allow_outside_modifications
    custom_instructions_path = args.custom_instructions_path
    
    if working_directory and os.path.isdir(working_directory):
        os.chdir(working_directory)
        logger.info(f"Changed working directory to: {working_directory}")
    else:
        logger.warning(f"Working directory '{working_directory}' is invalid. Using current directory.")

    if allow_outside_modifications:
        os.environ["ALLOW_OUTSIDE_MODIFICATIONS"] = "true"
        logger.info("Modifications to files outside the working directory are allowed.")
    else:
        os.environ["ALLOW_OUTSIDE_MODIFICATIONS"] = "false"
        logger.info("Modifications to files outside the working directory are NOT allowed.")

    # Create an agent using OpenAI ChatCompletion
    llm_config = AppConfig()
    agent_util = MSAIAgentUtil(app_config=llm_config)

    async with (agent_util.create_agent() as agent):
        # Create a thread for persistent conversation
        thread = agent.get_new_thread()
        while True:
            input_text = input("Enter your request: ")
            result = await agent.run(input_text, thread=thread)
            print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(async_main())
