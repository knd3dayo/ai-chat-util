import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLMRouter
from litellm.router import Router

from ..config.mcp_config import MCPConfigParser
from ..config.runtime import AiChatUtilConfig, get_runtime_config, get_runtime_config_path
from ..util.file_path_resolver import resolve_existing_file_path

class MCPClient:
    def __init__(self, runtime_config: AiChatUtilConfig):
        self.runtime_config = runtime_config
        mcp_config_path = (
            self.runtime_config.paths.mcp_config_path
            or self.runtime_config.paths.mcp_server_config_file_path
        )
        if not mcp_config_path:
            raise ValueError(
                "MCP 設定ファイルパスが未設定です。config.yml の paths.mcp_config_path（または互換の paths.mcp_server_config_file_path）を設定してください。"
            )

        # config.yml からの相対パスも解決できるよう、設定ファイルのディレクトリも探索対象に入れる
        config_dir = str(get_runtime_config_path().parent)
        resolved = resolve_existing_file_path(
            mcp_config_path,
            working_directory=self.runtime_config.paths.working_directory,
            extra_search_dirs=[config_dir],
        ).resolved_path

        self.config_parser = MCPConfigParser(resolved)
        # 2. LangChain用設定の生成
        self.mcp_config = self.config_parser.to_langchain_config()

    async def main(self):
        # 3. MultiServerMCPClientの初期化
        # この内部で各サーバーへの接続（stdioの起動やSSEのセッション開始）が行われます
        client = MultiServerMCPClient(self.mcp_config)

        # LangChainのツールリストを取得
        langchain_tools = await client.get_tools()
        
        # (オプション) allowedToolsによるフィルタリングが必要な場合
        allowed_map = self.config_parser.get_allowed_tools_map()
        # ここで langchain_tools をフィルタリングするロジックを挟めます
        
        # あとはこれを LangChain の Agent や LLM (bind_tools) に渡すだけ！
        # example: 
        # llm_with_tools = ChatOpenAI().bind_tools(langchain_tools)
        
        print(f"Loaded {len(langchain_tools)} tools from MCP servers.")



        # LLM + MCP ツールでエージェントを作成
        litellm_router = Router(model_list=self.runtime_config.llm.create_litellm_model_list())
        llm = ChatLiteLLMRouter(router=litellm_router, model_name=self.runtime_config.llm.completion_model)
        agent = create_agent(llm, langchain_tools)

        # 実行
        result = await agent.ainvoke({"input": "3 と 5 を足して"})  # type: ignore
        print(result)

if __name__ == "__main__":
    runtime_config = get_runtime_config()  # ここは適宜、実際の設定に合わせて初期化してください
    asyncio.run(MCPClient(runtime_config).main())