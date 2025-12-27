from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig

async def main():
    llm_config = LLMConfig()
    client = LLMClient.create_llm_client(llm_config)
    response = await client.simple_chat("こんにちは、今日の天気は？")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())