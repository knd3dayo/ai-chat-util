from ai_chat_util.llm.llm_factory import LLMFactory

async def main():

    client = LLMFactory.create_llm_client()
    response = await client.simple_chat("こんにちは、今日の天気は？")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())