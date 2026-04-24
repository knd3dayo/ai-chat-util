from ai_chat_util.ai_chat_util_base.chat import create_llm_client

async def main():

    client = create_llm_client()
    response = await client.simple_chat("こんにちは、今日の天気は？")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())