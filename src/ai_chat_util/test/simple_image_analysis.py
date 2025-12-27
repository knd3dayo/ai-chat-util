from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig

async def main(files):
    llm_config = LLMConfig()
    client = LLMClient.create_llm_client(llm_config)

    result = await client.analyze_image_files(
        file_list=files,
        prompt="この画像の内容を説明してください。",
        detail="auto"
    )
    print(result.output)

if __name__ == "__main__":
    import sys
    files = sys.argv[1:]
    import asyncio
    asyncio.run(main(files))

