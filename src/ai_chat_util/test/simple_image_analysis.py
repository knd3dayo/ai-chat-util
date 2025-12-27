from ai_chat_util.llm.llm_factory import LLMFactory

async def main(files):
    client = LLMFactory.create_llm_client()

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

