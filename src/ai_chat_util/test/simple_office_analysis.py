from ai_chat_util.llm.llm_factory import LLMFactory

async def main(files):
    client = LLMFactory.create_llm_client()

    result = await client.analyze_office_files(
        file_path_list=files,
        prompt="このExcelファイルの要約を作成してください。"
    )
    print(result.output)

if __name__ == "__main__":
    import sys
    files = sys.argv[1:]
    import asyncio
    asyncio.run(main(files))