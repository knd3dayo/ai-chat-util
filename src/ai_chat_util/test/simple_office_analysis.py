from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig

async def main(files):
    llm_config = LLMConfig()
    client = LLMClient.create_llm_client(llm_config)

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