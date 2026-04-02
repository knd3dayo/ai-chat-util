from ai_chat_util.base.chat import create_llm_client
from ai_chat_util.analysis import AnalysisService

async def main(files):
    client = create_llm_client()

    result = await AnalysisService.analyze_office_files(
        client,
        file_path_list=files,
        prompt="このExcelファイルの要約を作成してください。"
    )
    print(result.output)

if __name__ == "__main__":
    import sys
    files = sys.argv[1:]
    import asyncio
    asyncio.run(main(files))