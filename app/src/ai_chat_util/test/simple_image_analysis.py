from ai_chat_util.base.chat import create_llm_client
from ai_chat_util.analysis import AnalysisService

async def main(files):
    client = create_llm_client()

    result = await AnalysisService.analyze_image_files(
        client,
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

