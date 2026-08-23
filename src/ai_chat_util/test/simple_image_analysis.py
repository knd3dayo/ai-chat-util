from ai_chat_util.core.chat import create_llm_client
from ai_chat_util.core.analysis.analyze_image import AnalyzeImageUtil

async def main(files):
    client = create_llm_client()

    result = await AnalyzeImageUtil.analyze_image_files(
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

