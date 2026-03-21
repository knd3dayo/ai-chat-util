from ai_chat_util.base.llm.llm_client_factory import LLMFactory
from ai_chat_util.base.llm.llm_client_util import LLMClientUtil

async def main(files):
    client = LLMFactory.create_llm_client()

    result = await LLMClientUtil.analyze_image_files(
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

