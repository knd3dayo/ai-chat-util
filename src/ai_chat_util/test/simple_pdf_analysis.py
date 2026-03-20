from ai_chat_util.base.llm.llm_factory import LLMFactory
from ai_chat_util.base.llm.llm_client import LLMClientUtil


async def main(files):

    client = LLMFactory.create_llm_client()
    result = await LLMClientUtil.analyze_pdf_files(
        client,
        file_list=files,
        prompt="このPDFの要約を作成してください。"
    )
    print(result.output)

if __name__ == "__main__":
    import sys
    files = sys.argv[1:]
    import asyncio
    asyncio.run(main(files))