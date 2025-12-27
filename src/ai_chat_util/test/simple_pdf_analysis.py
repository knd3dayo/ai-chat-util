from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig

async def main(files):
    llm_config = LLMConfig()
    client = LLMClient.create_llm_client(llm_config)

    result = await client.simple_pdf_analysis(
        pdf_path_list=files,
        prompt="このPDFの要約を作成してください。"
    )
    print(result)

if __name__ == "__main__":
    import sys
    files = sys.argv[1:]
    import asyncio
    asyncio.run(main(files))