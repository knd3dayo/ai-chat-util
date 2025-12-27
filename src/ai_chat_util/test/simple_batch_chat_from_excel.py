from ai_chat_util.batch.batch_client import LLMBatchClient

async def main(input_file: str, output_file: str):

    batch = LLMBatchClient()
    prompt = "要約してください"
    await batch.run_batch_chat_from_excel(
        prompt=prompt,
        input_excel_path=input_file,
        output_excel_path=output_file
    )


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    import asyncio
    asyncio.run(main(input_file, output_file))