from ai_chat_util.batch.batch_client import LLMBatchClient

async def main():

    batch = LLMBatchClient()
    prompt = "英語に翻訳してください"
    messages = [
        "今日はどんな日？",
        "明日の天気は？",
        "今週のニュースを教えて"
    ]
    results = await batch.run_simple_batch_chat(prompt, messages, concurrency=3)
    for r in results:
        print(r)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())