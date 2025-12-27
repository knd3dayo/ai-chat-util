import asyncio
from tqdm.asyncio import tqdm_asyncio

from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.agent.agent_util import MSAIAgentFactory


class LLMBatchClient:
    def __init__(self, client: LLMClient):
        self.client = client

    async def _process_row_(
            self, row_num: int, prompt: str, input_message: str, agent_mode: bool, progress: tqdm_asyncio
            ) -> tuple[int, str, str]:
        content = f"{prompt}\n{input_message}"
        if agent_mode:
            agent_util = MSAIAgentFactory(llm_config=self.client.llm_config)

            async with (agent_util.create_agent() as agent):
                response = await agent.run(content)
                progress.update(1)  # Update progress after processing the row
                return (row_num, input_message, response.text)

        message = f"{prompt}\n{input_message}"
        response = await self.client.simple_chat(message)

        progress.update(1)  # Update progress after processing the row
        return (row_num, input_message, response)

    async def run(self, prompt: str, messages: list[str], agent_mode: bool = False, concurrency: int = 5) -> list[str]:

        progress = tqdm_asyncio(total=len(messages), desc="progress")
        # 進捗バーのフォーマット
        progress.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        sem = asyncio.Semaphore(concurrency)

        async def _run_one(i: int, msg: str):
            # Semaphore is effective only when each task acquires it.
            async with sem:
                return await self._process_row_(i, prompt, msg, agent_mode, progress)

        tasks = [asyncio.create_task(_run_one(i, msg)) for i, msg in enumerate(messages)]

        try:
            responses = await asyncio.gather(*tasks)
        finally:
            progress.close()
    
        # Sort responses by row number to maintain order
        responses.sort(key=lambda x: x[0])
        return [response for _, _, response in responses]
    
# テストコードは以下の通りです。
if __name__ == "__main__":
    import os
    from ai_chat_util.llm.llm_config import LLMConfig

    async def main():
        llm_config = LLMConfig()
        client = LLMClient.create_llm_client(llm_config)
        batch_client = LLMBatchClient(client)
        agent_mode = False

        prompt = "以下の言語のお昼のあいさつを教えて"
        messages = [
            "日本語",
            "英語",
            "中国語"
        ]

        results = await batch_client.run(prompt, messages, agent_mode)

        for i, res in enumerate(results):
            print(f"Prompt {i+1}: {res}")

    asyncio.run(main())
