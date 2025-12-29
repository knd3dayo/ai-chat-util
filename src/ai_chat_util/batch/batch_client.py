import os
import asyncio
from tqdm.asyncio import tqdm_asyncio

from ai_chat_util.llm.llm_factory import LLMFactory

from ai_chat_util.llm.model import ChatMessage, ChatResponse, ChatHistory, ChatContent, ChatRequest

import pandas as pd

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class LLMBatchClient:

    async def _process_row_(
            self, row_num: int, chat_history: ChatHistory, progress: tqdm_asyncio
            ) -> tuple[int, ChatResponse, ChatHistory]:

        if not chat_history.messages:
            # メッセージが空の場合はスキップして空のレスポンスを返す
            chat_response = ChatResponse(output="", input_tokens=0, output_tokens=0, documents=[])
            result_chat_history = chat_history
        else:
            chat_request = ChatRequest(chat_history=chat_history)
            llm_client = LLMFactory.create_llm_client(chat_request=chat_request)
            chat_response = await llm_client.chat()
            result_chat_history = llm_client.chat_request.chat_history

        progress.update(1)  # Update progress after processing the row
        return (row_num, chat_response, result_chat_history)


    async def run_batch_chat(
            self, chat_histories: list[ChatHistory], concurrency: int = 5
            ) -> list[tuple[int, ChatResponse, ChatHistory]]:
        '''
        指定されたメッセージリストに対して、指定されたプロンプトを用いてバッチ処理を行う。
        '''
        progress = tqdm_asyncio(total=len(chat_histories), desc="progress")
        # 進捗バーのフォーマット
        progress.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        sem = asyncio.Semaphore(concurrency)

        async def _run_one(i: int, chat_history: ChatHistory):
            # Semaphore is effective only when each task acquires it.
            async with sem:
                return await self._process_row_(i, chat_history, progress)

        tasks = [asyncio.create_task(_run_one(i, chat_history)) for i, chat_history in enumerate(chat_histories)]

        try:
            responses = await asyncio.gather(*tasks)
        finally:
            progress.close()
    
        # Sort responses by row number to maintain order
        responses.sort(key=lambda x: x[0])
        return responses

    async def run_simple_batch_chat(self, prompt: str, messages: list[str], concurrency: int = 5) -> list[str]:
        '''
        指定されたメッセージリストに対して、指定されたプロンプトを用いてバッチ処理を行う。
        '''
        llm_client = LLMFactory.create_llm_client()
        chat_histories: list[ChatHistory] = []
        for msg in messages:
            chat_content = llm_client.create_text_content(text=f"{prompt}\n{msg}")
            chat_message = ChatMessage(role="user", content=[chat_content])
            chat_history = ChatHistory(messages=[chat_message])
            chat_histories.append(chat_history)
    
        responses = await self.run_batch_chat(chat_histories, concurrency)
        response_messages = []
        for _, chat_response, _ in responses:
            response_messages.append(chat_response.output)
    
        return response_messages
    
    # Excelファイルからデータを読み込み、結果をExcelファイルに書き込むメソッド
    async def run_batch_chat_from_excel(
            self, prompt: str, 
            input_excel_path: str, output_excel_path: str = "output.xlsx",
            content_column: str = "content", 
            file_path_column: str = "file_path", 
            output_column: str = "output",
            detail: str = "auto",
            concurrency: int = 16,
        ) -> None:

        llm_client = LLMFactory.create_llm_client()
        use_custom_pdf_analyzer = llm_client.llm_config.use_custom_pdf_analyzer
        # Excelファイルを読み込む
        df = pd.read_excel(input_excel_path)

        # content_columnとfile_path_columnの両方がない場合はエラー
        if content_column not in df.columns and file_path_column not in df.columns:
            raise ValueError(f"Input Excel must have at least one of the columns: {content_column}, {file_path_column}")

        if content_column in df.columns:
            # pandas 3.0 以降では `df[col].method(..., inplace=True)` のような chained assignment は
            # intermediate が copy 扱いになり、inplace が効かなくなるため避ける。
            # また `astype(str)` は NaN を 'nan' 文字列にしてしまうので、先に fillna を行う。
            s = df[content_column].fillna("").astype(str)
            s = s.replace(to_replace=r"_x000D_", value="", regex=True)
            df[content_column] = s

        if file_path_column in df.columns:
            df[file_path_column] = df[file_path_column].fillna("").astype(str)

        # 指定された入力列からメッセージを取得
        # 1行ずつ読み込んで、ChatHistoryオブジェクトのリストを作成
        chat_histories: list[ChatHistory] = []
        for _, row in df.iterrows():
            contents: list[ChatContent] = []
            input_message = str(row[content_column])
            file_path = str(row[file_path_column])
            # input_messageとfile_pathの両方が空の場合はスキップ
            if not input_message and not file_path:
                chat_histories.append(ChatHistory(messages=[]))
                continue
            contents.append(llm_client.create_text_content(text=f"{prompt}\n{input_message}"))
            # ファイルが存在しない場合はfile_pathを無視
            if os.path.isfile(file_path):
                logger.info(f"Processing file: {file_path}")
                file_content = llm_client.create_multi_format_contents_from_file(
                    file_path=file_path, 
                    detail=detail,
                    use_custom_pdf_analyzer=use_custom_pdf_analyzer
                )
                contents.extend(file_content)

            chat_message = ChatMessage(
                role="user",
                content=contents
            )
            chat_history = ChatHistory(messages=[chat_message])
            chat_histories.append(chat_history)

        # バッチ処理を実行
        results = await self.run_batch_chat(chat_histories, concurrency)

        # 結果を指定された出力列に追加
        df[output_column] = [ response.output for _, response, _ in results ]

        # 結果を新しいExcelファイルに保存
        df.to_excel(output_excel_path, index=False)

