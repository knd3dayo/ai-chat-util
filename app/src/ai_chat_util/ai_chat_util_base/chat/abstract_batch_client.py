from abc import ABC, abstractmethod

from ai_chat_util.ai_chat_util_base.ai_chatl_util_models import ChatRequest, ChatResponse


class AbstractBatchClient(ABC):

    @abstractmethod
    async def run_batch_chat(
        self, chat_requests: list[ChatRequest], concurrency: int = 5
    ) -> list[tuple[int, ChatResponse]]:
        '''
        指定されたチャットリクエスト群に対してバッチ処理を行う。
        '''
        pass

    @abstractmethod
    async def run_simple_batch_chat(
        self, prompt: str, messages: list[str], concurrency: int = 5
    ) -> list[str]:
        '''
        文字列メッセージ群に対して簡易バッチ処理を行う。
        '''
        pass

    @abstractmethod
    async def run_batch_chat_from_excel(
        self,
        prompt: str,
        input_excel_path: str,
        output_excel_path: str = "output.xlsx",
        content_column: str = "content",
        file_path_column: str = "file_path",
        output_column: str = "output",
        detail: str = "auto",
        concurrency: int = 16,
    ) -> None:
        '''
        Excel ファイルを入力としてバッチ処理を行い、結果を Excel へ出力する。
        '''
        pass