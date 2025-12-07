import json, os

from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.model import ChatResponse, ChatContent, ChatMessage
import ai_chat_util.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

class PDFChatClient:

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def create_pdf_contents(self, pdf_path: str) -> list[ChatContent]:
        text_content = ChatContent(type="text", text=f"PDFName: {os.path.basename(pdf_path)}")
        pdf_content = self.llm_client.create_pdf_content_from_file(
            file_path=pdf_path,
        )
        return [pdf_content, text_content]

    async def analyze_pdf_async(self, pdf_path: str, prompt: str) -> str:
        '''
        画像解析を行う。テキスト抽出、画像説明、プロンプト応答を生成して、ImageAnalysisResponseで返す
        '''
        prompt_content = ChatContent(type="text", text=prompt)
        pdf_contents = self.create_pdf_contents(pdf_path)

        chat_message = ChatMessage(role="user", content=[prompt_content] + pdf_contents)

        chat_response: ChatResponse = await self.llm_client.run_chat([chat_message], request_context=None)

        return chat_response.output
    
    async def analyze_pdfs_async(self, pdf_path_list: list[str], prompt: str) -> str:
        '''
        複数の画像とプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答を生成して返す
        '''
        prompt_content = ChatContent(type="text", text=prompt)
        pdf_content_list = []
        for pdf_path in pdf_path_list:
            pdf_contents = self.create_pdf_contents(pdf_path)
            pdf_content_list.extend(pdf_contents)

        chat_message = ChatMessage(role="user", content=[prompt_content] + pdf_content_list)
        chat_response: ChatResponse = await self.llm_client.run_chat([chat_message],  request_context=None)
        return chat_response.output

