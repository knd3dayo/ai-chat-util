import json, os

from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.model import ChatResponse, ChatContent, ChatMessage
import ai_chat_util.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

class ImageChatClient:

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def create_image_contents(self, image_path: str) -> list[ChatContent]:
        text_content = ChatContent(type="text", text=f"ImageName: {os.path.basename(image_path)}")
        image_content = self.llm_client.create_image_content_from_file(
            file_path=image_path,
        )
        return [image_content, text_content]

    async def analyze_image_async(self, image_path: str, prompt: str) -> str:
        '''
        画像解析を行う。テキスト抽出、画像説明、プロンプト応答を生成して、ImageAnalysisResponseで返す
        '''
        prompt_content = ChatContent(type="text", text=prompt)
        image_contents = self.create_image_contents(image_path)

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_contents)

        chat_response: ChatResponse = await self.llm_client.run_chat([chat_message], request_context=None)

        return chat_response.output
    
    async def analyze_images_async(self, image_path_list: list[str], prompt: str) -> str:
        '''
        複数の画像とプロンプトから画像解析を行う。各画像のテキスト抽出、各画像の説明、プロンプト応答を生成して返す
        '''
        prompt_content = ChatContent(type="text", text=prompt)
        image_content_list = []
        for image_path in image_path_list:
            image_contents = self.create_image_contents(image_path)
            image_content_list.extend(image_contents)

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_content_list)
        chat_response: ChatResponse = await self.llm_client.run_chat([chat_message],  request_context=None)
        return chat_response.output


    async def analyze_image_groups_async(self, image_group1: list[str], image_group2: list[str], prompt: str) -> str:
        """
        画像グループ1と画像グループ2とプロンプトから画像解析を行う。
        各画像のプロンプト応答を生成して、回答を返す
        """
        image1_contents: list[ChatContent] = []
        for path1 in image_group1:
            image1_content = self.llm_client.create_image_content_from_file(
                file_path=path1
            )
            image1_contents.append(image1_content)
            
            text_content1 = ChatContent(type="text", text=f"ImageGroup: 1, ImageName: {os.path.basename(path1)}")
            image1_contents.append(text_content1)

        image2_contents: list[ChatContent] = []
        for path2 in image_group2:
            image2_content = self.llm_client.create_image_content_from_file(
                file_path=path2
            )
            image2_contents.append(image2_content)
            
            text_content2 = ChatContent(type="text", text=f"ImageGroup: 2, ImageName: {os.path.basename(path2)}")
            image2_contents.append(text_content2)
        
        prompt_content = ChatContent(type="text", text=prompt)

        all_contents = image1_contents + image2_contents + [prompt_content]
        chat_message = ChatMessage(role="user", content=all_contents)
        chat_response: ChatResponse = await self.llm_client.run_chat([chat_message])
        
        return chat_response.output
