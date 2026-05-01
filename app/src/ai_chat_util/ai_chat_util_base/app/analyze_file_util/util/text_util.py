import aiofiles

class TextUtil:
    # text/*のファイルを読み込んで文字列として返す関数
    @classmethod
    async def process_text_async(cls, filename, mime_type, encoding):
        result = ""
        if mime_type == "text/html":
            # text/htmlの場合
            from bs4 import BeautifulSoup
            # テキストを取得
            async with aiofiles.open(filename, "rb") as f:
                text_data = await f.read()
                soup = BeautifulSoup(text_data, "html.parser")
            result = soup.get_text()

        elif mime_type == "text/xml":
            # text/xmlの場合
            from bs4 import BeautifulSoup
            # テキストを取得
            async with aiofiles.open(filename, "rb") as f:
                text_data = await f.read()
                soup = BeautifulSoup(text_data, features="xml")
            result = soup.get_text()

        elif mime_type == "text/markdown":
            # markdownの場合
            from bs4 import BeautifulSoup
            from markdown import markdown # type: ignore
            # テキストを取得
            async with aiofiles.open(filename, "r" ,encoding=encoding, errors='ignore') as f:
                text_data = await f.read()
                md = markdown(text_data)
                soup = BeautifulSoup(md, "html.parser")
            result = soup.get_text()
        else:
            # その他のtext/*の場合
            async with aiofiles.open(filename, "r", encoding=encoding, errors='ignore') as f:
                result = await f.read()
            
        return result

