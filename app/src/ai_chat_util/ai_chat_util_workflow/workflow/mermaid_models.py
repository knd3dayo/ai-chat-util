from pydantic import BaseModel


class MermaidCodeBlock(BaseModel):
    code: str
    full_text: str
    start_index: int
    end_index: int