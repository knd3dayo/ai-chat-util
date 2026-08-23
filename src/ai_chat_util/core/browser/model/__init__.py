from pydantic import BaseModel, Field


class BrowserTaskResult(BaseModel):
    """Result of a browser-use agent task execution."""

    output: str = Field(description="Text output or extracted content from the task")
    is_done: bool = Field(description="Whether the agent completed the task successfully")
    n_steps: int = Field(description="Number of steps the agent took to complete the task")
