from pydantic import BaseModel, Field
from typing import List

class PDFConfig(BaseModel):
    filename: str = Field(..., title="The name of the PDF file")
    infer_table_structure: bool = True
    strategy: str = "hi_res"
    extract_image_block_types: List[str] = ["Image"]
    extract_image_block_to_payload: bool = True
    chunking_strategy: str = "by_title"
    max_characters: int = 10000
    combine_text_under_n_chars: int = 2000
    new_after_n_chars: int = 6000