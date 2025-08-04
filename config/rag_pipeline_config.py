from pydantic import BaseModel
from .pdf_config import PDFConfig
from .ollama_summaries_config import OllamaSummariesConfig


class RagConfig(BaseModel):
    embedding_model: str = "bge-m3:567m"
    collection_name: str = "multimodal-rag-system"
    llm_model: str = "gemini-2.0-flash"

    pdf_config: PDFConfig
    ollama_summaries_config: OllamaSummariesConfig = OllamaSummariesConfig()

    history_system_prompt: str = """
    Given the chat history and the user's latest question,
    rewrite the question to be fully self-contained and understandable without the history.
    Include relevant context from the history if needed. If no rewrite is necessary, return the question as is.
    Do not answer it.
    """
    llm_prompt: str = """
    *"You are a helpful AI assistant. You will be given a context that may include a table, text, and
    the below image. 
    Use this context to answer the user’s question accurately and concisely.
    
    If the context does not provide enough information to answer, respond only with:
    'I don’t know.'
    
    Context: {context}
    Question: {question}
    Answer:"*
    """
