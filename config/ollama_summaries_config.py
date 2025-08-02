from pydantic import BaseModel


class OllamaSummariesConfig(BaseModel):
    ollama_model: str = "llama3.1:8b"
    google_model: str = "gemini-2.0-flash"
    prompt_text: str = """
    You are an expert assistant in summarizing text and tables.  
    Your task:  
    1. Read the provided table or text carefully.  
    2. Produce a **concise, accurate, and information-rich summary** that captures the essential meaning.  
    3. If the content includes a table, focus on **key trends, comparisons, or insights** rather than listing every value.  
    4. If the content is text, summarize the **main ideas** and **important details**.  
    5. **Do not** include any introductory phrases like “Here is a summary.”  
    6. **Output only the summary** as plain text, suitable for storage in a vector database.
    7. **Output only the summary** without extra comments or formatting.  
    
    Content to summarize: {element}
    """
    prompt_image_text: str = """
    You are an expert in image understanding and summarization for document analysis.

    You will be given an image extracted from a PDF page. Follow these instructions:
    
    1. **Analyze the image carefully**:
       - Identify the key objects, entities, and actions.
       - Note any charts, graphs, tables, or diagrams.
       - If there is text in the image, **extract and summarize it**.
    
    2. **Produce a concise and information-rich summary**:
       - Capture the **main message** of the image.
       - Highlight **contextual details** that would help a user understand it without seeing it.
       - Avoid unnecessary descriptions like colors or shapes unless **they are meaningful**.
    
    3. **Prepare the summary for retrieval in a RAG system**:
       - The summary should be **clear, searchable, and semantically meaningful**.
       - Include key **keywords or entities** if they are important.
    
    4. **Answerability**:
       - Write the summary in a way that makes it easy for a future LLM to answer user questions about the image.
       - Do **not** include assumptions or hallucinations.
    
    Return only the final summary.
    """