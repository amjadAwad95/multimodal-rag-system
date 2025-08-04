from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import OllamaSummariesConfig
from .base_summaries import BaseSummaries


class OllamaSummaries(BaseSummaries):
    def __init__(self, config: OllamaSummariesConfig = OllamaSummariesConfig()):
        self.config = config
        self.summaries_texts = None
        self.summaries_tables = None
        self.summaries_images = None

        self.ollama_llm = ChatOllama(model = config.ollama_model)
        self.google_llm = ChatGoogleGenerativeAI(
            model = config.google_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )

        self.ollama_prompt = ChatPromptTemplate.from_template(config.prompt_text)
        self.image_prompt = [(
            "user",
            [
                {"type": "text", "text": config.prompt_image_text},
                {"type": "image_url", "image_url": "data:image/png;base64,{image}"}
            ]
        )]
        self.google_prompt = ChatPromptTemplate.from_messages(self.image_prompt)

        self.ollama_chain = {"element":lambda x:x} | self.ollama_prompt | self.ollama_llm | StrOutputParser()
        self.google_chain = {"image": lambda x:x}  | self.google_prompt | self.google_llm | StrOutputParser()


    def texts_summaries(self, texts):
        if self.summaries_texts:
            return self.summaries_texts

        self.summaries_texts = self.ollama_chain.batch(texts)
        return self.summaries_texts

    def tables_summaries(self, tables):
        if self.summaries_tables:
            return self.summaries_tables

        self.summaries_tables = self.ollama_chain.batch(tables)
        return self.summaries_tables

    def images_summaries(self, images):
        if self.summaries_images:
            return self.summaries_images

        self.summaries_images = self.google_chain.batch(images)
        return self.summaries_images


