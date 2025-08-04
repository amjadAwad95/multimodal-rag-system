import os
import json
from utils import is_base64
from config import RagConfig
from extractor import PdfExtractor
from summaries import OllamaSummaries
from vectorstore import MVRVectorStore
from .base_rag_pipeline import BaseRagPipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


GREEN = '\033[92m'
ENDC = '\033[0m'


class InMemoryRagPipeline(BaseRagPipeline):
    def __init__(self, config: RagConfig):
        self.config = config
        self.data =None
        self.chat_history = []
        self.chain = None

    def get_or_create_document(self):
        if not os.path.isfile(f"data/{self.config.pdf_config.filename}.json"):
            print(f"{GREEN} Prepare new data {ENDC}")
            pdf_extractor = PdfExtractor(self.config.pdf_config)
            texts, images, tables = pdf_extractor.extract(pdf_extractor.chunks)

            print(f"{GREEN} Extraction complete {ENDC}")

            ollama_summaries = OllamaSummaries(self.config.ollama_summaries_config)
            summaries_texts, summaries_images, summaries_tables = ollama_summaries.summaries(texts, images, tables)

            print(f"{GREEN} Summaries complete {ENDC}")

            self.data = {
                "original_data": {"texts": texts, "images": images, "tables": tables},
                "summaries_data": {"texts": summaries_texts, "images": summaries_images, "tables": summaries_tables},
            }

            if not os.path.isdir("data"):
                os.mkdir("data")

            with open(f"data/{self.config.pdf_config.filename}.json", 'w') as outfile:
                json.dump(self.data, outfile)

            print(f"{GREEN} New data is saved in data/{self.config.pdf_config.filename}.json {ENDC}")

        else:
            with open(f"data/{self.config.pdf_config.filename}.json", 'r') as outfile:
                print(f"{GREEN} Prepare exist data {ENDC}")
                self.data = json.load(outfile)
                print(f"{GREEN} Pull the data {ENDC}")


    def load_chat_history(self):
        return self.chat_history


    def save_chat_history(self, question, answer):
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(SystemMessage(content=answer))

    def __pars_doc(self, documents):
        texts_doc = []
        images_doc = []
        for document in documents:
            if is_base64(document):
                images_doc.append(document)
            else:
                texts_doc.append(document)

        return {"texts": texts_doc, "images": images_doc}


    def __prepare_prompt(self, kwargs):
        context = kwargs["context"]

        context_text = ""
        if len(context["texts"]) > 0:
            for element in context["texts"]:
                context_text += f"{element}\n\n"

        prompt_content = [{"type": "text", "text": self.config.llm_prompt.format_map(kwargs)}]

        if len(context["images"]) > 0:
            for image in context["images"]:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])


    def setup(self):
        print(f"{GREEN} Setting up {ENDC}")
        if not self.data:
            self.get_or_create_document()

        print(f"{GREEN} Start save in vectorstore {ENDC}")

        original_data = self.data["original_data"]
        summaries_data = self.data["summaries_data"]

        mvr_vectors = MVRVectorStore(self.config.embedding_model, self.config.collection_name)
        mvr_vectors.save_documents(original_data, summaries_data)

        print(f"{GREEN} Finish save in Vectorstore {ENDC}")

        llm = ChatGoogleGenerativeAI(model = self.config.llm_model, temperature=0.8, max_tokens=None, timeout=None, max_retries=2)

        history_prompt =  ChatPromptTemplate.from_messages([
            ("system", self.config.history_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        self.chain = history_prompt| llm | StrOutputParser() | {
            "context": RunnableLambda(mvr_vectors.retrieve) | RunnableLambda(self.__pars_doc),
            "question":RunnablePassthrough()
        } | RunnablePassthrough().assign(
            response=RunnableLambda(self.__prepare_prompt)| llm | StrOutputParser(),
        )

        print(f"{GREEN} Pipeline is ready {ENDC}")


    def run(self, query):
        if not self.chain:
            raise ValueError("Chain is empty, call setup() first")

        chat_history = self.load_chat_history()

        response = self.chain.invoke({"input": query, "chat_history": chat_history})

        self.save_chat_history(response["question"], response["response"])

        return response

