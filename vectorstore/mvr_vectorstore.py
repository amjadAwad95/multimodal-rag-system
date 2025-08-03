import uuid
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from .base_vectorstore import BaseVectorStore


class MVRVectorStore(BaseVectorStore):
    def __init__(self, model_name="bge-m3:567m", collection_name="multimodal-rag-system"):
        self.model_name = model_name
        self.collection_name = collection_name

        self.embedding=OllamaEmbeddings(model=model_name)
        vectorstore = Chroma(collection_name=collection_name, embedding_function=self.embedding)
        store = InMemoryStore()

        id_key = "doc_id"

        self.retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

    def save_documents(self, raw_documents, summarized_documents):
        texts_ids = [str(uuid.uuid4()) for _ in summarized_documents["texts"]]
        images_ids = [str(uuid.uuid4()) for _ in summarized_documents["images"]]
        tables_ids = [str(uuid.uuid4()) for _ in summarized_documents["tables"]]

        summary_texts = [
            Document(page_content=text, metadata={"doc_id": texts_ids[i]}) for i, text in enumerate(summarized_documents["texts"])
        ]
        summary_images = [
            Document(page_content=image, metadata={"doc_id": images_ids[i]}) for i, image in enumerate(summarized_documents["images"])
        ]
        summary_tables = [
            Document(page_content=table, metadata={"doc_id": tables_ids[i]}) for i, table in enumerate(summarized_documents["tables"])
        ]

        self.retriever.vectorstore.add_documents(summary_texts)
        self.retriever.docstore.mset(list(zip(texts_ids, raw_documents["texts"])))

        self.retriever.vectorstore.add_documents(summary_images)
        self.retriever.docstore.mset(list(zip(images_ids, raw_documents["images"])))

        self.retriever.vectorstore.add_documents(summary_tables)
        self.retriever.docstore.mset(list(zip(tables_ids, raw_documents["tables"])))


    def retrieve(self, query):
        return self.retriever.invoke(query)