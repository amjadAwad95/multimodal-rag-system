from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    @abstractmethod
    def save_documents(self, raw_documents, summarized_documents):
        pass

    @abstractmethod
    def retrieve(self, query):
        pass