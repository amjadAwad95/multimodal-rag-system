from abc import ABC, abstractmethod


class BaseRagPipeline(ABC):
    @abstractmethod
    def get_or_create_document(self):
        pass

    @abstractmethod
    def load_chat_history(self):
        pass

    @abstractmethod
    def save_chat_history(self, question, answer):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self, query):
        pass