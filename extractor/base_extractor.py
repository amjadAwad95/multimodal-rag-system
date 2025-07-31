from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    @abstractmethod
    def extract_texts(self, chunks):
        pass

    @abstractmethod
    def extract_images(self, chunks):
        pass

    @abstractmethod
    def extract_tables(self, chunks):
        pass

    def extract(self, chunks):
        texts = self.extract_texts(chunks)
        images = self.extract_images(chunks)
        tables = self.extract_tables(chunks)

        return texts, images, tables