from abc import ABC, abstractmethod

class BaseSummaries(ABC):
    @abstractmethod
    def texts_summaries(self, texts):
        pass

    @abstractmethod
    def tables_summaries(self, tables):
        pass

    @abstractmethod
    def images_summaries(self, images):
        pass

    def summaries(self, texts, images, tables):
        summaries_texts = self.texts_summaries(texts)
        summaries_tables = self.tables_summaries(tables)
        summaries_images = self.images_summaries(images)

        return summaries_texts, summaries_images, summaries_tables