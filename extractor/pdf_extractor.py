from unstructured.partition.pdf import partition_pdf
from .base_extractor import BaseExtractor
from config import PDFConfig


class PdfExtractor(BaseExtractor):
    def __init__(self, config: PDFConfig):
        self.config = config

        self.chunks = partition_pdf(**config.model_dump())


    def extract_texts(self, chunks):
        texts = []

        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                texts.append(chunk)

        return texts


    def extract_images(self, chunks):
        images = []

        for chunk in chunks:
            if "Image" in str(type(chunk)):
                images.append(chunk.metadata.image_base64)
            else:
                if "CompositeElement" in str(type(chunk)):
                    for element in chunk.metadata.orig_elements:
                        if "Image" in str(type(element)):
                            images.append(element.metadata.image_base64)

        return images


    def extract_tables(self, chunks):
        tables = []

        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            else:
                for element in chunk.metadata.orig_elements:
                    if "Table" in str(type(element)):
                        tables.append(element)

        return tables