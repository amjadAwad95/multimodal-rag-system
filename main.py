import os
import json
import base64
import binascii
from config import PDFConfig
from dotenv import load_dotenv
from extractor import PdfExtractor
from summaries import OllamaSummaries
from vectorstore import MVRVectorStore
from IPython.display import display, Image

load_dotenv()

pdf_config = PDFConfig(filename = "Attention-Is-All-You-Need.pdf")

pdf_extractor = PdfExtractor(pdf_config)

texts, images, tables = pdf_extractor.extract(pdf_extractor.chunks)

normal_data = {
"texts": texts,
"images": images,
"tables": tables
}

print("Extraction complete")


if not os.path.isfile("data/summaries_data.json"):

    ollama_summaries = OllamaSummaries()

    summaries_texts, summaries_images, summaries_tables = ollama_summaries.summaries(texts, images, tables)

    if not os.path.isdir("data"):
        os.mkdir("data")

    with open("data/summaries_data.json", "w", encoding='utf-8') as json_file:
        summaries_data = {
            "texts": summaries_texts,
            "images": summaries_images,
            "tables": summaries_tables
        }
        json.dump(summaries_data, json_file)

else:
    with open("data/summaries_data.json", "r", encoding='utf-8') as json_file:
        summaries_data = json.load(json_file)
        summaries_texts = summaries_data["texts"]
        summaries_images = summaries_data["images"]
        summaries_tables = summaries_data["tables"]

print("Summaries complete")

mvr_vectors = MVRVectorStore()
mvr_vectors.save_documents(normal_data, summaries_data)

values = mvr_vectors.retrieve("What is multi head")


def is_base64(value):
    try:
        base64.b64decode(value, validate=True)
        return True
    except binascii.Error:
        return False
    except Exception as e:
        return False

for value in values:
    if is_base64(value):
        image = base64.b64decode(value)
        display(Image(image))
    elif "CompositeElement" in str(type(value)):
        print(value.text)
    else:
        print(value.metadata.text_as_html)