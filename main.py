import os
from config import PDFConfig
from extractor import PdfExtractor
from summaries import OllamaSummaries
from dotenv import load_dotenv

load_dotenv()

pdf_config = PDFConfig(filename = "Attention-Is-All-You-Need.pdf")

pdf_extractor = PdfExtractor(pdf_config)

texts, images, tables = pdf_extractor.extract(pdf_extractor.chunks)

print("Extraction complete")

ollama_summaries = OllamaSummaries()

summaries_texts, summaries_images, summaries_tables = ollama_summaries.summaries(texts, images, tables)

if not os.path.isdir("summaries_return"):
    os.mkdir("summaries_return")

with open("summaries_return/summaries_text.txt", "w", encoding='utf-8') as text_file:
    text_file.write("\n----------------------------------------------\n".join(summaries_texts))

with open("summaries_return/summaries_images.txt", "w", encoding='utf-8') as images_file:
    images_file.write("\n----------------------------------------------\n".join(summaries_images))

with open("summaries_return/summaries_tables.txt", "w", encoding='utf-8') as tables_file:
    tables_file.write("\n----------------------------------------------\n".join(summaries_tables))


print("Summaries complete")