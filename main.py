from config import PDFConfig
from extractor import PdfExtractor

pdf_config = PDFConfig(filename = "Attention-Is-All-You-Need.pdf")

pdf_extractor = PdfExtractor(pdf_config)

texts, images, tables = pdf_extractor.extract(pdf_extractor.chunks)

print("texts", texts)
print("images", images)
print("tables", tables)