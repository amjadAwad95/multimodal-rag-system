from config import PDFConfig, RagConfig
from dotenv import load_dotenv
from rag_pipeline import InMemoryRagPipeline

load_dotenv()

pdf_config = PDFConfig(filename = "Attention-Is-All-You-Need.pdf")
rag_config = RagConfig(pdf_config = pdf_config)

rag = InMemoryRagPipeline(rag_config)

rag.setup()

response = rag.run("What the transformer")

print(response)
