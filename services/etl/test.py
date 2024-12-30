from pdf_parsing import PDFParser
import os
from minio import Minio
from loguru import logger

os.environ["NO_PROXY"] = "localhost"

# Example usage for MinIO stored PDFs
client = Minio(
    endpoint="localhost:9800",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)
logger.add("logs/pdf_parsing.log", rotation="10 MB")
logger.info("MinIO client is initialized")

minio_parser = PDFParser(client)


paper = client.get_object("arxiv-papers", "papers/20240105/2307.11196v2.pdf")
paper_data = paper.read()  # Read file bytes

# Parse the PDF
result = minio_parser.parse_pdf(file_data=paper_data)
logger.info(len(result))
with open("text.md", "w") as file:
    file.write(result)