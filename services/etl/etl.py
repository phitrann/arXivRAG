from datetime import datetime, timedelta

from loguru import logger
from llama_index.vector_stores.milvus import MilvusVectorStore

from pdf_parsing import ArxivPDFParser
from transform import Transformer


class ExtractTransformLoad:
    """ETL class to extract arXiv papers, transform them into text nodes as embeddings, and load them into the vector store."""

    def __init__(
        self,
        pdf_parser: ArxivPDFParser,
        transformer: Transformer,
        vector_store: MilvusVectorStore,
        skip_pdf_parsing: bool = False,
    ) -> None:
        self.pdf_parser = pdf_parser
        self.transformer = transformer
        self.vector_store = vector_store
        self.skip_pdf_parsing = skip_pdf_parsing

    def process(self, start_date: str, end_date: str):
        logger.info(f"Processing papers from {start_date} to {end_date}...")

        # Extract -> Transform [Parse PDF -> Dict
        if not self.skip_pdf_parsing:
            logger.info("Parsing PDFs...")
            self.pdf_parser.parse_pdf_to_dict(start_date, end_date)

        # Parse string to datetime
        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")

        # Generate dates from the start to end date
        while start_date <= end_date:
            date = start_date.strftime("%Y%m%d")

            # From Dict -> Chunking + Embedding -> Text Nodes]
            logger.info("Transforming PDFs to text nodes...")
            text_nodes = self.transformer.transform(date)

            # Load -> Vector Store
            logger.info("Loading text nodes into vector store...")
            self.vector_store.add(text_nodes)
            
            start_date += timedelta(days=1)
