import sys
import os
from typing import List
from dotenv import load_dotenv
import json

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from minio import Minio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.embedder import InstructorEmbeddings


class Chunker():
    def __init__(self, chunk_size: int=256, chunk_overlap: int=20) -> None:
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            ]
        )

    def chunk_batch(self, texts: List[str]) -> List[List[str]]:
        docs = [Document(text=text) for text in texts]
        nodes = self.pipeline.run(documents=docs)
        return nodes


class Embedder():
    def __init__(self, model_name: str="BAAI/llm-embedder", uri: str="http://172.16.87.76:8081") -> None:
        self.embedder = InstructorEmbeddings(model_name=model_name, uri=uri)

    def embed_doc(self, text: str) -> List[float]:
        return self.embedder._get_text_embedding(text)


class Transformer():
    def __init__(self, storage_client, chunker, embedder) -> None:
        self.storage_client = storage_client
        self.chunker = chunker
        self.embedder = embedder

    def transform(self, paper_path) -> List[List[float]]:
        # Get paper from Minio
        paper_obj = self.storage_client.get_object(
            bucket_name="arxiv-papers",
            object_name=f'processed_papers/{paper_path}'
        )

        paper_metadata_obj = self.storage_client.get_object(
            bucket_name="arxiv-papers",
            object_name=f'metadata/{paper_path}'
        )

        # Load paper as json
        paper = json.load(paper_obj)
        paper_metadata = json.load(paper_metadata_obj)

        # Create document from paper
        text_list = [section['text'] for section in paper['sections'] if len(section['text']) > 0]
        text_list.append(paper['abstract'])

        # Chunk documents
        nodes = self.chunker.chunk_batch(text_list)

        paper_metadata['published'] = paper['pub_date']
        for node in nodes:
            node.embedding = self.embedder.embed_doc(node.text)
            node.metadata = paper_metadata

        return nodes


if __name__ == "__main__":
    chunker = Chunker()
    embedder = Embedder()

    # Load environment variables
    load_dotenv('.env')

    os.environ['NO_PROXY'] = os.getenv("SERVER_HOST")

    # Get parsed paper from Minio
    client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
    )

    transformer = Transformer(client, chunker, embedder)
    nodes = transformer.transform(paper_path="20240101/2305.09126v3.json")
    print(len(nodes[0].embedding))
    print(nodes[0].metadata)
    print(nodes[0].text)
    