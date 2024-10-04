import sys
import os
from typing import List
from dotenv import load_dotenv
import json

from loguru import logger
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from minio import Minio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arXivRAG.src.core.embedder import InstructorEmbeddings
from configs import cfg


class Chunker:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 20) -> None:
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            ]
        )

    def chunk_batch(self, texts: List[str]) -> List[List[str]]:
        docs = [Document(text=text) for text in texts]
        nodes = self.pipeline.run(documents=docs)
        return nodes


class Transformer:
    def __init__(
        self,
        minio_client: Minio,
        minio_bucket: str,
        minio_json_prefix: str,
        minio_metadata_prefix: str,
        chunk_size: int,
        chunk_overlap: int,
        embedder: InstructorEmbeddings,
    ) -> None:
        self.minio_client = minio_client
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = embedder
        self.minio_bucket = minio_bucket
        self.minio_json_prefix = minio_json_prefix
        self.minio_metadata_prefix = minio_metadata_prefix

    def transform(self, date: str) -> List[List[float]]:
        # Get all objects in the date folder
        paper_objs = self.minio_client.list_objects(
            bucket_name=self.minio_bucket, prefix=f"{self.minio_json_prefix}/{date}/"
        )

        nodes = []
        for paper_obj in paper_objs:
            paper_path = paper_obj.object_name
            logger.info(f"Processing {paper_path}")

            # Get paper from Minio
            # paper_obj = self.minio_client.get_object(
            #     bucket_name=self.minio_bucket,
            #     object_name=paper_path,
            # )

            paper_metadata_obj = self.minio_client.get_object(
                bucket_name=self.minio_bucket,
                object_name=f"{self.minio_metadata_prefix}/{date}/{paper_path.split('/')[-1]}",
            )

            # Load paper as json
            paper = json.load(paper_obj)
            paper_metadata = json.load(paper_metadata_obj)
            paper_metadata["published"] = paper["pub_date"]

            # Create document from paper
            text_list = [paper['abstract']] if len(paper['abstract']) > 0 else []
            text_list += [
                section["text"]
                for section in paper["sections"]
                if len(section["text"]) > 0
            ]

            # Chunk documents
            tmp_nodes = self.chunker.chunk_batch(text_list)

            for i, node in enumerate(tmp_nodes):
                node.embedding = self.embedder._get_text_embedding(node.text)
                node.metadata = paper_metadata

            # Add this paper's nodes
            nodes.extend(tmp_nodes)

        return nodes


if __name__ == "__main__":
    chunker = Chunker()

    # Load environment variables
    load_dotenv(".env")

    os.environ["NO_PROXY"] = os.getenv("SERVER_HOST")

    # Get parsed paper from Minio
    client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False,
    )

    embedding_cfgs = cfg["embedding"]
    embedder = InstructorEmbeddings(
        uri=embedding_cfgs["uri"], model_name=embedding_cfgs["model_name"]
    )
    transformer = Transformer(client, chunker, embedder)
    nodes = transformer.transform(paper_path="20240101/2305.09126v3.json")
    print(len(nodes[0].embedding))
    print(nodes[0].metadata)
    print(nodes[0].text)
