import argparse
import sys
import os
import shutil
import re
import json
from io import BytesIO
from typing import List, Sequence, Iterator, Dict

from minio import Minio
from pymongo import MongoClient
from loguru import logger
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.schema import BaseNode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core import Document
from llama_index.retrievers.bm25 import BM25Retriever

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdf_parsing import PDFParser
from configs import cfg, pdf_parser_cfg
from utils.embedder import InstructorEmbeddings


class Extractor:
    """
    Extractor class to extract documents from MinIO.
    """
    
    def __init__(self, minio_client: Minio):
        self.minio_client = minio_client
        self.doc_bucket = "arxiv-papers"
        self.pdf_prefix = "papers"
        self.metadata_prefix = "metadatas"

    def extract_docs(self, date: str) -> Iterator[dict]:
        # Get all objects in the date folder
        paper_objs = self.minio_client.list_objects(
            bucket_name=self.doc_bucket, prefix=f"{self.pdf_prefix}/{date}/"
        )

        nodes = []
        for paper_obj in paper_objs:
            paper_path = paper_obj.object_name
            logger.info(f"Processing {paper_path}")

            paper_metadata_obj = self.minio_client.get_object(
                bucket_name=self.metadata_bucket,
                object_name=f"{date}/{paper_path.split('/')[-1]}",
            )
            paper_metadata = json.load(paper_metadata_obj)

            paper_obj = self.minio_client.get_object(
                bucket_name=self.doc_bucket,
                object_name=paper_path
            )

            yield {"metadata": paper_metadata, "pdf": paper_obj.data}


class Transformer:
    """
    Transformer class to parse PDFs, transform them into text nodes, and embed them.
    """
    def __init__(
        self,
        embedder: InstructorEmbeddings,
        metadata_db: MongoClient,
        pdf_parser: PDFParser,
        chunk_size: int = 256,
        chunk_overlap: int = 30,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = embedder
        self.pdf_parser = pdf_parser
        self.metadata_db = metadata_db

    def bm25_transform(self, nodes: List[BaseNode]):
        logger.info("BM25 Transforming ...")
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=10,
        )
        return bm25_retriever

    def parse_pdf(self, doc_dict: Dict, parse_cfgs: Dict):
        # Extract document text and path
        doc_path = doc_dict["path"]
        doc_pdf = doc_dict["pdf"]

        if doc_path in parse_cfgs:
            doc_cfg = parse_cfgs[doc_path]
        else:
            logger.error(f"Can not find {doc_path} configurations")

        logger.info(f"Parsing pdf file: {doc_path}")

        # Extract text, metadata from PDF
        texts, metadata = self.pdf_parser.parse_pdf(doc_pdf, doc_cfg)

        return texts, metadata

    def node_transform(self, doc_dict: Dict) -> List[BaseNode]:
        # Extract document text and path
        texts = doc_dict["texts"]
        metadata = doc_dict["metadata"]

        # logger.info(f"Transforming documents into nodes: {metadata}")

        # Define Ingestion Pipeline
        docs = [Document(text=texts)]
        ingestion_pipe = IngestionPipeline(
            transformations=[
                MarkdownNodeParser(),
                SentenceSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                ),
            ]
        )

        # Run Ingestion Pipeline
        raw_nodes = ingestion_pipe.run(documents=docs)

        # We do not want to add nodes that are too short
        cleaned_nodes = []
        for node in raw_nodes:
            if len(cleaned_nodes) > 0:
                prev_len = len(cleaned_nodes[-1].text)
                if prev_len < self.chunk_size * 4 and prev_len + len(node.text) < self.chunk_size * 8:
                    cleaned_nodes[-1].text += "\n"+ node.text
                else:
                    cleaned_nodes.append(node)
            else:
                cleaned_nodes.append(node)

        nodes = []

        for node in cleaned_nodes:
            node.embedding = self.embedder._get_text_embedding(node.text)
            node.metadata = metadata
            nodes.append(node)

        return nodes


class Loader:
    """
    Loader class to save metadata, markdown, visualization, BM25, and nodes to MinIO, MongoDB, and Milvus.
    """

    def __init__(
        self,
        minio_client: Minio,
        milvus_vector_store: MilvusVectorStore,
        metadata_db: MongoClient
    ):
        self.minio_client = minio_client
        self.vector_store = milvus_vector_store
        self.metadata_db = metadata_db
    
    def load_metadata(
        self, file_name: str, metadata: List[Dict]
    ):
        """
        Save metadata to MongoDB
        """

        logger.info(f"Saving Metadata of file {file_name} into MongoDB")
        # Check if the collection exists
        if file_name in self.metadata_db.list_collection_names():
            # Drop the collection if it exists
            metadata_db[file_name].drop()

        # Create (or recreate) the collection
        collection = self.metadata_db[file_name]

        collection.insert_many(metadata)

    def load_markdown(
        self, doc_bytes: bytes, file_name: str, bucket_name: str = "parsed-documents"
    ):
        """
        Save markdown to MinIO
        """

        # Ensure the bucket exists, create if not
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Upload document bytes to the bucket
        try:
            minio_client.put_object(
                bucket_name=bucket_name,
                object_name=file_name.replace("pdf", "md"),
                data=BytesIO(doc_bytes),
                length=len(doc_bytes),
                content_type="application/md",
            )
        except Exception as e:
            logger.error(f"Failed to upload {file_name}: {e}")

    def load_visualization(
        self, doc_bytes: bytes, file_name: str, bucket_name: str = "visualization"
    ):
        """
        Save visualization to MinIO
        """

        # Ensure the bucket exists, create if not
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Upload document bytes to the bucket
        try:
            minio_client.put_object(
                bucket_name=bucket_name,
                object_name=file_name,
                data=BytesIO(doc_bytes),
                length=len(doc_bytes),
                content_type="application/pdf",
            )
        except Exception as e:
            logger.error(f"Failed to upload {file_name}: {e}")

    def load_bm25(
        self,
        bm25_retriever: BM25Retriever,
        name: str,
        bucket_name: str = "bm25-persistence",
    ):
        """
        Save BM25 Retriever to MinIO
        """        

        folder_path = f"{name}_weights"

        # Save BM25
        logger.info(f"Saving BM25 Retriever at {folder_path}")
        bm25_retriever.persist(folder_path)

        # Ensure the bucket exists, create if not
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Upload all files from the folder to the bucket
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                object_name = os.path.join(
                    name, file_name
                )  # Relative path to use as object name
                try:
                    minio_client.fput_object(
                        bucket_name=bucket_name,
                        object_name=object_name,
                        file_path=file_path,
                    )
                except Exception as e:
                    logger.error(f"Failed to upload {file_name}: {e}")

        # Remove folder
        # Check if the folder exists before trying to remove it
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                logger.info(f"Folder '{folder_path}' has been removed.")
            except Exception as e:
                logger.error(f"Error while removing folder: {e}")
        else:
            logger.error(f"Folder '{folder_path}' does not exist.")

    def load_nodes(self, nodes: Sequence[BaseNode]):
        logger.info(f"Loading {len(nodes)} nodes into vector store")
        self.vector_store.add(nodes)


class ExtractTransformLoad:
    """ETL class to extract arXiv papers, transform them into text nodes as embeddings, and load them into the vector store."""

    def __init__(
        self,
        extractor: Extractor,
        transformer: Transformer,
        loader: Loader,
        pipe_flag: int,
    ) -> None:
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader

        # Flag to show which part of the pipeline to run
        # 2^0: Transform [PDF Parser], 2^1: Transform [Chunking + Embedding], 2^2: Transform [BM25]
        # 2^3: Load Vector store, 2^4: Load BM25
        self.pipe_flag = pipe_flag

    def process(self, tool: str) -> List[Dict]:
        logger.info(f"Processing documents of {tool}...")

        # Extract pdf documents
        pdf_docs = self.extractor.extract_docs(tool=tool)

        tool_nodes = []

        # For each document
        for item in pdf_docs:
            # Initialize variables

            if self.pipe_flag & 1:  # 000001
                texts, metadata = self.transformer.parse_pdf(doc_dict=item, parse_cfgs=pdf_parser_cfg[tool])
                item["texts"] = texts
                item["metadata"] = metadata
                
                self.loader.load_markdown(doc_bytes=texts.encode("utf-8"), file_name=item["path"])
                # self.loader.load_metadata(file_name=item["path"], metadata=item["metadata"])

            if self.pipe_flag & 3 == 3:  # 000011
                # Transform document to nodes
                nodes = self.transformer.node_transform(doc_dict=item)
                item["nodes"] = nodes
            
            if self.pipe_flag & 7 == 7:  # 000111
                tool_nodes.extend(nodes)

            if self.pipe_flag & 19 == 19:  # 010011
                ## Load to Vector Store
                self.loader.load_nodes(nodes)

        if self.pipe_flag & 7 == 7:  # 000111
            # Sparse Embedding with BM25
            bm25_retriever = self.transformer.bm25_transform(
                nodes=tool_nodes
            )
        if self.pipe_flag & 39 == 39:  # 100111
            self.loader.load_bm25(bm25_retriever=bm25_retriever, tool=tool)

        logger.info("ETL pipeline successfully completed")


if __name__ == "__main__":
    # -------- Argument Parser ---------
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=str, help="Tool name to process")
    args = parser.parse_args()

    # -------- MinIO ---------
    minio_cfgs = cfg.get("minio")
    minio_client = Minio(
        endpoint=minio_cfgs["uri"],
        access_key=minio_cfgs["access_key"],
        secret_key=minio_cfgs["secret_key"],
        secure=False,
    )

    # -------- Embedder ---------
    embedding_cfgs = cfg.get("embedding")
    embedder = InstructorEmbeddings(
        uri=embedding_cfgs["uri"], model_name=embedding_cfgs["model_name"]
    )

    # -------- PDF Parser ---------
    pdf_parser = PDFParser()

    # -------- ETL ---------
    etl_cfgs = cfg.get("etl")

    # -------- MongoDB ---------
    mongo_cfgs = cfg.get("mongodb")
    mongo_client = MongoClient(f"mongodb://{mongo_cfgs['username']}:{mongo_cfgs['password']}@{mongo_cfgs['uri']}")
    metadata_db = mongo_client[etl_cfgs["metadata_database"]]

    extractor = Extractor(
        minio_client=minio_client,
    )

    transformer = Transformer(
        embedder=embedder,
        metadata_db=metadata_db,
        pdf_parser=pdf_parser,
        chunk_size=etl_cfgs["chunk_size"],
        chunk_overlap=etl_cfgs["chunk_overlap"],
    )

    milvus_cfgs = cfg.get("milvus")

    milvus_vector_store = MilvusVectorStore(
        uri=milvus_cfgs["uri"],
        dim=embedding_cfgs["dim"],
        collection_name=f"{args.tool.lower()}_{etl_cfgs['chunk_size']}",
        overwrite=True,
    )

    loader = Loader(minio_client=minio_client, milvus_vector_store=milvus_vector_store, metadata_db=metadata_db)

    etl_core = ExtractTransformLoad(
        extractor=extractor,
        transformer=transformer,
        loader=loader,
        pipe_flag=int("111111",base=2)
    )

    etl_core.process(tool=args.tool)
