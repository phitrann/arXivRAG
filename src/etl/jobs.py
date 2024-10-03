import os
import sys
import argparse

from loguru import logger
from minio import Minio

from etl import (
    ArxivPDFParser,
    Transformer,
    ExtractTransformLoad,
    MilvusVectorStore,
)
from data_fetching import DataFetcher

# Import configs path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import cfg
from utils.embedder import InstructorEmbeddings

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Fetch ArXiv papers and process them.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (yyyymmdd)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (yyyymmdd)")
    parser.add_argument(
        "--job_type",
        type=str,
        default="etl",
        help="Type of job to run (etl, fetch, or all)",
    )
    # parser.add_argument("--device", type=str, default="cuda", help="Device to run the embedding model")
    parser.add_argument(
        "--overwrite_collection",
        action="store_true",
        help="Overwrite the existing collection in vector store",
    )
    parser.add_argument("--skip_pdf_parsing", action="store_true", help="Skip PDF parsing")
    args = parser.parse_args()

    logger.add(f"logs/jobs_{args.start_date}_{args.end_date}_{args.job_type}.log", rotation="10 MB")

    # Initialize MinIO client
    logger.info("Initializing MinIO client...")
    minio_cfgs = cfg["minio"]
    minio_client = Minio(
        endpoint=minio_cfgs["uri"],
        access_key=minio_cfgs["access_key"],
        secret_key=minio_cfgs["secret_key"],
        secure=False,
    )

    # Set up ETL
    etl_cfgs = cfg["etl"]
    bucket_name = etl_cfgs["minio_bucket"]

    # Create bucket if it does not exist
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        print(f"Bucket {bucket_name} is created")

    logger.info(f"Job type: {args.job_type}")

    if args.job_type in ["fetch", "all"]:
        # Fetch ArXiv papers
        logger.info("Fetching ArXiv papers...")
        fetcher = DataFetcher(
            categories_list=etl_cfgs["categories_list"],
            arxiv_category_file_path=etl_cfgs["arxiv_category_file_path"],
        )
        # TODO: Implement fetching papers
        list_of_papers_path, list_of_papers_code = fetcher.fetch_papers(
            start_date=args.start_date,
            end_date=args.end_date,
            MinIO_client=minio_client,
            max_results=args.max_results,
        )

    # Perform ETL to load the papers into the vector database as embeddings
    if args.job_type in ["etl", "all"]:

        # Initialize the PDF parser
        logger.info("Initializing PDF parser...")
        pdf_parser = ArxivPDFParser(minio_client=minio_client)

        # Initialize the transformer
        logger.info("Initializing transformer...")
        embedding_cfgs = cfg["embedding"]
        embedder = InstructorEmbeddings(
            uri=embedding_cfgs["uri"], model_name=embedding_cfgs["model_name"]
        )
        transformer = Transformer(
            minio_client=minio_client,
            minio_bucket=etl_cfgs["minio_bucket"],
            minio_json_prefix=etl_cfgs["minio_json_prefix"],
            minio_metadata_prefix=etl_cfgs["minio_metadata_prefix"],
            chunk_size=etl_cfgs["chunk_size"],
            chunk_overlap=etl_cfgs["chunk_overlap"],
            embedder=embedder,
        )

        # Initialize the vector store
        logger.info("Initializing vector store...")
        milvus_cfgs = cfg["milvus"]
        embedding_cfgs = cfg["embedding"]
        vector_store = MilvusVectorStore(
            dim=embedding_cfgs["dim"],
            collection_name=etl_cfgs["milvus_collection"],
            overwrite=True,  # overwrite the collection if it already exists
            uri=milvus_cfgs["uri"],
        )

        # Perform ETL
        logger.info("Performing ETL...")
        etl = ExtractTransformLoad(
            pdf_parser=pdf_parser,
            transformer=transformer,
            vector_store=vector_store,
            skip_pdf_parsing=args.skip_pdf_parsing,
        )
        etl.process(start_date=args.start_date, end_date=args.end_date)

