import os
import argparse
from dotenv import load_dotenv
from ultils import (
    ETL,
    Fetcher,
    OllamaEmbedding,
    MilvusVectorStore,
    Minio,
    PyMuPDFReader,
    SentenceSplitter
)

if __name__ == "__main__":
    folder_path = os.path.dirname(__file__)

    # Load environment variables
    load_dotenv(os.path.join(folder_path, ".env"))

    # Define argument parser
    parser = argparse.ArgumentParser(description="Fetch ArXiv papers and process them.")
    parser.add_argument("--start_day", type=int, required=True, help="Start day of latest updated papers")
    parser.add_argument("--start_month", type=int, required=True, help="Start month of latest updated papers")
    parser.add_argument("--start_year", type=int, required=True, help="Start year of latest updated papers")
    parser.add_argument("--max_results", type=int, default=10, help="Maximum number of papers to fetch")
    parser.add_argument("--embedding_model", type=str, default="llm-embedder-q4_k_m" ,help="Ollama embedding model name")
    parser.add_argument("--vector_dim", type=int, default=768, help="Dimension of the embedding vector")
    # parser.add_argument("--device", type=str, default="cuda", help="Device to run the embedding model")
    parser.add_argument("--overwrite_collection", action="store_true", help="Overwrite the existing collection in vector store")
    args = parser.parse_args()

    # Set which host to skip proxy
    os.environ['NO_PROXY'] = os.getenv("NO_PROXY_HOST")

    # Define categories of papers to fetch
    categories_list = [
        'Computer Science',
        'Electrical Engineering and Systems Science',
        'Statistics',
    ]

    # Initialize MinIO client
    client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
    )
    print("MinIO client is initialized")
    bucket_name = os.getenv("MINIO_BUCKET_NAME")

    # Create bucket if it does not exist
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Bucket {bucket_name} is created")

    # Initialize the vector store
    vector_store = MilvusVectorStore(
        dim=args.vector_dim,
        collection_name=os.getenv("MILVUS_COLLECTION_NAME"),
        overwrite=True, # overwrite the collection if it already exists
        uri=os.getenv("MILVUS_URI"),
    )
    print("Vector store is initialized")

    # Initialize the embedding model
    if not os.path.exists(os.path.join(folder_path, "model_cache")):
        os.makedirs(os.path.join(folder_path, "model_cache"))
        print("Model cache folder is created")
    embed_model = OllamaEmbedding(
        model_name=args.embedding_model,
        base_url=os.getenv("OLLAMA_EMBEDDING_URL"),
    )
    print("Embedding model is initialized")

    # Initialize the PDF reader and text parser
    pdf_reader = PyMuPDFReader()
    text_parser = SentenceSplitter(chunk_size=1024)
    print("PDF reader and text parser are initialized")

    # Fetch ArXiv papers
    print("\nFetching ArXiv papers...")
    fetcher = Fetcher(
        categories_list=categories_list,
        arxiv_category_file_path=os.path.join(folder_path, "arxiv_category.json")
    )
    list_of_papers_path, list_of_papers_code = fetcher.fetch_papers(
        start_day=args.start_day,
        start_month=args.start_month,
        start_year=args.start_year,
        MinIO_client=client,
        max_results=args.max_results
    )

    # Perform ETL to load the papers into the vector database as embeddings
    print("\nPerforming ETL...")
    etl = ETL(
        root_folder=folder_path,
        vector_store=vector_store,
        embed_model=embed_model,
        MinIO_client=client,
        bucket_name=bucket_name
    )
    etl.extract_transform_load(
        list_of_papers_path=list_of_papers_path,
        list_of_papers_code=list_of_papers_code,
        text_parser=text_parser,
        pdf_reader=pdf_reader
    )
