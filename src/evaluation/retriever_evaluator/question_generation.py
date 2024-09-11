import pandas as pd
import numpy as np
import os
from llama_index.llms.ollama import Ollama
import nest_asyncio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
def question_generation(
    csv_file_path,
    sample_size,
    model_generate,
    model_embedding,
    top_k=5,
    cache_dir_embedding="./model",
    data_dir="./data",
    chunk_size=512,
    save_eval_dataset=True
):
    """
    Load a CSV file, process abstracts, and generate question-context pairs for QA evaluation.

    Parameters:
    csv_file_path (str): The path to the CSV file.
    sample_size (int): The number of samples to process.
    model_generate (str): The model name for question generation.
    model_embedding (str): The embedding model name.
    top_k (int): Number of top similar results for retriever (default=5).
    cache_dir_embedding (str): Path to the cache directory for embeddings (default="./model").
    data_dir (str): Directory for storing abstract text files (default="./data").
    chunk_size (int): Chunk size for splitting abstracts (default=512).
    save_eval_dataset (bool): Whether to save the generated QA dataset (default=True).

    Returns:
    None
    """
    print("Step 1: Loading CSV file and sampling data...")
    # Load CSV file and sample data
    df = pd.read_csv(csv_file_path)
    df = df.dropna()  # Drop rows with missing values
    df = df.sample(n=sample_size)  # Sample the required number of rows
    print(f"Loaded {df.shape[0]} samples from CSV.")

    # Ensure the data directory exists
    print(f"Step 2: Ensuring directory {data_dir} exists and saving abstracts as text files...")
    os.makedirs(data_dir, exist_ok=True)

    # Save abstracts as individual text files
    for idx, abstract in enumerate(df['abstract']):
        # Handle non-string values in the abstract
        if not isinstance(abstract, str):
            abstract = ''
        # Define file path for each abstract
        file_path = os.path.join(data_dir, f'abstract_{idx}.txt')
        with open(file_path, 'w') as file:
            file.write(abstract)
    print(f"Saved {df.shape[0]} abstracts to {data_dir}.")

    # Configure the LLM and embedding models
    print("Step 3: Configuring LLM and embedding models...")
    nest_asyncio.apply()
    os.environ["NO_PROXY"] = "172.16.87.75"  # Update with your proxy settings if needed
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")

    llm = Ollama(base_url=ollama_base_url, model=model_generate, request_timeout=120.0)
    embed_model = HuggingFaceEmbedding(model_name=model_embedding, cache_folder=cache_dir_embedding)

    # Apply settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    print(f"LLM Model '{model_generate}' and Embedding Model '{model_embedding}' are set.")

    # Read the saved abstracts and create chunks for processing
    print(f"Step 4: Loading abstracts and creating chunks of size {chunk_size}...")
    reader = SimpleDirectoryReader(data_dir)
    documents = reader.load_data()

    # Chunk documents into smaller parts
    node_parser = SentenceSplitter(chunk_size=chunk_size)
    nodes = node_parser.get_nodes_from_documents(documents)

    # Assign unique IDs to nodes to ensure consistency
    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"
    print(f"Chunked abstracts into {len(nodes)} nodes.")

    # Create vector index and retriever
    print(f"Step 5: Creating vector index and retriever with top_k={top_k}...")
    vector_index = VectorStoreIndex(nodes)
    retriever = vector_index.as_retriever(similarity_top_k=top_k)

    # Generate question-context pairs
    print(f"Step 6: Generating question-context pairs...")
    qa_dataset = generate_question_context_pairs(
        nodes, llm=llm, num_questions_per_chunk=5
    )
    print("Question-context pairs generated successfully.")

    # Save the QA dataset if required
    if save_eval_dataset:
        os.makedirs("./output_evaluation", exist_ok=True)
        qa_dataset.save_json("./output_evaluation/pg_eval_dataset.json")
        print("QA evaluation dataset saved to 'pg_eval_dataset.json'.")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process CSV and generate questions.')
    parser.add_argument('--csv_file_path', type=str, required=True, help='Path to the CSV file.')
    parser.add_argument('--sample_size', type=int, default=100, help='Number of samples to process.')
    parser.add_argument('--model_generate', type=str, default="phi3", help='Model for question generation.')
    parser.add_argument('--model_embedding', type=str, default="BAAI/bge-small-en-v1.5", help='Embedding model name.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results for retriever.')
    parser.add_argument('--cache_dir_embedding', type=str, default="./model", help='Cache directory for embeddings.')
    parser.add_argument('--data_dir', type=str, default="./data", help='Directory for saving abstracts.')
    parser.add_argument('--chunk_size', type=int, default=512, help='Chunk size for splitting abstracts.')
    parser.add_argument('--save_eval_dataset', type=bool, default=True, help='Whether to save the QA dataset.')

    args = parser.parse_args()

    # Call the function with passed arguments
    question_generation(
        csv_file_path=args.csv_file_path,
        sample_size=args.sample_size,
        model_generate=args.model_generate,
        model_embedding=args.model_embedding,
        top_k=args.top_k,
        cache_dir_embedding=args.cache_dir_embedding,
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        save_eval_dataset=args.save_eval_dataset
    )
