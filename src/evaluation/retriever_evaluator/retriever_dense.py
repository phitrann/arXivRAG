import nest_asyncio
import os
import argparse
import asyncio
import pandas as pd
import Stemmer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from dotenv import load_dotenv

# Apply nest_asyncio to avoid asyncio issues in notebooks
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

class VectorStoreRetrieverSystem:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.chunk_size = args.chunk_size
        self.qa_file = args.qa_file
        self.llm_model_name = args.llm_model_name  # LLM model name
        self.embedding_model_name = args.embedding_model_name  # Embedding model name
        self.top_k = args.top_k  # Top K value for similarity retrieval
        self.metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
        
        self.llm, self.embed_model = None, None
        self.vector_retriever = None
        self.documents, self.nodes = [], []

    def initialize_models(self):
        """Initialize LLM and embedding models."""
        print(f"Initializing models... LLM: {self.llm_model_name}, Embedding: {self.embedding_model_name}")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        self.llm = Ollama(base_url=ollama_base_url, model=self.llm_model_name, request_timeout=120.0)
        self.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name, cache_folder="./model")

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        print(f"LLM and embedding models initialized. LLM: {self.llm_model_name}, Embedding: {self.embedding_model_name}")

    def load_and_chunk_documents(self):
        """Load documents and chunk them into nodes."""
        print(f"Loading and chunking documents from {self.data_dir}...")
        reader = SimpleDirectoryReader(self.data_dir)
        self.documents = reader.load_data()
        node_parser = SentenceSplitter(chunk_size=self.chunk_size)
        self.nodes = node_parser.get_nodes_from_documents(self.documents)

        for idx, node in enumerate(self.nodes):
            node.id_ = f"node_{idx}"
        
        print(f"Loaded {len(self.documents)} documents and chunked into {len(self.nodes)} nodes.")

    def initialize_vector_store_retriever(self):
        """Initialize the vector store retriever using Milvus."""
        print("Initializing vector store retriever with Milvus vector store...")
        # Milvus vector store
        milvus_uri = os.getenv("MILVUS_URI")
        embedding_dim= os.getenv("EMBEDDING_DIM")
        vector_store = MilvusVectorStore(
            uri=milvus_uri, dim=embedding_dim, overwrite=True
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes=self.nodes, storage_context=storage_context)

        # Set vector retriever with the specified top_k
        self.vector_retriever = vector_index.as_retriever(similarity_top_k=self.top_k)
        print(f"Vector store retriever initialized with top_k={self.top_k}.")

    def display_retrieved_nodes(self, query):
        """Retrieve and display nodes for a given query."""
        print(f"Retrieving top {self.top_k} nodes for query: {query}")
        retrieved_nodes = self.vector_retriever.retrieve(query)
        for node in retrieved_nodes:
            display_source_node(node, source_length=5000)

    def load_qa_dataset(self):
        """Load QA dataset from a JSON file."""
        print(f"Loading QA dataset from {self.qa_file}...")
        qa_dataset = EmbeddingQAFinetuneDataset.from_json(self.qa_file)
        print(f"QA dataset loaded with {len(qa_dataset.queries)} queries.")
        return qa_dataset

    async def evaluate_retriever(self, qa_dataset):
        """Perform evaluation using the vector-based retriever."""
        print(f"Evaluating vector store retriever on {len(qa_dataset.queries)} queries...")
        retriever_evaluator = RetrieverEvaluator.from_metric_names(self.metrics, retriever=self.vector_retriever)

        # Try it on a sample query
        sample_id, sample_query = list(qa_dataset.queries.items())[0]
        sample_expected = qa_dataset.relevant_docs[sample_id]
        eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
        print("Sample evaluation result:", eval_result)

        # Evaluate on the full dataset (await coroutine)
        eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
        return eval_results

    def display_results(self, name, eval_results):
        """Display evaluation results."""
        metric_dicts = [eval_result.metric_vals_dict for eval_result in eval_results]
        full_df = pd.DataFrame(metric_dicts)

        columns = {
            "retrievers": [name],
            **{k: [full_df[k].mean()] for k in self.metrics},
        }

        metric_df = pd.DataFrame(columns)
        print(f"Evaluation results for {name}:")
        print(metric_df)

    async def run(self):
        """Run the entire workflow."""
        self.initialize_models()
        self.load_and_chunk_documents()
        self.initialize_vector_store_retriever()

        # Retrieve nodes for a specific query (example)
        self.display_retrieved_nodes("A robust 3D triangular?")

        # Load QA dataset for evaluation
        qa_dataset = self.load_qa_dataset()

        # Perform evaluation and display results
        eval_results = await self.evaluate_retriever(qa_dataset)
        self.display_results(f"vector store eval {self.top_k} ", eval_results)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Vector Store Retriever System with Milvus.")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Directory containing data files.")
    parser.add_argument("--chunk_size", type=int, default=512, help="Size of document chunks.")
    parser.add_argument("--qa_file", type=str, default="./output_evaluation/pg_eval_dataset.json", help="QA dataset JSON file path.")
    parser.add_argument("--llm_model_name", type=str, default="phi3", help="LLM model name to use for the retrieval system.")
    parser.add_argument("--embedding_model_name", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model name for vector-based retrieval.")
    parser.add_argument("--top_k", type=int, default=10, help="Top K results for similarity retrieval.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retriever_system = VectorStoreRetrieverSystem(args)
    asyncio.run(retriever_system.run())
