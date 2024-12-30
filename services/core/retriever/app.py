import os
import sys
import shutil
from typing import List, Optional
from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI
from minio import Minio
from loguru import logger
from llama_index.core.settings import Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.retrievers import QueryFusionRetriever

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from rag.llm import EmbedderCore, RerankerCore

app = FastAPI()

# ---------- Input & Output Schemas -------------
class InputData(BaseModel):
    query: str

class OutputData(BaseModel):
    nodes: List[NodeWithScore]

# ---------- Custom Classes -------------
class VectorDBRetriever(BaseRetriever):
    """Retriever over a Milvus vector store."""

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embed_model: EmbedderCore,
        query_mode: str = "default",
        similarity_top_k: int = 5,
    ) -> None:
        """Init params."""
        # Load pre-trained model and tokenizer
        self._embed_model = embed_model
        self._vector_store = vector_store
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores

class CustomBM25Retriever(BaseRetriever):
    """
    Custom BM25 Retriever that loads BM25 weights from MinIO.
    """

    def __init__(self, minio_client: Minio, tool: str, bucket_name: str="bm25-persistence", similarity_top_k: int=5):
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.tool = tool
        self.similarity_top_k = similarity_top_k

        self.bm25_retriever = self._load_model()

    def _load_model(self) -> BM25Retriever:
        # Create model folder if it does not exist yet
        if not os.path.exists(".models"):
            os.makedirs(".models")

        # Download the BM25 weights from MinIO
        try:
            # List all objects in the specified "folder" (prefix)
            objects = self.minio_client.list_objects(self.bucket_name, prefix=self.tool, recursive=True)

            for obj in objects:
                # Download each object
                object_name = obj.object_name
                file_path = os.path.join(".models", object_name)  # Remove prefix from file path
                
                # Create any subdirectories if necessary
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Download the file
                self.minio_client.fget_object(self.bucket_name, object_name, file_path)
                logger.info(f"Downloaded BM25 weights: {object_name} -> {file_path}")

        except Exception as err:
            logger.error(f"Error occurred when dowloading BM25 weights: {err}")

        # Load model
        model_folder = os.path.join(".models", self.tool) 
        bm25_retriever = BM25Retriever.from_persist_dir(model_folder)
        logger.info("Loaded BM25 Retriever!")

        # Clean downloaded files
        try:
        # Remove the directory and all its contents
            shutil.rmtree(model_folder)
            print(f"Directory '{model_folder}' removed successfully.")
        except Exception as e:
            logger.error(f"Error occurred while removing directory: {e}")

        return bm25_retriever

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self.bm25_retriever.retrieve(query_bundle)[:self.similarity_top_k]

# ---------- API Endpoint -----------
@app.post("/retrieve", response_model=OutputData)
def retrieve(input_data: InputData):
    query = input_data.query


    # -------- MinIO ---------
    # minio_client = Minio(
    #     endpoint=settings.MINIO_URL,
    #     access_key=settings.MINIO_ACCESS_KEY,
    #     secret_key=settings.MINIO_SECRET_KEY,
    #     secure=False,
    # )

    # -------- Embedding ---------
    # embed_model = OllamaEmbedding(model_name="llm-embedder-q4_k_m", base_url="http://localhost:11434",)
    embed_model = EmbedderCore(
        uri=settings.EMB_SERVING_URL
    )

    # -------- Reranker ---------
    reranker = RerankerCore(
        uri=settings.RERANK_SERVING_URL, rerank_top_k=settings.RERANK_TOP_K
    )

    # -------- Retriever ---------
    retriever_list = []
    vector_store = MilvusVectorStore(
        dim=settings.EMBEDDING_DIM,
        collection_name=settings.VECTOR_STORE_COLLECTION,
        uri=settings.MILVUS_URL
    )

    retriever_list.append(
        VectorDBRetriever(
            vector_store=vector_store,
            embed_model=embed_model,
            query_mode="default",
            similarity_top_k=settings.SIMILARITY_TOP_K
        )
    )

    # retriever_list.append(
    #     CustomBM25Retriever(
    #         minio_client=minio_client,
    #         prefix="test",
    #         bucket_name="bm25-persistence",
    #         similarity_top_k=settings.SIMILARITY_TOP_K,
    #     )
    # )
    Settings.llm = None
    Settings.embedder = embed_model

    retriever = QueryFusionRetriever(
        retriever_list,
        similarity_top_k=settings.FUSION_TOP_K,
        num_queries=settings.NUM_QUERY,
        use_async=True,
        verbose=False,
    )
    nodes = retriever.retrieve(query)

    # Rerank
    nodes = reranker.rerank(query=query, nodes=nodes)

    return OutputData(nodes=nodes)

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")