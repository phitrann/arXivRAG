from typing import List, Optional
from loguru import logger

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.vector_stores import VectorStoreQuery

from utils.embedder import InstructorEmbeddings

class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embed_model: InstructorEmbeddings,
        query_mode: str = "default",
        similarity_top_k: int = 2,
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
                logger.info(f"Node: {node}, Score: {score}")
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

if __name__=="__main__":
    import os
    os.environ["NO_PROXY"]="localhost"
    vector_store = MilvusVectorStore(
        dim=768,
        collection_name="arxiv_test",
        uri="http://localhost:19530",
    )
    embed_model = InstructorEmbeddings(
        uri="http://localhost:8081",
        model_name="BAAI/llm-embedder",
    )

    print(embed_model._get_query_embedding("What is Retrieval Augment Generation?"))

    retriever = VectorDBRetriever(vector_store=vector_store, embed_model=embed_model)
    nodes = retriever.retrieve("What is Retrieval Augmented?")
    print(nodes)