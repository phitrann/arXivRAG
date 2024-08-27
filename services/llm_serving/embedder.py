from typing import Any, List
import requests
import dotenv
from loguru import logger
from llama_index.core.embeddings import BaseEmbedding

import os

dotenv.load_dotenv()

class InstructorEmbeddings(BaseEmbedding):
    def __init__(
        self,
        model_name: str = "BAAI/llm-embedder",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    def _APICall(self, call_type: str, embedding_type: str, prompt: str = "", prompts: List = []) -> str:
        logger.info(f"Calling {call_type} API with prompt: {prompt}")
        try:
            key_val = "key" if call_type == "key_embedding" else "keys"
            key_val = "query" if call_type == "query_embedding" else key_val
            response = requests.post(
                f"http://172.16.87.76:8081/{call_type}",
                json={
                    key_val: prompt if len(prompt) else prompts, 
                    "instruction": "qa",
                },
            )
        except Exception as e:
            logger.error(f"Error in {call_type} API call: {e}")
        return response.json()[embedding_type]
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        response = self._APICall(
            call_type="query_embedding", embedding_type="embedding", prompt=query
        )
        return response

    def _get_text_embedding(self, text: str) -> List[float]:
        response = self._APICall(
            call_type="key_embedding", embedding_type="embedding", prompt=text
        )
        return response

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self._APICall(
            call_type="key_embeddings", embedding_type="embeddings", prompt=texts
        )
        return response

if __name__ == "__main__":
    embedder = InstructorEmbeddings()
    query = "Encode this query and context for searching relevant passages: "
    key = "Encode this passage for retrieval: "
    print(embedder._get_query_embedding(query))
    # print(embedder._get_text_embedding(key))
    # print(embedder._get_text_embeddings([key, key]))