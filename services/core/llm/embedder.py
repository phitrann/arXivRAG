from typing import Any, List
import requests
import dotenv
from time import sleep
from loguru import logger
from llama_index.core.embeddings import BaseEmbedding

class InstructorEmbeddings(BaseEmbedding):
    model_name: str = "BAAI/llm-embedder"
    uri: str = "http://172.16.87.76:8081"

    def __init__(
        self,
        model_name: str = "BAAI/llm-embedder",
        uri: str = "http://172.16.87.76:8081",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.uri = uri

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    def _APICall(self, call_type: str, embedding_type: str, prompt: str = "", prompts: List = []) -> str:
        # logger.info(f"Calling {call_type} API with prompt: {prompt}")
        key_val = "key" if call_type == "key_embedding" else "keys"
        key_val = "query" if call_type == "query_embedding" else key_val

        # Retry the request if it fails
        for _ in range(3):
            try:
                response = requests.post(
                    f"{self.uri}/{call_type}",
                    json={
                        key_val: prompt if len(prompt) else prompts, 
                        "instruction": "qa",
                    },
                )

                if response.status_code != 200:
                    logger.error(f"Error in API call: {response.text}")
                    # sleep 1s
                    sleep(1)

                    continue

                return response.json()[embedding_type]
            except Exception as e:
                logger.error(f"Error in API call: {e}")
                continue

    
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
            call_type="key_embeddings", embedding_type="embeddings", prompts=texts
        )
        return response

if __name__ == "__main__":
    import os
    os.environ["NO_PROXY"] = "172.16.87.76"
    embedder = InstructorEmbeddings(uri="http://172.16.87.76:8081",model_name="BAAI/llm-embedder")
    query = "Encode this query and context for searching relevant passages: "
    key = "Encode this passage for retrieval: "
    # print(embedder._get_query_embedding(query))
    # print(embedder._get_text_embedding(key))
    print(embedder._get_text_embeddings([key, key]))