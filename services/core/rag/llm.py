import os
import sys
from time import sleep
from loguru import logger
from typing import Any, List, Dict
import requests

from loguru import logger
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.embeddings import BaseEmbedding

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RerankerCore:
    """
    Reranks a list of nodes based on their similarity to a query.
    """
    
    def __init__(self, uri: str, rerank_top_k: int = 2):
        self.uri = uri
        self.rerank_top_k = rerank_top_k

    def rerank(self, query: str, nodes: List[NodeWithScore]):
        pairs = [[query, node.text] for node in nodes]
        response = requests.post(
            url=f"{self.uri}/rerank", json={"pairs": pairs}, timeout=10
        )

        scores = response.json()["scores"]
        for node, score in zip(nodes, scores):
            node.score = score

        sorted_nodes = sorted(nodes, key=lambda node: node.score, reverse=True)

        return sorted_nodes[: self.rerank_top_k]

class EmbedderCore(BaseEmbedding):
    """
    Custom Embedding class that call API to get the embeddings.
    """

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
                    timeout=5
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

class LLMCore(CustomLLM):
    """
    Custom LLM class that calls the LLM API.
    """

    uri: str = "http://172.16.87.76:8088"
    max_new_tokens: int = 128
    model_name: str = "llama3.1:8b-instruct"
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    num_beams: int = 3
    do_sample: bool = True
    request_timeout: int = 30

    # def _stream_wrapper(self, stream_object):
    #     for chunk in stream_object:
    #         if chunk:
    #             text = chunk.decode("utf-8", errors="ignore")
    #             yield text

    def callAPI(self, call_type: str, messages: str) -> str:
        logger.info(f"Calling LLM API with type: {call_type}")
        logger.info(f"Message: {messages}")
        generation_params = {
            "max_new_tokens": self.max_new_tokens,
            "early_stopping": True,
        }

        payload = {"messages": messages, "generation_params": generation_params}

        if call_type == "stream":
            try:
                response = requests.post(
                    url=f"{self.uri}/stream",
                    json=payload,
                    stream=True,
                    timeout=self.request_timeout,
                )
                stream = response.iter_content(chunk_size=5, decode_unicode=True)
                return stream
            except Exception as e:
                logger.error(f"Error in stream API call: {e}")
        elif call_type == "generate":
            try:
                response = requests.post(
                    url=f"{self.uri}/generate",
                    json=payload,
                    timeout=self.request_timeout,
                )
                return response.json()["text"]
            except Exception as e:
                logger.error(f"Error in generate API call: {e}")
        else:
            logger.error("Invalid call type")

    def _text2msg(self, text: str) -> List[Dict]:
        lines = text.strip().split("\n")
        user = lines[-1]
        system = lines[:-1]

        return [
            {"role": "system", "content": "" + "\n".join(system)},
            {"role": "user", "content": user},
        ]

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = self._text2msg(prompt)
        response = self.callAPI("generate", messages)
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        messages = self._text2msg(prompt)
        itr_response = self.callAPI("stream", messages)
        response = ""
        for token in itr_response:
            response += token
            yield CompletionResponse(text=response, delta=token)
