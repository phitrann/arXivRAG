from typing import Any
from loguru import logger
import requests

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback


class LLMHandler(CustomLLM):
    uri: str = "http://172.16.87.76:8088"
    max_new_tokens: int = 128
    model_name: str = "llama3.1:8b-instruct"
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    def _stream_wrapper(self, stream_object):
        for chunk in stream_object:
            if chunk:
                text = chunk.decode("utf-8", errors="ignore")
                yield text

    def APICall(self, call_type: str, prompt: str) -> str:
        logger.info(f"Calling LLM API with type: {call_type}")
        generation_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "length_penalty": -0.1,
            "repetition_penalty": 1.5,
            "num_beams": 1,
            "do_sample": False
        }

        payload = {
            "prompt": prompt,
            "generation_params": generation_params
        }

        if call_type == "stream":
            try:
                response = requests.post(
                    url=f"{self.uri}/stream",
                    json=payload,
                    stream=True,
                )
                stream = self._stream_wrapper(response.iter_content(chunk_size=1))
                return stream
            except Exception as e:
                logger.error(f"Error in stream API call: {e}")
        elif call_type == "generate":
            try:
                response = requests.post(
                    url=f"{self.uri}/generate",
                    json=payload,
                )
                return response.json()["text"]
            except Exception as e:
                logger.error(f"Error in generate API call: {e}")
        else:
            logger.error("Invalid call type")

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
        response = self.APICall("generate", prompt)
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        itr_response = self.APICall("stream", prompt)
        response = ""
        for token in itr_response:
            response += token
            yield CompletionResponse(text=response, delta=token)