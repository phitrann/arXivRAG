from typing import Any
from loguru import logger
import requests
import os
import dotenv


from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

dotenv.load_dotenv()

class KoiLLM(CustomLLM):
    num_output: int = 256
    model_name: str = "llama3.1:8b-instruct"
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    def _stream_wrapper(self, stream_object):
        for chunk in stream_object:
            if chunk:
                text = chunk.decode("UTF-8")
                if text != '<|eot_id|>':
                    yield ""
                yield text


    def APICall(self, call_type: str, prompt: str) -> str:
        if call_type == "stream":
            try:
                response = requests.post(
                    url="http://localhost:8080/stream",
                    json={
                        "prompt": prompt,
                        "max_new_tokens": self.num_output,
                        "temperature": self.temperature,
                        "top_k": self.top_k,
                        "top_p": self.top_p,
                        "length_penalty": -0.1,
                        "repetition_penalty": 1.5,
                    },
                    stream=True,
                )
                stream = self._stream_wrapper(response)
                return stream
            except Exception as e:
                logger.error(f"Error in stream API call: {e}")
        elif call_type == "generate":
            try:
                response = requests.post(
                    url="http://localhost:8080/generate",
                    json={
                        "prompt": prompt,
                        "max_new_tokens": self.num_output,
                        "temperature": self.temperature,
                        "top_k": self.top_k,
                        "top_p": self.top_p,
                        "length_penalty": -0.1,
                        "repetition_penalty": 1.5,
                    },
                    stream=False,
                )
                return response.json().text
            except Exception as e:
                logger.error(f"Error in generate API call: {e}")
        else:
            logger.error("Invalid call type")

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            num_output=self.num_output,
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
