from typing import List

from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer

from llm import LLMCore
from prompts import qa_prompt
from config import settings

class SynthesizerCore:
    def __init__(self):
        self.llm = LLMCore(
            uri=settings.LLM_SERVING_URL,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            request_timeout=30
        )

        # -------- Setting LlamaIndex ---------
        self.synth = get_response_synthesizer(
            llm=self.llm,
            streaming=True,
            text_qa_template=qa_prompt.partial_format(
                max_num_tokens=int(settings["max_new_tokens"])
            ),
        )
    
    async def synthesize(self, query: str, nodes: List[NodeWithScore]) -> str:
        response = await self.synth.asynthesize(query, nodes)
        return response.response_gen

