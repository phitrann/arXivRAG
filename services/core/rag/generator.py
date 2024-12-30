from typing import List

from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer

from llm import LLMCore
from prompts import qa_prompt
from config import settings

class GeneratorCore:
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
                max_num_tokens=int(settings.MAX_NEW_TOKENS)
            ),
        )
    
    def generate(self, query: str, nodes: List[NodeWithScore]):
        response = self.synth.synthesize(query, nodes=nodes)
        return response.response_gen

if __name__ == "__main__":
    generator = GeneratorCore()
    # stream = generator.generate("Hello world", nodes=[])

    for x in generator.generate("Hello world, What is your name?", nodes=[]):
        print(x, end="", flush=True)
