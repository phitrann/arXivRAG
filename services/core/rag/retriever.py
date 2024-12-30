from typing import List
import requests

from llama_index.core.schema import NodeWithScore

from config import settings


class RetrieverCore:
    def retrieve(self, query: str) -> List[NodeWithScore]:
        response = requests.post(
            url=f"{settings.RETRIEVER_URL}/retrieve",
            json={
                "query": query
            },
            timeout=10
        )

        return response.json()["nodes"]

if __name__ == "__main__":
    retriever = RetrieverCore()
    print(retriever.retrieve("What is diffusion model?"))
