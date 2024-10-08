from typing import List
import requests

from llama_index.core.schema import NodeWithScore

from config import settings


class RetrieverCore:
    def retrieve(self, query: str) -> List[NodeWithScore]:
        response = requests.post(
            url=settings.RETRIEVER_URL,
            json={
                "query": query
            },
            timeout=10
        )

        return response.json()["nodes"]

        

