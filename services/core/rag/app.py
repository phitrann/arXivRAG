from typing import List, Dict

import uvicorn
from loguru import logger
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_index.core.schema import TextNode, NodeWithScore

from retriever import RetrieverCore
from generator import GeneratorCore 


class LLMInputData(BaseModel):
    query: str = ""


class RAGCore:
    def __init__(self) -> None:
        self.retriever = RetrieverCore()
        self.generator = GeneratorCore()

    # Function to reconstruct TextNode from JSON response
    def _reconstruct_nodes(self, dict_nodes: List[Dict]) -> List[NodeWithScore]:
        nodes = []
        
        # Iterate over the 'nodes' in the JSON response
        for dict_node in dict_nodes:
            node_data = dict_node['node']
            node_score = dict_node['score']

            # Reconstruct the TextNode object
            text_node = TextNode(
                id_=node_data['id_'],
                text=node_data['text'],
                mimetype=node_data['mimetype'],
                start_char_idx=node_data.get('start_char_idx'),
                end_char_idx=node_data.get('end_char_idx'),
                text_template=node_data.get('text_template', '{metadata_str}\n\n{content}'),
                metadata_template=node_data.get('metadata_template', '{key}: {value}'),
                metadata_separator=node_data.get('metadata_seperator', '\n'),
                metadata=node_data.get('extra_info', {}),
                excluded_embed_metadata_keys=node_data.get('excluded_embed_metadata_keys', []),
                excluded_llm_metadata_keys=node_data.get('excluded_llm_metadata_keys', []),
                relationships=node_data.get('relationships', {})
            )
            
            nodes.append(NodeWithScore(node=text_node, score=node_score))
        
        return nodes

    def query(self, query: str):
        dict_nodes = self.retriever.retrieve(query)
        nodes = self._reconstruct_nodes(dict_nodes)

        response = self.generator.generate(query=query, nodes=nodes)

        return response

app = FastAPI()
rag_core = RAGCore()

@app.post("/query")
async def stream_api(input_data: LLMInputData):
    query = input_data.query
    response = rag_core.query(query)
    return StreamingResponse(response)


if __name__ == "__main__":
    uvicorn.run(app, port=8800, host="0.0.0.0")
        

    