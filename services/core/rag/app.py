import dotenv
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from retriever import RetrieverCore
from generator import GeneratorCore 

dotenv.load_dotenv()

class LLMInputData(BaseModel):
    query: str = ""


class RAGCore:
    def __init__(self) -> None:
        self.retriever = RetrieverCore()
        self.generator = GeneratorCore()

    def query(self, query: str):
        nodes = self.retriever.retrieve(query)
        response = self.generator.generate(query=query, nodes=nodes)

        return response

app = FastAPI()
rag_core = RAGCore()


@app.post("/query")
async def stream_api(input_data: LLMInputData):
    query = await input_data.query
    response = rag_core.query(query)
    return StreamingResponse(response)


if __name__ == "__main__":
    uvicorn.run(app, port=8800, host="0.0.0.0")
        

    