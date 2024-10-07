from typing import List 
from pydantic import BaseModel

import dotenv
import uvicorn
from fastapi import FastAPI
from FlagEmbedding import FlagLLMReranker

dotenv.load_dotenv()
app = FastAPI()

class InputData(BaseModel):
    pairs: List[List]

class OutputData(BaseModel):
    scores: List[float]

reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True, device="cuda") # Setting use_fp16 to True speeds up computation with a slight performance degradation


@app.post("/rerank", response_model=OutputData)
async def generate(input_data: InputData):
    scores = reranker.compute_score(input_data.pairs, normalize=True) 
    return OutputData(scores=scores)

if __name__ == "__main__":
    uvicorn.run(app, port=8083, host="0.0.0.0")
    