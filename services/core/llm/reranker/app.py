from typing import List
from pydantic import BaseModel
from contextlib import asynccontextmanager

import dotenv
import uvicorn
from fastapi import FastAPI
from loguru import logger
from FlagEmbedding import FlagLLMReranker

dotenv.load_dotenv()

# ---------- Input & Output Schemas -------------
class RerankInputData(BaseModel):
    pairs: List[List]

class RerankOutputData(BaseModel):
    scores: List[float]
# ---------- Init models -------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global reranker

    # Initialize models
    logger.info('Initializing the Reranker...')
    reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True, device="cuda:0") # Setting use_fp16 to True speeds up computation with a slight performance degradation

    # Clean up the model
    yield
    del reranker 

app = FastAPI(lifespan=lifespan)


@app.post("/rerank", response_model=RerankOutputData)
async def rerank(input_data: RerankInputData):
    logger.info(f"Rerank Request: {input_data.pairs}")
    scores = reranker.compute_score(input_data.pairs, normalize=True) 
    return RerankOutputData(scores=scores)

if __name__ == "__main__":
    uvicorn.run(app, port=8002, host="0.0.0.0")