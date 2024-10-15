from typing import List
from pydantic import BaseModel

import dotenv
import torch
import uvicorn
from loguru import logger
from fastapi import FastAPI
from transformers import AutoModel, AutoTokenizer

dotenv.load_dotenv()
app = FastAPI()

# ---------- Input & Output Schemas -------------

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: "
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}

class EmbInputData(BaseModel):
    query: str = ""
    key: str = ""
    keys: List[str] = []
    instruction: str = "qa"

class EmbOutputData(BaseModel):
    embedding: List[float] = []
    embeddings: List[List[float]] = []

# ---------- Init models -------------

def init_model(model_name: str):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
emb_model, emb_tokenizer = init_model("BAAI/llm-embedder")
emb_model.to(device)

@app.post("/query_embedding", response_model=EmbOutputData)
async def query_embedding(input_data: EmbInputData):
    logger.info(f"Received request with data: {input_data}")
    instruction_prompt = INSTRUCTIONS[input_data.instruction]["query"]
    query = instruction_prompt + input_data.query
    inputs = emb_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = emb_model(**inputs)
    return EmbOutputData(embedding=outputs.last_hidden_state[:, 0].tolist()[0])

@app.post("/key_embedding", response_model=EmbOutputData)
async def key_embedding(input_data: EmbInputData):
    logger.info(f"Received request with data: {input_data}")
    instruction_prompt = INSTRUCTIONS[input_data.instruction]["key"]
    key = instruction_prompt + input_data.key
    inputs = emb_tokenizer(key, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = emb_model(**inputs)
    return EmbOutputData(embedding=outputs.last_hidden_state[:, 0].tolist()[0])

@app.post("/key_embeddings", response_model=EmbOutputData)
async def key_embeddings(input_data: EmbInputData):
    logger.info(f"Received request with data: {input_data}")
    instruction_prompt = INSTRUCTIONS[input_data.instruction]["key"]
    keys = [instruction_prompt + k for k in input_data.keys]
    inputs = emb_tokenizer(keys, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = emb_model(**inputs)
    return EmbOutputData(embeddings=outputs.last_hidden_state[:, 0].tolist())


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")