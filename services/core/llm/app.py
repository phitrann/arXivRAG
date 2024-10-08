from threading import Thread
from typing import Dict, Any, List
from pydantic import BaseModel, Field

import dotenv
import torch
import uvicorn
from loguru import logger
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import pipeline
from transformers import TextIteratorStreamer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import FlagLLMReranker

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

class LLMInputData(BaseModel):
    messages: List[Dict[str, str]]
    generation_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_new_tokens": 256,
            "early_stopping": True,
        }
    )

class LLMOutputData(BaseModel):
    text: str = ""

class RerankInputData(BaseModel):
    pairs: List[List]

class RerankOutputData(BaseModel):
    scores: List[float]


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
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def init_pipeline(model_name: str):
    llm_pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return llm_pipe, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
llm_pipe, llm_tokenizer = init_pipeline("meta-llama/Meta-Llama-3.1-8B-Instruct")
reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True, device="cuda") # Setting use_fp16 to True speeds up computation with a slight performance degradation

emb_model = AutoModel.from_pretrained("BAAI/llm-embedder")
emb_tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
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

@app.post("/rerank", response_model=RerankOutputData)
async def rerank(input_data: RerankInputData):
    scores = reranker.compute_score(input_data.pairs, normalize=True) 
    return RerankOutputData(scores=scores)

@app.post("/stream")
async def stream_api(input_data: LLMInputData):
    streamer = TextIteratorStreamer(
        llm_tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True
    )

    generation_kwargs = {
        **input_data.generation_params,
        "text_inputs": input_data.messages,
        "streamer": streamer,
    }
    
    thread = Thread(target=llm_pipe, kwargs=generation_kwargs)
    thread.start()
    
    return StreamingResponse(streamer)

@app.post("/generate", response_model=LLMOutputData)
async def generate(input_data: LLMInputData):
    generation_kwargs = {
        **input_data.generation_params,
        "text_inputs": input_data.messages,
    }
    outputs = llm_pipe(
        **generation_kwargs
    )
    response = outputs[0]["generated_text"][-1]
    
    return LLMOutputData(text=response)

if __name__ == "__main__":
    uvicorn.run(app, port=8088, host="0.0.0.0")