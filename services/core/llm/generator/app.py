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
from transformers import  AutoTokenizer

dotenv.load_dotenv()
app = FastAPI()

# ---------- Input & Output Schemas -------------
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

# ---------- Init models -------------

def init_pipeline(model_name: str):
    llm_pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return llm_pipe, tokenizer

device = "cuda:1" if torch.cuda.is_available() else "cpu"
llm_pipe, llm_tokenizer = init_pipeline("meta-llama/Meta-Llama-3.1-8B-Instruct")

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
    uvicorn.run(app, port=8001, host="0.0.0.0")