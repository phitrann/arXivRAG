import os
from threading import Thread
from typing import Dict, Any, List
from pydantic import BaseModel, Field

import dotenv
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import pipeline
from transformers import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

dotenv.load_dotenv()
app = FastAPI()

class InputData(BaseModel):
    messages: List[Dict[str, str]]
    generation_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_new_tokens": 256,
            # "temperature": 0.8,
            # "top_k": 50,
            # "top_p": 0.95,
            # "length_penalty": -0.1,
            # "repetition_penalty": 1.5,
            # "num_beams": 1,
            # "do_sample": True ,
            "early_stopping": True,
        }
    )

class OutputData(BaseModel):
    text: str = ""

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
# model, tokenizer = init_model("meta-llama/Meta-Llama-3.1-8B-Instruct")
# model, tokenizer = init_model("Qwen/Qwen2.5-7B-Instruct")
# model.to(device)
llm_pipe, tokenizer = init_pipeline("meta-llama/Meta-Llama-3.1-8B-Instruct")


@app.post("/stream")
async def stream_api(input_data: InputData):
    streamer = TextIteratorStreamer(
        tokenizer, 
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

@app.post("/generate", response_model=OutputData)
async def generate(input_data: InputData):
    generation_kwargs = {
        **input_data.generation_params,
        "text_inputs": input_data.messages,
    }
    outputs = llm_pipe(
        **generation_kwargs
    )
    response = outputs[0]["generated_text"][-1]
    
    return OutputData(text=response)

if __name__ == "__main__":
    uvicorn.run(app, port=8088, host="0.0.0.0")
    