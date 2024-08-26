import os
import torch
import dotenv
from typing import Iterable
from threading import Thread

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

dotenv.load_dotenv()

app = FastAPI()


class InputData(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    length_penalty: float = -0.1
    repetition_penalty: float = 1.5
    num_beams: int = 3


class OutputData(BaseModel):
    text: str = ""


device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model.to(device)


@app.post("/stream")
async def stream_api(input_data: InputData):
    inputs = tokenizer([input_data.prompt], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=input_data.max_new_tokens,
        temperature=input_data.temperature,
        top_k=input_data.top_k,
        top_p=input_data.top_p,
        length_penalty=input_data.length_penalty,
        repetition_penalty=input_data.repetition_penalty,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return StreamingResponse(streamer)


@app.post("/generate", response_model=OutputData)
async def generate(input_data: InputData):
    prompt = [{"role": "user", "content": input_data.prompt}]
    inputs = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=False, return_tensors="pt"
    )
    prompt_length = inputs[0].shape[0]
    tokens = model.generate(
        inputs.to(model.device), 
        max_new_tokens=input_data.max_new_tokens, 
        temperature=input_data.temperature, 
        do_sample=True,
        num_beams=input_data.num_beams,
        top_k=input_data.top_k,
        top_p=input_data.top_p,
        length_penalty=input_data.length_penalty,
        repetition_penalty=input_data.repetition_penalty,
    )
    response = tokenizer.decode(tokens[0][prompt_length:], skip_special_tokens=True)
    return OutputData(text=response)


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")
