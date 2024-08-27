import os
import torch
import dotenv
from typing import Iterable, Dict, Any
from threading import Thread
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

dotenv.load_dotenv()
app = FastAPI()

class InputData(BaseModel):
    prompt: str
    generation_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_new_tokens": 256,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "length_penalty": -0.1,
            "repetition_penalty": 1.5,
            "num_beams": 3,
            "do_sample": True
        }
    )

class OutputData(BaseModel):
    text: str = ""

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model.to(device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

@app.post("/stream")
async def stream_api(input_data: InputData):
    inputs = tokenizer([input_data.prompt], return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True
    )
    
    generation_kwargs = {
        **input_data.generation_params,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "streamer": streamer,
        "eos_token_id": terminators,
    }
    
    # Remove length_penalty if num_beams is 1 or not specified
    if generation_kwargs.get('num_beams', 1) == 1:
        generation_kwargs.pop('length_penalty', None)
    
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
    
    generation_kwargs = {
        **input_data.generation_params,
        "input_ids": inputs.to(model.device),
        "eos_token_id": terminators,
    }
    
    tokens = model.generate(**generation_kwargs)
    response = tokenizer.decode(tokens[0][prompt_length:], skip_special_tokens=True)
    
    return OutputData(text=response)

if __name__ == "__main__":
    uvicorn.run(app, port=8088, host="0.0.0.0")