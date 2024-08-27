import os
import chainlit as cl
from loguru import logger
import dotenv

from transformers import AutoTokenizer
from llama_index.core.callbacks import CallbackManager
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.settings import Settings

from utils import VectorDBRetriever
import sys
sys.path.append(os.path.abspath('../'))
from llm_serving.llm import KoiLLM
from llm_serving.embedder import InstructorEmbeddings
from chat_history import ChatHistory

dotenv.load_dotenv()

chat_history = ChatHistory()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

system_prompt = """
You are an expert research assistant specializing in machine learning and artificial intelligence. 
Your task involves processing a collection of recent research paper abstracts. Perform the following:

1. Provide a coherent and comprehensive summary of these abstracts tailored to the user's specific query.
2. Identify and summarize the key themes, methodologies, and implications of the research findings, highlighting any significant trends or innovations.
"""

try:
    vector_store = MilvusVectorStore(
        dim=768,
        collection_name="arxiv",
        uri="http://172.16.87.76:19530"
    )
except:
    logger.error("Error in loading vectorstore")

@cl.on_chat_start
async def start():
    llm = KoiLLM(num_output=256, temperature=0.5, top_k=50, top_p=0.95)
    embed_model = InstructorEmbeddings(model_name="BAAI/llm-embedder")
    retriever = VectorDBRetriever(vector_store=vector_store, embed_model=embed_model, query_mode="default", similarity_top_k=2)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.context_window = 4096

    synth = get_response_synthesizer(llm=llm, streaming=True)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        response_synthesizer=synth
    )

    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine
    
    prompt_history = chat_history.get_chat_history(limit=3)
    prompt = chat_history.format_chat_history(tokenizer, 
                                              system_prompt, 
                                              prompt_history, 
                                              message.content)
    logger.info(f"{prompt = }")

    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(query_engine.query)(prompt)
    
    response_content = ""
    for token in res.response_gen:
        response_content += token
        await msg.stream_token(token)

    # Save chat history
    chat_history.save_chat(message.content, response_content)
    await msg.send()