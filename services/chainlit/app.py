import os
import chainlit as cl
import loguru
import dotenv

from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core import Settings


from utils import VectorDBRetriever
import sys
sys.path.append(os.path.abspath('../'))
from llm_serving.llm import KoiLLM
from llm_serving.embedder import InstructorEmbeddings

dotenv.load_dotenv()

try:
    vector_store = MilvusVectorStore(
        dim=768,
        collection_name="arxiv",
        uri="http://localhost:19530"
    )
except:
    loguru.error("Error in loading vectorstore")

@cl.on_chat_start
async def start():
    # llm =  Ollama(base_url="http://localhost:11434", model="llama3.1", request_timeout=60.0)
    llm = KoiLLM(num_output=256, temperature=0.5, top_k=50, top_p=0.95)
    # embed_model = OllamaEmbedding(model_name="llm-embedder-q4_k_m", base_url="http://localhost:11434",)
    embed_model = InstructorEmbeddings(model_name="BAAI/llm-embedder")
    retriever = VectorDBRetriever(vector_store=vector_store, embed_model=embed_model, query_mode="default", similarity_top_k=2)    

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.context_window = 4096

    synth = get_response_synthesizer(llm=llm, streaming=True)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))

    query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm, response_synthesizer=synth, service_context=service_context)
    
    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine

    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()