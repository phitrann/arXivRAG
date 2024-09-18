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
from llm_serving.llm import LLMHandler
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
except Exception as e:
    logger.error(f"Error in loading vectorstore: {e}")

@cl.on_chat_start
async def start():
    llm = LLMHandler(num_output=256, temperature=0.5, top_k=50, top_p=0.95)
    embed_model = InstructorEmbeddings(model_name="BAAI/llm-embedder")
    retriever = VectorDBRetriever(vector_store=vector_store, embed_model=embed_model, query_mode="default", similarity_top_k=2)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.context_window = 4096

    synth = get_response_synthesizer(llm=llm, streaming=True)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        response_synthesizer=synth,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()])  # Add callback manager for progress tracking
    )

    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Assistant", content="Hello! I'm an AI assistant. How may I help you?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    
    prompt_history = chat_history.get_chat_history(limit=3)
    prompt = chat_history.format_chat_history(tokenizer, 
                                              system_prompt, 
                                              prompt_history, 
                                              message.content)
    # logger.info(f"{prompt = }")

    msg = cl.Message(content="", author="Assistant")

    # Custom callback handler for tracking progress
    class PostMessageHandler(CallbackManager):
        def __init__(self, msg: cl.Message):
            super().__init__()
            self.msg = msg
            self.sources = set()

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            # Track and display sources from retrieved documents
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                self.sources.add(source)
            sources_text = "\n".join(self.sources)
            if sources_text:
                self.msg.elements.append(cl.Text(name="Sources", content=sources_text, display="inline"))

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            # Finalize message after LLM response is complete
            if self.msg.elements:
                sources_text = "\n".join(self.sources)
                self.msg.elements.append(cl.Text(name="Sources", content=sources_text, display="inline"))

    # Run the query and track progress with the custom callback handler
    callback_handler = PostMessageHandler(msg)

    res = await cl.make_async(query_engine.query)(prompt)

    response_content = ""
    for token in res.response_gen:
        response_content += token
        await msg.stream_token(token)

    # Save chat history
    # chat_history.save_chat(message.content, response_content)
    await msg.send()
