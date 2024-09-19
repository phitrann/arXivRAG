import os
from loguru import logger

from transformers import AutoTokenizer
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.settings import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext

import chainlit as cl
from chainlit.input_widget import Switch

from configs import cfg
from utils.chat_history import ChatHistory
from utils.embedder import InstructorEmbeddings
from utils.llm import LLMHandler
from utils.retriever import VectorDBRetriever


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
        uri=cfg.get("milvus")["uri"],
    )
except Exception as e:
    logger.error(f"Error in loading Vector store: {e}")

try:
    llm_setting = cfg.get("llm_settings")

    llm = LLMHandler(
        uri=cfg.get("llm")["uri"],
        model_name=cfg.get("llm")["model_name"],
        max_new_tokens=llm_setting["max_new_tokens"],
        temperature=llm_setting["temperature"],
        top_k=llm_setting["top_k"],
        top_p=llm_setting["top_p"],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.get("llm").get("model_name"))

    embed_model = InstructorEmbeddings(
        uri=cfg.get("embedding")["uri"],
        model_name=cfg.get("embedding")["model_name"],
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.context_window = cfg.get("llm_settings")["context_window"]
except Exception as e:
    logger.error(f"Error in loading LLM: {e}")

try:
    retriever_settings = cfg.get("retriever")
    retriever = VectorDBRetriever(
        vector_store=vector_store,
        embed_model=embed_model,
        query_mode="default",
        similarity_top_k=retriever_settings["similarity_top_k"],
    )
except Exception as e:
    logger.error(f"Error in loading Utilities: {e}")


# @cl.password_auth_callback
# def auth_callback(username: str, password: str):
#     # Fetch the user matching username from your database
#     # and compare the hashed password with the value stored in the database
#     if (username, password) == ("admin", "admin"):
#         return cl.User(
#             identifier="admin", metadata={"role": "admin", "provider": "credentials"}
#         )
#     else:
#         return None


@cl.on_chat_start
async def start():
    synth = get_response_synthesizer(llm=llm, streaming=True)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))

    query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm, response_synthesizer=synth, service_context=service_context)

    cl.user_session.set("query_engine", query_engine)

    # Chat history
    # chat_history_settings = cfg.get("chat_history")
    # mongodb_settings = cfg.get("mongodb")
    # chat_history = ChatHistory(
    #     uri=f"mongodb://{mongodb_settings.get('username')}:{mongodb_settings.get('password')}@{mongodb_settings.get('uri')}",
    #     db_name=chat_history_settings.get("database_name"),
    #     collection_name=chat_history_settings.get("collection_name"),
    #     tokenizer=tokenizer,
    #     top_k=2,
    # )
    # settings = await cl.ChatSettings(
    #     [
    #         Switch(id="Chat History", label="Use Chat History", initial=False),
    #     ]
    # ).send()
    # value = settings["Chat History"]
    # cl.user_session.set("chat_history_engine", chat_history)
    # cl.user_session.set("chat_history", value)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    msg = cl.Message(content="", author="Assistant")

    # Use history
    # history_flag = cl.user_session.get("chat_history")
    # logger.info(f"History Flag: {history_flag}")

    # response = None
    # if history_flag:
    #     chat_history = cl.user_session.get("chat_history_engine")
    #     prompt_history = chat_history.get_chat_history()
    #     prompt = chat_history.format_chat_history(
    #         system_prompt, prompt_history, message.content
    #     )
    #     logger.info(f"{prompt = }")
    #     response = await cl.make_async(query_engine.query)(prompt)
    #     response_content = ""
    #     for token in response.response_gen:
    #         response_content += token
    #         await msg.stream_token(token)
    #     chat_history.save_chat(message.content, response_content)

    response = await cl.make_async(query_engine.query)(message.content) 
    response_content = ""
    for token in response.response_gen:
        response_content += token
        await msg.stream_token(token)

    await msg.send()
