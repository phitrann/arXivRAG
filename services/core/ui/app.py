import os
import requests
from loguru import logger

from llama_index.core.settings import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext

import chainlit as cl
from chainlit.input_widget import Switch


from configs import settings

@cl.on_chat_start
async def start():
    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="", author="Assistant")


    response = requests.post(
        url=f"{settings.RAG_CORE_URL}/query",
        json={
            "query": message.content
        },
        stream=True,
        timeout=60,
    )
    stream = response.iter_content(chunk_size=5, decode_unicode=True)

    for token in stream:
        # response_content += token
        await msg.stream_token(token)

    await msg.send()
