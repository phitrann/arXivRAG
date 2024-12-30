import requests
import chainlit as cl
from loguru import logger

from config import settings

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

    if response.status_code != 200:
        logger.error(f"Response with status code: {response.status_code}")

    stream = response.iter_content(chunk_size=5, decode_unicode=True)

    for chunk in stream:
        # response_content += token
        await msg.stream_token(chunk.decode("utf-8", errors="ignore"))

    await msg.send()
