"""The chat model. `get_llm()` is the one place to swap model or provider."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.ingestion.embedder import call_ollama

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
NUM_CTX = 4096  # tokens the model sees per request (prompt + answer); src/chat.py sizes the context to fit


def get_llm() -> BaseChatModel:
    # reasoning=False: qwen3 skips its "thinking" step, so no hidden reasoning reaches the answer.
    return ChatOllama(model=LLM_MODEL, temperature=0, reasoning=False, num_ctx=NUM_CTX)


def generate(messages: list[tuple[str, str]], llm: BaseChatModel | None = None) -> str:
    llm = llm or get_llm()
    response = call_ollama(llm.invoke, messages, model=LLM_MODEL)
    return response.text.strip()  # .text joins content parts into plain text
