"""Chat model and embedder builders.

Centralizing construction here means the Day-4 score-floor/temperature
tuning pass is one function change, not a search-and-replace across every
node that touches an LLM.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=1200,
    )


def build_embedder(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
        dimensions=settings.embed_dimensions,
    )
