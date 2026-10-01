from langchain_ollama import ChatOllama, OllamaEmbeddings


def get_chat_model(base_url: str, model: str) -> ChatOllama:
    """Create the local chat model used by the application."""
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=0.2,
    )


def get_embeddings(base_url: str, model: str) -> OllamaEmbeddings:
    """Create the local embedding model used for retrieval."""
    return OllamaEmbeddings(
        base_url=base_url,
        model=model,
    )
