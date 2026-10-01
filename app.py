import hashlib
from pathlib import Path

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag import build_vector_store, format_sources
from llm import get_chat_model, get_embeddings

st.set_page_config(
    page_title="Local LangChain AI Assistant",
    layout="wide",
)

SYSTEM_PROMPT = """You are a helpful local AI assistant.

You are running locally through Ollama and are being used inside a Streamlit application.

When document context is provided:
- Answer using the supplied context first.
- If the context does not contain enough information, clearly say so.
- Do not invent citations or page numbers.
- Keep answers concise but technically useful.

When no document context is provided:
- Answer normally using your model knowledge.
- Make uncertainty explicit when appropriate.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """Conversation so far:
{history}

Document context:
{context}

User question:
{question}

Answer the question. When document context is available, ground the answer in it.
""",
        ),
    ]
)

st.title("Local LangChain + Ollama AI Assistant")
st.caption("Streamlit • LangChain • Ollama • Local RAG • No cloud API key required")

with st.sidebar:
    st.header("Configuration")
    ollama_url = st.text_input(
        "Ollama URL",
        value="http://localhost:11434",
        help="Default Ollama server address.",
    )
    model_name = st.text_input(
        "Chat model",
        value="llama3.2:3b",
        help="Change this to any model you have pulled in Ollama.",
    )
    embedding_name = st.text_input(
        "Embedding model",
        value="nomic-embed-text",
        help="Used to create document/query embeddings.",
    )
    top_k = st.slider("Retrieved chunks", 2, 8, 4)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        """
**Architecture**

`Streamlit → LangChain → Ollama`

For document Q&A:

`PDF/Text → chunks → Ollama embeddings → vector store → retriever → prompt → Ollama`
"""
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_signature" not in st.session_state:
    st.session_state.document_signature = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

@st.cache_resource(show_spinner=False)
def load_chat_model(base_url: str, model: str):
    return get_chat_model(base_url=base_url, model=model)

@st.cache_resource(show_spinner=False)
def load_embeddings(base_url: str, model: str):
    return get_embeddings(base_url=base_url, model=model)

chat_model = None
embeddings = None

try:
    chat_model = load_chat_model(ollama_url, model_name)
    embeddings = load_embeddings(ollama_url, embedding_name)
except Exception as exc:
    st.error(
        "Could not initialize the Ollama-backed LangChain components.\n\n"
        f"Details: {exc}"
    )
    st.info(
        "Make sure Ollama is running and that the selected models have been pulled. "
        "See README.md for setup commands."
    )
    st.stop()

st.subheader("1. Add knowledge")
uploaded = st.file_uploader(
    "Upload a PDF, TXT, or Markdown file",
    type=["pdf", "txt", "md"],
    accept_multiple_files=False,
)

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    signature = hashlib.sha256(file_bytes).hexdigest()

    if signature != st.session_state.document_signature:
        with st.spinner("Reading, chunking, and embedding your document..."):
            try:
                store = build_vector_store(
                    file_name=uploaded.name,
                    file_bytes=file_bytes,
                    embeddings=embeddings,
                )
                st.session_state.vector_store = store
                st.session_state.document_signature = signature
                st.session_state.document_name = uploaded.name
                st.success(
                    f"Indexed **{uploaded.name}** successfully."
                )
            except Exception as exc:
                st.session_state.vector_store = None
                st.session_state.document_signature = None
                st.session_state.document_name = None
                st.error(f"Could not process the document: {exc}")

if st.session_state.document_name:
    st.info(f"Active knowledge source: **{st.session_state.document_name}**")
else:
    st.info("No document loaded. You can still use the assistant as a normal local LLM chat.")

st.subheader("2. Chat")

for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

question = st.chat_input("Ask something about your document or a general question")

if question:
    st.session_state.messages.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    history_parts = []
    for msg in st.session_state.messages[-7:-1]:
        speaker = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_parts.append(f"{speaker}: {msg.content}")
    history = "\n".join(history_parts) if history_parts else "(no previous turns)"

    context = ""
    source_text = ""

    if st.session_state.vector_store is not None:
        with st.spinner("Retrieving relevant context..."):
            docs = st.session_state.vector_store.similarity_search(
                question,
                k=top_k,
            )
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        source_text = format_sources(docs)

    with st.chat_message("assistant"):
        with st.spinner("Thinking with Ollama..."):
            try:
                chain = RAG_PROMPT | chat_model | StrOutputParser()
                answer = chain.invoke(
                    {
                        "history": history,
                        "context": context or "(no document context loaded)",
                        "question": question,
                    }
                )
                st.markdown(answer)

                if source_text:
                    with st.expander("Sources used"):
                        st.markdown(source_text)

            except Exception as exc:
                answer = (
                    "I couldn't complete that request. "
                    "Please check that Ollama is running and the selected model is available."
                )
                st.error(f"{answer}\n\nDetails: {exc}")

    st.session_state.messages.append(AIMessage(content=answer))

st.divider()
st.caption("Everything is designed to run locally. Your uploaded document is processed by this app and sent to your local Ollama model/embeddings.")
