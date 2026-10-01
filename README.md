# Local LangChain + Ollama + Streamlit AI Assistant

A local AI application that combines:

- Streamlit for the web UI
- LangChain for the LLM/RAG application layer
- Ollama for local chat and embedding models
- InMemoryVectorStore for semantic retrieval
- PDF/TXT/Markdown upload
- Conversation history
- Source/page display for retrieved context

No cloud LLM API key is required.

## 1. Prerequisites

Install:

- Python 3.10+
- Ollama

Start Ollama, then pull a chat model and an embedding model.

Example:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

You can use different Ollama models. Just enter the model name in the Streamlit sidebar.

Check that Ollama is responding:

```bash
ollama list
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 4. Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 5. What the application does

### Normal chat

User question -> LangChain prompt -> Ollama chat model -> answer

### Document Q&A

PDF/TXT/Markdown
-> LangChain document loader
-> recursive chunking
-> Ollama embeddings
-> in-memory vector store
-> similarity retrieval
-> LangChain prompt
-> Ollama chat model
-> grounded answer + sources


## 6. Why InMemoryVectorStore?

This version intentionally uses LangChain's in-memory vector store to keep the project easy to run and understand.

For a larger persistent application, replace it with a persistent vector database such as Chroma, FAISS, Qdrant, or another LangChain-supported store.

## 7. Important note

The vector index is session-local. Uploading a new document rebuilds the index for that session.

For production, add:

- Persistent vector storage
- Authentication/authorization
- Better document metadata/filtering
- Retrieval evaluation
- Reranking
- Observability/evaluation
- Background ingestion
- A dedicated backend for larger workloads

## 8. Suggested demo

Upload a lecture PDF, then ask:

1. "Summarize this document."
2. "Explain PCA in simple terms."
3. "What are the important formulas?"
4. "Give me likely interview questions from this document."
5. "Which page discusses dimensionality reduction?"

This makes the project easy to demonstrate during a technical-club interview.
