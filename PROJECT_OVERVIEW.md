# Project overview

## Architecture

```text
                 ┌─────────────────────┐
                 │     Streamlit UI    │
                 └──────────┬──────────┘
                            │
                ┌───────────▼───────────┐
                │       LangChain       │
                │ prompt + retrieval    │
                └───────┬─────────┬─────┘
                        │         │
              ┌─────────▼───┐   ┌─▼────────────────┐
              │ Vector Store │   │  ChatOllama      │
              │ InMemory     │   │ Local LLM        │
              └───────┬──────┘   └──────────────────┘
                      │
                OllamaEmbeddings
                      │
                 Local Ollama
```

