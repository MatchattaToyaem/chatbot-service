# O'Connors IMS Chatbot Agent

RAG-based document retrieval service for O'Connors Services IMS files. Connects to a remote ChromaDB instance hosted on Azure Container Apps, embeds queries with BGE-M3, and returns relevant document chunks via a FastAPI HTTP API.

## Architecture

```
User Question
      │
      ▼
  FastAPI (/search)
      │
      ▼
  RAGClient.retrieve()
      │
      ├── BGE-M3 embeds the query (1024-dim, normalised)
      │
      ├── ChromaDB query (explicit query_embeddings=, HTTPS to Azure)
      │
      └── Returns top-k chunks with metadata and distance
```

## Setup

```cmd
cd "D:\Capstone S1-43\chatbot-agent"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust if needed (defaults point to Mat's Azure ChromaDB).

## Run

```cmd
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for the Swagger UI.

## Endpoints

| Method | Path      | Description                                      |
|--------|-----------|--------------------------------------------------|
| POST   | /search   | Send a question, get back relevant IMS chunks    |
| GET    | /health   | Check ChromaDB connection and model status        |
| GET    | /         | Service info                                      |

## Example Request

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the after-hours call out procedure?", "top_k": 5}'
```

## File Structure

```
chatbot-agent/
├── app.py              # FastAPI endpoints
├── rag_client.py       # BGE-M3 embedding + ChromaDB retrieval
├── config.py           # Environment variable loader
├── .env                # Local config (not committed)
├── .env.example        # Config template
├── requirements.txt    # Dependencies
├── .gitignore          # Git exclusions
└── README.md           # This file
```

## Integration with Mat's Backend

Mat's backend can call the `/search` endpoint directly over HTTP, or import `RAGClient` as a Python class:

```python
from rag_client import RAGClient

rag = RAGClient()
hits = rag.retrieve("What is the after-hours call out procedure?", top_k=5)
for hit in hits:
    print(hit.document, hit.metadata, hit.distance)
```

## Notes

The embedding model (BGE-M3) must match what was used during ingestion. If the ChromaDB collection was built with a different model, queries will return poor results due to vector space mismatch.
