"""
FastAPI application for O'Connors IMS Chatbot Agent.

Endpoints:
    POST /search  — Retrieval only (returns chunks, no LLM)
    POST /ask     — Full RAG (retrieve → rerank → generate answer via Ollama)
    GET  /health  — System status check
    GET  /        — Service info

Mat's backend calls /ask for end-to-end Q&A or /search for retrieval-only.

CHANGES (v2.1 — 24 Apr 2026):
  - Added CORS middleware for frontend cross-origin requests
  - /ask response now returns clean source names (not raw file paths)
  - Removed chunk_id and raw scores from client-facing /ask response
  - /search still returns full debug info for internal use
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware      # ADDED
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig
from rag.service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

config = AppConfig()
rag = RAGService(config)

# Domain filter — set RAG_DOMAIN=IMS or RAG_DOMAIN=Service in the environment
# to restrict retrieval to one SharePoint folder. Leave unset to search all.
_raw_domain = os.getenv("RAG_DOMAIN", "").strip()
RAG_DOMAIN: str | None = _raw_domain if _raw_domain else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting O'Connors IMS Chatbot Agent")
    logger.info("Config: %s", config.summary())
    logger.info("Domain filter: %s", RAG_DOMAIN or "all (RAG_DOMAIN not set)")
    try:
        health = rag.health_check()
        logger.info("Health check: %s", health)
    except Exception as e:
        logger.warning("Startup health check failed: %s", e)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="O'Connors IMS Chatbot Agent",
    description="RAG-based chatbot for O'Connors IMS. /search for retrieval, /ask for full Q&A.",
    version="2.1.0",
    lifespan=lifespan,
)

# ADDED: CORS middleware — allows Shrid's frontend to call the API
# In production, replace "*" with the actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────

class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(5, ge=1, le=20)


class ChunkResult(BaseModel):
    chunk_id: str
    document: str
    metadata: dict
    distance: float
    reranked_score: float
    boost_reasons: list


class SearchResponse(BaseModel):
    question: str
    results: list[ChunkResult]
    count: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(5, ge=1, le=10)
    session_id: Optional[str] = Field(None, description="Chat session UUID for context-aware answers")


# CHANGED: Clean response for client-facing use
class SourceInfo(BaseModel):
    """A clean source reference — no file paths, no chunk IDs."""
    document: str       # e.g. "R134A Material Safety Data Sheet"
    category: str       # e.g. "MSDS's", "Policies", "Australian Standards"


class AskResponse(BaseModel):
    """Client-facing response — clean answer with source references."""
    question: str
    answer: str
    sources: list[SourceInfo]


class HealthResponse(BaseModel):
    status: str
    embedder: str
    chromadb: str
    ollama: str
    collection_count: int
    config: dict


# ── Endpoints ────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Retrieve and rerank relevant IMS chunks. No LLM generation.

    Internal/debug endpoint — returns raw chunk data with scores.
    Use /ask for client-facing Q&A.
    """
    try:
        reranked = rag.retrieve(
            question=request.question, retrieval_k=20, final_k=request.top_k,
            domain=RAG_DOMAIN,
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    results = [
        ChunkResult(
            chunk_id=r.chunk_id, document=r.document, metadata=r.metadata,
            distance=r.original_distance, reranked_score=r.reranked_score,
            boost_reasons=r.boost_reasons,
        )
        for r in reranked
    ]
    return SearchResponse(question=request.question, results=results, count=len(results))


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Full RAG Q&A: retrieve → rerank → generate answer via Ollama.

    Client-facing endpoint — returns a clean, professionally written
    answer with source references. No file paths, no chunk IDs,
    no confidence scores exposed.
    """
    try:
        response = rag.ask(
            question=request.question,
            retrieval_k=20,
            final_k=request.top_k,
            session_id=request.session_id or "",
            domain=RAG_DOMAIN,
        )
    except Exception as e:
        logger.error("RAG pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    # Convert sources to clean SourceInfo objects
    clean_sources = []
    for src in response.sources:
        clean_sources.append(SourceInfo(
            document=src.get("document", src.get("file", "Unknown")),
            category=src.get("category", src.get("subfolder", "")),
        ))

    return AskResponse(
        question=response.question,
        answer=response.answer,
        sources=clean_sources,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """System status check."""
    check = rag.health_check()
    all_ok = check["embedder"] == "ready" and check["chromadb"] == "connected"
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        embedder=check["embedder"], chromadb=check["chromadb"],
        ollama=check["ollama"], collection_count=check["collection_count"],
        config=config.summary(),
    )


@app.get("/")
async def root():
    """Landing page with endpoint documentation."""
    return {
        "service": "O'Connors IMS Chatbot Agent",
        "version": "2.1.0",
        "endpoints": {
            "/ask": "POST — Client-facing Q&A (clean answer + source names)",
            "/search": "POST — Internal retrieval (raw chunks + scores)",
            "/health": "GET — System status",
            "/docs": "GET — Swagger UI",
        },
    }
