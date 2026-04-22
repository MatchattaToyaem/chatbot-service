"""
FastAPI application for O'Connors IMS Chatbot Agent.

Endpoints:
    POST /search  — Retrieval only (returns chunks, no LLM)
    POST /ask     — Full RAG (retrieve → rerank → generate answer via Ollama)
    GET  /health  — System status check
    GET  /        — Service info

Mat's backend calls /ask for end-to-end Q&A or /search for retrieval-only.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting O'Connors IMS Chatbot Agent")
    logger.info("Config: %s", config.summary())
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
    version="2.0.0",
    lifespan=lifespan,
)


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


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    confidence: float
    model: str


class HealthResponse(BaseModel):
    status: str
    embedder: str
    chromadb: str
    ollama: str
    collection_count: int
    config: dict


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Retrieve and rerank relevant IMS chunks. No LLM generation."""
    try:
        reranked = rag.retrieve(
            question=request.question, retrieval_k=20, final_k=request.top_k,
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
    """Full RAG: retrieve → rerank → generate answer via Ollama llama3.2."""
    try:
        response = rag.ask(
            question=request.question, retrieval_k=20, final_k=request.top_k,
        )
    except Exception as e:
        logger.error("RAG pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    return AskResponse(
        question=response.question, answer=response.answer,
        sources=response.sources, confidence=response.confidence, model=response.model,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
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
    return {
        "service": "O'Connors IMS Chatbot Agent",
        "version": "2.0.0",
        "endpoints": {
            "/search": "POST — Retrieval only",
            "/ask": "POST — Full RAG Q&A (needs Ollama)",
            "/health": "GET — System status",
            "/docs": "GET — Swagger UI",
        },
    }
