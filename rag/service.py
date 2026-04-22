"""
RAG Service — the main orchestrator for the O'Connors IMS chatbot.

Ties together:
    1. Retrieval (BGE-M3 embedding -> ChromaDB query)
    2. Reranking (subfolder hints, keyword boosts, source diversity)
    3. Generation (Ollama llama3.2 with domain-specific prompts)

This is the single class Mat's backend calls for end-to-end Q&A.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from config import AppConfig
from ingest.embedder import Embedder
from ingest.store import ChromaStore
from rag.reranker import Reranker, RankedResult
from rag.generator import AnswerGenerator, GeneratedAnswer

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""
    question: str
    answer: str
    sources: list[dict]
    confidence: float
    model: str
    retrieval_count: int
    reranked_count: int


class RAGService:
    """
    End-to-end RAG service for O'Connors IMS Q&A.

    Usage:
        service = RAGService()
        response = service.ask("What is the after-hours call out procedure?")
        print(response.answer)
        print(response.sources)
        print(response.confidence)

    Pipeline:
        1. Embed the question with BGE-M3.
        2. Retrieve top-30 chunks from ChromaDB (over-fetch for reranking).
        3. Rerank using subfolder hints and keyword boosts.
        4. Take top-5 reranked results.
        5. Generate answer via Ollama with domain-specific prompts.
        6. Return answer + sources + confidence score.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self._config = config or AppConfig()

        # Shared embedder (same instance used by both ingest and query)
        self._embedder = Embedder(model_name=self._config.embedding.model_name)

        # ChromaDB store (connects to Mat's Azure server)
        self._store = ChromaStore(
            collection_name=self._config.chroma.collection,
            mode="remote",
            host=self._config.chroma.host,
            port=self._config.chroma.port,
            ssl=self._config.chroma.ssl,
        )

        # Reranker with domain-aware scoring
        self._reranker = Reranker(
            subfolder_boost_weight=1.0,
            keyword_boost_weight=1.5,
            max_per_source=3,
        )

        # Answer generator (Ollama llama3.2)
        self._generator = AnswerGenerator(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,
        )

        self._collection = None

    def _get_collection(self):
        """Get ChromaDB collection on first use."""
        if self._collection is None:
            self._collection = self._store.get_collection()
        return self._collection

    def retrieve(
        self,
        question: str,
        retrieval_k: int = None,
        final_k: int = None,
    ) -> list[RankedResult]:
        """
        Retrieve and rerank relevant chunks for a question.

        Args:
            question: The user's question.
            retrieval_k: Number of initial results from ChromaDB (over-fetch).
                         Defaults to config RAG_RETRIEVAL_K (30).
            final_k: Number of results after reranking.
                     Defaults to config RAG_TOP_K (5).

        Returns:
            List of RankedResult objects sorted by relevance.
        """
        if retrieval_k is None:
            retrieval_k = self._config.rag.retrieval_k
        if final_k is None:
            final_k = self._config.rag.top_k

        collection = self._get_collection()

        # Embed query with BGE-M3
        query_embedding = self._embedder.embed_query(question)

        # Retrieve from ChromaDB — explicit query_embeddings, never query_texts
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieval_k,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to reranker input format
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0.0,
            })

        logger.info("Retrieved %d chunks from ChromaDB for: '%s'", len(hits), question[:60])

        # Rerank
        reranked = self._reranker.rerank(
            query=question,
            results=hits,
            final_top_k=final_k,
        )

        return reranked

    def ask(
        self,
        question: str,
        retrieval_k: int = None,
        final_k: int = None,
    ) -> RAGResponse:
        """
        Full RAG pipeline: retrieve -> rerank -> generate answer.

        Args:
            question: The user's natural-language question.
            retrieval_k: Number of initial retrieval results.
                         Defaults to config RAG_RETRIEVAL_K (30).
            final_k: Number of chunks sent to the LLM.
                     Defaults to config RAG_TOP_K (5).

        Returns:
            RAGResponse with answer, sources, and confidence.
        """
        if retrieval_k is None:
            retrieval_k = self._config.rag.retrieval_k
        if final_k is None:
            final_k = self._config.rag.top_k

        # Step 1+2: Retrieve and rerank
        reranked = self.retrieve(question, retrieval_k, final_k)

        if not reranked:
            return RAGResponse(
                question=question,
                answer="No relevant documents found in the IMS collection.",
                sources=[],
                confidence=0.0,
                model=self._generator._model,
                retrieval_count=0,
                reranked_count=0,
            )

        # Step 3: Generate answer
        generated = self._generator.generate(question, reranked)

        return RAGResponse(
            question=question,
            answer=generated.answer,
            sources=generated.sources,
            confidence=generated.confidence,
            model=generated.model,
            retrieval_count=retrieval_k,
            reranked_count=len(reranked),
        )

    def health_check(self) -> dict:
        """Check all components."""
        status = {
            "embedder": "unknown",
            "chromadb": "unknown",
            "collection_count": 0,
            "ollama": "unknown",
        }

        try:
            self._embedder.embed_query("test")
            status["embedder"] = "ready"
        except Exception as e:
            status["embedder"] = f"error: {e}"

        try:
            col = self._get_collection()
            status["chromadb"] = "connected"
            status["collection_count"] = col.count()
        except Exception as e:
            status["chromadb"] = f"error: {e}"

        try:
            if self._generator._check_ollama():
                status["ollama"] = "ready"
            else:
                status["ollama"] = "not available"
        except Exception as e:
            status["ollama"] = f"error: {e}"

        return status
