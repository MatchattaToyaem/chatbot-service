"""
RAG Service — the main orchestrator for the O'Connors IMS chatbot.

Ties together:
    1. Retrieval (BGE-M3 embedding -> Hybrid BM25 + ChromaDB vector search)
    2. Ranking (Reciprocal Rank Fusion from hybrid retriever)
    3. Form detection (boosts exact document when user asks by code)
    4. Generation (LLM via Ollama / HuggingFace / Azure Foundry)

This is the single class Mat's backend calls for end-to-end Q&A.

CHANGES (v2.3 — 03 May 2026):
  - Fixed form code regex to handle letter suffixes (OF113a, OF15A, CF06_1)
CHANGES (v2.2 — 02 May 2026):
  - Added form/document code detection: when user asks for a specific
    form (OF140, CF18, SOP-001, etc.), chunks from that exact document
    are boosted to the top of retrieval results. Fixes Adam's feedback #5.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from config import AppConfig
from ingest.embedder import Embedder
from ingest.store import ChromaStore
from rag.reranker import Reranker, RankedResult
from rag.generator import AnswerGenerator, GeneratedAnswer
from rag.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


# Regex for O'Connors document codes
# Handles: OF140, OF113a, OF15A, CF18, CF06_1, SOP-001, SWMS 024, etc.
_FORM_CODE_RE = re.compile(
    r'\b(OF\d{1,4}[a-zA-Z_]?'
    r'|CF\d{1,4}[a-zA-Z_]?'
    r'|SOP-?\d{1,4}'
    r'|SWMS\s*\d{1,4}'
    r'|AHP\d{1,4}'
    r'|EP\d{1,4}'
    r'|FP\s*\d{1,4}'
    r'|WP\d{1,4})',
    re.IGNORECASE
)


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
        2. Hybrid retrieve: BM25 keyword + vector search, fused with RRF.
        3. Form detection: if user asks for OF140/CF18/SOP-001 etc.,
           boost chunks from that exact document to the top.
        4. Take top-5 fused results (reranker bypassed — RRF handles ranking).
        5. Generate answer via LLM with domain-specific prompts.
        6. Return answer + sources + confidence score.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self._config = config or AppConfig()

        # Shared embedder (same instance used by both ingest and query)
        self._embedder = Embedder(model_name=self._config.embedding.model_name)

        # ChromaDB store (connects to Mat's Azure server or local)
        self._store = ChromaStore(
            collection_name=self._config.chroma.collection,
            mode=self._config.chroma.mode,
            host=self._config.chroma.host,
            port=self._config.chroma.port,
            ssl=self._config.chroma.ssl,
            local_path=self._config.chroma.local_path,
        )

        # Reranker with domain-aware scoring (kept for future use,
        # currently bypassed in favor of RRF ranking from hybrid retriever)
        self._reranker = Reranker(
            subfolder_boost_weight=1.0,
            keyword_boost_weight=1.5,
            max_per_source=3,
        )

        # Answer generator (supports Ollama / HuggingFace / Azure Foundry)
        self._generator = AnswerGenerator(
            model=os.getenv("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
            temperature=0.1,
        )

        self._collection = None
        self._hybrid = None

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
        Retrieve relevant chunks for a question using hybrid search.
        Uses BM25 + vector search with Reciprocal Rank Fusion (RRF).

        If the question contains a form code (OF140, CF18, SOP-001, etc.),
        chunks from that exact document are boosted to the top.

        Args:
            question: The user's question.
            retrieval_k: Number of initial results from each retriever.
                         Defaults to config RAG_RETRIEVAL_K (50).
            final_k: Number of results after fusion.
                     Defaults to config RAG_TOP_K (5).

        Returns:
            List of RankedResult objects sorted by RRF relevance.
        """
        if retrieval_k is None:
            retrieval_k = self._config.rag.retrieval_k
        if final_k is None:
            final_k = self._config.rag.top_k

        collection = self._get_collection()

        # Initialize hybrid retriever on first use
        if self._hybrid is None:
            self._hybrid = HybridRetriever(collection, self._embedder)

        # Hybrid retrieve: BM25 + vector + RRF fusion
        hybrid_hits = self._hybrid.retrieve(
            question=question,
            retrieval_k=retrieval_k,
            final_k=retrieval_k,  # Pass all to allow selection of top final_k
        )

        # Convert hybrid results to standard format
        hits = []
        for h in hybrid_hits:
            hits.append({
                "chunk_id": h["chunk_id"],
                "document": h["document"],
                "metadata": h["metadata"],
                "distance": h.get("distance", 0.0),
            })

        logger.info("Hybrid retrieved %d chunks for: '%s'", len(hits), question[:60])

        # ── Form/document code detection ──────────────────────────
        # When user asks for a specific form (OF140, CF18, SOP-001, etc.),
        # boost chunks from that exact document to the top of results.
        # Without this, SWMS documents that *mention* a form can outrank
        # the actual form itself.
        form_match = _FORM_CODE_RE.search(question)
        if form_match:
            form_code = form_match.group(1).upper().replace(" ", "").replace("-", "")
            exact = [h for h in hits if form_code in
                     (h.get("metadata", {}).get("source_file", "") +
                      h.get("metadata", {}).get("source_path", ""))
                     .upper().replace(" ", "").replace("-", "")]
            others = [h for h in hits if h not in exact]
            hits = exact + others
            if exact:
                logger.info("Form detected: %s → %d exact chunks boosted",
                            form_match.group(1), len(exact))

        # With hybrid search, RRF already handles ranking
        # Convert hits to RankedResult format (skip reranker)
        ranked_results = []
        for h in hits[:final_k]:
            ranked_results.append(RankedResult(
                chunk_id=h["chunk_id"],
                document=h["document"],
                metadata=h.get("metadata", {}),
                original_distance=h.get("distance", 0.0),
                reranked_score=h.get("rrf_score", 0.5),
                boost_reasons=[],
            ))

        logger.info("Returning top %d hybrid results (reranker bypassed)", len(ranked_results))
        return ranked_results

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
                         Defaults to config RAG_RETRIEVAL_K (50).
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

        provider = os.getenv("LLM_PROVIDER", "ollama")
        status["ollama"] = f"provider: {provider}"

        return status
