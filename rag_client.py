"""
RAG Client for O'Connors IMS Chatbot Agent.

Connects to Mat's remote ChromaDB via HTTPS, embeds queries with BGE-M3,
and retrieves relevant document chunks. All embeddings are computed
explicitly — no ChromaDB embedding_function wrapper is used.

Follows class-based pattern per Vincent's coding standards.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import chromadb

from config import AppConfig
from ingest.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalHit:
    """A single retrieval result from ChromaDB."""
    document: str
    metadata: dict
    distance: float
    chunk_id: str


class RAGClient:
    """
    Retrieval-Augmented Generation client.

    Responsibilities:
        1. Load and manage the BGE-M3 embedding model.
        2. Connect to Mat's remote ChromaDB over HTTPS.
        3. Embed user queries and retrieve relevant IMS document chunks.

    All ChromaDB interactions use explicit embeddings (query_embeddings=)
    rather than ChromaDB's built-in embedding_function. This prevents
    the EF mismatch bug identified in Sprint 35.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self._config = config or AppConfig()
        self._embedder: Optional[Embedder] = None
        self._client: Optional[chromadb.HttpClient] = None
        self._collection = None

    # ------------------------------------------------------------------
    # Lazy initialisation — heavy resources load on first use, not import
    # ------------------------------------------------------------------

    def _get_embedder(self) -> Embedder:
        """Load BGE-M3 on first call, reuse thereafter."""
        if self._embedder is None:
            self._embedder = Embedder(model_name=self._config.embedding.model_name)
        return self._embedder

    def _get_collection(self):
        """Connect to ChromaDB and get the collection on first call."""
        if self._collection is None:
            cfg = self._config.chroma
            logger.info("Connecting to ChromaDB at %s:%d (ssl=%s)", cfg.host, cfg.port, cfg.ssl)
            self._client = chromadb.HttpClient(
                host=cfg.host,
                port=cfg.port,
                ssl=cfg.ssl,
            )
            # Verify connection
            heartbeat = self._client.heartbeat()
            logger.info("ChromaDB connected. Heartbeat: %s", heartbeat)

            # Get collection WITHOUT embedding_function — we pass embeddings explicitly
            self._collection = self._client.get_collection(name=cfg.collection)
            count = self._collection.count()
            logger.info(
                "Collection '%s' loaded: %d chunks", cfg.collection, count
            )
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string using BGE-M3.

        Returns a normalised 1024-dim vector as a Python list.
        """
        embedder = self._get_embedder()
        return embedder.embed_query(query)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievalHit]:
        """
        Retrieve the most relevant IMS document chunks for a given query.

        Args:
            query: The user's natural-language question.
            top_k: Number of results to return (defaults to RAG_TOP_K from config).

        Returns:
            A list of RetrievalHit objects sorted by relevance (closest first).
        """
        k = top_k or self._config.rag.top_k
        collection = self._get_collection()

        # Embed query explicitly — never use query_texts=
        query_embedding = self.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hit = RetrievalHit(
                chunk_id=results["ids"][0][i],
                document=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                distance=results["distances"][0][i] if results["distances"] else 0.0,
            )
            hits.append(hit)

        logger.info(
            "Query: '%s' → %d hits (top distance: %.4f)",
            query[:80],
            len(hits),
            hits[0].distance if hits else -1,
        )
        return hits

    def health_check(self) -> dict:
        """
        Check connectivity to ChromaDB and model readiness.

        Returns a dict with status info suitable for a /health endpoint.
        """
        status = {"chroma": "unknown", "model": "unknown", "collection_count": 0}

        try:
            collection = self._get_collection()
            status["chroma"] = "connected"
            status["collection_count"] = collection.count()
        except Exception as e:
            status["chroma"] = f"error: {e}"

        try:
            self._get_embedder()
            status["model"] = "loaded"
        except Exception as e:
            status["model"] = f"error: {e}"

        return status
