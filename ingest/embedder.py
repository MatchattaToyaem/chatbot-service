"""
BGE-M3 embedding model wrapper.

Provides a single class used by both the ingestion pipeline (batch embed)
and the RAG service (query embed). All embeddings are normalised for
cosine similarity — ChromaDB's default distance metric.

Dense retrieval only. BGE-M3 also supports sparse and multi-vector
retrieval, but ChromaDB only handles dense vectors.
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    BGE-M3 embedding model manager.

    Loads the model once on first use (lazy initialisation) and provides
    batch and single-query embedding methods. All output vectors are
    L2-normalised for cosine similarity.

    This class is the single source of truth for embedding in the entire
    project. Both ingest/store.py and rag_client.py use it, ensuring
    the same model and normalisation settings are applied everywhere.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self._model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _load_model(self) -> SentenceTransformer:
        """Load the model on first call."""
        if self._model is None:
            logger.info(
                "Loading embedding model: %s (~2.3 GB on first download)",
                self._model_name,
            )
            self._model = SentenceTransformer(self._model_name)
            # Verify dimension
            test = self._model.encode(["test"], normalize_embeddings=True)
            dim = len(test[0])
            logger.info("Embedding model ready. Dimension: %d", dim)
        return self._model

    def embed_batch(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = True
    ) -> list[list[float]]:
        """
        Embed a batch of texts. Used during ingestion.

        Args:
            texts: List of text strings to embed.
            batch_size: Encoding batch size (tune for GPU memory).
            show_progress: Show tqdm progress bar.

        Returns:
            List of normalised embedding vectors as Python lists.
        """
        model = self._load_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string. Used during retrieval.

        Returns:
            A single normalised 1024-dim vector as a Python list.
        """
        model = self._load_model()
        embedding = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (loads model if needed)."""
        self._load_model()
        return 1024
