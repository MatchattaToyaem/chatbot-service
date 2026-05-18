"""
Hybrid retriever: BM25 keyword search + ChromaDB vector search.
Uses Reciprocal Rank Fusion (RRF) to merge results.

Why: Vector search finds semantically similar content but misses
exact product names (e.g., "Raid Flying Insect Killer"). BM25
finds exact keyword matches. Combining both gives the best of
both worlds.
"""

import re
import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines BM25 keyword matching with ChromaDB vector search.

    Builds a BM25 index from all chunks in ChromaDB on first use,
    then each query runs both retrievers and merges via RRF.
    """

    def __init__(self, collection, embedder):
        self._collection = collection
        self._embedder = embedder
        self._bm25 = None
        self._doc_ids = []
        self._doc_texts = []
        self._doc_metas = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer."""
        return re.findall(r'\w+', text.lower())

    def _ensure_bm25_index(self):
        """Build BM25 index from ChromaDB (once, lazily)."""
        if self._bm25 is not None:
            return

        logger.info("Building BM25 index from ChromaDB (one-time)...")
        count = self._collection.count()
        batch_size = 5000

        for offset in range(0, count, batch_size):
            result = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"]
            )
            self._doc_ids.extend(result["ids"])
            self._doc_texts.extend(result["documents"])
            self._doc_metas.extend(result["metadatas"])
            logger.info("  Loaded %d / %d chunks...",
                        len(self._doc_ids), count)

        tokenized = [self._tokenize(doc) for doc in self._doc_texts]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index ready: %d documents", len(self._doc_ids))

    def retrieve(
        self,
        question: str,
        retrieval_k: int = 30,
        final_k: int = 5,
        where_filter: dict | None = None,
    ) -> list[dict]:
        """
        Hybrid retrieve: vector + BM25 with RRF fusion.

        Args:
            where_filter: optional ChromaDB metadata filter, e.g.
                          {"domain": "Service"} or {"domain": "IMS"}.
                          Applied to both vector search and BM25 post-filtering.

        Returns list of dicts with keys:
            chunk_id, document, metadata, distance, rrf_score
        """
        self._ensure_bm25_index()

        # --- Vector search ---
        query_embedding = self._embedder.embed_query(question)
        vector_query_kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=retrieval_k,
            include=["documents", "metadatas", "distances"],
        )
        if where_filter:
            vector_query_kwargs["where"] = where_filter
        vector_results = self._collection.query(**vector_query_kwargs)

        vector_ranked = {}
        vector_data = {}
        for i, doc_id in enumerate(vector_results["ids"][0]):
            vector_ranked[doc_id] = 1.0 / (60 + i + 1)  # RRF score
            vector_data[doc_id] = {
                "chunk_id": doc_id,
                "document": vector_results["documents"][0][i],
                "metadata": vector_results["metadatas"][0][i] or {},
                "distance": vector_results["distances"][0][i],
            }

        # --- BM25 keyword search ---
        query_tokens = self._tokenize(question)
        bm25_raw = self._bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(bm25_raw)),
            key=lambda i: bm25_raw[i],
            reverse=True,
        )[:retrieval_k]

        # Post-filter BM25 results to match the same domain constraint used
        # for vector search. BM25 index is built from all chunks, so we filter
        # here instead of rebuilding per-domain.
        if where_filter:
            top_indices = [
                idx for idx in top_indices
                if all(
                    self._doc_metas[idx].get(k) == v
                    for k, v in where_filter.items()
                )
            ]

        bm25_ranked = {}
        bm25_data = {}
        for rank, idx in enumerate(top_indices):
            doc_id = self._doc_ids[idx]
            bm25_ranked[doc_id] = 1.0 / (60 + rank + 1)
            if doc_id not in vector_data:
                bm25_data[doc_id] = {
                    "chunk_id": doc_id,
                    "document": self._doc_texts[idx],
                    "metadata": self._doc_metas[idx] or {},
                    "distance": 0.0,
                }

        # --- Reciprocal Rank Fusion ---
        all_ids = set(vector_ranked.keys()) | set(bm25_ranked.keys())
        fused = []
        for doc_id in all_ids:
            score = vector_ranked.get(doc_id, 0) + bm25_ranked.get(doc_id, 0)
            data = vector_data.get(doc_id) or bm25_data.get(doc_id)
            data["rrf_score"] = score
            fused.append(data)

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Log top result source
        if fused:
            top = fused[0]
            src = top["metadata"].get("source_file", "Unknown")
            logger.info("Hybrid top: score=%.4f (%s), vector=%d, bm25=%d, fused=%d",
                        top["rrf_score"], src[:50],
                        len(vector_ranked), len(bm25_ranked), len(fused))

        return fused[:final_k]
