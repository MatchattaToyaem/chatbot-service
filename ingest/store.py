"""
ChromaDB collection writer for IMS document chunks.

Writes embeddings and metadata to Mat's remote ChromaDB instance
on Azure Container Apps. All embeddings are computed externally
by the Embedder class and passed explicitly via embeddings= parameter.
No ChromaDB embedding_function is used anywhere.

Supports both remote (Azure HTTPS) and local (PersistentClient) modes,
controlled by the mode parameter.
"""

import logging
from typing import Optional

import chromadb

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    Manages a ChromaDB collection for IMS document storage.

    Two modes:
        - "remote": connects to Mat's Azure ChromaDB via HttpClient (HTTPS).
        - "local": writes to a local directory via PersistentClient.

    The collection is created with cosine distance and no embedding_function.
    All vectors must be passed explicitly on upsert.
    """

    def __init__(
        self,
        collection_name: str = "oconnors_ims_bge_m3",
        mode: str = "remote",
        # Remote settings
        host: str = "chroma-db-ai-platform.proudground-90080d26.australiaeast.azurecontainerapps.io",
        port: int = 443,
        ssl: bool = True,
        # Local settings
        local_path: str = "./chromadb_data_bge_m3",
    ):
        self._collection_name = collection_name
        self._mode = mode
        self._host = host
        self._port = port
        self._ssl = ssl
        self._local_path = local_path
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None

    def _connect(self):
        """Establish connection to ChromaDB."""
        if self._client is not None:
            return

        if self._mode in ("remote", "http"):
            logger.info(
                "Connecting to remote ChromaDB: %s:%d (ssl=%s)",
                self._host, self._port, self._ssl,
            )
            self._client = chromadb.HttpClient(
                host=self._host,
                port=self._port,
                ssl=self._ssl,
            )
        else:
            logger.info("Using local ChromaDB at: %s", self._local_path)
            import os
            os.makedirs(self._local_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._local_path)

        heartbeat = self._client.heartbeat()
        logger.info("ChromaDB connected. Heartbeat: %s", heartbeat)

    def create_collection(self, delete_existing: bool = True):
        """
        Create (or recreate) the target collection.

        Args:
            delete_existing: If True, deletes any existing collection with
                the same name before creating. Ensures a clean ingest.
        """
        self._connect()

        if delete_existing:
            existing = [c.name for c in self._client.list_collections()]
            if self._collection_name in existing:
                logger.info("Deleting existing collection '%s'...", self._collection_name)
                self._client.delete_collection(name=self._collection_name)

        # No embedding_function — embeddings are always passed explicitly
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Created collection '%s'.", self._collection_name)

    def get_collection(self):
        """Get an existing collection (for querying, not ingestion)."""
        self._connect()
        self._collection = self._client.get_collection(name=self._collection_name)
        return self._collection

    def upsert_batch(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        batch_size: int = 50,
    ) -> int:
        """
        Upsert chunks into ChromaDB in batches.

        All embeddings are pre-computed by the Embedder class and passed
        explicitly. ChromaDB never auto-embeds.

        Args:
            ids: Chunk IDs (SHA-1 hashes).
            documents: Raw text of each chunk.
            embeddings: Pre-computed BGE-M3 vectors (1024-dim, normalised).
            metadatas: Metadata dicts for each chunk.
            batch_size: Number of chunks per upsert call.

        Returns:
            Number of chunks successfully stored.
        """
        if self._collection is None:
            raise RuntimeError("Collection not initialised. Call create_collection() first.")

        stored = 0
        total = len(ids)

        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            batch_ids = ids[i:end]
            batch_docs = documents[i:end]
            batch_embs = embeddings[i:end]
            batch_meta = metadatas[i:end]

            try:
                self._collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embs,
                    metadatas=batch_meta,
                )
                stored += len(batch_ids)

                if (i // batch_size) % 10 == 0:
                    logger.info("  Stored %d / %d chunks...", stored, total)

            except Exception as e:
                logger.error("Batch upsert failed at index %d: %s", i, e)
                # Fall back to single upserts
                for j in range(len(batch_ids)):
                    try:
                        self._collection.upsert(
                            ids=[batch_ids[j]],
                            documents=[batch_docs[j]],
                            embeddings=[batch_embs[j]],
                            metadatas=[batch_meta[j]],
                        )
                        stored += 1
                    except Exception as e2:
                        logger.error("Single upsert failed %s: %s", batch_ids[j], e2)

        logger.info("Upsert complete: %d / %d chunks stored.", stored, total)
        return stored

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        if self._collection is None:
            return 0
        return self._collection.count()
