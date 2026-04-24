"""
Configuration loader for O'Connors RAG Chatbot Agent.

Reads all settings from environment variables (via .env file).
Follows class-based pattern per Vincent's coding standards.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ChromaConfig:
    """ChromaDB connection settings."""
    host: str = os.getenv(
        "CHROMA_HOST",
        "chroma-db-ai-platform.proudground-90080d26.australiaeast.azurecontainerapps.io",
    )
    port: int = int(os.getenv("CHROMA_PORT", "443"))
    ssl: bool = os.getenv("CHROMA_SSL", "true").lower() == "true"
    collection: str = os.getenv("CHROMA_COLLECTION", "oconnors_ims_bge_m3")
    mode: str = os.getenv("CHROMA_MODE", "remote")
    local_path: str = os.getenv("CHROMA_LOCAL_PATH", "./chromadb_local")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model settings."""
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    normalize: bool = True
    dimension: int = 1024


@dataclass(frozen=True)
class RAGConfig:
    """RAG retrieval settings."""
    retrieval_k: int = int(os.getenv("RAG_RETRIEVAL_K", "30"))
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))


@dataclass(frozen=True)
class ServerConfig:
    """FastAPI server settings."""
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))


class AppConfig:
    """Top-level application configuration. Single access point for all settings."""

    def __init__(self):
        self.chroma = ChromaConfig()
        self.embedding = EmbeddingConfig()
        self.rag = RAGConfig()
        self.server = ServerConfig()

    def summary(self) -> dict:
        """Return a dict summary of current config (safe for logging)."""
        return {
            "chroma_host": self.chroma.host,
            "chroma_port": self.chroma.port,
            "chroma_ssl": self.chroma.ssl,
            "chroma_collection": self.chroma.collection,
            "embedding_model": self.embedding.model_name,
            "embedding_dimension": self.embedding.dimension,
            "rag_retrieval_k": self.rag.retrieval_k,
            "rag_top_k": self.rag.top_k,
        }
