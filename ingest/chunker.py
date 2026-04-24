"""
Text chunker for IMS document processing.

Uses RecursiveCharacterTextSplitter from langchain-text-splitters
with 1200 character chunks and 200 character overlap. These settings
are tuned for O'Connors IMS documents (SOPs, Procedures, Policies)
and sit comfortably within BGE-M3's 8192-token context window.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with its metadata and unique ID."""
    chunk_id: str
    text: str
    metadata: dict


class DocumentChunker:
    """
    Splits extracted document text into overlapping chunks.

    Each chunk receives a deterministic SHA-1 ID based on its content
    and source file, ensuring idempotent upserts to ChromaDB.

    Metadata attached to each chunk:
        - file: filename only (used by eval scripts and reranker)
        - source_file: relative path within IMS folder
        - subfolder: top-level IMS folder (Procedures, SOPs, Policies, etc.)
        - subfolder_path: full relative folder path for nested subfolders
        - chunk_index: position within the document
        - total_chunks: number of chunks in the document
        - page_count: total pages in the source document
        - extraction_method: how the text was obtained (pdfplumber, ocr, docx, xlsx)
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @staticmethod
    def _make_chunk_id(source_file: str, chunk_index: int, text: str) -> str:
        """Generate a deterministic SHA-1 chunk ID for idempotent upserts."""
        content = f"{source_file}::chunk_{chunk_index}::{text[:100]}"
        return hashlib.sha1(content.encode("utf-8")).hexdigest()

    def chunk_document(
        self,
        text: str,
        file_path: str,
        ims_root: str,
        page_count: int = 0,
        extraction_method: str = "unknown",
    ) -> list[Chunk]:
        """
        Split document text into chunks with metadata.

        Args:
            text: The full extracted text of the document.
            file_path: Absolute path to the source file.
            ims_root: Root IMS directory (for computing relative paths).
            page_count: Number of pages in the source document.
            extraction_method: How the text was extracted.

        Returns:
            List of Chunk objects ready for embedding and storage.
        """
        if not text or not text.strip():
            return []

        # Compute relative path and subfolder
        try:
            rel_path = str(Path(file_path).relative_to(ims_root))
        except ValueError:
            rel_path = Path(file_path).name

        filename = Path(file_path).name
        parts = Path(rel_path).parts
        subfolder = parts[0] if len(parts) > 1 else "root"
        subfolder_path = str(Path(*parts[:-1])) if len(parts) > 1 else "root"

        # Split text
        splits = self._splitter.split_text(text)

        chunks = []
        for i, split_text in enumerate(splits):
            chunk_id = self._make_chunk_id(rel_path, i, split_text)
            chunk = Chunk(
                chunk_id=chunk_id,
                text=split_text,
                metadata={
                    "file": filename,
                    "source_file": rel_path,
                    "subfolder": subfolder,
                    "subfolder_path": subfolder_path,
                    "chunk_index": i,
                    "total_chunks": len(splits),
                    "page_count": page_count,
                    "extraction_method": extraction_method,
                },
            )
            chunks.append(chunk)

        return chunks

    def make_fallback_chunk(
        self,
        file_path: str,
        ims_root: str,
        reason: str = "no_text_extracted",
    ) -> Chunk:
        """
        Create a single fallback chunk for files that produced no text.

        Stores the filename and metadata so the file is still discoverable
        via retrieval, even if the content couldn't be extracted (DRM,
        scanned with failed OCR, empty file, etc.).
        """
        try:
            rel_path = str(Path(file_path).relative_to(ims_root))
        except ValueError:
            rel_path = Path(file_path).name

        filename = Path(file_path).name
        parts = Path(rel_path).parts
        subfolder = parts[0] if len(parts) > 1 else "root"
        subfolder_path = str(Path(*parts[:-1])) if len(parts) > 1 else "root"

        fallback_text = (
            f"Document: {filename}\n"
            f"Location: {rel_path}\n"
            f"Category: {subfolder}\n"
            f"Note: Text extraction was not possible ({reason}). "
            f"Please refer to the original document for full content."
        )

        chunk_id = self._make_chunk_id(rel_path, 0, fallback_text)

        return Chunk(
            chunk_id=chunk_id,
            text=fallback_text,
            metadata={
                "file": filename,
                "source_file": rel_path,
                "subfolder": subfolder,
                "subfolder_path": subfolder_path,
                "chunk_index": 0,
                "total_chunks": 1,
                "page_count": 0,
                "extraction_method": "fallback",
                "fallback_reason": reason,
            },
        )
