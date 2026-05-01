"""
IMS ingestion pipeline orchestrator.

Scans the IMS folder for all supported files (PDF, DOCX, DOC, XLSX),
extracts text, chunks it, embeds with BGE-M3, and stores in ChromaDB.

Supports folder filtering (Priority 1 = Procedures/SOPs/Policies)
and skip-ocr mode for fast dry runs.

SCRUM-37: Integrated dead-letter queue (DLQ) routes all failed
extractions to dead_letter/ with structured error records and
generates an alert report at the end of each run.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ingest.extractor import TextExtractor
from ingest.chunker import DocumentChunker
from ingest.embedder import Embedder
from ingest.store import ChromaStore
from dead_letter_queue import DeadLetterQueue

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    """Running statistics for the ingestion pipeline."""
    files_found: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    files_fallback: int = 0
    total_chunks: int = 0
    total_ocr_pages: int = 0
    elapsed_seconds: float = 0.0
    skipped_files: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    dlq_report_path: str = ""  # SCRUM-37: path to DLQ alert report

    def summary(self) -> dict:
        return {
            "files_found": self.files_found,
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "files_fallback": self.files_fallback,
            "files_failed": self.files_failed,
            "total_chunks": self.total_chunks,
            "total_ocr_pages": self.total_ocr_pages,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "skipped_files": self.skipped_files[:50],
            "errors": self.errors[:50],
            "dlq_report_path": self.dlq_report_path,
        }


class IngestPipeline:
    """
    End-to-end IMS document ingestion pipeline.

    Usage:
        pipeline = IngestPipeline(
            ims_root="/content/IMS",
            collection_name="oconnors_ims",
            store_mode="local",
        )
        stats = pipeline.run()
        print(stats.summary())

    The pipeline processes files in this order:
        1. Scan IMS folder for supported files (PDF, DOCX, DOC, XLSX, etc.).
        2. Filter by priority folders if specified.
        3. Extract text from each file (pdfplumber -> PaddleOCR fallback).
        4. Chunk text into overlapping segments (1200 chars, 200 overlap).
        5. If no text extracted, create a fallback chunk with filename metadata.
        6. Embed all chunks with BGE-M3 (batched, GPU-accelerated on Colab).
        7. Store chunks + embeddings in ChromaDB (explicit embeddings=).

    SCRUM-37: Failed documents are routed to a dead-letter queue (DLQ)
    at dead_letter/. An alert report is generated after each run.
    """

    PRIORITY_1 = ["Procedures", "SOP's", "Policies"]
    PRIORITY_2 = ["Australian Standards", "MSDS's"]
    PRIORITY_3 = ["Forms", "SWMS's"]
    OTHER = ["RDO Calendar"]
    ALL_FOLDERS = PRIORITY_1 + PRIORITY_2 + PRIORITY_3 + OTHER

    SKIP_EXTENSIONS = {".xps", ".tmp", ".bak", ".ds_store"}

    def __init__(
        self,
        ims_root: str,
        collection_name: str = "oconnors_ims",
        embedding_model: str = "BAAI/bge-m3",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        store_mode: str = "remote",
        chroma_host: str = "chroma-db-ai-platform.proudground-90080d26.australiaeast.azurecontainerapps.io",
        chroma_port: int = 443,
        chroma_ssl: bool = True,
        local_chroma_path: str = "./chromadb_data_bge_m3",
        enable_ocr: bool = True,
        ocr_timeout: int = 10,
        folders: Optional[list[str]] = None,
        dead_letter_dir: str = "./dead_letter",
    ):
        self._ims_root = ims_root
        self._folders = folders or self.PRIORITY_1
        self._enable_ocr = enable_ocr

        self._extractor = TextExtractor(
            enable_ocr=enable_ocr,
            ocr_timeout=ocr_timeout,
        )
        self._chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._embedder = Embedder(model_name=embedding_model)
        self._store = ChromaStore(
            collection_name=collection_name,
            mode=store_mode,
            host=chroma_host,
            port=chroma_port,
            ssl=chroma_ssl,
            local_path=local_chroma_path,
        )

        # SCRUM-37: Dead-letter queue for failed extractions
        self._dlq = DeadLetterQueue(base_dir=dead_letter_dir)

    def _discover_files(self) -> list[str]:
        """
        Scan IMS folder for all supported files in the specified subfolders.
        Recursively walks all nested subdirectories within each folder.
        """
        files = []
        root = Path(self._ims_root)
        skipped_ext = {}

        for folder_name in self._folders:
            folder_path = root / folder_name
            if not folder_path.exists():
                logger.warning("Folder not found: %s", folder_path)
                continue

            folder_files = []
            for f in sorted(folder_path.rglob("*")):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in self.SKIP_EXTENSIONS:
                    skipped_ext[ext] = skipped_ext.get(ext, 0) + 1
                    continue
                if self._extractor.is_supported(str(f)):
                    folder_files.append(str(f))
                else:
                    skipped_ext[ext] = skipped_ext.get(ext, 0) + 1

            files.extend(folder_files)
            logger.info("  %s: %d supported files", folder_name, len(folder_files))

        if skipped_ext:
            for ext, count in sorted(skipped_ext.items(), key=lambda x: -x[1]):
                logger.info("  Skipped %d files with unsupported extension: %s", count, ext)

        return files

    def run(self) -> IngestStats:
        """
        Execute the full ingestion pipeline.

        Returns:
            IngestStats with counts, timing, and any errors.
        """
        stats = IngestStats()
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("O'Connors IMS -- BGE-M3 Ingestion Pipeline")
        logger.info("=" * 70)
        logger.info("  IMS Root:     %s", self._ims_root)
        logger.info("  Folders:      %s", ", ".join(self._folders))
        logger.info("  Embedding:    %s", self._embedder.model_name)
        logger.info("  OCR:          %s", "ENABLED" if self._enable_ocr else "DISABLED")
        logger.info("  Store mode:   %s", self._store._mode)
        logger.info("  Dead letter:  %s", self._dlq.base_dir)
        logger.info("=" * 70)

        logger.info("Discovering files...")
        files = self._discover_files()
        stats.files_found = len(files)
        logger.info("Total files found: %d", stats.files_found)

        if not files:
            logger.warning("No files found. Check IMS_ROOT and folder names.")
            return stats

        self._store.create_collection(delete_existing=True)

        all_chunks = []

        for file_idx, file_path in enumerate(files):
            filename = Path(file_path).name
            logger.info("[%d/%d] %s", file_idx + 1, stats.files_found, filename)

            try:
                result = self._extractor.extract(file_path)

                if result.error:
                    stats.files_failed += 1
                    stats.errors.append(f"{filename}: {result.error}")
                    logger.warning("  FAILED: %s", result.error)

                    # SCRUM-37: Route to dead-letter queue
                    self._dlq.enqueue(
                        file_path=file_path,
                        stage="extraction",
                        error=result.error,
                        metadata={"extraction_method": result.method},
                    )

                    fallback = self._chunker.make_fallback_chunk(
                        file_path=file_path,
                        ims_root=self._ims_root,
                        reason=result.error,
                    )
                    all_chunks.append(fallback)
                    stats.files_fallback += 1
                    logger.info("  FALLBACK: stored filename chunk for discoverability")
                    continue

                if not result.text.strip():
                    stats.files_skipped += 1
                    stats.skipped_files.append(filename)
                    logger.info("  SKIPPED: no text extracted")

                    # SCRUM-37: Route to dead-letter queue
                    self._dlq.enqueue(
                        file_path=file_path,
                        stage="extraction",
                        error="No text extracted from document",
                        metadata={"extraction_method": result.method},
                    )

                    fallback = self._chunker.make_fallback_chunk(
                        file_path=file_path,
                        ims_root=self._ims_root,
                        reason="no_text_extracted",
                    )
                    all_chunks.append(fallback)
                    stats.files_fallback += 1
                    logger.info("  FALLBACK: stored filename chunk for discoverability")
                    continue

                stats.total_ocr_pages += result.ocr_pages

                chunks = self._chunker.chunk_document(
                    text=result.text,
                    file_path=file_path,
                    ims_root=self._ims_root,
                    page_count=result.page_count,
                    extraction_method=result.method,
                )

                if not chunks:
                    stats.files_skipped += 1
                    stats.skipped_files.append(filename)
                    logger.info("  SKIPPED: no chunks produced")

                    # SCRUM-37: Route to dead-letter queue
                    self._dlq.enqueue(
                        file_path=file_path,
                        stage="chunking",
                        error="Chunker produced zero chunks despite extracted text",
                        metadata={
                            "text_length": len(result.text),
                            "extraction_method": result.method,
                        },
                    )

                    fallback = self._chunker.make_fallback_chunk(
                        file_path=file_path,
                        ims_root=self._ims_root,
                        reason="chunker_produced_zero",
                    )
                    all_chunks.append(fallback)
                    stats.files_fallback += 1
                    continue

                all_chunks.extend(chunks)
                stats.files_processed += 1
                logger.info(
                    "  OK: %d chunks (%s, %d pages%s)",
                    len(chunks), result.method, result.page_count,
                    f", {result.ocr_pages} OCR" if result.ocr_pages else "",
                )

            except Exception as e:
                # SCRUM-37: Catch any unexpected crash (OOM, segfault, etc.)
                stats.files_failed += 1
                error_msg = f"Unexpected error: {type(e).__name__}: {e}"
                stats.errors.append(f"{filename}: {error_msg}")
                logger.error("  CRASH: %s", error_msg)

                self._dlq.enqueue(
                    file_path=file_path,
                    stage="unknown",
                    error=error_msg,
                )
                continue

        # ── SCRUM-37: Generate DLQ alert report ──────────────────────────
        dlq_report = self._dlq.generate_alert_report()
        stats.dlq_report_path = dlq_report

        logger.info("-" * 70)
        logger.info(
            "Extraction complete. %d chunks from %d files (%d fallback chunks).",
            len(all_chunks), stats.files_processed, stats.files_fallback,
        )

        if not all_chunks:
            logger.warning("No chunks to embed. Pipeline complete.")
            stats.elapsed_seconds = time.time() - start_time
            return stats

        logger.info("Embedding %d chunks with BGE-M3...", len(all_chunks))
        texts = [c.text for c in all_chunks]
        embeddings = self._embedder.embed_batch(texts, batch_size=32)
        logger.info("Embedding complete.")

        logger.info("Storing %d chunks in ChromaDB...", len(all_chunks))
        ids = [c.chunk_id for c in all_chunks]
        documents = [c.text for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]

        stored = self._store.upsert_batch(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        stats.total_chunks = stored

        stats.elapsed_seconds = time.time() - start_time
        logger.info("=" * 70)
        logger.info("INGESTION COMPLETE")
        logger.info("  Files processed: %d / %d", stats.files_processed, stats.files_found)
        logger.info("  Files skipped:   %d", stats.files_skipped)
        logger.info("  Files fallback:  %d", stats.files_fallback)
        logger.info("  Files failed:    %d", stats.files_failed)
        logger.info("  Total chunks:    %d", stats.total_chunks)
        logger.info("  OCR pages:       %d", stats.total_ocr_pages)
        logger.info("  Time:            %.1f seconds", stats.elapsed_seconds)
        logger.info("  Collection size: %d", self._store.count())
        logger.info("  DLQ report:      %s", stats.dlq_report_path)
        logger.info("=" * 70)

        return stats
