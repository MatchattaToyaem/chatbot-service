"""Resume ingestion: in-process OCR (fast) + signal.alarm timeout (safe, no threads).

Key fix from previous version: uses signal.alarm instead of threading.Thread
for file timeout. Threads caused PaddleOCR GPU memory corruption when timed-out
threads kept running in background. signal.alarm cleanly interrupts the main
thread with no zombie processes.

Usage on Colab:
  !cd /content/chatbot-agent && PYTHONUNBUFFERED=1 python run_resume_ingest.py
"""

import os, sys, time, logging, json, signal
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, '.')
from ingest.extractor import TextExtractor, ExtractionResult
from ingest.chunker import DocumentChunker
from ingest.embedder import Embedder
from ingest.store import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class FileTimeoutError(Exception):
    """Raised when file extraction exceeds time limit."""
    pass


def _alarm_handler(signum, frame):
    raise FileTimeoutError("Extraction timed out")


def extract_with_timeout(extractor, file_path, timeout=180):
    """
    Run extraction with signal.alarm timeout (Linux/Colab only).

    Unlike the thread-based approach, this cleanly interrupts the main
    thread. No background threads survive to corrupt PaddleOCR GPU state.
    """
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        result = extractor.extract(file_path)
        signal.alarm(0)  # Cancel alarm on success
        return result
    except FileTimeoutError:
        signal.alarm(0)
        logger.warning("  TIMEOUT after %ds: %s", timeout, Path(file_path).name)
        return ExtractionResult(
            file_path=file_path, text="", page_count=0,
            method="skipped", error=f"timeout_{timeout}s",
        )
    except Exception as e:
        signal.alarm(0)
        logger.error("  Extraction error: %s", str(e)[:100])
        return ExtractionResult(
            file_path=file_path, text="", page_count=0,
            method="skipped", error=str(e)[:200],
        )
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def flush_chunks(all_chunks, embedder, store, total_stored):
    """Embed and store a batch of chunks. Returns new total."""
    if not all_chunks:
        return total_stored
    print(f"\n--- Embedding {len(all_chunks)} chunks ---")
    sys.stdout.flush()
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_batch(texts, batch_size=32)
    ids = [c.chunk_id for c in all_chunks]
    docs = [c.text for c in all_chunks]
    metas = [c.metadata for c in all_chunks]
    store.upsert_batch(ids, docs, embeddings, metas)
    total_stored += len(all_chunks)
    print(f"--- Stored on Drive. Total NEW chunks this run: {total_stored} ---\n")
    sys.stdout.flush()
    return total_stored


def main():
    # ── Configuration ────────────────────────────────────────────────────
    IMS_ROOT = "/content/IMS_local"
    COLLECTION = "oconnors_ims_bge_m3"
    LOCAL_PATH = "/content/drive/MyDrive/Capstone/chromadb_local"
    FILE_TIMEOUT = 180  # seconds per file (signal.alarm, clean interrupt)

    # RESUME: skip files already processed across BOTH runs
    # Run 1 (run_safe_ingest.py): files 1-194 flushed → 10,162 chunks
    # Run 2 (run_resume_ingest.py v1): files 195-316 flushed → ~14,287 new chunks
    # Files 317-319 were in progress when crash happened, not flushed
    # This run starts from file 317
    SKIP_FILES = 316

    ALL_FOLDERS = ["Procedures", "SOP's", "Policies", "Australian Standards",
                   "MSDS's", "Forms", "SWMS's", "RDO Calendar"]
    SKIP_EXT = {".xps", ".tmp", ".bak"}
    SUPPORTED = {".pdf", ".docx", ".docm", ".xlsx", ".xlsm", ".xls", ".doc"}

    # ── Initialize components ────────────────────────────────────────────
    extractor = TextExtractor(
        enable_ocr=True,
        ocr_timeout=10,
        min_text_threshold=10,
        ocr_confidence=0.5,
    )

    chunker = DocumentChunker(chunk_size=1200, chunk_overlap=200)
    embedder = Embedder(model_name="BAAI/bge-m3")

    # Connect to EXISTING collection — do NOT delete!
    store = ChromaStore(
        collection_name=COLLECTION,
        mode="local",
        local_path=LOCAL_PATH,
    )

    # ── Discover files (same order as original run) ──────────────────────
    root = Path(IMS_ROOT)
    files = []
    for folder in ALL_FOLDERS:
        folder_path = root / folder
        if not folder_path.exists():
            logger.warning("Folder not found: %s", folder_path)
            continue
        folder_files = sorted([
            str(f) for f in folder_path.rglob("*")
            if f.is_file() and f.suffix.lower() not in SKIP_EXT
            and f.suffix.lower() in SUPPORTED
        ])
        files.extend(folder_files)
        logger.info("  %s: %d files", folder, len(folder_files))

    logger.info("Total files discovered: %d", len(files))

    # ── Verify existing collection ───────────────────────────────────────
    existing_col = store.get_collection()
    existing_count = existing_col.count()
    logger.info("Existing collection '%s': %d chunks already stored", COLLECTION, existing_count)
    logger.info("Resuming from file %d (skipping first %d)", SKIP_FILES + 1, SKIP_FILES)

    remaining_files = files[SKIP_FILES:]
    logger.info("Files to process: %d", len(remaining_files))

    if not remaining_files:
        logger.info("Nothing to resume — all files already processed!")
        return

    # ── Pre-warm PaddleOCR (load once, stays in GPU memory) ──────────────
    logger.info("Pre-warming PaddleOCR engine...")
    import numpy as np
    try:
        ocr_engine = extractor._get_ocr_engine()
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        ocr_engine.ocr(test_img, cls=True)
        logger.info("PaddleOCR warm — loaded in GPU memory, will stay resident.")
    except Exception as e:
        logger.warning("PaddleOCR warm-up failed: %s (OCR may still work)", e)

    # ── Process remaining files ──────────────────────────────────────────
    processed = 0
    skipped = 0
    failed = 0
    fallback = 0
    new_chunks = 0
    ocr_pages_done = 0
    all_chunks = []
    start_time = time.time()

    for idx, file_path in enumerate(remaining_files):
        global_idx = SKIP_FILES + idx + 1
        filename = Path(file_path).name
        sys.stdout.write(f"[{global_idx}/{len(files)}] {filename[:60]}...")
        sys.stdout.flush()

        # Extract with signal.alarm timeout (clean, no background threads)
        result = extract_with_timeout(extractor, file_path, timeout=FILE_TIMEOUT)

        if result is None or result.error:
            error_msg = result.error if result else "unknown_error"
            failed += 1
            fb = chunker.make_fallback_chunk(file_path, IMS_ROOT, error_msg[:80])
            all_chunks.append(fb)
            fallback += 1
            print(f" FAILED ({error_msg[:40]})")
            if len(all_chunks) >= 500:
                new_chunks = flush_chunks(all_chunks, embedder, store, new_chunks)
                all_chunks = []
            continue

        if not result.text.strip():
            skipped += 1
            fb = chunker.make_fallback_chunk(file_path, IMS_ROOT, "no_text")
            all_chunks.append(fb)
            fallback += 1
            print(" SKIP+FALLBACK")
            if len(all_chunks) >= 500:
                new_chunks = flush_chunks(all_chunks, embedder, store, new_chunks)
                all_chunks = []
            continue

        ocr_pages_done += result.ocr_pages
        chunks = chunker.chunk_document(
            text=result.text,
            file_path=file_path,
            ims_root=IMS_ROOT,
            page_count=result.page_count,
            extraction_method=result.method,
        )

        if not chunks:
            skipped += 1
            fb = chunker.make_fallback_chunk(file_path, IMS_ROOT, "zero_chunks")
            all_chunks.append(fb)
            fallback += 1
            print(" SKIP+FALLBACK")
            if len(all_chunks) >= 500:
                new_chunks = flush_chunks(all_chunks, embedder, store, new_chunks)
                all_chunks = []
            continue

        all_chunks.extend(chunks)
        processed += 1
        ocr_tag = f", {result.ocr_pages} OCR" if result.ocr_pages else ""
        print(f" OK {len(chunks)} chunks ({result.method}{ocr_tag})")

        # Batch embed + store every 500 chunks
        if len(all_chunks) >= 500:
            new_chunks = flush_chunks(all_chunks, embedder, store, new_chunks)
            all_chunks = []

    # ── Final batch ──────────────────────────────────────────────────────
    if all_chunks:
        print(f"\n--- Final batch ---")
        new_chunks = flush_chunks(all_chunks, embedder, store, new_chunks)

    # ── Report ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    final_count = store.count()

    logger.info("=" * 60)
    logger.info("RESUME INGESTION COMPLETE")
    logger.info("  Resumed from:  file %d", SKIP_FILES + 1)
    logger.info("  Processed:     %d / %d remaining", processed, len(remaining_files))
    logger.info("  Skipped:       %d", skipped)
    logger.info("  Failed:        %d", failed)
    logger.info("  Fallback:      %d", fallback)
    logger.info("  New chunks:    %d", new_chunks)
    logger.info("  OCR pages:     %d", ocr_pages_done)
    logger.info("  Collection:    %d total", final_count)
    logger.info("  Time:          %.1f min", elapsed / 60)
    logger.info("=" * 60)

    report = {
        "resumed_from": SKIP_FILES + 1,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "fallback": fallback,
        "new_chunks": new_chunks,
        "ocr_pages": ocr_pages_done,
        "total_collection": final_count,
        "previous_collection": existing_count,
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    report_path = "/content/drive/MyDrive/Capstone/resume_ingest_report_v2.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()
