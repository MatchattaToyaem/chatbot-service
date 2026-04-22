"""
CLI entry point for the IMS ingestion pipeline.

Usage:
    python run_ingest.py                          # Priority 1 only, with OCR
    python run_ingest.py --skip-ocr               # Priority 1, no OCR (fast)
    python run_ingest.py --all-folders             # All IMS folders
    python run_ingest.py --local                   # Write to local ChromaDB
    python run_ingest.py --folders "Procedures" "Policies"  # Specific folders
"""

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from ingest.pipeline import IngestPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="O'Connors IMS BGE-M3 Ingestion Pipeline")

    parser.add_argument(
        "--ims-root",
        default=os.getenv("IMS_ROOT", "/content/IMS"),
        help="Path to IMS root folder",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION", "oconnors_ims_bge_m3"),
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        help="Embedding model name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "1200")),
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "200")),
        help="Chunk overlap in characters",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Disable PaddleOCR (pdfplumber only — faster but misses scanned pages)",
    )
    parser.add_argument(
        "--ocr-timeout",
        type=int,
        default=10,
        help="Timeout per OCR page in seconds",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Write to local ChromaDB instead of Mat's Azure server",
    )
    parser.add_argument(
        "--local-path",
        default=os.getenv("CHROMA_LOCAL_PATH", "./chromadb_data_bge_m3"),
        help="Local ChromaDB directory (only used with --local)",
    )
    parser.add_argument(
        "--all-folders",
        action="store_true",
        help="Process all IMS folders (not just Priority 1)",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help="Specific folders to process (e.g., Procedures Policies)",
    )
    parser.add_argument(
        "--save-log",
        default=None,
        help="Save ingestion stats to JSON file",
    )

    args = parser.parse_args()

    # Determine folders
    if args.folders:
        folders = args.folders
    elif args.all_folders:
        folders = IngestPipeline.ALL_FOLDERS
    else:
        folders = IngestPipeline.PRIORITY_1

    # Determine store mode
    store_mode = "local" if args.local else "remote"

    # Build and run pipeline
    pipeline = IngestPipeline(
        ims_root=args.ims_root,
        collection_name=args.collection,
        embedding_model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        store_mode=store_mode,
        local_chroma_path=args.local_path,
        enable_ocr=not args.skip_ocr,
        ocr_timeout=args.ocr_timeout,
        folders=folders,
    )

    stats = pipeline.run()

    # Save log if requested
    if args.save_log:
        os.makedirs(os.path.dirname(args.save_log) or ".", exist_ok=True)
        with open(args.save_log, "w") as f:
            json.dump(stats.summary(), f, indent=2)
        logging.info("Stats saved to %s", args.save_log)

    # Exit code based on success
    if stats.files_failed > stats.files_processed:
        sys.exit(1)


if __name__ == "__main__":
    main()
