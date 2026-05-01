"""
dead_letter_queue.py  —  SCRUM-37: Dead-Letter Queue for failed document extractions.

Integrates with the existing ingest pipeline. When a document fails at any stage
(download, extraction, OCR, chunking, embedding), the DLQ:
  1. Copies / references the failed file into a dedicated dead-letter folder.
  2. Writes a structured JSON error record (filename, stage, error, timestamp, retries).
  3. Generates a summary alert report after each pipeline run.

Usage (standalone):
    dlq = DeadLetterQueue(base_dir="./dead_letter")
    dlq.enqueue(file_path="somefile.pdf", stage="extraction", error=str(e))
    dlq.generate_alert_report()

Usage (inside existing ingest loop):
    See integration example at bottom of this file.
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class DeadLetterQueue:
    """Manages failed document routing, logging, and alerting for the ingest pipeline."""

    def __init__(self, base_dir: str = "./dead_letter"):
        """
        Args:
            base_dir: Root folder for the dead-letter queue.
                      Structure created:
                        dead_letter/
                        ├── failed_files/      # copies of failed source files
                        ├── error_records/     # one JSON per failure
                        └── reports/           # summary alert reports
        """
        self.base_dir = Path(base_dir)
        self.failed_files_dir = self.base_dir / "failed_files"
        self.error_records_dir = self.base_dir / "error_records"
        self.reports_dir = self.base_dir / "reports"

        # Create directories
        for d in [self.failed_files_dir, self.error_records_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # In-memory ledger for current run
        self._current_run: list[dict] = []
        self._run_start = datetime.now(timezone.utc)

        logger.info(f"DeadLetterQueue initialized at: {self.base_dir}")

    # ------------------------------------------------------------------
    # Core: enqueue a failed document
    # ------------------------------------------------------------------
    def enqueue(
        self,
        file_path: str,
        stage: str,
        error: str,
        retry_count: int = 0,
        copy_file: bool = True,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Record a failed document in the dead-letter queue.

        Args:
            file_path:   Path to the original source file.
            stage:       Pipeline stage where it failed.
                         One of: 'download', 'extraction', 'ocr', 'chunking',
                         'embedding', 'vectorstore_write', 'unknown'.
            error:       The error message / traceback snippet.
            retry_count: How many times this file has been retried (default 0).
            copy_file:   If True, copy the source file into dead_letter/failed_files/.
            metadata:    Any extra info (file size, page count, etc.).

        Returns:
            The error record dict that was persisted.
        """
        src = Path(file_path)
        timestamp = datetime.now(timezone.utc).isoformat()

        # --- 1. Copy failed file (best-effort) ---
        copied_to = None
        if copy_file and src.exists():
            dest = self.failed_files_dir / src.name
            # Avoid overwriting: append timestamp stub if collision
            if dest.exists():
                stem = src.stem
                suffix = src.suffix
                ts_stub = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                dest = self.failed_files_dir / f"{stem}_{ts_stub}{suffix}"
            try:
                shutil.copy2(str(src), str(dest))
                copied_to = str(dest)
                logger.debug(f"Copied failed file to: {dest}")
            except Exception as copy_err:
                logger.warning(f"Could not copy failed file {src}: {copy_err}")

        # --- 2. Build structured error record ---
        record = {
            "file_name": src.name,
            "file_path": str(src),
            "copied_to": copied_to,
            "stage": stage,
            "error": error[:2000],  # truncate very long tracebacks
            "retry_count": retry_count,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }

        # --- 3. Persist as individual JSON ---
        safe_name = src.stem.replace(" ", "_")[:80]
        ts_stub = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        record_path = self.error_records_dir / f"{safe_name}_{ts_stub}.json"
        try:
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        except Exception as write_err:
            logger.error(f"Failed to write error record: {write_err}")

        # --- 4. Add to in-memory ledger ---
        self._current_run.append(record)

        logger.info(
            f"DLQ enqueued: {src.name} | stage={stage} | "
            f"error={error[:100]}..."
        )
        return record

    # ------------------------------------------------------------------
    # Alert report generation
    # ------------------------------------------------------------------
    def generate_alert_report(self) -> str:
        """
        Generate a summary report for the current pipeline run.
        Writes both JSON and human-readable TXT to dead_letter/reports/.

        Returns:
            Path to the generated TXT report.
        """
        run_end = datetime.now(timezone.utc)
        ts_stub = run_end.strftime("%Y%m%d_%H%M%S")

        # ---- Aggregate stats ----
        total_failures = len(self._current_run)
        by_stage: dict[str, int] = {}
        by_error_type: dict[str, int] = {}
        for rec in self._current_run:
            stage = rec["stage"]
            by_stage[stage] = by_stage.get(stage, 0) + 1
            # Classify error type from message
            err_type = self._classify_error(rec["error"])
            by_error_type[err_type] = by_error_type.get(err_type, 0) + 1

        report_data = {
            "run_start": self._run_start.isoformat(),
            "run_end": run_end.isoformat(),
            "total_failures": total_failures,
            "failures_by_stage": by_stage,
            "failures_by_error_type": by_error_type,
            "records": self._current_run,
        }

        # ---- Write JSON report ----
        json_path = self.reports_dir / f"dlq_report_{ts_stub}.json"
        json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

        # ---- Write human-readable TXT ----
        txt_path = self.reports_dir / f"dlq_report_{ts_stub}.txt"
        lines = [
            "=" * 70,
            "  DEAD-LETTER QUEUE  —  PIPELINE RUN ALERT REPORT",
            "=" * 70,
            f"  Run window : {self._run_start.isoformat()} → {run_end.isoformat()}",
            f"  Total failures : {total_failures}",
            "",
            "  Failures by stage:",
        ]
        for stage, count in sorted(by_stage.items()):
            lines.append(f"    {stage:<25} {count}")
        lines.append("")
        lines.append("  Failures by error type:")
        for etype, count in sorted(by_error_type.items()):
            lines.append(f"    {etype:<25} {count}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("  INDIVIDUAL FAILURE DETAILS")
        lines.append("-" * 70)
        for i, rec in enumerate(self._current_run, 1):
            lines.append(f"\n  [{i}] {rec['file_name']}")
            lines.append(f"      Stage    : {rec['stage']}")
            lines.append(f"      Error    : {rec['error'][:200]}")
            lines.append(f"      Time     : {rec['timestamp']}")
            if rec["copied_to"]:
                lines.append(f"      Copied to: {rec['copied_to']}")
        lines.append("\n" + "=" * 70)

        txt_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info(f"DLQ alert report written: {txt_path}")
        print(f"\n{'='*70}")
        print(f"  DLQ REPORT: {total_failures} failures logged")
        print(f"  JSON : {json_path}")
        print(f"  TXT  : {txt_path}")
        print(f"{'='*70}\n")

        return str(txt_path)

    # ------------------------------------------------------------------
    # Retry support
    # ------------------------------------------------------------------
    def get_failed_files(self) -> list[dict]:
        """Load all error records from disk (across all runs)."""
        records = []
        for p in sorted(self.error_records_dir.glob("*.json")):
            try:
                records.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return records

    def clear_record(self, file_name: str) -> int:
        """Remove all error records for a given file (e.g. after successful retry)."""
        removed = 0
        for p in self.error_records_dir.glob("*.json"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                if rec.get("file_name") == file_name:
                    p.unlink()
                    removed += 1
            except Exception:
                continue
        logger.info(f"Cleared {removed} DLQ record(s) for: {file_name}")
        return removed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_error(error_msg: str) -> str:
        """Classify an error message into a human-readable category."""
        lower = error_msg.lower()
        if "drm" in lower or "fileopen" in lower or "encrypted" in lower:
            return "DRM/Encrypted"
        if "ocr" in lower or "paddle" in lower:
            return "OCR failure"
        if "timeout" in lower or "timed out" in lower:
            return "Timeout"
        if "corrupt" in lower or "damaged" in lower or "invalid" in lower:
            return "Corrupt file"
        if "permission" in lower or "access denied" in lower:
            return "Permission denied"
        if "memory" in lower or "oom" in lower or "ram" in lower:
            return "Out of memory"
        if "connection" in lower or "network" in lower:
            return "Network error"
        return "Other"


# ======================================================================
# INTEGRATION EXAMPLE — drop this into your existing ingest loop
# ======================================================================
#
#   from dead_letter_queue import DeadLetterQueue
#
#   dlq = DeadLetterQueue(base_dir="./dead_letter")
#
#   for file_path in files_to_ingest:
#       try:
#           text = extractor.extract(file_path)       # stage: extraction
#           chunks = chunker.chunk(text)               # stage: chunking
#           store.add(chunks)                          # stage: vectorstore_write
#       except Exception as e:
#           dlq.enqueue(
#               file_path=file_path,
#               stage="extraction",  # or whichever stage failed
#               error=str(e),
#               metadata={"file_size": os.path.getsize(file_path)}
#           )
#
#   dlq.generate_alert_report()
#
