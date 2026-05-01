"""
test_dead_letter_queue.py  —  Verify SCRUM-37 DLQ without re-ingesting anything.

Creates 5 deliberately broken "files" that simulate real failure modes
from your pipeline, runs them through the DLQ, and checks everything works.

Usage:
    cd D:\Capstone S1-43\chatbot-agent
    python test_dead_letter_queue.py

No ChromaDB, no BGE-M3, no GPU needed — pure unit test.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

from dead_letter_queue import DeadLetterQueue


def create_test_files(tmp_dir: str) -> list[dict]:
    """
    Create fake files that simulate the failure modes you've actually seen
    in the O'Connors IMS pipeline.
    """
    test_cases = []

    # 1. DRM-locked PDF (simulates Techstreet/FileOpen Australian Standards)
    drm_path = Path(tmp_dir) / "AS_3823_2024_Electrical_Installations.pdf"
    drm_path.write_bytes(b"%PDF-1.4 encrypted DRM content")
    test_cases.append({
        "file_path": str(drm_path),
        "stage": "extraction",
        "error": "DRM/FileOpen protected — cannot extract text. "
                 "Techstreet license restricts programmatic access.",
        "metadata": {"file_size": drm_path.stat().st_size, "category": "Australian Standard"},
    })

    # 2. Corrupt PDF (simulates the 2 failed files from your 918-file run)
    corrupt_path = Path(tmp_dir) / "SOP-099_Valve_Maintenance.pdf"
    corrupt_path.write_bytes(b"NOT A REAL PDF AT ALL\x00\xff\xfe")
    test_cases.append({
        "file_path": str(corrupt_path),
        "stage": "extraction",
        "error": "PdfReadError: invalid PDF header, file appears corrupt or damaged.",
        "metadata": {"file_size": corrupt_path.stat().st_size, "category": "SOP"},
    })

    # 3. OCR timeout (simulates PaddleOCR hanging on a huge scanned doc)
    scanned_path = Path(tmp_dir) / "Procedure_042_Fire_Safety_Scanned.pdf"
    scanned_path.write_bytes(b"%PDF-1.4 scanned image only" + b"\x00" * 5000)
    test_cases.append({
        "file_path": str(scanned_path),
        "stage": "ocr",
        "error": "TimeoutError: PaddleOCR exceeded 120s limit on 48-page scanned document.",
        "metadata": {"file_size": scanned_path.stat().st_size, "pages": 48},
    })

    # 4. Embedding failure (simulates OOM when BGE-M3 hits a massive chunk)
    big_doc_path = Path(tmp_dir) / "Policy_Annual_Report_2025.docx"
    big_doc_path.write_bytes(b"PK\x03\x04 fake docx content")
    test_cases.append({
        "file_path": str(big_doc_path),
        "stage": "embedding",
        "error": "OutOfMemoryError: CUDA OOM — BGE-M3 failed on chunk batch "
                 "(batch_size=64, total_tokens=198000). Reduce batch size or chunk length.",
        "metadata": {"file_size": big_doc_path.stat().st_size, "chunks_attempted": 340},
    })

    # 5. ChromaDB write failure (simulates the Azure wipe issue)
    form_path = Path(tmp_dir) / "SWMS_Template_Blank.xlsx"
    form_path.write_bytes(b"PK\x03\x04 fake xlsx")
    test_cases.append({
        "file_path": str(form_path),
        "stage": "vectorstore_write",
        "error": "ConnectionError: ChromaDB server at azure-chroma:8000 refused connection. "
                 "Container may have restarted (no persistent volume).",
        "metadata": {"file_size": form_path.stat().st_size, "chunks_ready": 54},
    })

    return test_cases


def run_tests():
    """Run all DLQ tests and verify outputs."""
    tmp_dir = tempfile.mkdtemp(prefix="dlq_test_")
    dlq_dir = os.path.join(tmp_dir, "dead_letter")

    print(f"\n{'='*70}")
    print("  SCRUM-37 DLQ VERIFICATION TEST")
    print(f"{'='*70}")
    print(f"  Temp dir: {tmp_dir}\n")

    try:
        # ---- Initialize DLQ ----
        dlq = DeadLetterQueue(base_dir=dlq_dir)
        test_cases = create_test_files(tmp_dir)

        # ---- Enqueue all failures ----
        print("  Enqueuing test failures...\n")
        for i, tc in enumerate(test_cases, 1):
            record = dlq.enqueue(
                file_path=tc["file_path"],
                stage=tc["stage"],
                error=tc["error"],
                metadata=tc["metadata"],
            )
            print(f"    [{i}/5] {record['file_name']:<50} stage={record['stage']}")

        # ---- Generate alert report ----
        print()
        report_path = dlq.generate_alert_report()

        # ============================================================
        # VERIFICATION CHECKS
        # ============================================================
        print(f"\n{'-'*70}")
        print("  VERIFICATION CHECKS")
        print(f"{'-'*70}\n")

        passed = 0
        total = 6

        # Check 1: All 5 files copied to failed_files/
        copied = list(Path(dlq_dir, "failed_files").glob("*"))
        ok = len(copied) == 5
        print(f"  [{'PASS' if ok else 'FAIL'}] 1. Failed files copied: {len(copied)}/5")
        passed += int(ok)

        # Check 2: All 5 error records written
        records = list(Path(dlq_dir, "error_records").glob("*.json"))
        ok = len(records) == 5
        print(f"  [{'PASS' if ok else 'FAIL'}] 2. Error records written: {len(records)}/5")
        passed += int(ok)

        # Check 3: Each record has required fields
        required_fields = {"file_name", "file_path", "stage", "error", "timestamp", "metadata"}
        all_valid = True
        for rec_path in records:
            rec = json.loads(rec_path.read_text())
            if not required_fields.issubset(rec.keys()):
                all_valid = False
                break
        print(f"  [{'PASS' if all_valid else 'FAIL'}] 3. All records have required fields")
        passed += int(all_valid)

        # Check 4: Report files generated
        json_reports = list(Path(dlq_dir, "reports").glob("*.json"))
        txt_reports = list(Path(dlq_dir, "reports").glob("*.txt"))
        ok = len(json_reports) >= 1 and len(txt_reports) >= 1
        print(f"  [{'PASS' if ok else 'FAIL'}] 4. Alert reports generated (JSON + TXT)")
        passed += int(ok)

        # Check 5: Error classification works
        report_data = json.loads(json_reports[0].read_text())
        error_types = report_data.get("failures_by_error_type", {})
        ok = "DRM/Encrypted" in error_types and "Corrupt file" in error_types
        print(f"  [{'PASS' if ok else 'FAIL'}] 5. Error classification: {list(error_types.keys())}")
        passed += int(ok)

        # Check 6: clear_record works (retry simulation)
        cleared = dlq.clear_record("SOP-099_Valve_Maintenance.pdf")
        remaining = list(Path(dlq_dir, "error_records").glob("*.json"))
        ok = cleared == 1 and len(remaining) == 4
        print(f"  [{'PASS' if ok else 'FAIL'}] 6. clear_record (retry simulation): "
              f"removed {cleared}, remaining {len(remaining)}/4")
        passed += int(ok)

        # ---- Summary ----
        print(f"\n{'='*70}")
        if passed == total:
            print(f"  ALL {total} CHECKS PASSED  —  DLQ is working correctly")
        else:
            print(f"  {passed}/{total} CHECKS PASSED  —  review failures above")
        print(f"{'='*70}")

        # ---- Show the TXT report so you can see what it looks like ----
        print(f"\n  Sample TXT report preview:\n")
        txt_content = Path(report_path).read_text()
        # Print first 30 lines
        for line in txt_content.split("\n")[:35]:
            print(f"  {line}")
        print("  ...")

    finally:
        # Cleanup temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n  Cleaned up temp dir: {tmp_dir}\n")


if __name__ == "__main__":
    run_tests()
