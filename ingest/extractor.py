"""
Text extraction from PDF, DOCX, DOC, and XLSX files.

Extraction strategy (tiered):
    1. pdfplumber for embedded-text PDFs (fast, reliable).
    2. PaddleOCR 2.9.1 fallback for scanned pages or SAI Global watermarked PDFs
       where pdfplumber returns empty strings despite having visible content.
    3. python-docx for DOCX/DOCM files (paragraphs + tables).
    4. openpyxl for XLSX/XLSM/XLS files (extracts all cell values).
    5. subprocess antiword/catdoc for legacy .doc files.

PaddleOCR API: 2.9.1 uses ocr.ocr() with list-based output.
PaddlePaddle: 2.6.2 (GPU on Colab, CPU on Windows).
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of extracting text from a single file."""
    file_path: str
    text: str
    page_count: int
    method: str  # "pdfplumber", "paddleocr", "docx", "xlsx", "doc", "skipped"
    ocr_pages: int = 0
    error: Optional[str] = None


class TextExtractor:
    """
    Extracts text from PDF, DOCX, DOC, and XLSX files using a tiered strategy.

    For PDFs: pdfplumber first, PaddleOCR fallback for pages with
    insufficient text (below min_text_threshold). This handles both
    scanned documents and SAI Global watermarked Australian Standards
    where encoded text streams confuse pdfplumber.

    For DOCX/DOCM: python-docx paragraph + table extraction.
    For XLSX/XLSM/XLS: openpyxl cell value extraction across all sheets.
    For DOC: antiword subprocess fallback.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".docm",
        ".xlsx", ".xlsm", ".xls",
        ".doc",
    }

    # Known DRM patterns in PDF metadata or content
    DRM_INDICATORS = [
        "sai global", "techstreet", "licensed to",
        "copyright", "not for resale", "single user licence",
    ]

    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_timeout: int = 10,
        min_text_threshold: int = 10,
        ocr_confidence: float = 0.5,
    ):
        self._enable_ocr = enable_ocr
        self._ocr_timeout = ocr_timeout
        self._min_text_threshold = min_text_threshold
        self._ocr_confidence = ocr_confidence
        self._ocr_engine = None

    # ------------------------------------------------------------------
    # OCR engine — lazy load so non-OCR runs don't pay the import cost
    # ------------------------------------------------------------------

    def _get_ocr_engine(self):
        """Load PaddleOCR 2.9.1 on first use."""
        if self._ocr_engine is None:
            from paddleocr import PaddleOCR

            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=True,
            )
            logger.info("PaddleOCR 2.9.1 engine loaded (GPU enabled).")
        return self._ocr_engine

    # ------------------------------------------------------------------
    # DRM detection
    # ------------------------------------------------------------------

    def _check_drm(self, file_path: str) -> bool:
        """
        Check if a PDF is DRM-protected by examining metadata and first page.
        Returns True if DRM detected.
        """
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                meta = pdf.metadata or {}
                meta_str = " ".join(str(v).lower() for v in meta.values() if v)
                for indicator in self.DRM_INDICATORS:
                    if indicator in meta_str:
                        return True

                if hasattr(pdf, 'is_encrypted') and pdf.is_encrypted:
                    return True

        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # PDF extraction
    # ------------------------------------------------------------------

    def _extract_pdf(self, file_path: str) -> ExtractionResult:
        """Extract text from PDF using pdfplumber, with OCR fallback."""
        import pdfplumber

        if self._check_drm(file_path):
            filename = Path(file_path).name
            logger.warning("  DRM detected: %s — extracting what we can", filename)

        pages_text = []
        ocr_page_count = 0
        total_pages = 0

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    text = ""
                    try:
                        text = (page.extract_text() or "").strip()
                    except Exception as e:
                        logger.debug("pdfplumber failed on page %d: %s", page_num, e)

                    # Also try extracting tables as text
                    if len(text) < self._min_text_threshold:
                        try:
                            tables = page.extract_tables()
                            if tables:
                                table_text = []
                                for table in tables:
                                    for row in table:
                                        cells = [str(c).strip() for c in row if c]
                                        if cells:
                                            table_text.append(" | ".join(cells))
                                table_str = "\n".join(table_text).strip()
                                if len(table_str) > len(text):
                                    text = table_str
                        except Exception:
                            pass

                    # If still insufficient text and OCR is enabled, fall back
                    if len(text) < self._min_text_threshold and self._enable_ocr:
                        ocr_text = self._ocr_page(file_path, page_num)
                        if ocr_text and len(ocr_text) > len(text):
                            text = ocr_text
                            ocr_page_count += 1

                    if text:
                        pages_text.append(text)

        except Exception as e:
            error_str = str(e).lower()
            if "encrypted" in error_str or "password" in error_str:
                logger.warning("  DRM/encrypted PDF: %s", file_path)
                return ExtractionResult(
                    file_path=file_path, text="", page_count=0,
                    method="skipped", error=f"DRM/encrypted: {e}",
                )
            logger.error("PDF extraction failed for %s: %s", file_path, e)
            return ExtractionResult(
                file_path=file_path, text="", page_count=0,
                method="skipped", error=str(e),
            )

        combined = "\n\n".join(pages_text)
        method = "pdfplumber" if ocr_page_count == 0 else "pdfplumber+paddleocr"

        return ExtractionResult(
            file_path=file_path,
            text=combined,
            page_count=total_pages,
            method=method,
            ocr_pages=ocr_page_count,
        )

    def _ocr_page(self, pdf_path: str, page_num: int) -> str:
        """OCR a single PDF page using PaddleOCR 2.9.1 (ocr.ocr() API)."""
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=200,
            )
            if not images:
                return ""

            import numpy as np

            img_array = np.array(images[0])
            ocr = self._get_ocr_engine()

            result = ocr.ocr(img_array, cls=True)

            if not result or not result[0]:
                return ""

            lines = []
            for line in result[0]:
                if len(line) >= 2:
                    text_conf = line[1]
                    if isinstance(text_conf, tuple) and len(text_conf) >= 2:
                        text, conf = text_conf[0], text_conf[1]
                        if conf >= self._ocr_confidence:
                            lines.append(text)

            return "\n".join(lines)

        except Exception as e:
            logger.warning("OCR failed for %s page %d: %s", pdf_path, page_num, e)
            return ""

    # ------------------------------------------------------------------
    # DOCX / DOCM extraction
    # ------------------------------------------------------------------

    def _extract_docx(self, file_path: str) -> ExtractionResult:
        """Extract text from DOCX/DOCM using python-docx (paragraphs + tables)."""
        try:
            from docx import Document

            doc = Document(file_path)
            parts = []

            for p in doc.paragraphs:
                text = p.text.strip()
                if text:
                    parts.append(text)

            for table in doc.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        parts.append(" | ".join(cells))

            combined = "\n\n".join(parts)

            return ExtractionResult(
                file_path=file_path,
                text=combined,
                page_count=1,
                method="docx",
            )

        except Exception as e:
            logger.error("DOCX extraction failed for %s: %s", file_path, e)
            return ExtractionResult(
                file_path=file_path, text="", page_count=0,
                method="skipped", error=str(e),
            )

    # ------------------------------------------------------------------
    # DOC extraction (legacy Word format)
    # ------------------------------------------------------------------

    def _extract_doc(self, file_path: str) -> ExtractionResult:
        """Extract text from legacy .doc files using antiword or catdoc."""
        text = ""

        try:
            result = subprocess.run(
                ["antiword", file_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                text = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if not text:
            try:
                result = subprocess.run(
                    ["catdoc", file_path],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    text = result.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        if text:
            return ExtractionResult(
                file_path=file_path, text=text,
                page_count=1, method="doc",
            )

        logger.warning("No .doc converter available for %s (install antiword or catdoc)", file_path)
        return ExtractionResult(
            file_path=file_path, text="", page_count=0,
            method="skipped", error="No .doc converter (antiword/catdoc not found)",
        )

    # ------------------------------------------------------------------
    # XLSX / XLSM / XLS extraction
    # ------------------------------------------------------------------

    def _extract_xlsx(self, file_path: str) -> ExtractionResult:
        """Extract text from Excel files using openpyxl."""
        try:
            from openpyxl import load_workbook

            wb = load_workbook(file_path, read_only=True, data_only=True)
            parts = []
            sheet_count = len(wb.sheetnames)

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                sheet_rows = []

                for row in ws.iter_rows(values_only=True):
                    cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                    if cells:
                        sheet_rows.append(" | ".join(cells))

                if sheet_rows:
                    parts.append(f"--- Sheet: {sheet_name} ---")
                    parts.extend(sheet_rows)

            wb.close()
            combined = "\n".join(parts)

            return ExtractionResult(
                file_path=file_path,
                text=combined,
                page_count=sheet_count if parts else 0,
                method="xlsx",
            )

        except Exception as e:
            logger.error("XLSX extraction failed for %s: %s", file_path, e)
            return ExtractionResult(
                file_path=file_path, text="", page_count=0,
                method="skipped", error=str(e),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract text from a single file.

        Args:
            file_path: Absolute path to the file.

        Returns:
            ExtractionResult with extracted text, page count, method used.
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext in (".docx", ".docm"):
            return self._extract_docx(file_path)
        elif ext == ".doc":
            return self._extract_doc(file_path)
        elif ext in (".xlsx", ".xlsm", ".xls"):
            return self._extract_xlsx(file_path)
        else:
            return ExtractionResult(
                file_path=file_path, text="", page_count=0,
                method="skipped", error=f"Unsupported extension: {ext}",
            )

    def is_supported(self, file_path: str) -> bool:
        """Check if the file type is supported for extraction."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
