"""
Extraction Service - Extracts content from various file types
Supports: PDF, DOCX, HTML, XML, ZIP, TXT, JSON, CSV, Excel, and more

Pipeline routing
----------------
- PDF, DOCX, HTML, XML, ZIP, TXT, JSON  →  extract()          → text blob → normal chunking pipeline
- CSV / TSV                     →  extract_row_wise()  → CSVExtractor.extract_and_store()
                           Each row → one Qdrant point. No chunking step.
- XLSX / XLS / XLSM    →  extract_row_wise()  → ExcelExtractor.extract_and_store()
                           Every row of every sheet → one Qdrant point.
                           No classifier. No chunking step.
- .dbrows               →  extract_row_wise()  → _extract_dbrows_row_wise()
                           Used by DatabaseSource for all DB types (Postgres,
                           MySQL, MongoDB).  Each row/document is already
                           pre-formatted text in file_info['rows']; this
                           method embeds and stores them directly.
                           No fake filenames. No chunking step.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from .extractors.json_extractor import ExtractionResult, JSONExtractor
# PDF extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# DOCX extraction
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# CSV/Excel
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExtractionService:
    """Extract content from various file types."""

    # File types that bypass the chunking pipeline and store rows directly.
    # .dbrows is a synthetic sentinel used by DatabaseSource (Postgres/MySQL/MongoDB).
    ROW_WISE_TYPES = {'.csv', '.tsv', '.xlsx', '.xls', '.xlsm', '.dbrows'}
    IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    
    # Image and binary file types — should be skipped or handled separately
    # NOTE: .zip is now supported via ZIPExtractor, so removed from BINARY_TYPES
    BINARY_TYPES = {
        '.svg', '.ico',                                                     # unsupported images
        '.rar', '.7z', '.gz', '.tar', '.exe', '.dll', '.so',               # archives/binaries
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv',                     # media
        '.bin', '.dat', '.pth', '.pkl', '.pyc'                              # raw binary
    }

    def __init__(self):
        logger.info("Extraction Service initialized")

    # =========================================================================
    # PUBLIC — standard pipeline  (PDF, DOCX, TXT, JSON …)
    # =========================================================================

    async def extract(self, file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract content for the normal chunking → embedding → vector-store pipeline.
        CSV / Excel must go through extract_row_wise().
        """
        file_type = file_info.get('file_type', '').lower()
        file_path = file_info.get('file_path')
        content   = file_info.get('content')

        logger.info(f"Extracting: {file_info.get('filename')} ({file_type})")

        if file_type in self.ROW_WISE_TYPES:
            logger.warning(
                f"'{file_type}' must use extract_row_wise(), not extract(). "
                "Returning None to prevent incorrect chunking."
            )
            return None

        if file_type in self.IMAGE_TYPES:
            if not file_path:
                logger.error("No file_path provided for image extraction")
                return None
            text = self._extract_image(file_path)
            if not text or not text.strip():
                logger.warning(f"Empty OCR content extracted from {file_info.get('filename')}")
                return None
            return {
                'filename': file_info.get('filename'),
                'content': text,
                'file_type': file_type,
                'metadata': file_info
            }
        
        if file_type in self.BINARY_TYPES:
            logger.warning(
                f"Binary/image file type '{file_type}' is not supported for text extraction. "
                f"Skipping {file_info.get('filename')}. "
                f"(Supported raster images are handled separately through local OCR)"
            )
            return None

        try:
            if content and not file_path:
                return {
                    'filename': file_info.get('filename'),
                    'content': content,
                    'file_type': file_type,
                    'metadata': file_info
                }

            if not file_path:
                logger.error("No file_path or content provided")
                return None

            if file_type == '.pdf':
                text = self._extract_pdf(file_path)
            elif file_type in ('.docx', '.doc'):
                text = self._extract_docx(file_path)
            elif file_type in ('.html', '.htm'):
                text = self._extract_html(file_path)
            elif file_type == '.xml':
                text = self._extract_xml(file_path)
            elif file_type == '.zip':
                text = self._extract_zip(file_path)
            elif file_type in ('.txt', '.md', '.markdown'):
                text = self._extract_text(file_path)
            elif file_type == '.json':
                text = self._extract_json(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_type}, treating as plain text")
                text = self._extract_text(file_path)

            if not text or not text.strip():
                logger.warning(f"Empty content extracted from {file_info.get('filename')}")
                return None

            return {
                'filename': file_info.get('filename'),
                'content': text,
                'file_type': file_type,
                'metadata': file_info
            }

        except Exception as e:
            logger.error(f"Extraction failed for {file_info.get('filename')}: {e}")
            return None

    # =========================================================================
    # PUBLIC — row-wise pipeline  (CSV, TSV, XLSX, XLS, XLSM)
    # =========================================================================

    async def extract_row_wise(
        self,
        file_info:           Dict[str, Any],
        vector_store,
        knowledge_base_id:   str,
        knowledge_base_name: str,
        user_id:             str,
        file_id:             str,
        batch_size:          int = 8,
        embedding_details:   Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Row-wise extraction + direct Qdrant storage for CSV and Excel files.

        CSV / TSV  → CSVExtractor.extract_and_store()
        Excel      → ExcelExtractor.extract_and_store()
                     Every row of every sheet → one Qdrant point.

        Returns: (success, summary_dict)
        """
        file_type = file_info.get('file_type', '').lower()
        file_path = file_info.get('file_path')
        filename  = file_info.get(
            'filename', Path(file_path).name if file_path else 'unknown'
        )
        source_mapping_id = file_info.get('source_mapping_id')
        is_active = file_info.get('isActive', True)
        source_type_name = file_info.get('source_type_name') or file_info.get('source_type_from_config')

        # .dbrows has no file on disk — file_path is intentionally None
        if not file_path and file_type != '.dbrows':
            return False, {'error': 'No file_path provided', 'file_type': file_type}

        logger.info(f"Row-wise extraction: {filename} ({file_type})")

        try:
            if file_type in ('.csv', '.tsv'):
                return self._extract_csv_row_wise(
                    file_path=file_path, file_id=file_id, filename=filename,
                    knowledge_base_id=knowledge_base_id,
                    knowledge_base_name=knowledge_base_name,
                    user_id=user_id, vector_store=vector_store, batch_size=batch_size,
                    embedding_details=embedding_details,
                    source_mapping_id=source_mapping_id,
                    is_active=is_active,
                    source_type_name=source_type_name,
                )
            elif file_type in ('.xlsx', '.xls', '.xlsm'):
                return self._extract_excel_row_wise(
                    file_path=file_path, file_id=file_id, filename=filename,
                    knowledge_base_id=knowledge_base_id,
                    knowledge_base_name=knowledge_base_name,
                    user_id=user_id, vector_store=vector_store, batch_size=batch_size,
                    embedding_details=embedding_details,
                    source_mapping_id=source_mapping_id,
                    is_active=is_active,
                    source_type_name=source_type_name,
                )
            elif file_type == '.dbrows':
                return self._extract_dbrows_row_wise(
                    file_info=file_info,
                    file_id=file_id,
                    knowledge_base_id=knowledge_base_id,
                    knowledge_base_name=knowledge_base_name,
                    user_id=user_id, vector_store=vector_store, batch_size=batch_size,
                    embedding_details=embedding_details,
                )
            else:
                return False, {
                    'error': f"'{file_type}' is not row-wise. Use extract() instead.",
                    'file_type': file_type,
                }
        except Exception as exc:
            logger.error(f"extract_row_wise failed for '{filename}': {exc}", exc_info=True)
            return False, {'error': str(exc), 'file_type': file_type}

    # =========================================================================
    # PRIVATE — CSV / TSV row-wise
    # =========================================================================

    def _extract_csv_row_wise(
        self, file_path: str, file_id: str, filename: str,
        knowledge_base_id: str, knowledge_base_name: str,
        user_id: str, vector_store, batch_size: int,
        embedding_details: Optional[Dict[str, Any]] = None,
        source_mapping_id: Optional[Any] = None,
        is_active: bool = True,
        source_type_name: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Delegate to CSVExtractor.extract_and_store(). Each row → one Qdrant point."""
        from services.extraction.extractors.csv_extractor import CSVExtractor

        extractor = CSVExtractor()
        success, summary = extractor.extract_and_store(
            file_path=file_path, file_id=file_id, filename=filename,
            knowledge_base_id=knowledge_base_id,
            knowledge_base_name=knowledge_base_name,
            user_id=user_id, vector_store=vector_store, batch_size=batch_size,
            embedding_details=embedding_details,
            source_mapping_id=source_mapping_id,
            is_active=is_active,
            source_type_name=source_type_name,
        )
        summary['file_type'] = Path(file_path).suffix.lower()

        if success:
            logger.info(
                f"✓ CSV row-wise: '{filename}' — "
                f"{summary.get('total_stored', 0)}/{summary.get('total_rows', '?')} rows stored"
            )
        else:
            logger.error(f"✗ CSV row-wise failed '{filename}': {summary.get('error')}")
        return success, summary

    # =========================================================================
    # PRIVATE — Excel row-wise
    # =========================================================================

    def _extract_excel_row_wise(
        self, file_path: str, file_id: str, filename: str,
        knowledge_base_id: str, knowledge_base_name: str,
        user_id: str, vector_store, batch_size: int,
        embedding_details: Optional[Dict[str, Any]] = None,
        source_mapping_id: Optional[Any] = None,
        is_active: bool = True,
        source_type_name: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Delegate to ExcelExtractor.extract_and_store().
        Every row of every sheet → one Qdrant point. No chunking pipeline.
        """
        from services.extraction.extractors.excel_extractor import ExcelExtractor

        extractor = ExcelExtractor()
        success, summary, _ = extractor.extract_and_store(
            file_path=file_path, file_id=file_id, filename=filename,
            knowledge_base_id=knowledge_base_id,
            knowledge_base_name=knowledge_base_name,
            user_id=user_id, vector_store=vector_store, batch_size=batch_size,
            embedding_details=embedding_details,
            source_mapping_id=source_mapping_id,
            is_active=is_active,
            source_type_name=source_type_name,
        )
        summary['file_type'] = Path(file_path).suffix.lower()
        # No fallback_content — Excel rows always go straight to Qdrant
        summary['fallback_content'] = None

        if success:
            logger.info(
                f"✓ Excel row-wise: '{filename}' — "
                f"{summary.get('total_rows_stored', 0)} rows stored"
            )
        else:
            logger.error(f"✗ Excel row-wise failed '{filename}': {summary.get('error')}")
        return success, summary

    # =========================================================================
    # PRIVATE — Database row-wise  (.dbrows sentinel from DatabaseSource)
    # =========================================================================

    def _extract_dbrows_row_wise(
        self,
        file_info:           Dict[str, Any],
        file_id:             str,
        knowledge_base_id:   str,
        knowledge_base_name: str,
        user_id:             str,
        vector_store,
        batch_size:          int,
        embedding_details:   Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Embed and store database rows (Postgres / MySQL / MongoDB) directly
        in Qdrant.  No chunking, no fake filenames, no file I/O.

        DatabaseSource.discover() passes all rows pre-formatted as plain text
        in file_info['rows']:
            [
                {'text': 'col1: val1 | col2: val2', 'row_index': 1, ...},
                ...
            ]

        Each entry becomes one Qdrant point.
        """
        from services.embedding_service import create_embedding_service

        rows:     list = file_info.get('rows', [])
        db_type:  str  = file_info.get('db_type', 'database')
        filename: str  = file_info.get('filename', f'{db_type}_export')
        source_mapping_id = file_info.get('source_mapping_id')
        is_active = file_info.get('isActive', True)
        source_type_name = file_info.get('source_type_name') or file_info.get('source_type_from_config')
        import uuid as _uuid

        if not rows:
            logger.warning("[DB row-wise] No rows in file_info for '%s'", filename)
            return False, {'error': 'No rows provided', 'file_type': '.dbrows'}

        logger.info(
            "[DB row-wise] Embedding %d row(s) from '%s' (%s)",
            len(rows), filename, db_type,
        )

        embedding_service = create_embedding_service(embedding_details or {})
        total_stored = 0
        total_failed = 0
        error_message = None

        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start : batch_start + batch_size]

            # Build chunk dicts — text already formatted by DatabaseSource
            chunks = []
            for entry in batch:
                text      = entry.get('text', '')
                row_index = entry.get('row_index', batch_start + 1)
                chunk_id  = str(_uuid.uuid5(
                    _uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8'),
                    f"{file_id}::dbrow::{row_index}",
                ))
                chunks.append({
                    'text':        text,
                    'chunk_id':    chunk_id,
                    'chunk_index': row_index - 1,
                    'metadata': {
                        'chunk_type':          'db_row',
                        'db_type':             db_type,
                        'row_index':           row_index,
                        'source_db':           filename,
                        'file_id':             file_id,
                        'source_mapping_id':   source_mapping_id,
                        'isActive':            is_active,
                        'source_type_name':    source_type_name,
                        'knowledge_base_id':   knowledge_base_id,
                        'knowledge_base_name': knowledge_base_name,
                        'user_id':             user_id,
                        **{k: v for k, v in entry.items()
                           if k not in ('text', 'row_index')},
                    },
                })

            try:
                texts      = [c['text'] for c in chunks]
                # DEBUG: Log first text to see what we're sending
                if batch_start == 0 and texts:
                    logger.info(f"[DB DEBUG] First chunk text (len={len(texts[0])}): {texts[0][:300]}...")
                embeddings = embedding_service.embed_documents_with_sparse(texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk['embedding_dense'] = emb['dense']
                    chunk['embedding_sparse'] = emb['sparse']
                    # Count tokens (rough estimate: ~1 token per word)
                    chunk['token_count'] = len(chunk['text'].split())
                    # Count actual characters
                    chunk['char_count'] = len(chunk['text'])

                stored = vector_store.store_chunks_sync(chunks)
                total_stored += stored
                logger.debug(
                    "  [DB] batch %d–%d: stored %d",
                    batch_start + 1, batch_start + len(batch), stored,
                )
            except Exception as exc:
                logger.error(
                    "  [DB] batch %d–%d failed: %s",
                    batch_start + 1, batch_start + len(batch), exc,
                )
                # Capture the error message (model error takes priority)
                if not error_message:
                    error_message = str(exc)
                total_failed += len(batch)

        success = total_stored > 0
        
        # Calculate total tokens and characters from all rows
        total_tokens = sum(len(row.get('text', '').split()) for row in rows)
        total_characters = sum(len(row.get('text', '')) for row in rows)
        
        summary = {
            'file_type':      '.dbrows',
            'db_type':        db_type,
            'total_rows':     len(rows),
            'total_stored':   total_stored,
            'total_chunks':   total_stored,
            'total_tokens':   total_tokens,
            'total_characters': total_characters,
            'total_failed':   total_failed,
            'extraction_method': 'db_row_wise_direct',
        }

        # Add error message if extraction failed
        if error_message:
            summary['error'] = error_message
        elif not success:
            summary['error'] = f"row-wise extraction failed: 0/{len(rows)} rows stored"

        if success:
            logger.info(
                "✓ DB row-wise: '%s' — %d/%d rows stored",
                filename, total_stored, len(rows),
            )
        else:
            logger.error("✗ DB row-wise: no rows stored for '%s' (%s)", filename, error_message or "unknown error")

        return success, summary

    # =========================================================================
    # PRIVATE — standard extractors  (PDF, DOCX, TXT, JSON)
    # =========================================================================

    def _extract_pdf(self, file_path: str) -> str:
        from services.extraction.extractors.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()
        markdown_content = extractor.extract(file_path)
        self._write_markdown_file(file_path, markdown_content)
        return markdown_content

    def _extract_docx(self, file_path: str) -> str:
        from services.extraction.extractors.docx_extractor import DocxExtractor
        extractor = DocxExtractor()
        markdown_content = extractor.extract(file_path)
        self._write_markdown_file(file_path, markdown_content)
        return markdown_content

    def _extract_text(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _extract_json(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)

    def _extract_html(self, file_path: str) -> str:
        """Extract HTML file using HTMLExtractor"""
        from services.extraction.extractors.html_extractor import HTMLExtractor
        extractor = HTMLExtractor()
        markdown_content, success, metadata = extractor.extract(file_path)
        if success:
            self._write_markdown_file(file_path, markdown_content)
        return markdown_content if success else ""

    def _extract_xml(self, file_path: str) -> str:
        """Extract XML file using XMLExtractor"""
        from services.extraction.extractors.xml_extractor import XMLExtractor
        extractor = XMLExtractor()
        markdown_content, success, metadata = extractor.extract(file_path)
        if success:
            self._write_markdown_file(file_path, markdown_content)
        return markdown_content if success else ""

    def _extract_zip(self, file_path: str) -> str:
        """Extract ZIP archive using ZIPExtractor"""
        from services.extraction.extractors.zip_extractor import ZIPExtractor
        extractor = ZIPExtractor()
        markdown_content, success, metadata = extractor.extract(file_path)
        if success:
            self._write_markdown_file(file_path, markdown_content)
        return markdown_content if success else ""

    def _extract_image(self, file_path: str) -> str:
        """Extract text from an image using local OCR."""
        from services.extraction.extractors.image_extractor import ImageExtractor
        extractor = ImageExtractor()
        markdown_content = extractor.extract(file_path)
        self._write_markdown_file(file_path, markdown_content)
        return markdown_content

    # =========================================================================
    # PRIVATE — shared utility
    # =========================================================================

    def _write_markdown_file(self, source_file_path: str, markdown_content: str) -> None:
        """Persist extracted markdown under output_files/markdown (PDF/DOCX only)."""
        if not markdown_content or not markdown_content.strip():
            return
        source_path = Path(source_file_path)
        output_dir = Path.cwd() / "output_files" / "markdown"
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = output_dir / f"{source_path.stem}.md"
        markdown_path.write_text(markdown_content, encoding="utf-8")
        logger.info(f"Saved markdown: {markdown_path}")
