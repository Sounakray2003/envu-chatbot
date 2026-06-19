"""
Plain Text File Extractor
- No BaseExtractor dependency
- Handles .txt, .text, and .log files
- Supports multiple encodings (UTF-8, Latin-1, etc.)
- Returns markdown string with page markers
"""

from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Supported formats
SUPPORTED_FORMATS = ['txt', 'text', 'log']


class TextExtractor:
    """
    Plain Text Extractor: Standalone text file processing.

    Features:
    - Handles various encodings (UTF-8, Latin-1, cp1252, etc.)
    - Estimates pages based on content (500 words per page)
    - Adds page markers to markdown
    - Preserves line structure
    """

    def __init__(self):
        """Initialize text extractor"""
        logger.info("Text Extractor initialized")

    # =========================================================================
    # Public entry point
    # =========================================================================

    def extract(self, file_path: str) -> str:
        """
        Extract plain text file and return Markdown string.

        Flow:
          1. Validate file exists and has supported extension
          2. Read content with encoding detection
          3. Check if file is not empty
          4. Estimate page count based on word count
          5. Build markdown with page markers
          6. Return markdown string

        Args:
            file_path: Path to the text file.

        Returns:
            Markdown string of extracted content.

        Raises:
            RuntimeError: If extraction fails (empty file, encoding error, etc.)
        """
        logger.info(f"Extracting text: {file_path}")

        # Validate file
        path = Path(file_path)
        if not path.exists():
            raise RuntimeError(f"File does not exist: {file_path}")

        file_ext = path.suffix.lower().lstrip('.')
        if file_ext not in SUPPORTED_FORMATS:
            raise RuntimeError(
                f"Unsupported file type: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )

        # Read content with encoding detection
        content = self._read_with_encoding(file_path)
        if not content or not content.strip():
            raise RuntimeError("File is empty or contains no text")

        # Estimate pages and build markdown
        txt_stem = path.stem
        word_count = len(content.split())
        estimated_pages = max(1, (word_count // 500) + 1)

        markdown_content = self._build_markdown_with_pages(
            content, txt_stem, file_ext, estimated_pages
        )

        logger.info(f"✓ Text extraction complete: {estimated_pages} pages, {word_count} words")

        return markdown_content


    # =========================================================================
    # Encoding detection
    # =========================================================================

    def _read_with_encoding(self, file_path: str) -> str:
        """
        Read text file, trying multiple encodings.

        Encodings tried in order: UTF-8, UTF-16, Latin-1, CP1252, ISO-8859-1.
        Falls back to binary read with 'ignore' errors if all else fails.
        """
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.info(f"Successfully read with {encoding} encoding")
                return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"Error reading with {encoding}: {e}")
                continue

        # Last resort: binary read and decode with errors='ignore'
        try:
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
            logger.warning("Read with error handling (some characters may be lost)")
            return content
        except Exception as e:
            raise RuntimeError(f"Could not read file with any encoding: {e}")

    # =========================================================================
    # Markdown formatting
    # =========================================================================

    def _build_markdown_with_pages(
        self,
        content: str,
        txt_stem: str,
        file_ext: str,
        total_pages: int
    ) -> str:
        """
        Return raw text content with original formatting preserved.
        No auto-generated headers, page markers, or metadata.
        """
        return content
