"""
Standalone Image File Extractor
- Uses the shared ImageDescriptionService for direct image ingestion
- Supports common raster image formats
- Returns OCR-style markdown text that can flow through semantic chunking
"""

from pathlib import Path
import logging

from .image_description_service import get_image_description_service

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif"]


class ImageExtractor:
    """
    Image extractor for standalone image files.

    The extractor delegates OCR-style text extraction to ImageDescriptionService
    and wraps the returned text with a small amount of source metadata so the
    resulting content is useful inside the RAG pipeline.
    """

    def __init__(self):
        self.image_service = get_image_description_service()
        logger.info("Image Extractor initialized")

    def extract(self, file_path: str) -> str:
        """
        Extract text from an image file and return markdown content.

        Raises:
            RuntimeError: If the file is missing, unsupported, or the vision
            service cannot produce extracted text.
        """
        path = Path(file_path)
        if not path.exists():
            raise RuntimeError(f"File does not exist: {file_path}")

        file_ext = path.suffix.lower().lstrip(".")
        if file_ext not in SUPPORTED_FORMATS:
            raise RuntimeError(
                "Unsupported image type: "
                f"{file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )

        if not self.image_service.is_available():
            raise RuntimeError(
                "Image text extraction service is not available. "
                "Set IMAGE_DESCRIPTION=true and ensure local Tesseract OCR is installed."
            )

        success, extracted_text = self.image_service.describe_image_from_path(
            file_path,
            context=f"Standalone image file: {path.name}",
        )
        cleaned_text = str(extracted_text or "").strip()
        if not cleaned_text:
            raise RuntimeError("Image text extraction service returned empty content")

        if not success:
            raise RuntimeError(cleaned_text)

        return self._build_markdown(path, cleaned_text, file_ext)

    @staticmethod
    def _build_markdown(path: Path, extracted_text: str, file_ext: str) -> str:
        """Build markdown content for extracted image text."""
        return "\n".join(
            [
                f"# Image: {path.name}",
                "",
                f"- File type: {file_ext}",
                f"- Source file: {path.name}",
                "",
                "## Extracted Text",
                "",
                extracted_text.strip(),
            ]
        )
