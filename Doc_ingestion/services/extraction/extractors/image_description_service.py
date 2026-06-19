"""
Image text extraction service backed by local Tesseract OCR.

This service extracts visible text from images so the output can flow through
the ingestion and RAG pipeline without calling a remote vision model.
"""

import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image, ImageOps
import pytesseract

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_OCR_LANG = "eng"
DEFAULT_OCR_PSM = 6
DEFAULT_OCR_TIMEOUT = 20
DEFAULT_MIN_TEXT_LENGTH = 2


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ImageDescriptionService:
    """
    OCR-style image text extraction service using local Tesseract OCR.

    Reads all configuration from environment variables:
        IMAGE_DESCRIPTION   - Enable image text extraction: "true"/"false"
        IMAGE_OCR_LANG      - Tesseract language code(s), default "eng"
        IMAGE_OCR_PSM       - Tesseract page segmentation mode, default 6
        IMAGE_OCR_TIMEOUT   - OCR timeout in seconds, default 20
        TESSERACT_CMD       - Optional explicit tesseract executable path
    """

    def __init__(self):
        """Initialise the local OCR service."""
        self.enabled = os.getenv("IMAGE_DESCRIPTION", "true").strip().lower() == "true"
        self.tesseract_available = False

        if not self.enabled:
            logger.info("Image text extraction disabled (IMAGE_DESCRIPTION != true)")
            return

        self.lang = str(os.getenv("IMAGE_OCR_LANG", DEFAULT_OCR_LANG)).strip() or DEFAULT_OCR_LANG
        self.psm = _safe_int(os.getenv("IMAGE_OCR_PSM", DEFAULT_OCR_PSM), DEFAULT_OCR_PSM)
        self.timeout = _safe_int(
            os.getenv("IMAGE_OCR_TIMEOUT", DEFAULT_OCR_TIMEOUT),
            DEFAULT_OCR_TIMEOUT,
        )
        self.min_text_length = _safe_int(
            os.getenv("IMAGE_OCR_MIN_TEXT_LENGTH", DEFAULT_MIN_TEXT_LENGTH),
            DEFAULT_MIN_TEXT_LENGTH,
        )

        tesseract_cmd = str(os.getenv("TESSERACT_CMD", "")).strip()
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        try:
            version = pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info(
                "Image OCR service initialised (provider=tesseract, lang=%s, psm=%s, timeout=%ss, version=%s)",
                self.lang,
                self.psm,
                self.timeout,
                version,
            )
        except Exception as exc:
            logger.error(
                "Tesseract OCR is not available. Image text extraction will be disabled: %s",
                exc,
            )
            self.enabled = False

    def is_available(self) -> bool:
        """Return True if the local OCR service is ready to use."""
        return self.enabled and self.tesseract_available

    def describe_image_from_base64(
        self,
        base64_image: str,
        media_type: str,
        context: Optional[str] = None,
        page_number: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Extract visible text from a base64-encoded image.

        Returns:
            (success, extracted_text)
        """
        del media_type, context, page_number

        if not self.is_available():
            logger.warning("Local OCR service not available - skipping image OCR")
            return False, "[Image text extraction unavailable]"

        try:
            image_bytes = base64.b64decode(base64_image)
            with Image.open(io.BytesIO(image_bytes)) as image:
                return self._extract_text_from_image(image, filename=filename)
        except Exception as exc:
            logger.error("Unexpected image extraction error: %s", exc, exc_info=True)
            label = f"Image: {filename}" if filename else "Image"
            return False, f"[{label} - text extraction unavailable: {str(exc)}]"

    def describe_image_from_path(
        self,
        image_path: str,
        context: Optional[str] = None,
        page_number: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Extract visible text from an image file on disk.

        Returns:
            (success, extracted_text)
        """
        del context, page_number

        if not self.is_available():
            logger.warning("Local OCR service not available - skipping image OCR")
            return False, "[Image text extraction unavailable]"

        try:
            with Image.open(image_path) as image:
                return self._extract_text_from_image(image, filename=Path(image_path).name)
        except FileNotFoundError:
            logger.error("Image file not found: %s", image_path)
            return False, f"[Image file not found: {Path(image_path).name}]"
        except Exception as exc:
            logger.error("Error reading image %s: %s", image_path, exc)
            return False, f"[Error reading image: {str(exc)}]"

    def _extract_text_from_image(
        self,
        image: Image.Image,
        *,
        filename: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Run local OCR over the given image."""
        try:
            prepared = self._prepare_image(image)
            config = f"--oem 1 --psm {self.psm}"
            extracted_text = pytesseract.image_to_string(
                prepared,
                lang=self.lang,
                config=config,
                timeout=self.timeout,
            )
            normalized = self._normalize_extracted_text(extracted_text)
            if len(normalized) < self.min_text_length:
                return False, "[No extractable text]"
            return True, normalized
        except RuntimeError as exc:
            logger.error("Tesseract OCR failed for %s: %s", filename or "image", exc)
            label = f"Image: {filename}" if filename else "Image"
            return False, f"[{label} - text extraction unavailable]"

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        """Apply lightweight preprocessing that improves OCR while staying fast."""
        prepared = ImageOps.exif_transpose(image)
        if prepared.mode not in {"L", "RGB"}:
            prepared = prepared.convert("RGB")

        prepared = prepared.convert("L")
        prepared = ImageOps.autocontrast(prepared)

        width, height = prepared.size
        if max(width, height) < 1400 and width > 0 and height > 0:
            prepared = prepared.resize(
                (max(width * 2, 1), max(height * 2, 1)),
                Image.Resampling.LANCZOS,
            )

        prepared = prepared.point(lambda pixel: 0 if pixel < 170 else 255, mode="1")
        return prepared

    @staticmethod
    def _normalize_extracted_text(text: str) -> str:
        """Clean OCR output into a compact, ingestion-friendly text block."""
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()


_image_description_service: Optional[ImageDescriptionService] = None


def get_image_description_service() -> ImageDescriptionService:
    """Return the module-level singleton ImageDescriptionService."""
    global _image_description_service
    if _image_description_service is None:
        _image_description_service = ImageDescriptionService()
    return _image_description_service
