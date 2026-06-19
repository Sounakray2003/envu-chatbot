"""
PDF Extractor using Docling.
- No BaseExtractor dependency
- All config from environment variables
- Returns markdown string
- Fallback to Unstructured when Docling page-count validation fails
"""

import os
import re
import base64
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from .header_footer_detector import get_header_footer_detector
from .image_cache_manager import get_image_cache
from .image_description_service import get_image_description_service

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default

# PyMuPDF (ground truth page count)
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available - page count validation disabled")

# Unstructured (fallback extractor)
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
    logger.info("Unstructured library available")
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning("Unstructured not available - fallback disabled")


class PDFExtractor:
    """
    PDF Extractor: Docling primary + Unstructured fallback.

    Page-count validation (via PyMuPDF as ground truth) selects the extractor.
    Reads all config from environment variables — no settings/config file used.

    Relevant env vars:
        IMAGE_DESCRIPTION              - 'true'/'false', controls image OCR
        PDF_OCR_MODE                   - auto/true/false
        PDF_TABLE_STRUCTURE            - true/false
        PDF_IMAGES_SCALE               - image scale for Docling
        PDF_GENERATE_PICTURE_IMAGES    - true/false/blank
    """

    def __init__(self):
        self.header_footer_detector = get_header_footer_detector()
        self.image_cache = get_image_cache()
        self.image_service = get_image_description_service()
        self.image_text_enabled = self.image_service.is_available()
        self.pdf_ocr_mode = str(os.getenv("PDF_OCR_MODE", "auto")).strip().lower() or "auto"
        self.pdf_table_structure = _env_bool("PDF_TABLE_STRUCTURE", False)
        self.pdf_images_scale = _env_float("PDF_IMAGES_SCALE", 1.0)
        self.max_images_per_doc = max(0, _env_int("IMAGE_DESCRIPTION_MAX_IMAGES_PER_DOC", 0))
        self.min_image_bytes = max(0, _env_int("IMAGE_DESCRIPTION_MIN_BYTES", 0))
        generate_picture_images_raw = os.getenv("PDF_GENERATE_PICTURE_IMAGES")
        if generate_picture_images_raw in (None, ""):
            self.generate_picture_images = self.image_text_enabled
        else:
            self.generate_picture_images = _env_bool(
                "PDF_GENERATE_PICTURE_IMAGES",
                self.image_text_enabled,
            )

        # Statistics
        self.extraction_stats = {
            "total_extractions": 0,
            "docling_used": 0,
            "unstructured_used": 0,
            "validation_failures": 0,
        }

        logger.info("PDF Extractor initialized (Docling + Unstructured fallback)")
        logger.info(
            "PDF pipeline config | ocr_mode=%s table_structure=%s images_scale=%s generate_picture_images=%s image_text=%s max_images_per_doc=%s min_image_bytes=%s",
            self.pdf_ocr_mode,
            self.pdf_table_structure,
            self.pdf_images_scale,
            self.generate_picture_images,
            self.image_text_enabled,
            self.max_images_per_doc,
            self.min_image_bytes,
        )

    # =========================================================================
    # Public entry point
    # =========================================================================

    def extract(self, file_path: str) -> str:
        """
        Extract PDF content and return Markdown string.

        Flow:
          1. Get ground-truth page count via PyMuPDF.
          2. Quick page-count check with Docling.
          3. If counts match -> use Docling full extraction.
          4. If mismatch and Unstructured available -> use Unstructured extraction.
          5. If both fail validation -> raise RuntimeError.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Markdown string of extracted content.

        Raises:
            RuntimeError: If extraction fails or both extractors fail validation.
        """
        self.extraction_stats["total_extractions"] += 1

        pdf_stem = Path(file_path).stem
        image_output_dir = self._create_image_directory(file_path, pdf_stem)

        logger.info(f"Extracting PDF: {file_path}")

        # Step 1: Ground-truth page count
        actual_page_count = self._get_pymupdf_page_count(file_path)
        if actual_page_count == 0:
            raise RuntimeError("Could not determine PDF page count via PyMuPDF")

        logger.info(f"PDF has {actual_page_count} pages (PyMuPDF - ground truth)")

        # Step 2: Quick Docling page-count check
        docling_page_count = self._get_docling_page_count(file_path)
        logger.info(f"Docling detected {docling_page_count} pages")

        unstructured_page_count = 0
        if UNSTRUCTURED_AVAILABLE:
            unstructured_page_count = self._get_unstructured_page_count(file_path)
            logger.info(f"Unstructured detected {unstructured_page_count} pages")

        # Detect header/footer zones
        zones = self.header_footer_detector.detect(file_path, sample_pages=5)
        if zones:
            logger.info(f"Header/footer zones: {zones}")

        # Step 3: Choose extractor
        if docling_page_count == actual_page_count:
            logger.info(
                f"Validation passed: Docling={docling_page_count} == Actual={actual_page_count} "
                f"-> Using Docling"
            )
            markdown = self._extract_with_docling_full(
                file_path, pdf_stem, image_output_dir, actual_page_count, zones
            )
            self.extraction_stats["docling_used"] += 1
            
            # If Docling returned empty content, try Unstructured fallback
            if not markdown.strip() and UNSTRUCTURED_AVAILABLE:
                logger.warning(
                    f"Docling extraction returned empty content despite page count match. "
                    f"Falling back to Unstructured."
                )
                markdown = self._extract_with_unstructured_full(
                    file_path, pdf_stem, image_output_dir, actual_page_count, zones
                )
                self.extraction_stats["docling_used"] -= 1
                self.extraction_stats["unstructured_used"] += 1
            
            return markdown

        elif UNSTRUCTURED_AVAILABLE and unstructured_page_count == actual_page_count:
            logger.warning(
                f"Docling validation failed ({docling_page_count} != {actual_page_count}). "
                f"Unstructured matches ({unstructured_page_count}) -> Using Unstructured"
            )
            markdown = self._extract_with_unstructured_full(
                file_path, pdf_stem, image_output_dir, actual_page_count, zones
            )
            self.extraction_stats["unstructured_used"] += 1
            return markdown

        else:
            self.extraction_stats["validation_failures"] += 1
            raise RuntimeError(
                f"Both extractors failed page-count validation. "
                f"Actual={actual_page_count}, Docling={docling_page_count}, "
                f"Unstructured={unstructured_page_count}"
            )

    # =========================================================================
    # Page count helpers
    # =========================================================================

    def _get_pymupdf_page_count(self, file_path: str) -> int:
        if not PYMUPDF_AVAILABLE:
            return 0
        try:
            doc = fitz.open(file_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.error(f"PyMuPDF page count error: {e}")
            return 0

    def _get_docling_page_count(self, file_path: str) -> int:
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = False
            pipeline_options.generate_picture_images = False

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            result = converter.convert(file_path)
            return len(result.document.pages)
        except Exception as e:
            logger.error(f"Docling page count failed: {e}", exc_info=True)
            return 0

    def _get_unstructured_page_count(self, file_path: str) -> int:
        if not UNSTRUCTURED_AVAILABLE:
            return 0
        try:
            elements = partition_pdf(
                filename=file_path,
                extract_images_in_pdf=False,
                infer_table_structure=False,
                strategy="fast",
            )
            if not elements:
                return 0
            page_numbers = {
                elem.metadata.page_number
                for elem in elements
                if hasattr(elem, "metadata") and elem.metadata
                and hasattr(elem.metadata, "page_number") and elem.metadata.page_number
            }
            return max(page_numbers) if page_numbers else 0
        except Exception as e:
            logger.error(f"Unstructured page count failed: {e}", exc_info=True)
            return 0

    # =========================================================================
    # Docling full extraction
    # =========================================================================

    def _extract_with_docling_full(
        self,
        file_path: str,
        pdf_stem: str,
        image_output_dir: str,
        total_pages: int,
        zones,
    ) -> str:
        logger.info("Full Docling extraction...")

        result, extracted_pages = self._run_docling(file_path)
        logger.info(f"Docling extracted {extracted_pages} pages")

        # Check if document has actual content
        has_content = (
            len(result.document.texts) > 0
            or len(result.document.tables) > 0
            or len(result.document.pictures) > 0
        )
        
        if not has_content:
            logger.warning(
                "Docling extracted no content (texts=0, tables=0, pictures=0). "
                "PDF may be image-only or OCR failed."
            )
            return ""

        content_map = self._build_content_page_map(result.document)
        logger.info(f"Content map: {len(content_map['pictures'])} pictures mapped")

        image_data = self._process_images_with_pages(
            result.document, image_output_dir, pdf_stem, file_path, zones, content_map
        )

        markdown = self._build_markdown_from_pages(
            result.document, total_pages, image_data, pdf_stem, zones, content_map
        )

        logger.info(
            f"Docling extraction complete: {total_pages} pages, {len(image_data)} images"
        )
        return markdown

    def _run_docling(self, file_path: str) -> Tuple:
        """Run Docling with full pipeline options."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self._should_enable_docling_ocr(file_path)
        pipeline_options.do_table_structure = self.pdf_table_structure
        pipeline_options.images_scale = self.pdf_images_scale
        pipeline_options.generate_picture_images = self.generate_picture_images

        try:
            pipeline_options.generate_page_images = False
            pipeline_options.artifacts_path = None
        except AttributeError:
            pass

        logger.info(
            "Running Docling | do_ocr=%s table_structure=%s images_scale=%s generate_picture_images=%s",
            pipeline_options.do_ocr,
            pipeline_options.do_table_structure,
            pipeline_options.images_scale,
            pipeline_options.generate_picture_images,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(file_path)
        total_pages = len(result.document.pages)
        return result, total_pages

    def _should_enable_docling_ocr(self, file_path: str) -> bool:
        """Decide whether Docling OCR should run for this PDF."""
        if self.pdf_ocr_mode in {"true", "always", "on", "yes"}:
            return True
        if self.pdf_ocr_mode in {"false", "never", "off", "no"}:
            return False

        if not PYMUPDF_AVAILABLE:
            return True

        try:
            doc = fitz.open(file_path)
            sample_pages = min(3, len(doc))
            extracted_chars = 0
            for page_index in range(sample_pages):
                extracted_chars += len((doc[page_index].get_text("text") or "").strip())
                if extracted_chars >= 120:
                    doc.close()
                    return False
            doc.close()
        except Exception as exc:
            logger.warning("Auto OCR decision failed, defaulting to OCR enabled: %s", exc)
            return True

        return True

    def _build_content_page_map(self, document) -> Dict:
        content_map = {"texts": {}, "pictures": {}, "tables": {}}

        try:
            for idx, picture in enumerate(getattr(document, "pictures", [])):
                for prov in getattr(picture, "prov", []):
                    if hasattr(prov, "page_no"):
                        content_map["pictures"][idx] = prov.page_no
                        break

            for idx, table in enumerate(getattr(document, "tables", [])):
                for prov in getattr(table, "prov", []):
                    if hasattr(prov, "page_no"):
                        content_map["tables"][idx] = prov.page_no
                        break

            for idx, text in enumerate(getattr(document, "texts", [])):
                for prov in getattr(text, "prov", []):
                    if hasattr(prov, "page_no"):
                        content_map["texts"][idx] = prov.page_no
                        break

        except Exception as e:
            logger.warning(f"Error building content map: {e}")

        return content_map

    def _process_images_with_pages(
        self,
        document,
        image_output_dir: str,
        pdf_stem: str,
        pdf_path: str,
        zones,
        content_map: Dict,
    ) -> Dict[int, Dict]:
        try:
            markdown_with_base64 = document.export_to_markdown(image_mode="embedded")
        except Exception as e:
            logger.warning(f"Failed to export with embedded images: {e}")
            return {}

        pattern = r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)"
        matches = re.findall(pattern, markdown_with_base64)

        if not matches:
            logger.info("No images found in markdown")
            return {}

        if not self.image_text_enabled:
            logger.info("Image OCR disabled or unavailable - skipping embedded PDF images")
            return {}

        logger.info(f"Found {len(matches)} images in markdown")

        self._duplicates_skipped = 0
        self._headers_skipped = 0
        self._size_skipped = 0
        self._limit_skipped = 0
        seen_hashes = set()
        image_data = {}
        doc_context = f"PDF document: {pdf_stem}"

        for extract_idx, match in enumerate(matches, start=1):
            try:
                image_format = match[1]
                base64_image = match[2]
                image_bytes = base64.b64decode(base64_image)

                if self.min_image_bytes and len(image_bytes) < self.min_image_bytes:
                    self._size_skipped += 1
                    continue

                if self.max_images_per_doc and len(image_data) >= self.max_images_per_doc:
                    self._limit_skipped += 1
                    continue

                picture_idx = extract_idx - 1
                page_number = content_map["pictures"].get(picture_idx, 1)

                img_hash = self.image_cache.compute_hash(image_bytes)
                if img_hash in seen_hashes:
                    self._duplicates_skipped += 1
                    continue
                seen_hashes.add(img_hash)

                if zones and self.header_footer_detector.is_image_in_zone(
                    pdf_path, page_number, image_bytes, zones
                ):
                    self._headers_skipped += 1
                    continue

                cached_desc = self.image_cache.get(image_bytes)
                if cached_desc:
                    description = cached_desc
                    from_cache = True
                    success = True
                else:
                    success, description = self.image_service.describe_image_from_base64(
                        base64_image,
                        self._get_media_type(image_format),
                        context=doc_context,
                        page_number=None,
                    )
                    from_cache = False
                    if success:
                        self.image_cache.put(image_bytes, description)

                ext = image_format.lower().replace("jpeg", "jpg")
                filename = f"image_{extract_idx:03d}.{ext}"
                image_path = os.path.join(image_output_dir, filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                image_data[extract_idx] = {
                    "extract_idx": extract_idx,
                    "picture_idx": picture_idx,
                    "filename": filename,
                    "page_number": page_number,
                    "format": image_format,
                    "description": description,
                    "success": success,
                    "from_cache": from_cache,
                }

                cache_str = "cached" if from_cache else "ocr"
                logger.info(f"  Image {extract_idx} -> {filename} (page {page_number}, {cache_str})")

            except Exception as e:
                logger.warning(f"Error processing image {extract_idx}: {e}")

        logger.info(
            f"Processed {len(image_data)} images "
            f"(skipped {self._duplicates_skipped} dupes, {self._headers_skipped} headers, "
            f"{self._size_skipped} too-small, {self._limit_skipped} over-limit)"
        )
        return image_data

    def _build_markdown_from_pages(
        self,
        document,
        total_pages: int,
        image_data: Dict[int, Dict],
        pdf_stem: str,
        zones,
        content_map: Dict,
    ) -> str:
        """
        Extract content from all pages and replace image placeholders with descriptions.
        At this point we know the document has content (checked in _extract_with_docling_full).
        """
        try:
            # Export full document markdown with placeholder mode
            full_markdown = document.export_to_markdown(
                image_mode="placeholder",
                image_placeholder="[IMAGE_PLACEHOLDER_{image_name}]",
            )

            if not full_markdown or not full_markdown.strip():
                logger.warning("Full markdown export returned empty despite content checks")
                return ""

            logger.info(f"Full markdown extracted: {len(full_markdown)} characters")

            # Create a mapping of image indices to their descriptions
            image_descriptions = {}
            for img_info in sorted(image_data.values(), key=lambda x: x["extract_idx"]):
                image_descriptions[img_info["extract_idx"]] = img_info["description"]

            # Process the markdown and replace image placeholders with descriptions
            result_lines = []
            image_counter = 0
            for line in full_markdown.split("\n"):
                if "[IMAGE_PLACEHOLDER_" in line:
                    image_counter += 1
                    if image_counter in image_descriptions:
                        # Replace placeholder with description text
                        result_lines.append(image_descriptions[image_counter])
                else:
                    # Keep all non-image lines
                    result_lines.append(line)

            markdown_output = "\n".join(result_lines)
            logger.info(f"Markdown with image OCR text: {len(markdown_output)} characters")
            return markdown_output

        except Exception as e:
            logger.error(f"Error building markdown from pages: {e}", exc_info=True)
            return ""

    # =========================================================================
    # Unstructured full extraction (fallback)
    # =========================================================================

    def _extract_with_unstructured_full(
        self,
        file_path: str,
        pdf_stem: str,
        image_output_dir: str,
        total_pages: int,
        zones,
    ) -> str:
        logger.info("Full Unstructured extraction (fallback)...")

        temp_image_dir = Path(image_output_dir) / "unstructured_temp"
        temp_image_dir.mkdir(exist_ok=True)

        try:
            elements = partition_pdf(
                filename=file_path,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_output_dir=str(temp_image_dir),
            )
            logger.info(f"Unstructured extracted {len(elements)} elements")

            result_lines = []

            image_data = {}
            table_counter = 0
            image_counter = 0
            current_page = 0
            self._duplicates_skipped = 0
            self._headers_skipped = 0
            seen_hashes = set()

            for elem in elements:
                page_num = (elem.metadata.page_number or 1) if elem.metadata else 1

                if page_num != current_page:
                    current_page = page_num

                if elem.category == "Image":
                    image_counter += 1
                    if elem.metadata.image_path and Path(elem.metadata.image_path).exists():
                        image_path = elem.metadata.image_path
                        with open(image_path, "rb") as f:
                            image_bytes = f.read()

                        if not self.image_text_enabled:
                            continue

                        if self.min_image_bytes and len(image_bytes) < self.min_image_bytes:
                            continue

                        if self.max_images_per_doc and len(image_data) >= self.max_images_per_doc:
                            continue

                        img_hash = self.image_cache.compute_hash(image_bytes)
                        if img_hash in seen_hashes:
                            self._duplicates_skipped += 1
                            continue
                        seen_hashes.add(img_hash)

                        cached_desc = self.image_cache.get(image_bytes)
                        if cached_desc:
                            description = cached_desc
                            from_cache = True
                            success = True
                        else:
                            success, description = self.image_service.describe_image_from_path(
                                image_path, context=f"PDF document: {pdf_stem}", page_number=page_num
                            )
                            from_cache = False
                            if success:
                                self.image_cache.put(image_bytes, description)

                        final_filename = f"page_{page_num:03d}_image_{image_counter:03d}.png"
                        shutil.copy(image_path, Path(image_output_dir) / final_filename)

                        result_lines.append(description)
                        result_lines.append("")

                        image_data[image_counter] = {
                            "filename": final_filename,
                            "page_number": page_num,
                            "description": description,
                            "success": success,
                            "from_cache": from_cache,
                        }

                elif elem.category == "Table":
                    table_counter += 1
                    result_lines.append(f"**Table {table_counter}**:")
                    result_lines.append("")

                    if hasattr(elem.metadata, "text_as_markdown") and elem.metadata.text_as_markdown:
                        result_lines.append(elem.metadata.text_as_markdown)
                    elif hasattr(elem.metadata, "text_as_html") and elem.metadata.text_as_html:
                        result_lines.append(elem.metadata.text_as_html)
                    else:
                        result_lines.append(str(elem))

                    result_lines.append("")

                else:
                    result_lines.append(str(elem))
                    result_lines.append("")

        finally:
            try:
                shutil.rmtree(temp_image_dir)
            except Exception as e:
                logger.warning(f"Could not clean up temp dir: {e}")

        logger.info(
            f"Unstructured extraction complete: {total_pages} pages, "
            f"{len(image_data)} images, {table_counter} tables"
        )
        return "\n".join(result_lines)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _create_image_directory(self, file_path: str, pdf_stem: str) -> str:
        image_dir = os.path.join(
            str(Path.cwd()), "output_files", "extracted_images", pdf_stem
        )
        os.makedirs(image_dir, exist_ok=True)
        return image_dir

    def _get_media_type(self, extension: str) -> str:
        return {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
        }.get(extension.lower(), "image/jpeg")

    def get_extraction_stats(self) -> Dict:
        total = self.extraction_stats["total_extractions"]
        if total == 0:
            return self.extraction_stats
        return {
            **self.extraction_stats,
            "docling_usage_rate": round(self.extraction_stats["docling_used"] / total * 100, 1),
            "unstructured_usage_rate": round(self.extraction_stats["unstructured_used"] / total * 100, 1),
            "validation_failure_rate": round(self.extraction_stats["validation_failures"] / total * 100, 1),
        }
