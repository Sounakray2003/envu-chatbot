"""
XML File Extractor
Processes XML files for the RAG pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging
import re
import xml.etree.ElementTree as ET

try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None
    TRAFILATURA_AVAILABLE = False

logger = logging.getLogger(__name__)


class XMLExtractor:
    """
    Extractor for XML files.

    Features:
    - Uses trafilatura for primary content extraction when it can recover text
    - Falls back to XML-aware structured extraction for data-heavy XML files
    - Preserves key XML metadata such as root tag, namespace, and attributes
    - Converts the extracted result to markdown
    """

    def __init__(self):
        self.supported_formats = {"xml"}

    def _get_supported_formats(self) -> List[str]:
        return ["xml"]

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate XML file."""
        path = Path(file_path)

        if not path.exists():
            return False, "File does not exist"

        ext = path.suffix.lower().lstrip(".")
        if ext not in self.supported_formats:
            return False, "Not an XML file"

        try:
            ET.parse(file_path)
            return True, "Valid XML file"
        except ET.ParseError as exc:
            return False, f"Invalid XML: {exc}"
        except Exception as exc:
            return False, f"Could not read XML: {exc}"

    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract an XML file."""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {"error": msg}

            logger.info("Extracting XML: %s", file_path)

            xml_content = self._read_with_encoding(file_path)
            if not xml_content.strip():
                return "", False, {"error": "Empty XML file"}

            xml_stem = Path(file_path).stem
            parsed_data = self._parse_xml(xml_content)

            if not parsed_data.get("text"):
                return "", False, {"error": "No extractable XML content"}

            markdown_content = self._build_markdown(xml_stem, parsed_data)

            metadata = self.get_metadata(file_path)
            metadata.update(
                {
                    "total_pages": 1,
                    "title": parsed_data.get("title", xml_stem),
                    "root_element": parsed_data.get("root_tag", ""),
                    "element_count": parsed_data.get("element_count", 0),
                    "attribute_count": parsed_data.get("attribute_count", 0),
                    "has_namespace": bool(parsed_data.get("namespace")),
                    "extraction_method": parsed_data.get(
                        "content_extractor",
                        "xml_structured_fallback",
                    ),
                }
            )

            logger.info(
                "XML extraction complete: %s elements via %s",
                parsed_data.get("element_count", 0),
                metadata["extraction_method"],
            )
            return markdown_content, True, metadata

        except Exception as exc:
            logger.error("XML extraction error: %s", exc, exc_info=True)
            return "", False, {"error": str(exc)}

    def _read_with_encoding(self, file_path: str) -> str:
        """Read XML file with a few common encodings."""
        encodings = ["utf-8", "utf-8-sig", "utf-16", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as handle:
                    return handle.read()
            except UnicodeDecodeError:
                continue

        with open(file_path, "rb") as handle:
            return handle.read().decode("utf-8", errors="ignore")

    def _parse_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse XML and combine trafilatura output with XML-aware fallback data."""
        root = ET.fromstring(xml_content)
        structural_data = self._extract_structural_data(root)
        trafilatura_data = self._extract_with_trafilatura(xml_content)

        if trafilatura_data.get("text"):
            structural_data["text"] = trafilatura_data["text"]
            structural_data["content_extractor"] = trafilatura_data["content_extractor"]

        if trafilatura_data.get("title"):
            structural_data["title"] = trafilatura_data["title"]

        return structural_data

    def _extract_with_trafilatura(self, xml_content: str) -> Dict[str, str]:
        """Use trafilatura as the primary extractor for text-heavy XML."""
        if not TRAFILATURA_AVAILABLE:
            logger.warning("trafilatura not available, using XML fallback extraction")
            return {}

        extraction_kwargs = {
            "favor_precision": True,
            "deduplicate": True,
            "include_comments": False,
            "include_tables": True,
            "include_images": False,
            "include_links": True,
            "include_formatting": True,
        }

        try:
            markdown_content = trafilatura.extract(
                xml_content,
                output_format="markdown",
                **extraction_kwargs,
            )
            extracted_data = trafilatura.bare_extraction(
                xml_content,
                output_format="python",
                with_metadata=True,
                as_dict=True,
                **extraction_kwargs,
            )
        except Exception as exc:
            logger.warning(
                "trafilatura XML extraction failed, using XML fallback: %s",
                exc,
            )
            return {}

        title = ""
        plain_text = ""
        if isinstance(extracted_data, dict):
            title = str(extracted_data.get("title") or "").strip()
            plain_text = str(
                extracted_data.get("text")
                or extracted_data.get("raw_text")
                or ""
            ).strip()

        normalized_markdown = (markdown_content or "").strip()
        if normalized_markdown:
            normalized_markdown = self._deduplicate_markdown_blocks(normalized_markdown)
            normalized_markdown = self._strip_duplicate_title_heading(
                normalized_markdown,
                title,
            )
            return {
                "title": title,
                "text": normalized_markdown,
                "content_extractor": "xml_trafilatura",
            }

        if plain_text:
            logger.info("trafilatura returned plain-text XML content")
            plain_text = self._deduplicate_markdown_blocks(plain_text)
            return {
                "title": title,
                "text": plain_text,
                "content_extractor": "xml_trafilatura_text_fallback",
            }

        logger.info("trafilatura returned no XML content, using structured fallback")
        return {}

    def _extract_structural_data(self, root: ET.Element) -> Dict[str, Any]:
        """Build fallback data directly from the XML tree."""
        root_tag = self._strip_namespace(root.tag)
        namespace = self._extract_namespace(root.tag)
        text = self._extract_text_content(root).strip()

        if not text:
            text = self._xml_structure_to_text(root).strip()
            content_extractor = "xml_structure_fallback"
        else:
            content_extractor = "xml_structured_text_fallback"

        return {
            "title": self._extract_title(root),
            "text": text,
            "root_tag": root_tag,
            "namespace": namespace,
            "element_count": sum(1 for _ in root.iter()),
            "attribute_count": sum(len(element.attrib) for element in root.iter()),
            "root_attributes": dict(root.attrib),
            "content_extractor": content_extractor,
        }

    def _extract_title(self, root: ET.Element) -> str:
        """Find a title-like value from common XML tags."""
        candidate_tags = {
            "title",
            "name",
            "subject",
            "headline",
            "label",
            "caption",
        }

        for element in root.iter():
            tag = self._strip_namespace(element.tag).lower()
            if tag not in candidate_tags:
                continue

            text = " ".join(part.strip() for part in element.itertext() if part.strip()).strip()
            if text:
                return text[:300]

        return self._strip_namespace(root.tag)

    def _extract_text_content(self, element: ET.Element, depth: int = 0) -> str:
        """Recursively extract readable text from XML elements."""
        lines = []
        indent = "  " * depth
        tag = self._strip_namespace(element.tag)

        text = " ".join(part.strip() for part in [element.text] if part and part.strip()).strip()
        if text:
            lines.append(f"{indent}[{tag}] {text}")

        for child in element:
            child_text = self._extract_text_content(child, depth + 1)
            if child_text:
                lines.append(child_text)

            tail = (child.tail or "").strip()
            if tail:
                lines.append(f"{indent}{tail}")

        return "\n".join(lines)

    def _xml_structure_to_text(self, element: ET.Element, depth: int = 0) -> str:
        """Convert XML structure to a readable text fallback."""
        lines = []
        indent = "  " * depth
        tag = self._strip_namespace(element.tag)

        attr_str = ""
        if element.attrib:
            attrs = [f'{key}="{value}"' for key, value in element.attrib.items()]
            attr_str = " " + " ".join(attrs)

        lines.append(f"{indent}<{tag}{attr_str}>")

        text = (element.text or "").strip()
        if text:
            lines.append(f"{indent}  {text}")

        for child in element:
            lines.append(self._xml_structure_to_text(child, depth + 1))

        lines.append(f"{indent}</{tag}>")
        return "\n".join(lines)

    @staticmethod
    def _deduplicate_markdown_blocks(text: str) -> str:
        """Remove repeated paragraph-sized markdown blocks while preserving order."""
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text or "") if block.strip()]
        if not blocks:
            return ""

        unique_blocks = []
        seen = set()
        for block in blocks:
            normalized = re.sub(r"\s+", " ", block).strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_blocks.append(block)

        return "\n\n".join(unique_blocks)

    @staticmethod
    def _strip_duplicate_title_heading(text: str, title: str) -> str:
        """Drop a leading markdown heading when it exactly repeats the document title."""
        normalized_title = (title or "").strip()
        if not normalized_title or not text:
            return text

        lines = text.splitlines()
        if not lines:
            return text

        first_line = lines[0].strip()
        if first_line.lstrip("#").strip() != normalized_title:
            return text

        remaining_lines = lines[1:]
        while remaining_lines and not remaining_lines[0].strip():
            remaining_lines.pop(0)

        return "\n".join(remaining_lines).strip()

    @staticmethod
    def _strip_namespace(tag: str) -> str:
        """Remove XML namespace from a tag name."""
        return tag.split("}", 1)[-1] if "}" in tag else tag

    @staticmethod
    def _extract_namespace(tag: str) -> str:
        """Extract namespace URI from a tag name."""
        if tag.startswith("{") and "}" in tag:
            return tag[1:].split("}", 1)[0]
        return ""

    def _build_markdown(self, xml_stem: str, parsed_data: Dict[str, Any]) -> str:
        """Build markdown from parsed XML."""
        lines = []

        title = parsed_data.get("title") or f"XML Document: {xml_stem}"
        lines.append(f"# {title}")
        lines.append("**Format**: XML Document")
        lines.append(f"**Root Element**: {parsed_data.get('root_tag', xml_stem)}")
        lines.append(f"**Total Elements**: {parsed_data.get('element_count', 0)}")
        lines.append(f"**Total Attributes**: {parsed_data.get('attribute_count', 0)}")

        namespace = parsed_data.get("namespace")
        if namespace:
            lines.append(f"**Namespace**: {namespace}")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Content")
        lines.append("")

        text = parsed_data.get("text", "")
        extractor = parsed_data.get("content_extractor", "")
        if text:
            if extractor.startswith("xml_trafilatura"):
                lines.append(text)
            else:
                lines.append("```")
                lines.append(text)
                lines.append("```")
        else:
            lines.append("*(No XML content extracted)*")

        lines.append("")

        root_attributes = parsed_data.get("root_attributes", {})
        if root_attributes:
            lines.append("## Root Attributes")
            lines.append("")
            for key, value in root_attributes.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        return "\n".join(lines)

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata."""
        path = Path(file_path)
        return {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "extension": path.suffix.lower(),
        }
