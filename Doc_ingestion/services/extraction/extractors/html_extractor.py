"""
HTML/HTM File Extractor
Processes HTML web pages for the RAG pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging
import re

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None
    TRAFILATURA_AVAILABLE = False

logger = logging.getLogger(__name__)


class HTMLExtractor:
    """
    Extractor for HTML/HTM files.

    Features:
    - Uses trafilatura for primary content extraction
    - Collects lightweight metadata such as title, links, images, and headings
    - Falls back to local parsing if trafilatura yields no content
    - Converts the extracted result to markdown
    """

    def __init__(self):
        self.supported_formats = {"html", "htm"}

    def _get_supported_formats(self) -> List[str]:
        return ["html", "htm"]

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate HTML file."""
        path = Path(file_path)

        if not path.exists():
            return False, "File does not exist"

        ext = path.suffix.lower().lstrip(".")
        if ext not in self.supported_formats:
            return False, "Not an HTML file"

        return True, "Valid HTML file"

    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract an HTML file."""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {"error": msg}

            logger.info("Extracting HTML: %s", file_path)

            html_content = self._read_with_encoding(file_path)
            if not html_content:
                return "", False, {"error": "Empty HTML file"}

            html_stem = Path(file_path).stem
            parsed_data = self._parse_html(html_content)

            if not parsed_data.get("text"):
                return "", False, {"error": "No text content in HTML"}

            markdown_content = self._build_markdown(html_stem, parsed_data)

            metadata = self.get_metadata(file_path)
            metadata.update({
                "total_pages": 1,
                "title": parsed_data.get("title", html_stem),
                "link_count": len(parsed_data.get("links", [])),
                "image_count": len(parsed_data.get("images", [])),
                "heading_count": len(parsed_data.get("headings", [])),
                "extraction_method": parsed_data.get(
                    "content_extractor",
                    "html_fallback",
                ),
            })

            logger.info("HTML extraction complete")
            return markdown_content, True, metadata

        except Exception as exc:
            logger.error("HTML extraction error: %s", exc, exc_info=True)
            return "", False, {"error": str(exc)}

    def _read_with_encoding(self, file_path: str) -> str:
        """Read HTML file with a few common encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as handle:
                    return handle.read()
            except UnicodeDecodeError:
                continue

        with open(file_path, "rb") as handle:
            return handle.read().decode("utf-8", errors="ignore")

    def _parse_html(self, html_content: str) -> Dict[str, Any]:
        """Parse HTML and combine trafilatura content with local metadata."""
        structural_data = self._parse_document_structure(html_content)
        trafilatura_data = self._extract_with_trafilatura(html_content)

        if trafilatura_data.get("text"):
            structural_data["text"] = trafilatura_data["text"]
            structural_data["content_extractor"] = trafilatura_data["content_extractor"]

        if trafilatura_data.get("title"):
            structural_data["title"] = trafilatura_data["title"]

        return structural_data

    def _extract_with_trafilatura(self, html_content: str) -> Dict[str, str]:
        """Use trafilatura as the primary HTML content extractor."""
        if not TRAFILATURA_AVAILABLE:
            logger.warning("trafilatura not available, using local HTML fallback")
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
                html_content,
                output_format="markdown",
                **extraction_kwargs,
            )
            extracted_data = trafilatura.bare_extraction(
                html_content,
                output_format="python",
                with_metadata=True,
                as_dict=True,
                **extraction_kwargs,
            )
        except Exception as exc:
            logger.warning(
                "trafilatura extraction failed, using local HTML fallback: %s",
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
                "content_extractor": "html_trafilatura",
            }

        if plain_text:
            logger.info("trafilatura returned plain-text HTML content")
            plain_text = self._deduplicate_markdown_blocks(plain_text)
            return {
                "title": title,
                "text": plain_text,
                "content_extractor": "html_trafilatura_text_fallback",
            }

        logger.warning("trafilatura returned no HTML content, using local fallback")
        return {}

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

    def _parse_document_structure(self, html_content: str) -> Dict[str, Any]:
        """Parse structural metadata with local HTML parsing."""
        try:
            from bs4 import BeautifulSoup
            return self._parse_with_beautifulsoup(html_content)
        except ImportError:
            logger.warning("BeautifulSoup not available, using regex HTML fallback")
            return self._parse_basic(html_content)

    def _parse_with_beautifulsoup(self, html_content: str) -> Dict[str, Any]:
        """Parse HTML using BeautifulSoup for lightweight metadata."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()

        headings = []
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                headings.append({
                    "level": level,
                    "text": heading.get_text().strip(),
                })

        links = []
        for link in soup.find_all("a", href=True):
            link_text = link.get_text().strip()
            href = link["href"]
            if link_text or href:
                links.append({
                    "text": link_text or href,
                    "url": href,
                })

        images = []
        for img in soup.find_all("img"):
            src = img.get("src", "")
            alt = img.get("alt", "")
            if src:
                images.append({
                    "src": src,
                    "alt": alt,
                })

        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("body")
            or soup
        )

        text_parts = []
        for element in main_content.descendants:
            if element.name in [
                "h1", "h2", "h3", "h4", "h5", "h6",
                "p", "div", "li", "td", "th", "span",
            ]:
                text = element.get_text().strip()
                if text and text not in text_parts:
                    text_parts.append(text)

        if not text_parts:
            text_parts = [main_content.get_text(separator="\n").strip()]

        return {
            "title": title,
            "headings": headings,
            "links": links,
            "images": images,
            "text": "\n\n".join(text_parts).strip(),
            "content_extractor": "html_beautifulsoup_fallback",
        }

    def _parse_basic(self, html_content: str) -> Dict[str, Any]:
        """Basic HTML parsing without BeautifulSoup."""
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>",
            html_content,
            re.IGNORECASE | re.DOTALL,
        )
        title = title_match.group(1).strip() if title_match else ""

        html_content = re.sub(
            r"<script[^>]*>.*?</script>",
            "",
            html_content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        html_content = re.sub(
            r"<style[^>]*>.*?</style>",
            "",
            html_content,
            flags=re.IGNORECASE | re.DOTALL,
        )

        headings = []
        for level in range(1, 7):
            for match in re.finditer(
                rf"<h{level}[^>]*>(.*?)</h{level}>",
                html_content,
                re.IGNORECASE | re.DOTALL,
            ):
                text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
                if text:
                    headings.append({"level": level, "text": text})

        links = []
        for match in re.finditer(
            r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
            html_content,
            re.IGNORECASE | re.DOTALL,
        ):
            url = match.group(1)
            text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            links.append({"url": url, "text": text or url})

        images = []
        for match in re.finditer(
            r"<img[^>]+src=[\"']([^\"']+)[\"'][^>]*>",
            html_content,
            re.IGNORECASE,
        ):
            src = match.group(1)
            alt_match = re.search(
                r"alt=[\"']([^\"']+)[\"']",
                match.group(0),
                re.IGNORECASE,
            )
            alt = alt_match.group(1) if alt_match else ""
            images.append({"src": src, "alt": alt})

        text = re.sub(r"<[^>]+>", "\n", html_content)
        text = re.sub(r"\s+", " ", text).strip()

        return {
            "title": title,
            "headings": headings,
            "links": links,
            "images": images,
            "text": text,
            "content_extractor": "html_regex_fallback",
        }

    def _build_markdown(self, html_stem: str, parsed_data: Dict[str, Any]) -> str:
        """Build markdown from parsed HTML."""
        lines = []

        title = parsed_data.get("title", html_stem)
        lines.append(f"# {title}")
        lines.append("**Format**: HTML Document")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## Content")
        lines.append("")

        text = parsed_data.get("text", "")
        if text:
            lines.append(text)

        lines.append("")

        links = parsed_data.get("links", [])
        if links:
            lines.append("## Links")
            lines.append("")
            for index, link in enumerate(links[:50], start=1):
                lines.append(f"{index}. [{link['text']}]({link['url']})")

            if len(links) > 50:
                lines.append("")
                lines.append(f"*(... and {len(links) - 50} more links)*")

            lines.append("")

        images = parsed_data.get("images", [])
        if images:
            lines.append("## Images")
            lines.append("")
            for index, image in enumerate(images[:20], start=1):
                if image["alt"]:
                    lines.append(f"{index}. ![{image['alt']}]({image['src']})")
                else:
                    lines.append(f"{index}. Image: {image['src']}")

            if len(images) > 20:
                lines.append("")
                lines.append(f"*(... and {len(images) - 20} more images)*")

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
