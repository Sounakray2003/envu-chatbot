"""Website source crawler for full-site HTML ingestion."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree as ET

import httpx

from services.extraction.extractors.html_extractor import HTMLExtractor

logger = logging.getLogger(__name__)

WEBSITE_FILE_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class WebsiteSource:
    """Crawl same-site HTML pages and emit one ingestion document per page."""

    DEFAULT_MAX_PAGES = 500
    DEFAULT_TIMEOUT_SECONDS = 20.0
    DEFAULT_DELAY_SECONDS = 0.0
    DEFAULT_MAX_SITEMAPS = 20
    DEFAULT_USER_AGENT = "Q0KnowledgeBaseCrawler/1.0"
    HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
    SKIP_EXTENSIONS = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".webp",
        ".zip",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".css",
        ".js",
        ".json",
        ".xml",
        ".csv",
    }

    @staticmethod
    def _sanitize_failed_filename_component(value: Any, default: str) -> str:
        raw_value = str(value or "").strip()
        if not raw_value:
            return default

        normalized = raw_value.replace("\\", "/").rstrip("/")
        normalized = re.sub(r"^https?://", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"[^A-Za-z0-9._/-]+", "_", normalized).strip("._/-")
        normalized = normalized.replace("/", "_")
        return normalized or default

    @classmethod
    def build_discovery_failed_filename(cls, source_details: Dict[str, Any]) -> str:
        return cls._sanitize_failed_filename_component(
            source_details.get("start_url")
            or source_details.get("root_url")
            or source_details.get("website_url")
            or source_details.get("url"),
            "website",
        )

    @staticmethod
    def _normalize_file_id_seed_component(value: Any) -> str:
        """Normalize arbitrary values into stable website file-id seed components."""
        if value in (None, ""):
            return ""
        return str(value).strip()

    @classmethod
    def normalize_source_url_for_file_id(
        cls,
        start_url: Any,
        *,
        include_query_string: bool = False,
    ) -> str:
        """Normalize a website start URL for deterministic file-id generation."""
        candidate = str(start_url or "").strip()
        if not candidate:
            return ""

        candidate, _ = urldefrag(candidate)
        parsed = urlparse(candidate)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ""

        path = parsed.path or "/"
        query = parsed.query if include_query_string else ""
        normalized = urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                path,
                "",
                query,
                "",
            )
        )
        if normalized.endswith("/") and path != "/":
            normalized = normalized.rstrip("/")
        return normalized

    @classmethod
    def build_generated_file_id(
        cls,
        *,
        start_url: Any,
        knowledge_base_id: Any = None,
        source_mapping_id: Any = None,
        source_type: Any = "website",
        include_query_string: bool = False,
    ) -> Optional[str]:
        """Generate a deterministic website-level file_id from crawl scope."""
        normalized_start_url = cls.normalize_source_url_for_file_id(
            start_url,
            include_query_string=include_query_string,
        )
        if not normalized_start_url:
            return None

        normalized_source_type = (
            cls._normalize_file_id_seed_component(source_type or "website").lower()
            or "website"
        )

        scope_parts = [
            cls._normalize_file_id_seed_component(knowledge_base_id),
            cls._normalize_file_id_seed_component(source_mapping_id),
            normalized_source_type,
            normalized_start_url,
        ]
        seed = "|".join(part for part in scope_parts if part)
        return str(uuid.uuid5(WEBSITE_FILE_ID_NAMESPACE, seed))

    def __init__(self, request_data: Dict[str, Any]):
        self.request_data = request_data
        self.source_details = request_data.get("source_details", {}) or {}
        self.source_type = str(request_data.get("source_type_name", "")).strip().lower()

        self.start_url = str(
            self.source_details.get("start_url")
            or self.source_details.get("root_url")
            or self.source_details.get("website_url")
            or self.source_details.get("url")
            or ""
        ).strip()
        self.max_pages = self._safe_int(
            self.source_details.get("max_pages"),
            default=self.DEFAULT_MAX_PAGES,
        )
        self.max_depth = self._safe_optional_int(
            self.source_details.get("max_depth")
        )
        self.timeout_seconds = self._safe_float(
            self.source_details.get("timeout_seconds")
            or self.source_details.get("timeout"),
            default=self.DEFAULT_TIMEOUT_SECONDS,
        )
        self.delay_seconds = self._safe_float(
            self.source_details.get("delay_seconds")
            or self.source_details.get("request_delay_seconds"),
            default=self.DEFAULT_DELAY_SECONDS,
        )
        self.verify_ssl = self._coerce_bool(
            self.source_details.get("verify_ssl"),
            default=True,
        )
        self.include_query_string = self._coerce_bool(
            self.source_details.get("include_query_string"),
            default=False,
        )
        self.allow_subdomains = self._coerce_bool(
            self.source_details.get("allow_subdomains"),
            default=False,
        )
        self.respect_robots_txt = self._coerce_bool(
            self.source_details.get("respect_robots_txt"),
            default=True,
        )
        self.discover_sitemaps = self._coerce_bool(
            self.source_details.get("discover_sitemaps"),
            default=True,
        )
        self.scope_to_start_path = self._coerce_bool(
            self.source_details.get("scope_to_start_path"),
            default=False,
        )
        self.user_agent = str(
            self.source_details.get("user_agent") or self.DEFAULT_USER_AGENT
        ).strip()
        self.custom_headers = self._parse_headers(self.source_details.get("headers"))
        self.include_url_patterns = self._parse_string_list(
            self.source_details.get("include_url_patterns")
            or self.source_details.get("include_patterns")
        )
        self.exclude_url_patterns = self._parse_string_list(
            self.source_details.get("exclude_url_patterns")
            or self.source_details.get("exclude_patterns")
        )
        self.source_name = (
            self.source_details.get("source_name")
            or request_data.get("name")
            or "Website Source"
        )

        if not self.start_url:
            raise ValueError(
                "Website source requires source_details.start_url, root_url, website_url, or url."
            )

        normalized_start_url = self._normalize_url(self.start_url)
        if not normalized_start_url:
            raise ValueError(f"Invalid website start URL: {self.start_url!r}")
        self.start_url = normalized_start_url
        self.source_mapping_id = (
            self.source_details.get("source_mapping_id")
            or self.source_details.get("id")
            or self.request_data.get("source_mapping_id")
        )
        self.knowledge_base_id = (
            self.request_data.get("knowledge_base_id")
            or self.request_data.get("kb_id")
        )
        self.generated_file_id = str(
            self.source_details.get("file_id")
            or self.build_generated_file_id(
                start_url=self.start_url,
                knowledge_base_id=self.knowledge_base_id,
                source_mapping_id=self.source_mapping_id,
                source_type=self.source_type or "website",
                include_query_string=self.include_query_string,
            )
            or ""
        ).strip()

        parsed_start = urlparse(self.start_url)
        self.root_scheme = parsed_start.scheme.lower()
        self.root_netloc = parsed_start.netloc.lower()
        self.root_path = parsed_start.path or "/"
        self.html_extractor = HTMLExtractor()
        self.robots_parser: Optional[RobotFileParser] = None
        self.robots_text: Optional[str] = None

        logger.info("Website Source initialized")
        logger.info("  Start URL       : %s", self.start_url)
        logger.info("  Source type     : %s", self.source_type or "website")
        logger.info("  Website file_id : %s", self.generated_file_id or "N/A")
        logger.info("  Max pages       : %s", self.max_pages)
        logger.info("  Max depth       : %s", self.max_depth if self.max_depth is not None else "unlimited")
        logger.info("  Delay seconds   : %s", self.delay_seconds)
        logger.info("  Respect robots  : %s", self.respect_robots_txt)
        logger.info("  Discover sitemap: %s", self.discover_sitemaps)

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            if value in (None, ""):
                return default
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_optional_int(value: Any) -> Optional[int]:
        try:
            if value in (None, ""):
                return None
            return max(0, int(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            if value in (None, ""):
                return default
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_bool(value: Any, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0

        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
        return default

    @staticmethod
    def _parse_headers(value: Any) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {}
        headers: Dict[str, str] = {}
        for key, item in value.items():
            if key in (None, "") or item in (None, ""):
                continue
            headers[str(key)] = str(item)
        return headers

    @staticmethod
    def _parse_string_list(value: Any) -> List[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(part).strip() for part in value if str(part).strip()]
        return []

    def _build_client_headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml",
        }
        headers.update(self.custom_headers)
        return headers

    async def discover(self) -> List[Dict[str, Any]]:
        """Crawl the configured website and emit page documents for ingestion."""
        discovered_pages: List[Dict[str, Any]] = []
        queued_urls: Set[str] = set()
        seen_urls: Set[str] = set()
        queue: Deque[Tuple[str, int]] = deque()
        queue.append((self.start_url, 0))
        queued_urls.add(self.start_url)

        client_kwargs = {
            "timeout": self.timeout_seconds,
            "follow_redirects": True,
            "verify": self.verify_ssl,
            "headers": self._build_client_headers(),
        }

        async with httpx.AsyncClient(**client_kwargs) as client:
            if self.respect_robots_txt or self.discover_sitemaps:
                await self._load_robots(client)

            if self.discover_sitemaps:
                sitemap_urls = await self._discover_sitemap_page_urls(client)
                for sitemap_url in sitemap_urls:
                    if sitemap_url not in queued_urls:
                        queue.append((sitemap_url, 0))
                        queued_urls.add(sitemap_url)

            while queue and len(discovered_pages) < self.max_pages:
                current_url, depth = queue.popleft()
                if current_url in seen_urls:
                    continue
                seen_urls.add(current_url)

                if self.max_depth is not None and depth > self.max_depth:
                    continue
                if not self._is_in_scope(current_url):
                    continue
                if not self._is_allowed_by_patterns(current_url):
                    continue
                if self.respect_robots_txt and self.robots_parser is not None:
                    if not self.robots_parser.can_fetch(self.user_agent, current_url):
                        logger.info("Skipping robots-disallowed URL: %s", current_url)
                        continue

                response = await self._fetch_page(client, current_url)
                if response is None:
                    continue

                final_url = self._normalize_url(str(response.url)) or current_url
                if not self._is_in_scope(final_url):
                    logger.info(
                        "Skipping redirected out-of-scope URL: %s -> %s",
                        current_url,
                        final_url,
                    )
                    continue
                if not self._is_allowed_by_patterns(final_url):
                    logger.info(
                        "Skipping redirected URL excluded by filters: %s -> %s",
                        current_url,
                        final_url,
                    )
                    continue
                if final_url != current_url:
                    queued_urls.add(final_url)
                    if final_url in seen_urls:
                        continue

                page_document, child_urls = self._build_page_document(
                    final_url,
                    response.text,
                    depth,
                    response,
                )
                if page_document is not None:
                    discovered_pages.append(page_document)

                for child_url in child_urls:
                    if child_url in seen_urls or child_url in queued_urls:
                        continue
                    if self.max_depth is not None and depth + 1 > self.max_depth:
                        continue
                    queue.append((child_url, depth + 1))
                    queued_urls.add(child_url)

                if self.delay_seconds > 0:
                    await asyncio.sleep(self.delay_seconds)

        logger.info(
            "Website crawl complete | start_url=%s | pages=%d",
            self.start_url,
            len(discovered_pages),
        )
        return discovered_pages

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> Optional[httpx.Response]:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Website fetch failed for %s: %s", url, exc)
            return None

        content_type = str(response.headers.get("content-type", "")).lower()
        if not self._is_html_response(content_type, response.text):
            logger.info("Skipping non-HTML URL: %s (%s)", url, content_type or "unknown")
            return None
        return response

    async def _load_robots(self, client: httpx.AsyncClient) -> None:
        robots_url = f"{self.root_scheme}://{self.root_netloc}/robots.txt"
        try:
            response = await client.get(robots_url)
            if response.status_code >= 400:
                return
        except httpx.HTTPError:
            return

        self.robots_text = response.text or ""
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(self.robots_text.splitlines())
        self.robots_parser = parser

    async def _discover_sitemap_page_urls(
        self,
        client: httpx.AsyncClient,
    ) -> List[str]:
        candidates: Deque[str] = deque()
        seen_sitemaps: Set[str] = set()
        page_urls: List[str] = []

        if self.robots_text:
            for line in self.robots_text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    normalized = self._normalize_url(sitemap_url)
                    if normalized:
                        candidates.append(normalized)

        default_sitemaps = [
            f"{self.root_scheme}://{self.root_netloc}/sitemap.xml",
            f"{self.root_scheme}://{self.root_netloc}/sitemap_index.xml",
        ]
        for sitemap_url in default_sitemaps:
            normalized = self._normalize_url(sitemap_url)
            if normalized:
                candidates.append(normalized)

        while candidates and len(seen_sitemaps) < self.DEFAULT_MAX_SITEMAPS:
            sitemap_url = candidates.popleft()
            if sitemap_url in seen_sitemaps:
                continue
            seen_sitemaps.add(sitemap_url)

            try:
                response = await client.get(sitemap_url, headers={"Accept": "application/xml,text/xml"})
                if response.status_code >= 400:
                    continue
            except httpx.HTTPError:
                continue

            nested_sitemaps, discovered_urls = self._parse_sitemap_xml(response.text)
            for nested_url in nested_sitemaps:
                normalized_nested = self._normalize_url(nested_url)
                if normalized_nested and normalized_nested not in seen_sitemaps:
                    candidates.append(normalized_nested)

            for discovered_url in discovered_urls:
                normalized_page_url = self._normalize_url(discovered_url)
                if normalized_page_url and self._is_in_scope(normalized_page_url):
                    page_urls.append(normalized_page_url)

        deduped_urls: List[str] = []
        seen_urls: Set[str] = set()
        for page_url in page_urls:
            if page_url in seen_urls:
                continue
            seen_urls.add(page_url)
            deduped_urls.append(page_url)
        return deduped_urls

    @staticmethod
    def _parse_sitemap_xml(xml_text: str) -> Tuple[List[str], List[str]]:
        if not xml_text.strip():
            return [], []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return [], []

        def local_name(tag: str) -> str:
            return tag.rsplit("}", 1)[-1] if "}" in tag else tag

        root_name = local_name(root.tag)
        sitemap_urls: List[str] = []
        page_urls: List[str] = []

        if root_name == "sitemapindex":
            for child in root:
                if local_name(child.tag) != "sitemap":
                    continue
                loc = child.findtext(".//{*}loc")
                if loc:
                    sitemap_urls.append(loc.strip())
        elif root_name == "urlset":
            for child in root:
                if local_name(child.tag) != "url":
                    continue
                loc = child.findtext(".//{*}loc")
                if loc:
                    page_urls.append(loc.strip())

        return sitemap_urls, page_urls

    def _build_page_document(
        self,
        page_url: str,
        html_content: str,
        depth: int,
        response: httpx.Response,
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        parsed = self.html_extractor._parse_html(html_content)
        normalized_links = self._normalize_page_links(
            page_url,
            parsed.get("links", []),
        )
        parsed["links"] = normalized_links
        parsed["title"] = parsed.get("title") or self._title_from_url(page_url)

        markdown_content = self.html_extractor._build_markdown(
            self._slug_from_url(page_url),
            parsed,
        ).strip()
        if not markdown_content:
            return None, [link["url"] for link in normalized_links]

        page_title = str(parsed.get("title") or "").strip() or self._title_from_url(page_url)
        content_size_bytes = len(markdown_content.encode("utf-8"))
        discovered_file = {
            "filename": self._make_page_filename(page_url, page_title),
            "file_path": None,
            "file_type": ".html",
            "content": markdown_content,
            "content_size_bytes": content_size_bytes,
            "source": "website",
            "url": page_url,
            "page_url": page_url,
            "page_title": page_title,
            "crawl_depth": depth,
            "discovered_links": len(normalized_links),
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "site_root_url": self._normalize_url(self.start_url),
            "source_name": self.source_name,
            "file_id": self.generated_file_id or None,
        }
        return discovered_file, [link["url"] for link in normalized_links]

    def _normalize_page_links(
        self,
        base_url: str,
        raw_links: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        normalized_links: List[Dict[str, str]] = []
        seen_urls: Set[str] = set()

        for link in raw_links:
            if not isinstance(link, dict):
                continue
            href = str(link.get("url") or "").strip()
            if not href:
                continue

            absolute_url = self._normalize_url(urljoin(base_url, href))
            if not absolute_url:
                continue
            if not self._is_in_scope(absolute_url):
                continue
            if not self._is_allowed_by_patterns(absolute_url):
                continue
            if not self._is_html_candidate(absolute_url):
                continue
            if absolute_url in seen_urls:
                continue

            seen_urls.add(absolute_url)
            normalized_links.append(
                {
                    "text": str(link.get("text") or absolute_url).strip() or absolute_url,
                    "url": absolute_url,
                }
            )

        return normalized_links

    def _is_in_scope(self, candidate_url: str) -> bool:
        parsed = urlparse(candidate_url)
        if parsed.scheme.lower() not in {"http", "https"}:
            return False

        candidate_host = parsed.netloc.lower()
        if self.allow_subdomains:
            if candidate_host != self.root_netloc and not candidate_host.endswith(f".{self.root_netloc}"):
                return False
        elif candidate_host != self.root_netloc:
            return False

        if self.scope_to_start_path:
            candidate_path = parsed.path or "/"
            root_path = self.root_path or "/"
            if not candidate_path.startswith(root_path.rstrip("/") or "/"):
                return False

        return True

    def _is_allowed_by_patterns(self, candidate_url: str) -> bool:
        lowered_url = candidate_url.lower()

        if self.include_url_patterns:
            if not any(pattern.lower() in lowered_url for pattern in self.include_url_patterns):
                return False

        if self.exclude_url_patterns:
            if any(pattern.lower() in lowered_url for pattern in self.exclude_url_patterns):
                return False

        return True

    def _normalize_url(self, raw_url: str) -> str:
        candidate = str(raw_url or "").strip()
        if not candidate:
            return ""

        candidate, _ = urldefrag(candidate)
        parsed = urlparse(candidate)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ""

        path = parsed.path or "/"
        query = parsed.query if self.include_query_string else ""
        normalized = urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                path,
                "",
                query,
                "",
            )
        )
        if normalized.endswith("/") and path != "/":
            normalized = normalized.rstrip("/")
        return normalized

    def _is_html_candidate(self, candidate_url: str) -> bool:
        parsed = urlparse(candidate_url)
        path = (parsed.path or "").lower()
        return not any(path.endswith(ext) for ext in self.SKIP_EXTENSIONS)

    @staticmethod
    def _is_html_response(content_type: str, body: str) -> bool:
        lowered = (content_type or "").lower()
        if any(html_type in lowered for html_type in WebsiteSource.HTML_CONTENT_TYPES):
            return True

        stripped = (body or "").lstrip().lower()
        return stripped.startswith("<!doctype html") or "<html" in stripped[:500]

    @staticmethod
    def _title_from_url(page_url: str) -> str:
        parsed = urlparse(page_url)
        path = parsed.path.rstrip("/")
        if not path:
            return parsed.netloc
        slug = path.rsplit("/", 1)[-1]
        slug = re.sub(r"[-_]+", " ", slug).strip()
        return slug or parsed.netloc

    @staticmethod
    def _slug_from_url(page_url: str) -> str:
        parsed = urlparse(page_url)
        slug = parsed.path.strip("/") or "home"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug).strip("._-")
        return slug or "page"

    def _make_page_filename(self, page_url: str, title: str) -> str:
        slug = self._slug_from_url(page_url)
        title_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._-")
        digest = hashlib.md5(page_url.encode("utf-8")).hexdigest()[:8]
        if title_slug:
            return f"{slug}_{title_slug[:40]}_{digest}.html"
        return f"{slug}_{digest}.html"
