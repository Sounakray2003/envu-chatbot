import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from services.api_upload import (
    APISourceInput,
    AuthType,
    PaginationType,
    PostPaginationStrategy,
    WRITE_METHODS,
)
from services.extraction.extractors.json_extractor import JSONExtractor

logger = logging.getLogger(__name__)


class APISourceService:
    """Fetch JSON from an API and prepare the aggregated payload for ingestion."""

    DEFAULT_TIMEOUT_SECONDS = 30.0
    MAX_PAGES = 50

    async def fetch_and_extract(self, payload: APISourceInput) -> Dict[str, Any]:
        """Fetch paginated API data and return the JSON text used downstream."""
        headers, auth = await self._build_auth(payload)
        extractor = JSONExtractor(
            {
                "fmt": "json",
                "json_path": payload.json_path,
                "field_paths": payload.field_paths,
            }
        )

        aggregated_entries: List[Any] = []
        pages_fetched = 0
        page_marker: Any = payload.pagination.start_page or 1
        is_write_method = payload.http_method in WRITE_METHODS
        client_kwargs = self._build_client_kwargs(payload)

        async with httpx.AsyncClient(
            timeout=self.DEFAULT_TIMEOUT_SECONDS,
            follow_redirects=True,
            **client_kwargs,
        ) as client:
            while pages_fetched < self.MAX_PAGES:
                query_params, request_body = self._build_request_params(
                    payload=payload,
                    page_marker=page_marker,
                    is_write_method=is_write_method,
                )

                logger.info(
                    "Fetching API page %s | method=%s | params=%s | has_body=%s",
                    page_marker,
                    payload.http_method.value,
                    query_params,
                    request_body is not None,
                )

                response = await client.request(
                    method=payload.http_method.value,
                    url=str(payload.url),
                    headers=headers,
                    params=query_params if query_params else None,
                    json=request_body if is_write_method else None,
                    auth=auth,
                )
                response.raise_for_status()

                response_data = response.json()
                result = extractor.extract_data_result(
                    response_data,
                    source_name=payload.source_name,
                    base_metadata={"page_index": pages_fetched + 1},
                )

                if not result.success:
                    logger.warning(
                        "Extraction failed on page %s: %s",
                        page_marker,
                        result.error,
                    )
                    break

                page_entries = self._attach_source_url(
                    [entry.raw_object for entry in result.entries],
                    str(response.url),
                )
                if not page_entries:
                    logger.info("No records found on page %s. Stopping.", page_marker)
                    break

                aggregated_entries.extend(page_entries)
                pages_fetched += 1

                logger.info(
                    "Fetched page %s with %s record(s); total=%s",
                    page_marker,
                    len(page_entries),
                    len(aggregated_entries),
                )

                next_marker = self._resolve_next_page_marker(
                    payload=payload,
                    current_marker=page_marker,
                    page_size=payload.pagination.page_size_value or 20,
                    response_data=response_data,
                    page_entries_count=len(page_entries),
                )
                if next_marker is None:
                    logger.info("Last API page detected. Stopping pagination.")
                    break

                page_marker = next_marker

        if not aggregated_entries:
            raise ValueError("No entries extracted from API")

        json_text = json.dumps(aggregated_entries, indent=2, ensure_ascii=False)
        content_size_bytes = len(json_text.encode("utf-8"))

        logger.info(
            "API pagination finished | pages=%s | records=%s",
            pages_fetched,
            len(aggregated_entries),
        )

        return {
            "source_name": payload.source_name,
            "url": str(payload.url),
            "pages_fetched": pages_fetched,
            "entries_extracted": len(aggregated_entries),
            "data": aggregated_entries,
            "json_chunk_text": json_text,
            "content_size_bytes": content_size_bytes,
        }

    def _build_request_params(
        self,
        payload: APISourceInput,
        page_marker: Any,
        is_write_method: bool,
    ) -> Tuple[Dict[str, Any], Any]:
        """Return the query params and request body for one API page."""
        strategy = payload.pagination.post_pagination_strategy
        base_query_params: Dict[str, Any] = dict(payload.query_params)
        pagination_params = self._build_pagination_params(payload, page_marker)

        if not is_write_method:
            return {**base_query_params, **pagination_params}, None

        if strategy == PostPaginationStrategy.QUERY or not pagination_params:
            return {**base_query_params, **pagination_params}, payload.request_body

        base_body = payload.request_body
        if base_body is None:
            merged_body = dict(pagination_params)
        elif isinstance(base_body, dict):
            merged_body, merged_at_json_path = self._merge_into_request_body(
                request_body=base_body,
                json_path=payload.json_path,
                values=pagination_params,
            )
            if not merged_at_json_path:
                merged_body = {**copy.deepcopy(base_body), **pagination_params}
        else:
            logger.warning(
                "Body pagination requested but request_body is %s. Falling back to query params.",
                type(base_body).__name__,
            )
            return {**base_query_params, **pagination_params}, base_body

        return base_query_params, merged_body

    def _build_pagination_params(
        self,
        payload: APISourceInput,
        page_marker: Any,
    ) -> Dict[str, Any]:
        """Return the pagination key-value pairs for the current page."""
        if payload.pagination.pagination_type == PaginationType.NONE:
            return {}

        page_param = payload.pagination.page_param_name
        size_param = payload.pagination.page_size_param_name
        page_size = payload.pagination.page_size_value

        if not page_param or not size_param or page_size is None:
            raise ValueError("Pagination parameters are incomplete")

        return {
            size_param: page_size,
            page_param: page_marker,
        }

    def _merge_into_request_body(
        self,
        request_body: Any,
        json_path: Optional[str],
        values: Dict[str, Any],
    ) -> Tuple[Any, bool]:
        """Merge pagination values into a nested request-body object."""
        if (
            not isinstance(request_body, (dict, list))
            or not values
            or not str(json_path or "").strip()
        ):
            return request_body, False

        cloned_body = copy.deepcopy(request_body)
        target_node, matched = self._resolve_json_path_reference(cloned_body, json_path)
        if not matched or not isinstance(target_node, dict):
            return request_body, False

        target_node.update(values)
        return cloned_body, True

    def _resolve_json_path_reference(
        self,
        data: Any,
        path: Optional[str],
    ) -> Tuple[Any, bool]:
        """Resolve a simple JSON path against a request body."""
        tokens = self._parse_json_path(path)
        current = data

        for token_type, token_value in tokens:
            if token_type == "key":
                if not isinstance(current, dict) or token_value not in current:
                    return None, False
                current = current[token_value]
                continue

            if token_type == "index":
                if not isinstance(current, list):
                    return None, False
                if not (-len(current) <= token_value < len(current)):
                    return None, False
                current = current[token_value]
                continue

            return None, False

        return current, True

    @staticmethod
    def _parse_json_path(path: Optional[str]) -> List[Tuple[str, Any]]:
        """Parse a minimal JSONPath expression into key/index tokens."""
        raw_path = str(path or "").strip()
        if not raw_path or raw_path == "$":
            return []

        if raw_path.startswith("$."):
            raw_path = raw_path[2:]
        elif raw_path.startswith("$"):
            raw_path = raw_path[1:]
        elif not raw_path.startswith("["):
            raw_path = "." + raw_path

        tokens: List[Tuple[str, Any]] = []
        index = 0

        while index < len(raw_path):
            char = raw_path[index]

            if char == ".":
                index += 1
                start = index
                while index < len(raw_path) and raw_path[index] not in ".[":
                    index += 1
                key = raw_path[start:index].strip()
                if key:
                    tokens.append(("key", key))
                continue

            if char == "[":
                end = raw_path.find("]", index)
                if end == -1:
                    raise ValueError(f"Invalid JSON path {path!r}: missing ']'")
                content = raw_path[index + 1 : end].strip()
                if (
                    len(content) >= 2
                    and content[0] in {"'", '"'}
                    and content[-1] == content[0]
                ):
                    tokens.append(("key", content[1:-1]))
                else:
                    try:
                        tokens.append(("index", int(content)))
                    except ValueError:
                        tokens.append(("key", content))
                index = end + 1
                continue

            start = index
            while index < len(raw_path) and raw_path[index] not in ".[":
                index += 1
            key = raw_path[start:index].strip()
            if key:
                tokens.append(("key", key))

        return tokens

    async def _build_auth(
        self,
        payload: APISourceInput,
    ) -> Tuple[Dict[str, str], Optional[httpx.Auth]]:
        """Build request headers and optional httpx auth objects."""
        headers = {
            "Accept": "application/json",
            **payload.headers,
        }
        if payload.http_method in WRITE_METHODS and payload.request_body is not None:
            headers.setdefault("Content-Type", "application/json")

        auth: Optional[httpx.Auth] = None

        if payload.auth.type == AuthType.BASIC:
            auth = httpx.BasicAuth(
                payload.auth.username,
                payload.auth.password.get_secret_value(),
            )
        elif payload.auth.type == AuthType.BEARER:
            headers["Authorization"] = (
                f"Bearer {payload.auth.token.get_secret_value()}"
            )
        elif payload.auth.type == AuthType.API_KEY:
            headers[payload.auth.header_name] = payload.auth.api_key.get_secret_value()
        elif payload.auth.type == AuthType.OAUTH2:
            headers["Authorization"] = (
                f"Bearer {await self._get_oauth2_token(payload)}"
            )

        return headers, auth

    async def _get_oauth2_token(self, payload: APISourceInput) -> str:
        """Fetch an OAuth2 client-credentials token when configured."""
        auth = payload.auth
        if auth.type != AuthType.OAUTH2:
            raise ValueError("OAuth2 token requested for non-OAuth2 configuration")

        request_data = {"grant_type": "client_credentials"}
        if auth.scope:
            request_data["scope"] = auth.scope

        client_kwargs = self._build_client_kwargs(payload)
        async with httpx.AsyncClient(
            timeout=self.DEFAULT_TIMEOUT_SECONDS,
            follow_redirects=True,
            **client_kwargs,
        ) as client:
            response = await client.post(
                str(auth.token_url),
                data=request_data,
                auth=httpx.BasicAuth(
                    auth.client_id,
                    auth.client_secret.get_secret_value(),
                ),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            token_payload = response.json()

        access_token = token_payload.get("access_token")
        if not access_token:
            raise ValueError("OAuth2 token response did not include access_token")
        return access_token

    def _build_client_kwargs(self, payload: APISourceInput) -> Dict[str, Any]:
        """Build shared httpx client kwargs for TLS verification behavior."""
        ca_bundle_path = (payload.ca_bundle_path or "").strip() or None

        if ca_bundle_path:
            normalized_path = os.path.expandvars(os.path.expanduser(ca_bundle_path))
            if not os.path.exists(normalized_path):
                raise ValueError(
                    f"Custom CA bundle path does not exist: {normalized_path}"
                )
            logger.info("API source using custom CA bundle: %s", normalized_path)
            return {"verify": normalized_path}

        if not payload.verify_ssl:
            logger.warning(
                "TLS certificate verification is disabled for API source '%s' (%s).",
                payload.source_name,
                payload.url,
            )
            return {"verify": False}

        return {}

    def _next_page_marker(
        self,
        payload: APISourceInput,
        page_marker: Any,
        page_size: int,
    ) -> Any:
        """Calculate the next page marker when the response has no explicit hint."""
        if payload.pagination.pagination_type == PaginationType.PAGE_NUMBER:
            return page_marker + 1
        if payload.pagination.pagination_type == PaginationType.OFFSET:
            return page_marker + page_size
        return page_marker

    def _resolve_next_page_marker(
        self,
        payload: APISourceInput,
        current_marker: Any,
        page_size: int,
        response_data: Any,
        page_entries_count: int,
    ) -> Optional[Any]:
        """Resolve the next page marker from response pagination hints."""
        pagination = (
            response_data.get("pagination", {}) if isinstance(response_data, dict) else {}
        )

        has_next = pagination.get("has_next")
        if has_next is False:
            return None

        next_page = self._safe_int(pagination.get("next_page"))
        if next_page is not None:
            return next_page

        next_offset = self._safe_int(pagination.get("next_offset"))
        if next_offset is not None:
            return next_offset

        total_pages = self._safe_int(pagination.get("total_pages"))
        current_page = self._safe_int(pagination.get("page"))
        if total_pages is not None and current_page is not None:
            if current_page >= total_pages:
                return None
            return current_page + 1

        total_items = self._safe_int(pagination.get("total_items"))
        if (
            payload.pagination.pagination_type == PaginationType.OFFSET
            and total_items is not None
        ):
            next_marker = current_marker + page_size
            return next_marker if next_marker < total_items else None

        if page_entries_count < page_size:
            return None

        return self._next_page_marker(payload, current_marker, page_size)

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _attach_source_url(entries: List[Any], page_url: str) -> List[Any]:
        """Annotate each extracted entry with the response URL."""
        annotated_entries: List[Any] = []
        for entry in entries:
            if isinstance(entry, dict):
                annotated_entries.append({**entry, "_source_url": page_url})
            else:
                annotated_entries.append({"value": entry, "_source_url": page_url})
        return annotated_entries
