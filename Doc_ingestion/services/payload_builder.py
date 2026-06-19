"""
Standardized Qdrant payload builder.

Constructs a unified payload format for all data sources and keeps source
metadata consistent across file, API, and website ingestion paths.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

logger = logging.getLogger(__name__)

_INGESTION_COLLECTION_NAME = "main_memory"
_DEFAULT_OPENAI_EMBEDDING_MODEL = (
    str(os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")).strip()
    or "text-embedding-3-large"
)
_DEFAULT_OPENAI_EMBEDDING_DIMENSIONS = int(
    str(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1024")).strip() or "1024"
)

_SENSITIVE_DETAIL_TOKENS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "private_key",
    "access_key",
    "credential",
    "dsn",
)


class PayloadBuilder:
    """Build standardized Qdrant payloads from chunk and request data."""

    @staticmethod
    def _coerce_bool(value: Any, default: bool = True) -> bool:
        """Convert common truthy / falsy request values into a boolean."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0

        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on", "active", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "inactive", "disabled"}:
            return False
        return default

    def __init__(self, request_data: Dict[str, Any]):
        """Initialize builder with request context."""
        self.request_data = request_data
        self.knowledge_base_id = request_data.get("knowledge_base_id")
        self.org_id = request_data.get("org_id")
        self.member_id = str(request_data.get("member_id", ""))
        self.knowledge_base_name = request_data.get("name", "")
        self.source_type_id = request_data.get("source_type_id")
        self.source_type_name = str(
            request_data.get("source_type_name", "")
        ).strip()

        raw_source_details = request_data.get("source_details", {})
        self.source_details = raw_source_details if isinstance(raw_source_details, dict) else {}
        self.source_mapping_id = (
            self.source_details.get("source_mapping_id")
            or self.source_details.get("id")
            or request_data.get("source_mapping_id")
        )
        
        self.source_is_active = self._coerce_bool(
            request_data.get(
                "isActive",
                request_data.get(
                    "is_active",
                    self.source_details.get(
                        "isActive",
                        self.source_details.get("is_active", True),
                    ),
                ),
            ),
            default=True,
        )

        self.chunking_details = request_data.get("chunking_details", {})
        self.embedding_details = request_data.get("embedding_details", {})
        self.vector_store_details = request_data.get("vector_store_details", {})
        
        # Handle both list and dict formats for source_details
        # When source_details is a list (multi-source), use the first element
        # When source_details is a dict (single source), use it directly
        source_details_raw = request_data.get("source_details", {})
        if isinstance(source_details_raw, list):
            self.source_details = source_details_raw[0] if source_details_raw else {}
        else:
            self.source_details = source_details_raw

    def build_payload(
        self,
        chunk: Dict[str, Any],
        ingestion_source: str = "request",
    ) -> Dict[str, Any]:
        """Build a complete standardized payload for a chunk."""
        chunk_id = chunk.get("chunk_id") or str(uuid4())
        metadata = self._get_chunk_metadata(chunk)
        source_type_name = self._resolve_source_type_name(chunk)
        source_mapping_id = self._resolve_source_id(chunk)
        is_active = self._resolve_is_active(chunk)
        source_details_payload = self._build_source_details(chunk)
        file_id = self._resolve_file_id(chunk, source_details_payload)

        ingestion_metadata = {
            "ingestion_timestamp": datetime.utcnow().isoformat() + "Z",
            "ingestion_user_id": self.member_id,
            "ingestion_source": ingestion_source,
        }
        if chunk.get("processing_duration_ms"):
            ingestion_metadata["ingestion_duration_ms"] = chunk.get(
                "processing_duration_ms"
            )

        payload = {
            "chunk_id": chunk_id,
            "chunk_index": chunk.get("chunk_index", 0),
            "char_count": chunk.get("char_count", 0),
            "token_count": chunk.get("token_count", 0),
            "text": chunk.get("text", ""),
            "knowledge_base_id": self.knowledge_base_id,
            "org_id": self.org_id,
            "member_id": self.member_id,
            "knowledge_base_name": self.knowledge_base_name,
            "source_type_id": self.source_type_id,
            "source_type_name": source_type_name,
            "source_mapping_id": source_mapping_id,
            "source_id": source_mapping_id,
            "isActive": is_active,
            "file_id": file_id,
            "chunking_details": self._build_chunking_details(),
            "embedding_details": self._build_embedding_details(),
            "vector_store_details": self._build_vector_store_details(),
            "source_details": source_details_payload,
            "ingestion_metadata": ingestion_metadata,
        }

        return payload

    def _get_chunk_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        metadata = chunk.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    def _resolve_source_id(self, chunk: Dict[str, Any]) -> Any:
        metadata = self._get_chunk_metadata(chunk)
        source_details = self._resolve_source_details_context(chunk)
        source_mapping_id = (
            chunk.get("source_mapping_id")
            or metadata.get("source_mapping_id")
            or source_details.get("source_mapping_id")
            or source_details.get("id")
            or self.source_mapping_id
        )
        
        if source_mapping_id in (None, ""):
            return None
        return str(source_mapping_id)

    def _resolve_is_active(self, chunk: Dict[str, Any]) -> bool:
        metadata = self._get_chunk_metadata(chunk)
        return self._coerce_bool(
            chunk.get(
                "isActive",
                chunk.get(
                    "is_active",
                    metadata.get(
                        "isActive",
                        metadata.get("is_active", self.source_is_active),
                    ),
                ),
            ),
            default=True,
        )

    def _resolve_source_type_name(self, chunk: Dict[str, Any]) -> str:
        metadata = self._get_chunk_metadata(chunk)
        return (
            chunk.get("source_type_name")
            or metadata.get("source_type_name")
            or metadata.get("source_type_from_config")
            or self.source_type_name
            or ""
        ).strip()

    def _resolve_source_details_context(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        metadata = self._get_chunk_metadata(chunk)
        source_details = metadata.get("source_details")
        if isinstance(source_details, dict):
            return source_details
        return self.source_details

    def _resolve_file_id(
        self,
        chunk: Dict[str, Any],
        source_details_payload: Dict[str, Any],
    ) -> Any:
        """Resolve the file identifier for file-based payloads when available."""
        metadata = self._get_chunk_metadata(chunk)
        source_details = self._resolve_source_details_context(chunk)
        file_id = (
            chunk.get("file_id")
            or metadata.get("file_id")
            or source_details_payload.get("file_id")
            or source_details.get("file_id")
        )
        if file_id in (None, ""):
            return None
        return str(file_id)

    def _build_chunking_details(self) -> Dict[str, Any]:
        """Extract relevant chunking details from the request."""
        details = {
            "chunking_type": self.chunking_details.get(
                "chunking_type", "SEMANTIC"
            )
        }
        for key in [
            "chunkSize",
            "chunkOverlap",
            "delimiter",
            "parentChunkSize",
            "childChunkSize",
            "similarityThreshold",
        ]:
            if key in self.chunking_details:
                details[key] = self.chunking_details[key]
        return details

    def _build_embedding_details(self) -> Dict[str, Any]:
        """Extract embedding model details from the request."""
        return {
            "embedding_model_name": self.embedding_details.get(
                "embedding_model_name", _DEFAULT_OPENAI_EMBEDDING_MODEL
            ),
            "dimensions": self.embedding_details.get(
                "dimensions",
                _DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
            ),
            "embedded_model_id": self.embedding_details.get("embedded_model_id"),
        }

    def _build_vector_store_details(self) -> Dict[str, Any]:
        """Extract vector store configuration from the request."""
        if not isinstance(self.vector_store_details, dict):
            return {"collection_name": _INGESTION_COLLECTION_NAME}

        details = self._sanitize_detail_mapping(self.vector_store_details)

        if "vector_store_name" not in details:
            fallback_name = self.request_data.get("vector_store_name")
            if fallback_name not in (None, ""):
                details["vector_store_name"] = fallback_name

        if "vector_store_id" not in details:
            fallback_id = self.request_data.get("vector_store_id")
            if fallback_id not in (None, ""):
                details["vector_store_id"] = fallback_id

        details["collection_name"] = _INGESTION_COLLECTION_NAME

        return details

    @classmethod
    def _sanitize_detail_mapping(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Return a payload-safe copy of a config/details mapping."""
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            sanitized[key] = cls._sanitize_detail_value(key, item)
        return sanitized

    @classmethod
    def _sanitize_detail_value(cls, key: str, value: Any) -> Any:
        """Recursively redact sensitive values while preserving safe config."""
        normalized_key = str(key or "").strip().lower()
        if any(token in normalized_key for token in _SENSITIVE_DETAIL_TOKENS):
            return "[REDACTED]"

        if isinstance(value, dict):
            return cls._sanitize_detail_mapping(value)

        if isinstance(value, list):
            return [
                cls._sanitize_detail_value(key, item)
                for item in value
            ]

        return value

    def _build_source_details(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Route source details building based on source type."""
        source_type = self._resolve_source_type_name(chunk).lower()
        source_details = self._resolve_source_details_context(chunk)

        if "database" in source_type:
            return self._build_database_source_details(chunk, source_details)
        if "website" in source_type or "web" in source_type:
            return self._build_website_source_details(chunk, source_details)
        if "api" in source_type:
            return self._build_api_source_details(chunk, source_details)
        if "cloud" in source_type or "s3" in source_type or "gcs" in source_type:
            return self._build_cloud_source_details(chunk, source_details)
        return self._build_file_source_details(chunk, source_details)

    def _build_database_source_details(
        self,
        chunk: Dict[str, Any],
        source_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build database-specific source details."""
        metadata = self._get_chunk_metadata(chunk)
        details = {
            "database_type": str(
                source_details.get("database_type_name")
                or metadata.get("db_type")
                or ""
            ).lower(),
            "database_name": (
                source_details.get("database_name")
                or source_details.get("database", "")
                or metadata.get("source_db", "")
            ),
        }

        if source_details.get("host"):
            details["host"] = source_details["host"]
        if source_details.get("table_name") or source_details.get("table"):
            details["table_name"] = (
                source_details.get("table_name")
                or source_details.get("table")
            )
        if source_details.get("custom_query") or source_details.get("customQuery"):
            details["query"] = (
                source_details.get("custom_query")
                or source_details.get("customQuery")
            )

        if "mongodb" in details.get("database_type", ""):
            if source_details.get("collection_name") or metadata.get("collection_name"):
                details["collection_name"] = (
                    source_details.get("collection_name")
                    or metadata.get("collection_name")
                )
            if source_details.get("query_filter"):
                details["query_filter"] = source_details["query_filter"]

        if "row_index" in metadata:
            details["row_index"] = metadata["row_index"]
        if "total_rows" in metadata:
            details["total_rows"] = metadata["total_rows"]

        return details

    def _build_api_source_details(
        self,
        chunk: Dict[str, Any],
        source_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build API-specific source details."""
        metadata = self._get_chunk_metadata(chunk)
        details = {
            "api_url": metadata.get("api_url") or source_details.get("url", ""),
            "api_method": str(
                source_details.get("method")
                or source_details.get("api_method")
                or "GET"
            ).upper(),
        }

        if "pages_fetched" in metadata:
            details["pages_fetched"] = metadata["pages_fetched"]
        if "records_fetched" in metadata:
            details["records_fetched"] = metadata["records_fetched"]
        if "current_page" in metadata:
            details["current_page"] = metadata["current_page"]
        if "records_per_page" in metadata:
            details["records_per_page"] = metadata["records_per_page"]

        return details

    def _build_website_source_details(
        self,
        chunk: Dict[str, Any],
        source_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build website-crawl-specific source details."""
        metadata = self._get_chunk_metadata(chunk)
        details = {
            "start_url": (
                source_details.get("start_url")
                or source_details.get("root_url")
                or source_details.get("website_url")
                or source_details.get("url")
                or metadata.get("site_root_url")
                or metadata.get("page_url")
                or ""
            ),
            "page_url": metadata.get("page_url") or metadata.get("url") or "",
        }

        file_id = (
            chunk.get("file_id")
            or metadata.get("file_id")
            or source_details.get("file_id")
        )
        if file_id not in (None, ""):
            details["file_id"] = str(file_id)

        page_title = metadata.get("page_title")
        if page_title:
            details["page_title"] = page_title

        for field in (
            "crawl_depth",
            "discovered_links",
            "status_code",
            "content_type",
            "site_root_url",
        ):
            if field in metadata:
                details[field] = metadata[field]

        for field in (
            "max_pages",
            "max_depth",
            "respect_robots_txt",
            "discover_sitemaps",
            "scope_to_start_path",
            "allow_subdomains",
        ):
            if field in source_details:
                details[field] = source_details[field]

        return details

    def _build_cloud_source_details(
        self,
        chunk: Dict[str, Any],
        source_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build cloud-storage-specific source details."""
        metadata = self._get_chunk_metadata(chunk)
        details = {
            "provider": str(
                source_details.get("provider_name")
                or metadata.get("source")
                or ""
            ).lower()
        }

        folder_path = source_details.get("folder_path", "")
        if folder_path.startswith("s3://"):
            parts = folder_path.replace("s3://", "").split("/", 1)
            details["bucket_name"] = parts[0]
            details["object_key"] = parts[1] if len(parts) > 1 else ""
        elif folder_path.startswith("gs://"):
            parts = folder_path.replace("gs://", "").split("/", 1)
            details["bucket_name"] = parts[0]
            details["object_key"] = parts[1] if len(parts) > 1 else ""
        elif folder_path.startswith("azure://"):
            parts = folder_path.replace("azure://", "").split("/", 1)
            details["container_name"] = parts[0]
            details["object_key"] = parts[1] if len(parts) > 1 else ""
        else:
            if source_details.get("container_name"):
                details["container_name"] = source_details["container_name"]
            elif source_details.get("bucket_name"):
                details["container_name"] = source_details["bucket_name"]

            details["object_key"] = (
                metadata.get("s3_key")
                or metadata.get("gcs_blob")
                or metadata.get("azure_blob")
                or folder_path
            )

        if "file_size_bytes" in metadata:
            details["file_size_bytes"] = metadata["file_size_bytes"]
        if "last_modified" in metadata:
            details["last_modified"] = metadata["last_modified"]

        return details

    def _build_file_source_details(
        self,
        chunk: Dict[str, Any],
        source_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build file, CSV, Excel, and JSON source details."""
        metadata = self._get_chunk_metadata(chunk)
        source_type_name = self._resolve_source_type_name(chunk).lower()
        file_type = str(
            metadata.get("file_type")
            or metadata.get("file_ext")
            or ""
        ).lower()

        details: Dict[str, Any] = {}

        if "filename" in metadata:
            details["filename"] = metadata["filename"]
        elif "source_file" in metadata:
            details["filename"] = metadata["source_file"]
        elif "filename" in chunk:
            details["filename"] = chunk["filename"]
        elif source_details.get("filename"):
            details["filename"] = source_details["filename"]

        file_id = (
            chunk.get("file_id")
            or metadata.get("file_id")
            or source_details.get("file_id")
        )
        if file_id not in (None, ""):
            details["file_id"] = str(file_id)

        if file_type == ".csv" or "csv" in source_type_name:
            details["delimiter"] = metadata.get("delimiter", ",")
            if "total_rows" in metadata:
                details["total_rows"] = metadata["total_rows"]
            if "total_columns" in metadata:
                details["total_columns"] = metadata["total_columns"]
            if "column_names" in metadata:
                details["column_names"] = metadata["column_names"]
            if "columns" in metadata:
                details["column_names"] = metadata["columns"]
            if "row_index" in metadata:
                details["row_index"] = metadata["row_index"]

        elif file_type in {".xlsx", ".xls", ".xlsm"} or "excel" in source_type_name:
            if "sheet_name" in metadata:
                details["sheet_name"] = metadata["sheet_name"]
            if "total_sheets" in metadata:
                details["total_sheets"] = metadata["total_sheets"]
            if "total_rows" in metadata:
                details["total_rows"] = metadata["total_rows"]
            if "sheet_row_index" in metadata:
                details["sheet_row_index"] = metadata["sheet_row_index"]
            elif "row_index" in metadata:
                details["sheet_row_index"] = metadata["row_index"]

        elif file_type == ".json" or "json" in source_type_name:
            if "json_path" in metadata:
                details["json_path"] = metadata["json_path"]
            if "total_records" in metadata:
                details["total_records"] = metadata["total_records"]
            if "record_index" in metadata:
                details["record_index"] = metadata["record_index"]

        return details
