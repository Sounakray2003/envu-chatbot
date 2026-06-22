"""
Main Ingestion Service - Routes to appropriate data source handler

Extraction routing
------------------
ROW-WISE  (.csv, .tsv, .xlsx, .xls, .xlsm)
    ExtractionService.extract_row_wise() stores each row directly in the
    configured vector store.
    These files SKIP the chunking → embedding → store pipeline.
    Excel TEXT_HEAVY sheets are the only exception: their markdown is
    returned into the chunking pipeline exactly like a PDF/DOCX.

STANDARD  (.pdf, .docx, .txt, .json, …)
    ExtractionService.extract() returns a text blob that flows through the
    normal chunking → embedding → store pipeline.
"""

import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# File types that bypass chunking and store rows directly in the configured
# vector store
_ROW_WISE_TYPES = {'.csv', '.tsv', '.xlsx', '.xls', '.xlsm', '.dbrows'}
_EMBEDDING_BATCH_SIZE = 50
_OPENAI_EMBEDDING_MODEL_NAME = (
    str(os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")).strip()
    or "text-embedding-3-large"
)
_OPENAI_EMBEDDING_DIMENSIONS = int(
    str(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1024")).strip() or "1024"
)
_INGESTION_COLLECTION_NAME = "main_memory"
_MAX_EXTRACTED_CHARS_PER_FILE = int(
    str(os.getenv("MAX_EXTRACTED_CHARS_PER_FILE", "2000000")).strip() or "2000000"
)
_MAX_CHUNKS_PER_JOB = int(
    str(os.getenv("MAX_CHUNKS_PER_JOB", "1500")).strip() or "1500"
)
_SOURCE_LOG_DETAIL_KEYS = (
    "source_mapping_id",
    "source_id",
    "source_name",
    "provider_name",
    "region_name",
    "folder_path",
    "file_path",
    "filename",
    "start_url",
    "root_url",
    "website_url",
    "url",
    "max_pages",
    "max_depth",
    "host",
    "port",
    "database_type_name",
    "database_name",
    "schema_name",
    "table_name",
    "collection_name",
)


def _normalize_model_name(model_name: str) -> str:
    """Normalize model names for collection suffixes."""
    normalized = str(model_name or "").strip().lower()
    normalized = normalized.replace("_", "-").replace(" ", "-")
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")


class IngestionService:
    """
    Main ingestion orchestrator.
    Routes requests to appropriate data source handlers.
    """

    def __init__(self, request_data: Dict[str, Any]):
        """
        Initialize ingestion service with request data.

        Args:
            request_data: Complete request JSON
        """
        self.request_data = request_data
        self.knowledge_base_id = request_data.get("knowledge_base_id")
        if self.knowledge_base_id is None:
            self.knowledge_base_id = request_data.get("kb_id")
        self.job_id = request_data.get("job_id")
        
        # Normalize source_details to always be a list
        source_details_raw = request_data.get("source_details", {})
        
        if isinstance(source_details_raw, dict):
            # Single source (legacy format) → convert to list
            self.source_details_list = [source_details_raw]
            self.is_multi_source_format = False
        elif isinstance(source_details_raw, list):
            # Multiple sources (new format)
            self.source_details_list = source_details_raw
            self.is_multi_source_format = True
        else:
            logger.warning("source_details is neither dict nor list, using empty list")
            self.source_details_list = []
            self.is_multi_source_format = False
        
        # For backwards compatibility: keep accessing first source as self.source_details
        self.source_details = self.source_details_list[0] if self.source_details_list else {}
        self.source_types = [
            self._get_source_type_from_config(source_config)
            for source_config in self.source_details_list
        ]
        if len(self.source_types) == 1:
            self.source_type = self.source_types[0]
        elif len(self.source_types) > 1:
            self.source_type = "multi-source"
        else:
            self.source_type = ""

        self.chunking_details = request_data.get("chunking_details", {})
        if not isinstance(self.chunking_details, dict):
            self.chunking_details = {}
        self.embedding_details = request_data.get("embedding_details", {})
        if not isinstance(self.embedding_details, dict):
            self.embedding_details = {}
        self.vector_store_details = request_data.get("vector_store_details", {})
        if not isinstance(self.vector_store_details, dict):
            self.vector_store_details = {}
        self._enforce_semantic_chunking()
        self._enforce_openai_embedding()
        self._enforce_main_memory_collection()
        self.status_service = None

        logger.info(
            "Ingestion Service initialized | source_type=%s | resolved_source_types=%s | sources=%d",
            self.source_type,
            ", ".join(self.source_types) if self.source_types else "none",
            len(self.source_details_list)
        )
        total_sources = len(self.source_details_list)
        for source_index, source_config in enumerate(self.source_details_list):
            logger.info(
                "[SOURCE-CONFIG] %d/%d | source_type=%s | details=%s",
                source_index + 1,
                total_sources,
                self._get_source_type_from_config(source_config),
                self._sanitize_source_log_details(source_config),
            )

    def _enforce_semantic_chunking(self) -> None:
        """Normalize chunking settings so semantic chunking is always used."""
        requested_type = str(
            self.chunking_details.get("chunking_type", "SEMANTIC")
        ).strip().upper()
        if requested_type and requested_type != "SEMANTIC":
            logger.info(
                "Requested chunking type '%s' is no longer supported; using SEMANTIC.",
                requested_type,
            )

        self.chunking_details["chunking_type"] = "SEMANTIC"
        self.request_data["chunking_details"] = self.chunking_details

    def _enforce_openai_embedding(self) -> None:
        """Normalize embedding settings so OpenAI embeddings are always used."""
        requested_model_name = _normalize_model_name(
            self.embedding_details.get(
                "embedding_model_name",
                _OPENAI_EMBEDDING_MODEL_NAME,
            )
        )
        requested_dimensions = self.embedding_details.get("dimensions")
        resolved_model_name = _normalize_model_name(_OPENAI_EMBEDDING_MODEL_NAME)

        if requested_model_name and requested_model_name != resolved_model_name:
            logger.info(
                "Requested embedding model '%s' is no longer supported; using %s.",
                self.embedding_details.get("embedding_model_name"),
                _OPENAI_EMBEDDING_MODEL_NAME,
            )

        if requested_dimensions not in (None, "", _OPENAI_EMBEDDING_DIMENSIONS):
            logger.info(
                "Requested embedding dimensions '%s' are no longer supported; using %s.",
                requested_dimensions,
                _OPENAI_EMBEDDING_DIMENSIONS,
            )

        self.embedding_details["embedding_model_name"] = _OPENAI_EMBEDDING_MODEL_NAME
        self.embedding_details["dimensions"] = _OPENAI_EMBEDDING_DIMENSIONS
        self.request_data["embedding_details"] = self.embedding_details

    def _enforce_main_memory_collection(self) -> None:
        """Normalize vector store settings so ingestion always targets main_memory."""
        requested_collection_name = (
            self.vector_store_details.get("collection_name")
            or self.vector_store_details.get("QDRANT_COLLECTION_NAME")
            or self.vector_store_details.get("qdrant_collection_name")
        )
        if (
            requested_collection_name
            and str(requested_collection_name).strip() != _INGESTION_COLLECTION_NAME
        ):
            logger.info(
                "Requested collection '%s' is no longer supported; using %s.",
                requested_collection_name,
                _INGESTION_COLLECTION_NAME,
            )

        self.vector_store_details["collection_name"] = _INGESTION_COLLECTION_NAME
        self.vector_store_details["QDRANT_COLLECTION_NAME"] = _INGESTION_COLLECTION_NAME
        self.vector_store_details["qdrant_collection_name"] = _INGESTION_COLLECTION_NAME
        self.request_data["vector_store_details"] = self.vector_store_details

    # =========================================================================
    # HELPER METHODS - SOURCE ROUTING
    # =========================================================================

    @staticmethod
    def _sanitize_source_log_details(source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Return safe, concise source details for logging."""
        if not isinstance(source_config, dict):
            return {}

        details: Dict[str, Any] = {}
        for key in _SOURCE_LOG_DETAIL_KEYS:
            value = source_config.get(key)
            if value not in (None, "", []):
                details[key] = value
        return details

    @staticmethod
    def _coerce_bool(value: Any, default: bool = True) -> bool:
        """Convert common truthy/falsy payload values into a boolean."""
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

    def _get_source_type_from_config(self, source_config: Dict[str, Any]) -> str:
        """Resolve source type for one source_details item.

        Supported sources are file/folder, API, and website. Legacy cloud and
        database inference is preserved so unsupported payloads fail with a
        clear message instead of being mis-routed.
        """
        if not isinstance(source_config, dict):
            source_config = {}

        explicit_source_type = (
            source_config.get("source_type_name")
            or source_config.get("sourceTypeName")
            or source_config.get("source_type")
            or source_config.get("type")
            or ""
        )
        normalized = str(explicit_source_type).strip().lower()
        if normalized:
            return normalized

        # Fallback inference for legacy payloads missing explicit source type.
        folder_path = str(source_config.get("folder_path", "")).strip().lower()
        file_path = str(source_config.get("file_path", "")).strip()

        if any(
            key in source_config
            for key in (
                "database_type_name",
                "connection_uri",
                "connectionUri",
                "custom_query",
                "customQuery",
            )
        ):
            return "database"

        if any(
            key in source_config
            for key in (
                "start_url",
                "root_url",
                "website_url",
                "max_pages",
                "max_depth",
                "discover_sitemaps",
                "respect_robots_txt",
                "scope_to_start_path",
            )
        ):
            return "website"

        if "url" in source_config:
            return "api"

        if folder_path.startswith(("s3://", "gs://", "azure://")):
            return "cloud storage"
        if source_config.get("provider_name"):
            return "cloud storage"
        if folder_path or file_path:
            return "file upload"

        return "file upload"

    def _resolve_source_id_from_config(self, source_config: Dict[str, Any]) -> Optional[str]:
        """Resolve source mapping ID from per-source config with request fallback."""
        if not isinstance(source_config, dict):
            source_config = {}

        source_mapping_id = (
            source_config.get("source_mapping_id")
            or source_config.get("sourceMappingId")
            or source_config.get("source_id")
            or source_config.get("sourceId")
            or source_config.get("id")
            or self.request_data.get("source_mapping_id")
            or self.request_data.get("sourceMappingId")
            or self.request_data.get("source_id")
            or self.request_data.get("sourceId")
        )
        if source_mapping_id in (None, ""):
            return None
        return str(source_mapping_id)

    def _resolve_source_active_from_config(self, source_config: Dict[str, Any]) -> bool:
        """Resolve source active status from per-source config with request fallback."""
        if not isinstance(source_config, dict):
            source_config = {}

        is_active = source_config.get(
            "isActive",
            source_config.get(
                "is_active",
                source_config.get(
                    "active",
                    self.request_data.get(
                        "isActive",
                        self.request_data.get(
                            "is_active",
                            self.request_data.get("active", True),
                        ),
                    ),
                ),
            ),
        )
        return self._coerce_bool(is_active, default=True)

    def _resolve_discovery_failed_filename(
        self,
        source_config: Dict[str, Any],
        source_type: str,
        source_index: int,
    ) -> str:
        """Resolve failed-files names, delegating source-specific logic when available."""
        normalized_source_type = str(source_type or "").strip().lower()

        if "api" in normalized_source_type:
            from services.sources.api_source import APISource
            return APISource.build_discovery_failed_filename(source_config)

        if "website" in normalized_source_type or "web" in normalized_source_type:
            from services.sources.website_source import WebsiteSource
            return WebsiteSource.build_discovery_failed_filename(source_config)

        if "folder" in normalized_source_type or "file" in normalized_source_type:
            from services.sources.folder_source import FolderSource
            return FolderSource.build_discovery_failed_filename(source_config)

        if "database" in normalized_source_type:
            return "database_source"

        if "cloud" in normalized_source_type or "storage" in normalized_source_type:
            return "cloud_storage_source"

        return f"source_{source_index + 1}"

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    async def run_ingestion(self) -> Dict[str, Any]:
        """
        Main ingestion pipeline.

        STEP 1  Discover files from the data source(s).
        STEP 2  Extract:
                  • Row-wise files (CSV/Excel) → rows stored directly in the
                    configured vector store.
                  • Standard files (PDF/DOCX/…) → text blob returned for step 3.
        STEP 3  Chunk standard documents.
        STEP 4  Embed chunks.
        STEP 5  Store chunk vectors in the configured vector store.

        Returns:
            Results dictionary with status and metrics
        """
        results = {
            'status':                 'success',
            'source_type':            self.source_type,
            'knowledge_base_id':      self.knowledge_base_id,
            'collection_name':        self._resolve_collection_name(),
            'total_sources':          len(self.source_details_list),
            'sources_processed':      0,
            'sources_failed':         0,
            'total_files_discovered': 0,
            'total_files_processed':  0,
            'total_files_failed':     0,
            'total_content_size_bytes': 0,
            'processed_files':        [],
            'failed_files':           [],
            'total_chunks_created':   0,
            'total_vectors_stored':   0,
            'total_tokens':           0,
            'total_characters':       0,
            'errors':                 [],
            'warnings':               [],
        }

        try:
            # ------------------------------------------------------------------
            # STEP 1: Discover
            # ------------------------------------------------------------------
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP 1: Discovering data sources")
            logger.info("=" * 70)

            if self.status_service:
                self.status_service.send_preparing_data({
                    "step": 1,
                    "total_steps": 5,
                    "source_type": self.source_type,
                    "total_sources": len(self.source_details_list),
                    "source_mapping_ids": [
                        s.get("source_mapping_id")
                        for s in self.source_details_list
                        if s.get("source_mapping_id") is not None
                    ],
                })

            files, discovery_failures = await self._discover_sources()
            discovered_file_count = len(files)
            results["total_files_discovered"] = discovered_file_count
            results["sources_processed"] = max(0, len(self.source_details_list) - len(discovery_failures))
            results["sources_failed"] = len(discovery_failures)

            # Collect all source_mapping_ids from discovered files for downstream status events
            _source_mapping_ids = list(dict.fromkeys(
                f["source_mapping_id"] for f in files
                if f.get("source_mapping_id") is not None
            ))

            discovery_failed_files = [
                {
                    "filename": failure.get("failed_file"),
                    "source_mapping_id": failure.get("source_mapping_id"),
                }
                for failure in discovery_failures
                if failure.get("failed_file")
            ]
            if discovery_failed_files:
                results["failed_files"].extend(discovery_failed_files)
                results["total_files_failed"] += len(discovery_failed_files)
                results["total_files_discovered"] += len(discovery_failed_files)

            if discovery_failures:
                results["warnings"].extend(
                    failure.get("message", failure.get("error", "Discovery failed"))
                    for failure in discovery_failures
                )

            logger.info(
                "Discovered %s file(s)%s",
                discovered_file_count,
                f" with {len(discovery_failed_files)} discovery failure(s)"
                if discovery_failed_files
                else "",
            )

            if not files:
                # Use actual discovery errors if available, otherwise generic message
                if discovery_failures:
                    error_message = "; ".join(
                        failure.get("message", failure.get("error", "Discovery failed"))
                        for failure in discovery_failures[:3]
                    )
                    results['errors'].extend(
                        failure.get("message", failure.get("error", "Discovery failed"))
                        for failure in discovery_failures
                    )
                else:
                    error_message = (
                        "No files discovered. Verify source_details.file_path for File Upload "
                        "(or source_details.folder_path for folder ingestion, or "
                        "source_details.start_url for website ingestion)."
                    )
                    results['errors'].append(error_message)
                logger.error(error_message)
                results['status'] = 'FAILED'
                if self.status_service:
                    self.status_service.send_error(error_message)
                return results

            # ------------------------------------------------------------------
            # STEP 2: Extract
            # ------------------------------------------------------------------
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP 2: Extracting content")
            logger.info("=" * 70)

            # _extract_content mutates `results` for row-wise files and returns
            # only the docs that still need chunking → embedding → storing.
            docs_for_chunking = await self._extract_content(files, results)
            logger.info(
                f"Row-wise complete. "
                f"{len(docs_for_chunking)} doc(s) queued for chunking pipeline."
            )

            # Send extraction completion status with file counts
            if self.status_service:
                self.status_service.send_processing_documents({
                    "step": 2,
                    "total_steps": 5,
                    "source_mapping_ids": _source_mapping_ids,
                    "files_discovered": results['total_files_discovered'],
                    "files_processed": results['total_files_processed'],
                    "files_failed": results['total_files_failed'],
                    "processed_files": results['processed_files'],
                    "failed_files": results['failed_files'],
                    "extraction_complete": True,
                })

            # If nothing needs chunking AND no row-wise vectors were stored,
            # there was nothing to do.
            if not docs_for_chunking and results['total_vectors_stored'] == 0:
                logger.warning("No content extracted from any file")
                # If there are errors, send them to the status sink
                if results['errors'] and self.status_service:
                    error_summary = "; ".join(results['errors'][:3])  # First 3 errors
                    self.status_service.send_error(error_summary)
                return self._finalize_results_status(results)

            # Skip chunking steps if there are no standard docs to process.
            if not docs_for_chunking:
                logger.info(
                    "All files were row-wise — skipping chunking / embedding steps."
                )

                # Still emit all pipeline steps so downstream consumers receive
                # a complete status sequence (steps 3 → 4 → 5).
                logger.info("")
                logger.info("=" * 70)
                logger.info("STEP 3: Chunking documents (skipped — row-wise source)")
                logger.info("=" * 70)
                if self.status_service:
                    self.status_service.send_processing_documents({
                        "step": 3, "total_steps": 5,
                        "source_mapping_ids": _source_mapping_ids,
                        "skipped": True,
                        "reason": "row-wise source; no chunking needed",
                    })

                logger.info("")
                logger.info("=" * 70)
                logger.info("STEP 4: Generating embeddings (skipped — row-wise source)")
                logger.info("=" * 70)
                if self.status_service:
                    self.status_service.send_creating_embeddings({
                        "step": 4, "total_steps": 5,
                        "source_mapping_ids": _source_mapping_ids,
                        "skipped": True,
                        "reason": "row-wise source; embeddings generated inline",
                    })

                logger.info("")
                logger.info("=" * 70)
                logger.info("STEP 5: Storing in vector database (complete)")
                logger.info("=" * 70)
                if self.status_service:
                    self.status_service.send_saving_to_knowledge_base({
                        "step": 5, "total_steps": 5,
                        "source_mapping_ids": _source_mapping_ids,
                        "vectors_stored": results['total_vectors_stored'],
                        "files_processed": results['total_files_processed'],
                        "files_failed": results['total_files_failed'],
                        "total_content_size_bytes": results['total_content_size_bytes'],
                        "processed_files": results['processed_files'],
                        "failed_files": results['failed_files'],
                        "completed": True,
                    })

                logger.info("")
                logger.info("=" * 70)
                logger.info("INGESTION COMPLETED SUCCESSFULLY")
                logger.info("=" * 70)

                return self._finalize_results_status(results)

            # ------------------------------------------------------------------
            # STEP 3: Chunk
            # ------------------------------------------------------------------
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP 3: Chunking documents")
            logger.info("=" * 70)

            if self.status_service:
                self.status_service.send_processing_documents({
                    "step": 3, "total_steps": 5,
                    "source_mapping_ids": _source_mapping_ids,
                    "files_processed": results['total_files_processed'],
                    "processed_files": results['processed_files'],
                    "failed_files": results['failed_files'],
                })

            chunks = await self._chunk_documents(docs_for_chunking)
            if _MAX_CHUNKS_PER_JOB > 0 and len(chunks) > _MAX_CHUNKS_PER_JOB:
                raise RuntimeError(
                    f"Chunk limit exceeded: created {len(chunks)} chunks, "
                    f"max allowed is {_MAX_CHUNKS_PER_JOB}. Reduce file size "
                    "or increase MAX_CHUNKS_PER_JOB for a larger worker."
                )
            results['total_chunks_created'] += len(chunks)
            
            # Track tokens and characters from standard document chunks
            chunked_tokens = sum(chunk.get('token_count', 0) for chunk in chunks)
            chunked_chars = sum(chunk.get('char_count', 0) for chunk in chunks)
            results['total_tokens'] += chunked_tokens
            results['total_characters'] += chunked_chars
            
            logger.info(
                f"Created {len(chunks)} chunk(s) | "
                f"{chunked_tokens} tokens | {chunked_chars} characters"
            )

            if not chunks:
                logger.warning("No chunks created")
                return results

            # ------------------------------------------------------------------
            # STEP 4: Embed
            # ------------------------------------------------------------------
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP 4: Generating embeddings")
            logger.info("=" * 70)

            if self.status_service:
                self.status_service.send_creating_embeddings({
                    "step": 4, "total_steps": 5,
                    "source_mapping_ids": _source_mapping_ids,
                    "files_processed": results['total_files_processed'],
                    "chunks_created": len(chunks),
                })

            embedded_chunks = await self._generate_embeddings(chunks)
            logger.info("Generated embeddings for %s chunk(s)", len(embedded_chunks))

            # ------------------------------------------------------------------
            # STEP 5: Store chunk vectors
            # ------------------------------------------------------------------
            logger.info("")
            logger.info("=" * 70)
            logger.info("STEP 5: Storing in vector database")
            logger.info("=" * 70)

            if self.status_service:
                self.status_service.send_saving_to_knowledge_base({
                    "step": 5, "total_steps": 5,
                    "source_mapping_ids": _source_mapping_ids,
                    "chunks_to_store": len(embedded_chunks),
                })

            chunk_vectors_stored = await self._store_vectors(embedded_chunks)
            results['total_vectors_stored'] += chunk_vectors_stored
            logger.info(
                "Stored %s chunk vector(s) in the configured vector store "
                "(total: %s)",
                chunk_vectors_stored,
                results['total_vectors_stored'],
            )

            if self.status_service:
                self.status_service.send_saving_to_knowledge_base({
                    "step": 5, "total_steps": 5,
                    "source_mapping_ids": _source_mapping_ids,
                    "vectors_stored": results['total_vectors_stored'],
                    "files_processed": results['total_files_processed'],
                    "files_failed": results['total_files_failed'],
                    "total_content_size_bytes": results['total_content_size_bytes'],
                    "processed_files": results['processed_files'],
                    "failed_files": results['failed_files'],
                    "completed": True,
                })

            logger.info("")
            logger.info("=" * 70)
            logger.info("INGESTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)

        except Exception as exc:
            logger.error(f"Ingestion pipeline failed: {exc}", exc_info=True)
            results['status'] = 'FAILED'
            results['errors'].append(str(exc))
            if self.status_service:
                try:
                    self.status_service.send_error(str(exc))
                except Exception as status_exc:
                    logger.error(
                        "Failed to send FAILED status event: %s",
                        status_exc,
                        exc_info=True,
                    )

        return self._finalize_results_status(results)

    # =========================================================================
    # STEP IMPLEMENTATIONS
    # =========================================================================

    async def _discover_sources(self) -> tuple:
        """Discover files from the configured data source(s).
        
        Returns:
            Tuple of (all_files, discovery_failures) where discovery_failures
            contains source-aware details from failed discovery attempts.
        """
        all_files = []
        discovery_failures = []
        
        # Loop through each source in source_details_list
        for source_index, source_config in enumerate(self.source_details_list):
            source_type = "unknown"
            source_mapping_id = None
            try:
                logger.info(f"[MULTI-SOURCE] Discovering source {source_index + 1}/{len(self.source_details_list)}")

                if not isinstance(source_config, dict):
                    raise ValueError("Source configuration must be an object")
                
                # Detect source type for this specific config
                source_type = self._get_source_type_from_config(source_config)
                source_mapping_id = self._resolve_source_id_from_config(source_config)
                source_is_active = self._resolve_source_active_from_config(source_config)
                logger.info(
                    "[MULTI-SOURCE] source_index=%s, type=%s, details=%s",
                    source_index,
                    source_type,
                    self._sanitize_source_log_details(source_config),
                )
                
                # Create a request_data variant with just this source
                request_variant = dict(self.request_data)
                request_variant['source_details'] = source_config
                request_variant['source_type_name'] = (
                    str(source_config.get("source_type_name", "")).strip()
                    or source_type
                )
                request_variant['source_type_id'] = (
                    source_config.get("source_type_id")
                    or request_variant.get("source_type_id")
                )
                
                # Route to appropriate handler
                files = []
                
                if "folder" in source_type or "file" in source_type:
                    from services.sources.folder_source import FolderSource
                    source = FolderSource(request_variant)
                    files = await source.discover()
                
                elif "website" in source_type or "web" in source_type:
                    from services.sources.website_source import WebsiteSource
                    files = await WebsiteSource(request_variant).discover()

                elif "api" in source_type:
                    from services.sources.api_source import APISource
                    files = await APISource(request_variant).discover()

                elif "cloud" in source_type or "storage" in source_type:
                    raise ValueError(
                        "Cloud storage sources are no longer supported. "
                        "Use File Upload/Folder or API sources instead."
                    )

                elif "database" in source_type:
                    raise ValueError(
                        "Database sources are no longer supported. "
                        "Use File Upload/Folder or API sources instead."
                    )
                
                else:
                    raise ValueError(f"Unsupported source type: {source_type}")
                
                # Tag files with source_index for traceability
                for file_dict in files:
                    source_details_payload = (
                        dict(source_config) if isinstance(source_config, dict) else {}
                    )
                    if file_dict.get("file_id") not in (None, ""):
                        source_details_payload["file_id"] = str(file_dict["file_id"])

                    file_dict['source_index'] = source_index
                    file_dict['source_type_name'] = request_variant['source_type_name']
                    file_dict['source_type_from_config'] = source_type
                    file_dict['source_mapping_id'] = source_mapping_id
                    file_dict['isActive'] = source_is_active
                    file_dict['source_details'] = source_details_payload
                
                logger.info(f"[MULTI-SOURCE] Source {source_index} discovered {len(files)} file(s)")
                all_files.extend(files)

                # Notify the status sink: this specific source was discovered successfully
                if self.status_service:
                    self.status_service.send_preparing_data({
                        "step": 1,
                        "total_steps": 5,
                        "source_mapping_id": source_mapping_id,
                        "source_type_name": str(source_config.get("source_type_name", source_type)),
                        "provider_name": source_config.get("provider_name"),
                        "current_source_index": source_index + 1,
                        "total_sources": len(self.source_details_list),
                        "files_discovered": len(files),
                    })

            except Exception as e:
                error_msg = str(e)
                failure_label = self._resolve_discovery_failed_filename(
                    source_config if isinstance(source_config, dict) else {},
                    source_type,
                    source_index,
                )
                failure_message = f"Discovery failed for {failure_label}: {error_msg}"
                logger.error(f"[MULTI-SOURCE] Error discovering source {source_index}: {failure_message}")
                discovery_failures.append(
                    {
                        "source_index": source_index,
                        "source_type": source_type,
                        "source_mapping_id": source_mapping_id,
                        "failed_file": failure_label,
                        "error": error_msg,
                        "message": failure_message,
                    }
                )

                # Notify the status sink: this specific source failed discovery
                if self.status_service:
                    self.status_service.send_error(failure_message)

                continue
        
        if not all_files:
            logger.warning("[MULTI-SOURCE] No files discovered from any source")
        
        return all_files, discovery_failures

    async def _extract_content(
        self,
        files: List[Dict[str, Any]],
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract content from all discovered files.

        ROW-WISE files (.csv, .tsv, .xlsx, .xls, .xlsm)
        -------------------------------------------------
        Calls ExtractionService.extract_row_wise() which:
          1. Embeds and upserts every data row directly to the vector store.
          2. For Excel: any TEXT_HEAVY sheet yields markdown in
             summary['fallback_content']; that markdown is added to the
             returned list so it flows through the chunking pipeline.

        STANDARD files (.pdf, .docx, .txt, .json, …)
        ----------------------------------------------
        Calls ExtractionService.extract() and appends the result to the
        returned list for the chunking pipeline.

        Args:
            files:   file_info dicts from _discover_sources().
            results: top-level results dict — mutated here to track
                     row-wise vectors stored and failures.

        Returns:
            Docs that still need chunking → embedding → storing.
        """
        from services.extraction.extraction_service import ExtractionService
        from services.vectore_store import get_vector_store

        extractor = ExtractionService()

        # Single vector store instance shared across all row-wise files.
        # The factory picks the right Qdrant mode (internal or user-provided)
        # based on vector_store_details.vector_store_id.
        vector_store = get_vector_store(
            collection_name=self._resolve_collection_name(),
            vector_size=self.embedding_details.get('dimensions', 0),
            request_data=self.request_data,
            vector_type="dense",
            enable_sparse_vectors=True,
            ensure_collection=False,
        )
        logger.info(
            "[ROW-WISE] Using vector backend=%s | collection=%s",
            getattr(vector_store, "BACKEND_NAME", "unknown"),
            self._resolve_collection_name(),
        )

        knowledge_base_name = self.request_data.get('name', '')
        user_id = str(self.request_data.get('member_id', ''))

        docs_for_chunking: List[Dict[str, Any]] = []

        for file_info in files:
            filename  = file_info.get('filename', '')
            file_type = file_info.get('file_type', '').lower()
            file_id   = str(
                file_info.get('file_id') or file_info.get('id') or filename
            )

            # ------------------------------------------------------------------
            # ZIP fan-out path  (expand archive and recurse through pipeline)
            # ------------------------------------------------------------------
            if file_type == '.zip':
                try:
                    from services.extraction.extractors.zip_extractor import ZIPExtractor

                    logger.info(f"[ZIP] Expanding archive '{filename}' into pipeline inputs")

                    with tempfile.TemporaryDirectory(prefix="zip_ingest_") as temp_dir:
                        zip_extractor = ZIPExtractor()
                        extracted_files = zip_extractor.extract_file_infos(
                            file_path=file_info.get('file_path'),
                            destination_root=temp_dir,
                            parent_file_info=file_info,
                        )

                        if not extracted_files:
                            warning_msg = f"{filename}: ZIP archive has no supported files to ingest"
                            results['warnings'].append(warning_msg)
                            logger.warning(f"[ZIP] {warning_msg}")
                            continue

                        # Replace the archive with its supported contents in progress metrics.
                        results['total_files_discovered'] += max(0, len(extracted_files) - 1)
                        logger.info(
                            "[ZIP] '%s' expanded to %d supported file(s)",
                            filename,
                            len(extracted_files),
                        )

                        docs_for_chunking.extend(
                            await self._extract_content(extracted_files, results)
                        )
                    continue

                except Exception as exc:
                    logger.error(
                        f"ZIP expansion error for '{filename}': {exc}",
                        exc_info=True,
                    )
                    results['total_files_failed'] += 1
                    results['failed_files'].append({
                        "filename": filename,
                        "source_mapping_id": file_info.get("source_mapping_id"),
                    })
                    results['errors'].append(f"{filename}: {exc}")
                    continue

            # ------------------------------------------------------------------
            # ROW-WISE path  (CSV / TSV / Excel)
            # ------------------------------------------------------------------
            if file_type in _ROW_WISE_TYPES:
                try:
                    logger.info(f"[ROW-WISE] {filename} ({file_type})")
                    success, summary = await extractor.extract_row_wise(
                        file_info=file_info,
                        vector_store=vector_store,
                        knowledge_base_id=str(self.knowledge_base_id),
                        knowledge_base_name=knowledge_base_name,
                        user_id=user_id,
                        file_id=file_id,
                        batch_size=_EMBEDDING_BATCH_SIZE,
                        embedding_details=self.embedding_details,
                    )

                    if success:
                        # CSV → total_stored  |  Excel → total_rows_stored
                        row_vectors = (
                            summary.get('total_stored', 0)
                            or summary.get('total_rows_stored', 0)
                        )
                        row_chunks = (
                            summary.get('total_chunks', 0)
                            or summary.get('total_rows_stored', 0)
                        )
                        row_tokens = summary.get('total_tokens', 0)
                        row_chars = summary.get('total_characters', 0)
                        
                        results['total_vectors_stored'] += row_vectors
                        results['total_chunks_created'] += row_chunks
                        results['total_tokens'] += row_tokens
                        results['total_characters'] += row_chars
                        results['total_files_processed'] += 1
                        results['processed_files'].append({
                            "filename": filename,
                            "source_mapping_id": file_info.get("source_mapping_id"),
                        })
                        # Only count size for files successfully extracted into KB
                        results['total_content_size_bytes'] += file_info.get('content_size_bytes', 0)

                        logger.info(
                            f"  {row_vectors} chunk(s) | {row_tokens} tokens | "
                            f"{row_chars} characters stored for '{filename}'"
                        )
                    else:
                        results['total_files_failed'] += 1
                        results['failed_files'].append({
                            "filename": filename,
                            "source_mapping_id": file_info.get("source_mapping_id"),
                        })
                        err = summary.get('error', 'row-wise extraction failed')
                        results['errors'].append(f"{filename}: {err}")
                        logger.error(f"  Row-wise failed for '{filename}': {err}")

                    # Excel text-heavy sheets → also push through chunking
                    fallback = summary.get('fallback_content')
                    if fallback and fallback.strip():
                        docs_for_chunking.append({
                            'filename':  filename,
                            'content':   fallback,
                            'file_type': file_type,
                            'metadata':  file_info,
                        })
                        logger.info(
                            f"  Text-heavy sheet(s) of '{filename}' "
                            "queued for chunking pipeline"
                        )

                except Exception as exc:
                    logger.error(
                        f"Row-wise extraction error for '{filename}': {exc}",
                        exc_info=True,
                    )
                    results['total_files_failed'] += 1
                    results['failed_files'].append({
                        "filename": filename,
                        "source_mapping_id": file_info.get("source_mapping_id"),
                    })
                    results['errors'].append(f"{filename}: {exc}")

            # ------------------------------------------------------------------
            # STANDARD path  (PDF, DOCX, TXT, JSON, …)
            # ------------------------------------------------------------------
            else:
                try:
                    result = await extractor.extract(file_info)
                    if result:
                        content = str(result.get("content") or "")
                        if (
                            _MAX_EXTRACTED_CHARS_PER_FILE > 0
                            and len(content) > _MAX_EXTRACTED_CHARS_PER_FILE
                        ):
                            raise ValueError(
                                f"Extracted content is too large "
                                f"({len(content)} characters). Max allowed is "
                                f"{_MAX_EXTRACTED_CHARS_PER_FILE} characters."
                            )
                        docs_for_chunking.append(result)
                        results['total_files_processed'] += 1
                        results['processed_files'].append({
                            "filename": filename,
                            "source_mapping_id": file_info.get("source_mapping_id"),
                        })
                        # Only count size for files successfully extracted into KB
                        results['total_content_size_bytes'] += file_info.get('content_size_bytes', 0)
                        logger.info(f"Successfully extracted '{filename}'")
                    else:
                        results['total_files_failed'] += 1
                        results['failed_files'].append({
                            "filename": filename,
                            "source_mapping_id": file_info.get("source_mapping_id"),
                        })
                        logger.warning(f"No content extracted from '{filename}'")
                        results['errors'].append(f"{filename}: No content extracted")
                except Exception as exc:
                    results['total_files_failed'] += 1
                    results['failed_files'].append({
                        "filename": filename,
                        "source_mapping_id": file_info.get("source_mapping_id"),
                    })
                    logger.error(
                        f"Standard extraction error for '{filename}': {exc}",
                        exc_info=True,
                    )
                    results['errors'].append(f"{filename}: {exc}")

        return docs_for_chunking

    @staticmethod
    def _finalize_results_status(results: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize the overall status based on the recorded outcomes.
        
        - Success: if any files were processed and vectors stored
        - Failed: if no files were processed despite being discovered
        - Partial success includes files with failures but vectors stored
        """
        if results.get('status') == 'FAILED':
            return results

        # Success if vectors were stored (documents made it to DB)
        if results.get('total_vectors_stored', 0) > 0 or results.get('total_files_processed', 0) > 0:
            results['status'] = 'success'
        else:
            # Only FAILED if nothing was processed/stored
            results['status'] = 'FAILED'

        return results

    async def _chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk documents into smaller pieces.

        Each document's chunker resets chunk_index to 0.  After collecting all
        chunks we reassign a single monotonically increasing global_chunk_index
        so every chunk across all documents and all sources has a unique index.
        This prevents duplicate chunk_index values when multiple source files are
        ingested into the same table / collection.
        """
        from services.chunking_service import ChunkingService

        chunker = ChunkingService(
            chunking_details=self.chunking_details,
            embedding_details=self.embedding_details
        )

        all_chunks: List[Dict[str, Any]] = []
        for doc in documents:
            metadata = doc.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                doc["metadata"] = metadata

            metadata.setdefault("knowledge_base_id", self.knowledge_base_id)
            metadata.setdefault("kb_id", self.request_data.get("kb_id"))
            metadata.setdefault("job_id", self.job_id)
            metadata.setdefault("org_id", self.request_data.get("org_id"))
            metadata.setdefault("member_id", self.request_data.get("member_id"))

            all_chunks.extend(chunker.chunk(doc))

        # Re-assign chunk_index globally so there are no duplicates across docs
        for global_idx, chunk in enumerate(all_chunks):
            chunk["chunk_index"] = global_idx

        return all_chunks

    async def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks."""
        from services.embedding_service import create_embedding_service

        embedding_service = create_embedding_service(self.embedding_details)
        model_info = embedding_service.get_model_info()

        logger.info(
            "Resolved embedding model | requested=%s | endpoint=%s | requested_dim=%s",
            model_info.get("requested_model_name"),
            model_info.get("base_url"),
            model_info.get("requested_dimensions"),
        )

        texts = [chunk["text"] for chunk in chunks]
        embeddings: List[Dict[str, Any]] = []

        for batch_start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
            batch_end = min(batch_start + _EMBEDDING_BATCH_SIZE, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_embeddings = embedding_service.embed_documents_with_sparse(batch_texts)
            embeddings.extend(batch_embeddings)
            logger.info(
                "Embedded chunk batch %s-%s of %s",
                batch_start + 1,
                batch_end,
                len(texts),
            )

        resolved_dimension = (
            len(embeddings[0]["dense"])
            if embeddings
            else model_info.get("embedding_dim")
            or self.embedding_details.get("dimensions")
            or 0
        )

        self.embedding_details["dimensions"] = resolved_dimension
        self.embedding_details["resolved_dimensions"] = resolved_dimension
        self.embedding_details["resolved_model_name"] = model_info.get("model_name")
        self.embedding_details["embedded_model_id"] = model_info.get("embedded_model_id")

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding_dense"] = embedding["dense"]
            chunk["embedding_sparse"] = embedding["sparse"]
            chunk["embedding_model"] = self.embedding_details["resolved_model_name"]
            chunk["embedded_model_id"] = self.embedding_details["embedded_model_id"]
            chunk["requested_embedding_model_name"] = model_info.get("requested_model_name")
            chunk["embedding_dimension"] = len(embedding["dense"])

        return chunks

    def _resolve_collection_name(self) -> str:
        """Return the fixed collection used for every ingestion."""
        return _INGESTION_COLLECTION_NAME

    async def _store_vectors(self, chunks: List[Dict[str, Any]]) -> int:
        """Store vectors in the configured vector store backend."""
        from services.vectore_store import get_vector_store

        vector_size = (
            self.embedding_details.get("resolved_dimensions")
            or self.embedding_details.get("dimensions")
            or (
                len(chunks[0]["embedding_dense"])
                if chunks and chunks[0].get("embedding_dense")
                else 0
            )
        )

        vector_store = get_vector_store(
            collection_name=self._resolve_collection_name(),
            vector_size=vector_size,
            request_data=self.request_data,
            vector_type="dense",
            enable_sparse_vectors=True,
            ensure_collection=True,
        )
        logger.info(
            "Storing vectors | backend=%s | collection=%s | chunk_count=%d | vector_size=%d",
            getattr(vector_store, "BACKEND_NAME", "unknown"),
            self._resolve_collection_name(),
            len(chunks),
            vector_size,
        )
        return await vector_store.store_chunks(chunks)
