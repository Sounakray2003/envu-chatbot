"""
Qdrant Vector Store Backend
---------------------------
Handles both:
  - Internal Qdrant  (vector_store_id=1): URL from request/env, no API key
  - User-provided Qdrant (vector_store_id=2): URL + optional API key from
    vector_store_details (keys: QDRANT_URL, QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME)

All Qdrant communication is via direct HTTP REST calls (no qdrant-client SDK)
so there are no extra dependencies and the connection config is fully
transparent.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from services.payload_builder import PayloadBuilder
from services.vectore_store import BaseVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level defaults (mirror what vector_store_service.py used)
# ---------------------------------------------------------------------------
DEFAULT_QDRANT_URL     = ""
DEFAULT_QDRANT_TIMEOUT = 300

DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"


# ---------------------------------------------------------------------------
# Connection URL parser (unchanged from vector_store_service.py)
# ---------------------------------------------------------------------------

def _parse_qdrant_connection(raw_url: str) -> Dict[str, Any]:
    """
    Parse a Qdrant URL into the pieces the HTTP client needs.

    Handles reverse-proxied deployments that serve on port 443 with a
    path prefix (e.g. https://host/qdrant-dev) without defaulting to
    qdrant-client's hardcoded port 6333.
    """
    normalized_url = str(raw_url or "").strip()
    if not normalized_url:
        normalized_url = DEFAULT_QDRANT_URL

    if "://" not in normalized_url:
        normalized_url = f"http://{normalized_url}"

    parsed = urlparse(normalized_url)

    if not parsed.hostname:
        raise ValueError(f"Invalid QDRANT_URL: {raw_url!r}")

    scheme = (parsed.scheme or "http").lower()
    if scheme not in {"http", "https"}:
        raise ValueError(
            f"Unsupported QDRANT_URL scheme '{scheme}'. Use http or https."
        )

    port   = parsed.port or (443 if scheme == "https" else 80)
    prefix = parsed.path.rstrip("/")
    prefix = None if prefix in {"", "/"} else prefix.lstrip("/")

    return {
        "base_url":    f"{scheme}://{parsed.hostname}",
        "display_url": normalized_url.rstrip("/"),
        "host":        parsed.hostname,
        "port":        port,
        "prefix":      prefix,
        "scheme":      scheme,
    }


# ---------------------------------------------------------------------------
# QdrantVectorStore
# ---------------------------------------------------------------------------

class QdrantVectorStore(BaseVectorStore):
    """
    Store and manage vectors in Qdrant via direct HTTP REST API calls.

    Supports:
      - Dense vectors only
      - Dense + Sparse vectors (named 'dense' / 'sparse')
      - Sparse vectors only

    Works for both the internal cluster (no API key) and user-supplied
    Qdrant instances (optional API key injected into every request header).
    """

    BACKEND_NAME = "qdrant"
    SUPPORTS_SPARSE_NATIVE = True
    SUPPORTS_COLLECTION_CLONE = True
    SUPPORTS_LIST_COLLECTIONS = True

    def __init__(
        self,
        collection_name: str,
        vector_size: int = 0,
        request_data: Optional[Dict[str, Any]] = None,
        enable_sparse_vectors: bool = True,
        vector_type: str = "dense",
        ensure_collection: bool = True,
        qdrant_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        collection_name:
            Target Qdrant collection name.
        vector_size:
            Dimensionality of dense vectors (0 = infer from first batch).
        request_data:
            Full request JSON — forwarded to PayloadBuilder.
        enable_sparse_vectors:
            When vector_type='dense', also store a named sparse vector.
        vector_type:
            'dense' or 'sparse'.
        ensure_collection:
            Create / validate the collection during __init__.
        qdrant_url:
            Explicit Qdrant base URL. Falls back to DEFAULT_QDRANT_URL.
        api_key:
            Optional API key sent as 'api-key' header (user-provided clusters).
        """
        self.collection_name      = collection_name
        self.vector_size          = vector_size
        self.request_data         = request_data or {}
        self.enable_sparse_vectors = enable_sparse_vectors
        self.vector_type          = vector_type
        self.api_key              = api_key
        self.payload_builder      = PayloadBuilder(self.request_data) if request_data else None
        self._dense_vector_payload_name: Optional[str] = (
            DENSE_VECTOR_NAME
            if self.vector_type == "dense" and self.enable_sparse_vectors
            else None
        )
        self._sparse_vector_payload_name: Optional[str] = (
            SPARSE_VECTOR_NAME
            if self.enable_sparse_vectors or self.vector_type == "sparse"
            else None
        )

        if ensure_collection and vector_size < 0:
            raise ValueError("vector_size must be >= 0")

        connection    = _parse_qdrant_connection(qdrant_url or DEFAULT_QDRANT_URL)
        self.base_url = connection["display_url"].rstrip("/")
        self.timeout  = DEFAULT_QDRANT_TIMEOUT
        self._collection_ready = False

        logger.info("QdrantVectorStore initialized")
        logger.info("  Collection      : %s", self.collection_name)
        logger.info("  Vector type     : %s", self.vector_type)
        logger.info("  Dense size      : %s", self.vector_size)
        logger.info("  Sparse enabled  : %s", self.enable_sparse_vectors)
        logger.info("  URL             : %s", self.base_url)
        logger.info("  API key present : %s", bool(self.api_key))
        logger.info("  Ensure on init  : %s", ensure_collection)

        if ensure_collection:
            self._ensure_collection()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        """Build HTTP headers, including the API key when present."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        return headers

    def _get(self, path: str) -> requests.Response:
        return requests.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            timeout=self.timeout,
        )

    def _put(self, path: str, json: Any) -> requests.Response:
        return requests.put(
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=json,
            timeout=self.timeout,
        )

    def _delete(self, path: str) -> requests.Response:
        return requests.delete(
            f"{self.base_url}{path}",
            headers=self._headers(),
            timeout=self.timeout,
        )

    def _post(self, path: str, json: Any) -> requests.Response:
        return requests.post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=json,
            timeout=self.timeout,
        )

    # -----------------------------------------------------------------------
    # Collection schema builders
    # -----------------------------------------------------------------------

    def _build_collection_payload(self) -> Dict[str, Any]:
        """Return the Qdrant collection-create body for the current config."""
        if self.vector_type == "dense" and self.enable_sparse_vectors:
            return {
                "vectors": {
                    DENSE_VECTOR_NAME: {
                        "size":     self.vector_size,
                        "distance": "Cosine",
                    }
                },
                "sparse_vectors": {
                    SPARSE_VECTOR_NAME: {
                        "index": {"datatype": "float32"}
                    }
                },
            }

        if self.vector_type == "dense":
            return {
                "vectors": {
                    "size":     self.vector_size,
                    "distance": "Cosine",
                }
            }

        if self.vector_type == "sparse":
            return {
                "vectors": {},
                "sparse_vectors": {
                    SPARSE_VECTOR_NAME: {
                        "index": {"datatype": "float32"}
                    }
                },
            }

        raise ValueError(f"Unknown vector_type: {self.vector_type!r}")

    # -----------------------------------------------------------------------
    # Point vector builder
    # -----------------------------------------------------------------------

    def _build_point_vector(self, chunk: Dict[str, Any]) -> Any:
        """
        Assemble the Qdrant vector payload from a chunk's embedding fields.

        Raises ValueError if required embeddings are missing.
        """
        dense  = chunk.get("embedding_dense")
        sparse = chunk.get("embedding_sparse")

        if self.vector_type == "dense":
            if not dense:
                raise ValueError(
                    f"Chunk '{chunk.get('chunk_id', '?')}' must have "
                    "'embedding_dense' for Qdrant storage."
                )

            if self._dense_vector_payload_name is None and self._sparse_vector_payload_name is None:
                return dense

            vector_payload: Dict[str, Any] = {}
            if self._dense_vector_payload_name is not None:
                vector_payload[self._dense_vector_payload_name] = dense

            if self._sparse_vector_payload_name is not None:
                if not sparse or not sparse.get("indices") or not sparse.get("values"):
                    raise ValueError(
                        f"Chunk '{chunk.get('chunk_id', '?')}' must have "
                        "'embedding_sparse' for Qdrant sparse storage."
                    )
                vector_payload[self._sparse_vector_payload_name] = sparse

            return vector_payload or dense

        if self.vector_type == "sparse":
            if not sparse or not sparse.get("indices") or not sparse.get("values"):
                return None
            sparse_name = self._sparse_vector_payload_name or SPARSE_VECTOR_NAME
            return {sparse_name: sparse}

        # dense-only
        return dense or None

    # -----------------------------------------------------------------------
    # Collection management (public interface)
    # -----------------------------------------------------------------------

    def _create_collection(self) -> None:
        """Create the collection with the current vector schema."""
        resp = self._put(
            f"/collections/{self.collection_name}",
            self._build_collection_payload(),
        )
        resp.raise_for_status()
        logger.info("Collection created: %s", self.collection_name)
        
        # Create payload indexes for filterable fields
        # Qdrant Cloud requires this before filtering on these fields
        self._create_payload_indexes()
        
        self._collection_ready = True

    def _create_payload_indexes(self) -> None:
        """
        Create payload indexes on common filterable fields.
        
        Qdrant Cloud requires indexes on fields before they can be used in filters.
        This creates indexes on: isActive (boolean), source_id (keyword),
        file_id (keyword), source_details.file_id (keyword)
        """
        # Fields that need indexes for filtering
        filterable_fields = {
            "isActive": "bool",
            "source_id": "keyword",
            "file_id": "keyword",
            "source_details.file_id": "keyword",
        }
        
        for field_name, field_type in filterable_fields.items():
            try:
                payload = {
                    "field_name": field_name,
                    "field_schema": field_type,
                }
                # Correct Qdrant API endpoint: /collections/{name}/index (not /index/{field})
                resp = self._put(
                    f"/collections/{self.collection_name}/index",
                    payload,
                )
                
                # 409 Conflict means index already exists - that's OK
                if resp.status_code == 409:
                    logger.info(
                        "Payload index '%s' already exists on collection '%s'",
                        field_name, self.collection_name,
                    )
                else:
                    resp.raise_for_status()
                    logger.info(
                        "Created payload index '%s' (%s) on collection '%s'",
                        field_name, field_type, self.collection_name,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to create payload index '%s' on collection '%s': %s",
                    field_name, self.collection_name, exc,
                )

    def ensure_collection(self) -> None:
        """Public alias — delegates to _ensure_collection."""
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Idempotently ensure the Qdrant collection exists with the right schema.

        - If it does not exist: create it.
        - If it exists: validate the vector schema matches the requested config
          and raise ValueError on mismatch so callers fail fast.
        """
        try:
            logger.info("Checking Qdrant collections at %s ...", self.base_url)
            resp = self._get("/collections")
            resp.raise_for_status()

            collections = resp.json().get("result", {}).get("collections", [])
            exists = any(c.get("name") == self.collection_name for c in collections)

            if not exists:
                logger.info(
                    "Creating collection '%s' (mode=%s, size=%d)",
                    self.collection_name, self.vector_type, self.vector_size,
                )
                self._create_collection()
                return

            # --- validate existing collection schema ---
            info_resp = self._get(f"/collections/{self.collection_name}")
            info_resp.raise_for_status()

            info          = info_resp.json().get("result", {})
            config        = info.get("config", {})
            params        = config.get("params", {})
            vectors_cfg   = params.get("vectors", {})
            sparse_cfg    = (
                params.get("sparse_vectors")
                or config.get("sparse_vectors")
                or info.get("sparse_vectors")
                or {}
            )

            dense_vector_name: Optional[str] = None
            sparse_vector_name: Optional[str] = None

            # Named dense vector
            has_named_dense  = (
                isinstance(vectors_cfg, dict)
                and "size" not in vectors_cfg
                and "distance" not in vectors_cfg
                and bool(vectors_cfg)
            )
            # Legacy unnamed dense vector
            has_legacy_dense = (
                isinstance(vectors_cfg, dict)
                and "size" in vectors_cfg
                and "distance" in vectors_cfg
            )
            has_sparse = isinstance(sparse_cfg, dict) and bool(sparse_cfg)

            if has_named_dense:
                dense_vector_name = (
                    DENSE_VECTOR_NAME
                    if DENSE_VECTOR_NAME in vectors_cfg
                    else next(iter(vectors_cfg))
                )

            if has_sparse:
                sparse_vector_name = (
                    SPARSE_VECTOR_NAME
                    if SPARSE_VECTOR_NAME in sparse_cfg
                    else next(iter(sparse_cfg))
                )

            if has_named_dense:
                existing_size = vectors_cfg[dense_vector_name].get("size", 0)
                logger.info(
                    "Collection has named dense vector '%s' (size=%d)",
                    dense_vector_name, existing_size,
                )
                if self.vector_type == "dense" and self.vector_size and existing_size != self.vector_size:
                    logger.warning(
                        "Collection '%s' already exists with dense size %d; "
                        "requested size was %d. Reusing the existing collection "
                        "schema and appending new points.",
                        self.collection_name,
                        existing_size,
                        self.vector_size,
                    )
                    self.vector_size = existing_size
            elif has_legacy_dense:
                existing_size = vectors_cfg.get("size", 0)
                logger.info("Collection has unnamed dense vector (size=%d)", existing_size)
                if self.vector_type == "dense" and self.vector_size and existing_size != self.vector_size:
                    logger.warning(
                        "Collection '%s' already exists with dense size %d; "
                        "requested size was %d. Reusing the existing collection "
                        "schema and appending new points.",
                        self.collection_name,
                        existing_size,
                        self.vector_size,
                    )
                    self.vector_size = existing_size

            if self.vector_type == "dense":
                self._dense_vector_payload_name = dense_vector_name
                self._sparse_vector_payload_name = sparse_vector_name

                if has_legacy_dense:
                    self._dense_vector_payload_name = None
                    if self.enable_sparse_vectors and sparse_vector_name is not None:
                        logger.warning(
                            "Collection '%s' uses an unnamed dense vector with sparse metadata. "
                            "Appending dense vectors only for compatibility.",
                            self.collection_name,
                        )
                        self.enable_sparse_vectors = False
                        self._sparse_vector_payload_name = None

                if self.enable_sparse_vectors and not has_sparse:
                    logger.warning(
                        "Collection '%s' already exists without sparse vectors. "
                        "Appending dense vectors only.",
                        self.collection_name,
                    )
                    self.enable_sparse_vectors = False
                    self._sparse_vector_payload_name = None

            if self.vector_type == "sparse":
                self._sparse_vector_payload_name = sparse_vector_name or SPARSE_VECTOR_NAME

            if self.vector_type == "dense" and self.enable_sparse_vectors:
                if not has_named_dense and not has_legacy_dense:
                    raise ValueError(
                        f"Collection '{self.collection_name}' does not expose a dense vector "
                        "schema compatible with appending."
                    )

            if self.vector_type == "sparse" and not has_sparse:
                raise ValueError(
                    f"Collection '{self.collection_name}' lacks sparse vector "
                    f"'{SPARSE_VECTOR_NAME}', but sparse storage was requested."
                )

            points_count = info.get("points_count", 0)
            logger.info(
                "Collection '%s' OK (points=%d)", self.collection_name, points_count
            )
            
            # Ensure payload indexes exist for filterable fields
            # (important for Qdrant Cloud where indexes must exist before filtering)
            self._create_payload_indexes()
            
            self._collection_ready = True

        except requests.exceptions.Timeout:
            logger.error("Timeout connecting to Qdrant at %s", self.base_url)
            raise
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connection error to Qdrant at %s: %s", self.base_url, exc)
            raise
        except Exception as exc:
            logger.error("Failed to ensure collection: %s", exc, exc_info=True)
            raise



    def recreate_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ) -> bool:
        """Drop-and-recreate a collection. Returns True on success."""
        target             = collection_name or self.collection_name
        orig_collection    = self.collection_name
        orig_vector_size   = self.vector_size
        try:
            self.collection_name = target
            if vector_size is not None:
                self.vector_size = vector_size
            self._create_collection()
            logger.info("Recreated collection: %s (size=%s)", target, self.vector_size)
            return True
        except Exception as exc:
            logger.error("Failed to recreate collection %s: %s", target, exc)
            return False
        finally:
            self.collection_name = orig_collection
            self.vector_size     = orig_vector_size

    def _build_file_id_filter(self, file_id: str) -> Dict[str, Any]:
        normalized_file_id = str(file_id or "").strip()
        if not normalized_file_id:
            raise ValueError("file_id is required")

        return {
            "must": [
                {
                    "key": "file_id",
                    "match": {"value": normalized_file_id},
                }
            ]
        }

    def count_by_file_id(self, file_id: str) -> int:
        """Count points belonging to one file."""
        payload = {
            "exact": True,
            "filter": self._build_file_id_filter(file_id),
        }
        resp = self._post(
            f"/collections/{self.collection_name}/points/count",
            payload,
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})
        return int(result.get("count", 0))

    def set_file_active(self, file_id: str, is_active: bool) -> int:
        """
        Set isActive for all chunks belonging to a file.

        Retrieval filters isActive=True, so setting False hides a document
        immediately before physical deletion finishes.
        """
        file_filter = self._build_file_id_filter(file_id)
        matched_count = self.count_by_file_id(file_id)
        if matched_count <= 0:
            logger.info(
                "No points found for file_id=%s in collection=%s",
                file_id,
                self.collection_name,
            )
            return 0

        payload = {
            "payload": {"isActive": bool(is_active)},
            "filter": file_filter,
        }
        resp = self._post(
            f"/collections/{self.collection_name}/points/payload",
            payload,
        )
        resp.raise_for_status()
        logger.info(
            "Set isActive=%s for %d point(s) with file_id=%s",
            is_active,
            matched_count,
            file_id,
        )
        return matched_count

    def delete_by_file_id(self, file_id: str) -> int:
        """Delete every point in the collection that belongs to file_id."""
        file_filter = self._build_file_id_filter(file_id)
        matched_count = self.count_by_file_id(file_id)
        if matched_count <= 0:
            logger.info(
                "No points to delete for file_id=%s in collection=%s",
                file_id,
                self.collection_name,
            )
            return 0

        payload = {"filter": file_filter}
        resp = self._post(
            f"/collections/{self.collection_name}/points/delete",
            payload,
        )
        resp.raise_for_status()
        logger.info(
            "Deleted %d point(s) with file_id=%s from collection=%s",
            matched_count,
            file_id,
            self.collection_name,
        )
        return matched_count

    def get_collection_info(
        self, collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return basic stats for a collection."""
        target = collection_name or self.collection_name
        try:
            resp = self._get(f"/collections/{target}")
            if resp.status_code == 404:
                return {"name": target, "exists": False, "vectors_count": 0, "points_count": 0}
            resp.raise_for_status()
            info = resp.json().get("result", {})
            return {
                "name":          target,
                "exists":        True,
                "vectors_count": info.get("vectors_count", 0),
                "points_count":  info.get("points_count", 0),
                "status":        info.get("status", "unknown"),
            }
        except Exception as exc:
            logger.error("Failed to get collection info for %s: %s", target, exc)
            return {}

    def list_collections(self, prefix: Optional[str] = None) -> List[str]:
        """List Qdrant collections, optionally filtering by prefix."""
        try:
            response = self._get("/collections")
            response.raise_for_status()

            collections = response.json().get("result", {}).get("collections", [])
            names = [
                collection.get("name")
                for collection in collections
                if collection.get("name")
            ]
            if prefix is not None:
                names = [name for name in names if name.startswith(prefix)]
            return names
        except Exception as exc:
            logger.error("Failed to list collections: %s", exc)
            return []

    def _get_collection_details(
        self,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch the raw Qdrant collection details payload."""
        target = collection_name or self.collection_name
        try:
            response = self._get(f"/collections/{target}")
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            return response.json().get("result", {})
        except Exception as exc:
            logger.error("Failed to get collection details for %s: %s", target, exc)
            return {}

    def _build_collection_payload_from_details(
        self,
        collection_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a create-collection payload from existing collection config."""
        config = collection_details.get("config", {})
        params = config.get("params", {})
        payload: Dict[str, Any] = {}

        vectors_config = params.get("vectors")
        sparse_vectors_config = (
            params.get("sparse_vectors")
            or config.get("sparse_vectors")
            or collection_details.get("sparse_vectors")
        )

        if vectors_config is not None:
            payload["vectors"] = vectors_config
        if sparse_vectors_config:
            payload["sparse_vectors"] = sparse_vectors_config
        if not payload:
            payload = self._build_collection_payload()
        return payload

    def _scroll_collection_batch(
        self,
        collection_name: str,
        *,
        limit: int = 256,
        offset: Optional[Any] = None,
    ) -> tuple[List[Dict[str, Any]], Any]:
        """Fetch a single scroll batch including vectors and payload."""
        body: Dict[str, Any] = {
            "with_payload": True,
            "with_vector": True,
            "limit": limit,
        }
        if offset is not None:
            body["offset"] = offset

        response = requests.post(
            f"{self.base_url}/collections/{collection_name}/points/scroll",
            headers=self._headers(),
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json().get("result", {})
        points = result.get("points", []) if isinstance(result, dict) else []
        next_page_offset = (
            result.get("next_page_offset") if isinstance(result, dict) else None
        )
        return points, next_page_offset

    def clone_collection(
        self,
        source_collection: str,
        target_collection: str,
        *,
        batch_size: int = 256,
    ) -> int:
        """Clone one collection into another by copying schema and points."""
        source_details = self._get_collection_details(source_collection)
        if not source_details:
            raise ValueError(
                f"Source collection '{source_collection}' does not exist or could not be read."
            )

        target_info = self.get_collection_info(target_collection)
        if target_info.get("exists"):
            raise ValueError(
                f"Target collection '{target_collection}' already exists."
            )

        create_payload = self._build_collection_payload_from_details(source_details)
        create_response = self._put(
            f"/collections/{target_collection}",
            create_payload,
        )
        create_response.raise_for_status()

        total_copied = 0
        offset: Optional[Any] = None

        while True:
            points, next_page_offset = self._scroll_collection_batch(
                source_collection,
                limit=batch_size,
                offset=offset,
            )
            if not points:
                break

            cloned_points = [
                {
                    "id": point.get("id"),
                    "vector": point.get("vector"),
                    "payload": point.get("payload") or {},
                }
                for point in points
            ]

            upsert_response = self._put(
                f"/collections/{target_collection}/points",
                {"points": cloned_points},
            )
            upsert_response.raise_for_status()

            total_copied += len(cloned_points)
            if next_page_offset is None:
                break
            offset = next_page_offset

        logger.info(
            "Cloned collection %s -> %s (%d point(s))",
            source_collection,
            target_collection,
            total_copied,
        )
        return total_copied

    # -----------------------------------------------------------------------
    # Lazy collection init from actual embeddings
    # -----------------------------------------------------------------------

    def _infer_vector_size_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if self.vector_type != "dense":
            return
        dense = next(
            (
                c.get("embedding_dense")
                for c in chunks
                if isinstance(c.get("embedding_dense"), list) and c["embedding_dense"]
            ),
            None,
        )
        if not dense:
            return
        detected = len(dense)
        if detected <= 0:
            return
        if self.vector_size and self.vector_size != detected:
            logger.warning(
                "Adjusting dense vector size for '%s': %d -> %d",
                self.collection_name, self.vector_size, detected,
            )
        self.vector_size = detected

    def _ensure_collection_for_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if self._collection_ready:
            return
        self._infer_vector_size_from_chunks(chunks)
        self._ensure_collection()

    # -----------------------------------------------------------------------
    # Payload builder
    # -----------------------------------------------------------------------

    def _build_payload(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        if self.payload_builder:
            return self.payload_builder.build_payload(chunk, ingestion_source="request")

        # Legacy/test fallback for callers that do not provide request_data.
        return {
            "chunk_id":    chunk.get("chunk_id"),
            "text":        chunk.get("text"),
            "chunk_index": chunk.get("chunk_index"),
            "token_count": chunk.get("token_count"),
            "char_count":  chunk.get("char_count"),
            "metadata":    chunk.get("metadata", {}),
        }

    # -----------------------------------------------------------------------
    # Vector I/O (public interface)
    # -----------------------------------------------------------------------

    BATCH_SIZE = 25
    UPSERT_MAX_ATTEMPTS = 4
    UPSERT_RETRY_DELAY_SECONDS = 2.0

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Store embedded chunks in Qdrant (async, batched).

        Returns the number of points successfully upserted.
        """
        if not chunks:
            logger.warning("store_chunks called with empty list")
            return 0

        self._ensure_collection_for_chunks(chunks)
        logger.info("Storing %d vector(s) in Qdrant ...", len(chunks))

        points = self._build_points(chunks)
        if not points:
            logger.warning("No valid points to store (all chunks lacked embeddings)")
            return 0

        total_batches = (len(points) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        for i in range(0, len(points), self.BATCH_SIZE):
            batch = points[i : i + self.BATCH_SIZE]
            batch_number = i // self.BATCH_SIZE + 1
            resp = self._upsert_batch_with_retry(
                batch,
                batch_number=batch_number,
                total_batches=total_batches,
            )
            resp.raise_for_status()
            logger.info(
                "Inserted batch %d/%d (%d points)",
                batch_number,
                total_batches,
                len(batch),
            )

        logger.info("Stored %d vector(s) successfully", len(points))
        return len(points)

    def store_chunks_sync(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Synchronous variant of store_chunks for non-async callers.

        Returns the number of points successfully upserted.
        """
        if not chunks:
            logger.warning("store_chunks_sync called with empty list")
            return 0

        self._ensure_collection_for_chunks(chunks)
        logger.info("Storing %d vector(s) in Qdrant (sync) ...", len(chunks))

        points = self._build_points(chunks)
        if not points:
            logger.warning("No valid points to store (all chunks lacked embeddings)")
            return 0

        resp = self._put(
            f"/collections/{self.collection_name}/points",
            {"points": points},
        )
        resp.raise_for_status()
        logger.info("Stored %d vector(s) successfully (sync)", len(points))
        return len(points)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _upsert_batch_with_retry(
        self,
        batch: List[Dict[str, Any]],
        *,
        batch_number: int,
        total_batches: int,
    ) -> requests.Response:
        """Upsert one Qdrant batch with retries for transient network failures."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.UPSERT_MAX_ATTEMPTS + 1):
            try:
                response = self._put(
                    f"/collections/{self.collection_name}/points",
                    {"points": batch},
                )
                if response.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"Qdrant returned HTTP {response.status_code}",
                        response=response,
                    )
                return response
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as exc:
                response = getattr(exc, "response", None)
                is_retryable_http = (
                    isinstance(exc, requests.exceptions.HTTPError)
                    and response is not None
                    and response.status_code >= 500
                )
                if (
                    not is_retryable_http
                    and not isinstance(
                        exc,
                        (
                            requests.exceptions.ConnectionError,
                            requests.exceptions.Timeout,
                        ),
                    )
                ):
                    raise

                last_error = exc
                if attempt >= self.UPSERT_MAX_ATTEMPTS:
                    break

                delay_seconds = self.UPSERT_RETRY_DELAY_SECONDS * attempt
                logger.warning(
                    "Qdrant upsert retry %d/%d for batch %d/%d after error: %s",
                    attempt,
                    self.UPSERT_MAX_ATTEMPTS,
                    batch_number,
                    total_batches,
                    exc,
                )
                time.sleep(delay_seconds)

        assert last_error is not None
        raise last_error

    def _build_points(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert a list of chunks into Qdrant point dicts, skipping invalid ones."""
        points: List[Dict[str, Any]] = []
        for chunk in chunks:
            vectors = self._build_point_vector(chunk)
            if not vectors:
                logger.warning(
                    "Chunk '%s' has no valid embeddings — skipping",
                    chunk.get("chunk_id", "?"),
                )
                continue
            payload = self._build_payload(chunk)
            chunk_id = str(payload.get("chunk_id") or chunk.get("chunk_id") or uuid.uuid4())
            points.append({
                "id":      str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                "vector":  vectors,
                "payload": payload,
            })
        return points
