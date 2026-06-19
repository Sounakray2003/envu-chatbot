"""
Qdrant vector store factory and backend exports.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

VECTOR_STORE_INTERNAL_QDRANT = 1
VECTOR_STORE_USER_QDRANT = 2


def _resolve_vector_store_id(
    request_data: Optional[Dict[str, Any]] = None,
) -> int:
    """Resolve vector store ID from request JSON using common frontend shapes."""
    request_data = request_data or {}
    vs_details: Dict[str, Any] = request_data.get("vector_store_details") or {}

    raw_id = (
        vs_details.get("vector_store_id")
        or vs_details.get("vectorStoreId")
        or vs_details.get("id")
        or request_data.get("vector_store_id")
        or request_data.get("vectorStoreId")
        or request_data.get("id")
        or 1
    )
    return int(raw_id)


def _resolve_qdrant_url(
    request_data: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve Qdrant URL from .env first, then request payload fallbacks."""
    request_data = request_data or {}
    vs_details: Dict[str, Any] = request_data.get("vector_store_details") or {}

    return (
        os.getenv("QDRANT_URL")
        or vs_details.get("QDRANT_URL")
        or vs_details.get("qdrant_url")
        or request_data.get("QDRANT_URL")
        or request_data.get("qdrant_url")
        or request_data.get("qdrant_db_url")
    )


def _resolve_qdrant_api_key(
    request_data: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve Qdrant API key from .env first, then request payload fallbacks."""
    request_data = request_data or {}
    vs_details: Dict[str, Any] = request_data.get("vector_store_details") or {}

    return (
        os.getenv("QDRANT_API_KEY")
        or vs_details.get("QDRANT_API_KEY")
        or vs_details.get("qdrant_api_key")
        or request_data.get("QDRANT_API_KEY")
        or request_data.get("qdrant_api_key")
        or request_data.get("qdrantApiKey")
    )


class BaseVectorStore:
    """Shared contract for the Qdrant backends used by ingestion."""

    BACKEND_NAME: str = "base"
    SUPPORTS_SPARSE_NATIVE: bool = False
    SUPPORTS_COLLECTION_CLONE: bool = False
    SUPPORTS_LIST_COLLECTIONS: bool = False

    def ensure_collection(self) -> None:
        """Create or validate the target collection / namespace / index."""
        raise NotImplementedError

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete the logical collection target. Returns True on success."""
        raise NotImplementedError

    def set_file_active(self, file_id: str, is_active: bool) -> int:
        """Update the active flag for all points belonging to one file."""
        raise NotImplementedError

    def delete_by_file_id(self, file_id: str) -> int:
        """Delete all points belonging to one file and return affected count when known."""
        raise NotImplementedError

    def recreate_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ) -> bool:
        """Recreate the logical collection target. Returns True on success."""
        raise NotImplementedError

    def get_collection_info(
        self,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return normalized collection info such as existence and point count."""
        raise NotImplementedError

    async def store_chunks(self, chunks: list[Dict[str, Any]]) -> int:
        """Persist a batch of embedded chunks asynchronously."""
        raise NotImplementedError

    def store_chunks_sync(self, chunks: list[Dict[str, Any]]) -> int:
        """Persist a batch of embedded chunks synchronously."""
        raise NotImplementedError

    def list_collections(self, prefix: Optional[str] = None) -> list[str]:
        """List logical collections when supported by the backend."""
        raise NotImplementedError

    def clone_collection(
        self,
        source_collection: str,
        target_collection: str,
        *,
        batch_size: int = 100,
    ) -> int:
        """Clone one logical collection into another and return copied point count."""
        raise NotImplementedError


class _NotImplementedVectorStore(BaseVectorStore):
    """Placeholder backend for stores that are not implemented yet."""

    def __init__(self, name: str):
        self.BACKEND_NAME = name

    def _raise(self, method: str) -> None:
        raise NotImplementedError(
            f"Vector store backend '{self.BACKEND_NAME}' is not yet implemented "
            f"(called: {method}). Implement the corresponding module in "
            "services/vectore_store/."
        )

    def ensure_collection(self) -> None:
        self._raise("ensure_collection")

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        self._raise("delete_collection")

    def set_file_active(self, file_id: str, is_active: bool) -> int:
        self._raise("set_file_active")

    def delete_by_file_id(self, file_id: str) -> int:
        self._raise("delete_by_file_id")

    def recreate_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ) -> bool:
        self._raise("recreate_collection")

    def get_collection_info(
        self,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._raise("get_collection_info")

    async def store_chunks(self, chunks: list[Dict[str, Any]]) -> int:
        self._raise("store_chunks")

    def store_chunks_sync(self, chunks: list[Dict[str, Any]]) -> int:
        self._raise("store_chunks_sync")

    def list_collections(self, prefix: Optional[str] = None) -> list[str]:
        self._raise("list_collections")

    def clone_collection(
        self,
        source_collection: str,
        target_collection: str,
        *,
        batch_size: int = 100,
    ) -> int:
        self._raise("clone_collection")


def get_vector_store(
    *,
    collection_name: str,
    vector_size: int = 0,
    request_data: Optional[Dict[str, Any]] = None,
    enable_sparse_vectors: bool = True,
    vector_type: str = "dense",
    ensure_collection: bool = True,
) -> BaseVectorStore:
    """
    Instantiate and return the configured Qdrant backend.
    """
    request_data = request_data or {}
    vs_details: Dict[str, Any] = request_data.get("vector_store_details") or {}
    vector_store_id = _resolve_vector_store_id(request_data)

    # Force dense-only embedding by disabling sparse vectors for main_memory and cache_memory
    if collection_name in ("main_memory", "cache_memory"):
        logger.info("Collection is '%s' — programmatically disabling sparse vectors for dense-only RAG.", collection_name)
        enable_sparse_vectors = False

    logger.info(
        "Vector store factory | vector_store_id=%s | collection=%s",
        vector_store_id,
        collection_name,
    )

    if vector_store_id == VECTOR_STORE_INTERNAL_QDRANT:
        from services.vectore_store.qudrant_vector import QdrantVectorStore

        qdrant_url = _resolve_qdrant_url(request_data)
        api_key = _resolve_qdrant_api_key(request_data)
        if not qdrant_url:
            raise ValueError(
                "Internal Qdrant (vector_store_id=1) requires 'QDRANT_URL' "
                "in .env or request_data."
            )

        return QdrantVectorStore(
            collection_name=collection_name,
            vector_size=vector_size,
            request_data=request_data,
            enable_sparse_vectors=enable_sparse_vectors,
            vector_type=vector_type,
            ensure_collection=ensure_collection,
            qdrant_url=qdrant_url,
            api_key=api_key,
        )

    if vector_store_id == VECTOR_STORE_USER_QDRANT:
        from services.vectore_store.qudrant_vector import QdrantVectorStore

        # DEBUG: log all keys present in vs_details so we can confirm the
        # exact field names the frontend is sending.
        logger.info(
            "[USER QDRANT] vector_store_details keys received: %s",
            list(vs_details.keys()),
        )
        logger.info(
            "[USER QDRANT] vector_store_details values (redacted): %s",
            {
                k: (v[:6] + "..." if isinstance(v, str) and len(v) > 6 else v)
                for k, v in vs_details.items()
            },
        )

        qdrant_url = _resolve_qdrant_url(request_data)
        api_key = _resolve_qdrant_api_key(request_data)
        user_collection = (
            vs_details.get("QDRANT_COLLECTION_NAME")
            or vs_details.get("qdrant_collection_name")
            or collection_name
        )

        if not qdrant_url:
            raise ValueError(
                "User-provided Qdrant (vector_store_id=2) requires "
                "'QDRANT_URL' either in .env or in vector_store_details."
            )

        logger.info(
            "Using user-provided Qdrant | url=%s | collection=%s",
            qdrant_url,
            user_collection,
        )
        return QdrantVectorStore(
            collection_name=user_collection,
            vector_size=vector_size,
            request_data=request_data,
            enable_sparse_vectors=enable_sparse_vectors,
            vector_type=vector_type,
            ensure_collection=ensure_collection,
            qdrant_url=qdrant_url,
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown vector_store_id={vector_store_id}. Supported IDs: "
        "1 (Internal Qdrant), 2 (User Qdrant)."
    )


__all__ = [
    "BaseVectorStore",
    "VECTOR_STORE_INTERNAL_QDRANT",
    "VECTOR_STORE_USER_QDRANT",
    "_resolve_vector_store_id",
    "get_vector_store",
]
