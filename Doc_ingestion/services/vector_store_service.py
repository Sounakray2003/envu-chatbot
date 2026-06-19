"""
Vector Store Service — backward-compatible shim
------------------------------------------------
All real logic now lives in:
  services/vectore_store/                  (Qdrant factory)
  services/vectore_store/qudrant_vector.py (Qdrant implementation)

This module keeps the original VectorStoreService class name so that any
existing call-sites outside ingestion_service.py continue to work without
changes.  Internally it delegates every call to the backend returned by the
factory's get_vector_store().

For NEW code, import get_vector_store() from services.vectore_store directly.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from services.vectore_store import BaseVectorStore, get_vector_store

logger = logging.getLogger(__name__)

# Re-export defaults so any module that imported them from here keeps working.
DEFAULT_QDRANT_URL     = ""
DEFAULT_QDRANT_TIMEOUT = 300
DENSE_VECTOR_NAME      = "dense"
SPARSE_VECTOR_NAME     = "sparse"


class VectorStoreService:
    """
    Thin shim over the Qdrant vector store factory.

    Accepts the same constructor arguments as the original class and exposes
    the same public methods: ensure_collection, store_chunks, store_chunks_sync,
    delete_collection, recreate_collection, get_collection_info.

    The active backend is selected via vector_store_details.vector_store_id:
      1  -> Internal Qdrant (default)
      2  -> User-provided Qdrant
    """

    def __init__(
        self,
        collection_name: str,
        vector_size: int = 0,
        request_data: Optional[Dict[str, Any]] = None,
        enable_sparse_vectors: bool = True,
        vector_type: str = "dense",
        ensure_collection: bool = True,
        qdrant_url: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.vector_size     = vector_size
        self.request_data    = request_data or {}

        # qdrant_url kwarg is kept for call-sites that pass it explicitly
        # (e.g. tests). Inject it into vector_store_details so the factory sees it.
        if qdrant_url:
            vector_store_details = self.request_data.get("vector_store_details", {})
            vector_store_details["QDRANT_URL"] = qdrant_url
            self.request_data = {**self.request_data, "vector_store_details": vector_store_details}

        self._backend: BaseVectorStore = get_vector_store(
            collection_name=collection_name,
            vector_size=vector_size,
            request_data=self.request_data,
            enable_sparse_vectors=enable_sparse_vectors,
            vector_type=vector_type,
            ensure_collection=ensure_collection,
        )

        logger.info(
            "VectorStoreService shim -> backend=%s | collection=%s",
            self._backend.BACKEND_NAME,
            collection_name,
        )

    # -----------------------------------------------------------------------
    # Delegated public API
    # -----------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return self._backend.BACKEND_NAME

    def ensure_collection(self) -> None:
        self._backend.ensure_collection()

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        return await self._backend.store_chunks(chunks)

    def store_chunks_sync(self, chunks: List[Dict[str, Any]]) -> int:
        return self._backend.store_chunks_sync(chunks)

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        return self._backend.delete_collection(collection_name)

    def list_collections(self, prefix: Optional[str] = None) -> List[str]:
        return self._backend.list_collections(prefix)

    def clone_collection(
        self,
        source_collection: str,
        target_collection: str,
        *,
        batch_size: int = 100,
    ) -> int:
        return self._backend.clone_collection(
            source_collection=source_collection,
            target_collection=target_collection,
            batch_size=batch_size,
        )

    def recreate_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ) -> bool:
        return self._backend.recreate_collection(collection_name, vector_size)

    def get_collection_info(
        self, collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._backend.get_collection_info(collection_name)
